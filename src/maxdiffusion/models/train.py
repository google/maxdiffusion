"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import datetime
import logging
import os
import time
from typing import Sequence
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import optax
import transformers
from absl import app
from maxdiffusion import (
    FlaxDDPMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxAutoencoderKL,
    FlaxUNet2DConditionModel,
    max_logging,
    max_utils,
    pyconfig,
    mllog_utils,
)
from maxdiffusion.maxdiffusion_utils import vae_apply, transform_images, calculate_unet_tflops

from maxdiffusion.train_utils import (
    get_first_step,
    load_next_batch,
    validate_train_config,
    record_scalar_metrics,
    write_metrics,
    get_params_to_save,
    compute_snr,
    generate_timestep_weights,
    save_checkpoint
)

from transformers import FlaxCLIPTextModel

from maxdiffusion.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P, PositionalSharding
from transformers import CLIPImageProcessor, set_seed

from maxdiffusion.input_pipeline.input_pipeline_interface import (
  make_pokemon_train_iterator,
  make_laion400m_train_iterator
)

def get_shaped_batch(config, pipeline):
  """Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078."""
  vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
  total_train_batch_size = config.per_device_batch_size * jax.device_count()
  if config.cache_latents_text_encoder_outputs:
    batch_image_shape = (total_train_batch_size, 4,
            config.resolution // vae_scale_factor,
            config.resolution // vae_scale_factor)
    #bs, encoder_input, seq_length
    batch_ids_shape = (total_train_batch_size,
                       pipeline.text_encoder.config.max_position_embeddings,
                       pipeline.text_encoder.config.hidden_size)
  else:
    batch_image_shape = (total_train_batch_size, 3, config.resolution, config.resolution)
    batch_ids_shape = (total_train_batch_size, pipeline.text_encoder.config.max_position_embeddings)
  shaped_batch = {}
  shaped_batch["pixel_values"] = jax.ShapeDtypeStruct(batch_image_shape, jnp.float32)
  shaped_batch["input_ids"] = jax.ShapeDtypeStruct(batch_ids_shape, jnp.float32)
  return shaped_batch

def encode(input_ids, text_encoder, text_encoder_params):
  return text_encoder(
    input_ids,
    params=text_encoder_params,
    train=False
  )[0]

def tokenize_captions(examples, caption_column, tokenizer, p_encode=None):
    captions = list(examples[caption_column])
    text_inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True
    )

    if p_encode:
        encoder_hidden_states = p_encode(np.stack(text_inputs.input_ids))
        examples["input_ids"] = encoder_hidden_states
    else:
        examples["input_ids"] = text_inputs.input_ids
    return examples

def train(config):
    rng = jax.random.PRNGKey(config.seed)

    writer = max_utils.initialize_summary_writer(config)
    if config.dataset_name is None and config.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # Setup Mesh
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if jax.process_index() == 0:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    data_sharding = jax.sharding.NamedSharding(mesh,P(*config.data_sharding))

    total_train_batch_size = config.per_device_batch_size * jax.device_count()

    weight_dtype = max_utils.get_dtype(config)
    flash_block_sizes = max_utils.get_flash_block_sizes(config)
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision,
        dtype=weight_dtype,
        safety_checker=None,
        feature_extractor=None,
        from_pt=config.from_pt,
        split_head_dim=config.split_head_dim,
        norm_num_groups=config.norm_num_groups,
        attention_kernel=config.attention,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
    )
    if len(config.unet_checkpoint) > 0:
        unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
                config.unet_checkpoint,
                split_head_dim=config.split_head_dim,
                norm_num_groups=config.norm_num_groups,
                attention_kernel=config.attention,
                flash_block_sizes=flash_block_sizes,
                mesh=mesh
            )
        params["unet"] = unet_params
        pipeline.unet = unet
    params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)

    noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(config.pretrained_model_name_or_path,
        revision=config.revision, subfolder="scheduler", dtype=jnp.float32)

    pipeline.scheduler = noise_scheduler
    params["scheduler"] = noise_scheduler_state

    sharding = PositionalSharding(devices_array).replicate()
    partial_device_put_replicated = partial(max_utils.device_put_replicated, sharding=sharding)
    params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])

    # Optimization
    if config.scale_lr:
        config.learning_rate = config.learning_rate * total_train_batch_size

    learning_rate_scheduler = max_utils.create_learning_rate_schedule(config)

    tx = optax.adamw(
        learning_rate=learning_rate_scheduler,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
    )

    (unet_state,
    unet_state_mesh_shardings,
    vae_state, vae_state_mesh_shardings) = max_utils.get_states(mesh,
                                                                tx, rng, config,
                                                                pipeline, params["unet"],
                                                                params["vae"], training=True)

    per_device_tflops = calculate_unet_tflops(config, pipeline, rng, train=True)
    max_logging.log(f"Per train step, estimated total TFLOPs will be {per_device_tflops:.2f}")

    if config.dataset_name == "diffusers/pokemon-gpt4-captions":
        p_encode = None
        p_vae_apply = None
        if config.cache_latents_text_encoder_outputs:
            p_encode = jax.jit(partial(encode,text_encoder=pipeline.text_encoder,text_encoder_params=params["text_encoder"]))
            p_vae_apply = jax.jit(partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
        tokenize_fn = partial(tokenize_captions,caption_column=config.caption_column, tokenizer=pipeline.tokenizer, p_encode=p_encode)
        image_transforms_fn = partial(transform_images,
                                      image_column=config.image_column,
                                      image_resolution=config.resolution,
                                      rng=rng,
                                      global_batch_size=total_train_batch_size,
                                      p_vae_apply=p_vae_apply)

        data_iterator = make_pokemon_train_iterator(
           config,
           mesh,
           total_train_batch_size,
           tokenize_fn,
           image_transforms_fn
        )
    else:
        data_iterator = make_laion400m_train_iterator(
           config, mesh, total_train_batch_size
        )

    # Initialize our training
    _, train_rngs = jax.random.split(rng)

    def train_step(unet_state, vae_state, batch, train_rng, cache_latents_text_encoder_outputs):
        _, gen_dummy_rng = jax.random.split(train_rng)
        sample_rng, timestep_bias_rng, new_train_rng = jax.random.split(gen_dummy_rng, 3)
        def compute_loss(unet_params):

            if cache_latents_text_encoder_outputs:
               latents = batch["pixel_values"]
               encoder_hidden_states = batch["input_ids"]
            else:
                # Convert images to latent space
                vae_outputs = pipeline.vae.apply(
                {"params": vae_state.params}, batch["pixel_values"], deterministic=True, method=pipeline.vae.encode
                )
                latents = vae_outputs.latent_dist.sample(sample_rng)
                # (NHWC) -> (NCHW)
                latents = jnp.transpose(latents, (0, 3, 1, 2))
                latents = latents * pipeline.vae.config.scaling_factor

                # Get the text embedding for conditioning
                encoder_hidden_states = encode(batch["input_ids"], pipeline.text_encoder, params["text_encoder"])

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            if config.timestep_bias["strategy"] == "none":
                timesteps = jax.random.randint(
                    timestep_rng,
                    (bsz,),
                    0,
                    noise_scheduler.config.num_train_timesteps,
                )
            else:
                weights = generate_timestep_weights(config, noise_scheduler.config.num_train_timesteps)
                timesteps = jax.random.categorical(timestep_bias_rng, logits=jnp.log(weights), shape=(bsz,))

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

            # TODO - laion dataset was prepared with an extra dim.
            # need to preprocess the dataset with dim removed.
            if len(encoder_hidden_states.shape) == 4:
                encoder_hidden_states = jnp.squeeze(encoder_hidden_states)

            # Predict the noise residual and compute loss
            model_pred = pipeline.unet.apply(
                {"params": unet_params}, noisy_latents, timesteps, encoder_hidden_states, train=True
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            loss = (target - model_pred) ** 2

            # snr
            if config.snr_gamma > 0:
                snr = jnp.array(compute_snr(timesteps, noise_scheduler_state))
                snr_loss_weights = jnp.where(snr < config.snr_gamma, snr, jnp.ones_like(snr) * config.snr_gamma)
                if noise_scheduler.config.prediction_type == "epsilon":
                    snr_loss_weights = snr_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    snr_loss_weights = snr_loss_weights / (snr + 1)
                loss = loss * snr_loss_weights[:, None, None, None]

            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(unet_state.params)

        new_state = unet_state.apply_gradients(grads=grad)
        metrics = {'scalar' : {'learning/loss' : loss}, 'scalars': {}}

        return new_state, metrics, new_train_rng

    num_model_parameters = max_utils.calculate_num_params_from_pytree(unet_state.params)
    max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")

    my_data_sharding = {'input_ids': data_sharding, 'pixel_values': data_sharding}

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        p_train_step = jax.jit(
            partial(train_step, cache_latents_text_encoder_outputs=config.cache_latents_text_encoder_outputs),
            in_shardings=(unet_state_mesh_shardings, vae_state_mesh_shardings, my_data_sharding, None),
            out_shardings=(unet_state_mesh_shardings, None, None),
            donate_argnums=(0,)
        )
        max_logging.log("Precompiling...")
        s = time.time()
        dummy_batch = get_shaped_batch(config, pipeline)
        p_train_step = p_train_step.lower(unet_state,
                                          vae_state,
                                          dummy_batch,
                                          train_rngs)
        p_train_step = p_train_step.compile()
        max_logging.log(f"Compile time: {(time.time() - s )}")

    # Train!
    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
    max_utils.add_config_to_summary_writer(config, writer)

    if jax.process_index() == 0:
        max_logging.log("***** Running training *****")
        max_logging.log(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
        max_logging.log(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
        max_logging.log(f"  Total optimization steps = {config.max_train_steps}")

    last_step_completion = datetime.datetime.now()

    local_metrics_file = open(config.metrics_file, 'a', encoding="utf8") if config.metrics_file else None
    running_gcs_metrics = [] if config.gcs_metrics else None
    example_batch = None

    first_profiling_step = config.skip_first_n_steps_for_profiler
    if config.enable_profiler and first_profiling_step >= config.max_train_steps:
       raise ValueError("Profiling requested but initial profiling step set past training final step")
    last_profiling_step = np.clip(first_profiling_step + config.profiler_steps -1, first_profiling_step, config.max_train_steps - 1)
    # ======================== Training ================================
    # train

    start_step = get_first_step(unet_state)
    mllog_utils.train_init_print(config)
    mllog_utils.train_init_stop(config)
    mllog_utils.train_run_start(config)
    mllog_utils.train_step_start(config, start_step)
    for step in np.arange(start_step, config.max_train_steps+1):
        example_batch = load_next_batch(data_iterator, example_batch, config)
        unet_state, train_metric, train_rngs = p_train_step(unet_state,
                                                            vae_state,
                                                            example_batch,
                                                            train_rngs)
        samples_count = total_train_batch_size * (step + 1)
        new_time = datetime.datetime.now()
        record_scalar_metrics(train_metric, new_time - last_step_completion, per_device_tflops, learning_rate_scheduler(step))
        if config.write_metrics:
            write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, config)
        last_step_completion = new_time
        # Start profiling at end of first step to avoid compilation.
        # Move before for loop to include.
        if step == first_profiling_step:
            max_utils.activate_profiler(config)
        if step == last_profiling_step:
            max_utils.deactivate_profiler(config)

        mllog_utils.maybe_train_step_log(config, start_step, step, train_metric)
        if step != 0 and config.checkpoint_every != -1 and samples_count % config.checkpoint_every == 0:
            checkpoint_name = f"UNET-samples-{samples_count}"
            save_checkpoint(pipeline.unet.save_pretrained,
                            get_params_to_save(unet_state.params),
                            config,
                            os.path.join(config.checkpoint_dir, checkpoint_name))

    # save the last checkpoint
    if jax.process_index() == 0:
        checkpoint_name = "final"
        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker", from_pt=True
        )
        # Restore vae and text encoder if we cached latents and encoder outputs.
        if config.cache_latents_text_encoder_outputs:
            text_encoder = FlaxCLIPTextModel.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="text_encoder", dtype=weight_dtype, from_pt=config.from_pt
            )
            vae, vae_params = FlaxAutoencoderKL.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="vae", dtype=weight_dtype, from_pt=config.from_pt
            )
            pipeline.vae = vae
            pipeline.text_encoder = text_encoder
            params["text_encoder"] = text_encoder.params
            params["vae"] = vae_params

        pipeline = FlaxStableDiffusionPipeline(
            text_encoder=pipeline.text_encoder,
            vae=pipeline.vae,
            unet=pipeline.unet,
            tokenizer=pipeline.tokenizer,
            scheduler=noise_scheduler,
            safety_checker=safety_checker,
            feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        )

        params = {
            "text_encoder": get_params_to_save(params["text_encoder"]),
            "vae": get_params_to_save(params["vae"]),
            "unet": get_params_to_save(unet_state.params),
            "safety_checker": safety_checker.params,
        }
        save_checkpoint(pipeline.save_pretrained, params, config, os.path.join(config.checkpoint_dir, checkpoint_name))

    max_utils.close_summary_writer(writer)

def main(argv: Sequence[str]) -> None:
    max_logging.log(f"Found {jax.device_count()} devices.")
    pyconfig.initialize(argv)
    config = pyconfig.config
    if len(config.cache_dir) > 0:
        jax.config.update("jax_compilation_cache_dir", config.cache_dir)
    mllog_utils.train_init_start(config)
    validate_train_config(config)
    train(config)
if __name__ == "__main__":
    app.run(main)
