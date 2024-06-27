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
from typing import Sequence
from functools import partial
import time
import numpy as np

import jax
import jax.numpy as jnp
import optax
import transformers
from absl import app
from maxdiffusion import (
    FlaxDDPMScheduler,
    FlaxStableDiffusionXLPipeline,
    FlaxAutoencoderKL,
    max_logging,
    max_utils,
    pyconfig,
    mllog_utils,
)

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

from transformers import FlaxCLIPTextModel, FlaxCLIPTextModelWithProjection

from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P, PositionalSharding
from transformers import set_seed

from maxdiffusion.input_pipeline.input_pipeline_interface import (
  make_pokemon_train_iterator
)

from maxdiffusion.maxdiffusion_utils import (
  vae_apply,
  transform_images,
  get_add_time_ids,
  calculate_unet_tflops
)

def get_shaped_batch(config, pipeline):
    """Return the shape of the batch - this is what eval_shape would return for the
    output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078."""

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) -1)
    total_train_batch_size = config.per_device_batch_size * jax.device_count()

    batch_latent_shape = (
        total_train_batch_size,
        pipeline.unet.config.in_channels,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor
    )

    text_embeds_dim = pipeline.unet.config.projection_class_embeddings_input_dim - (
        6 * pipeline.unet.config.addition_time_embed_dim
    )
    text_embeds_shape = (
        total_train_batch_size,
        text_embeds_dim
    )
    input_ids_shape = (
        total_train_batch_size,
        2,
        pipeline.text_encoder.config.max_position_embeddings
    )
    prompt_embeds_shape = (
        total_train_batch_size,
        pipeline.text_encoder.config.max_position_embeddings,
        2048
    )

    shaped_batch = {}
    shaped_batch["pixel_values"] = jax.ShapeDtypeStruct(batch_latent_shape, jnp.float32)
    shaped_batch["prompt_embeds"] = jax.ShapeDtypeStruct(prompt_embeds_shape, jnp.float32)
    shaped_batch["text_embeds"] = jax.ShapeDtypeStruct(text_embeds_shape, jnp.float32)
    shaped_batch["input_ids"] = jax.ShapeDtypeStruct(input_ids_shape, jnp.int32)
    return shaped_batch

def encode_xl(input_ids, text_encoders, text_encoder_params):
  te_1_inputs = input_ids[:, 0, :]
  te_2_inputs = input_ids[:, 1, :]

  prompt_embeds = text_encoders[0](
    te_1_inputs, params=text_encoder_params[0], output_hidden_states=True
  )
  prompt_embeds = prompt_embeds["hidden_states"][-2]

  prompt_embeds_2_out = text_encoders[1](
    te_2_inputs, params=text_encoder_params[1], output_hidden_states=True
  )
  prompt_embeds_2 = prompt_embeds_2_out["hidden_states"][-2]
  text_embeds = prompt_embeds_2_out["text_embeds"]
  prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)

  return prompt_embeds, text_embeds

def tokenize_captions_xl(examples, caption_column, tokenizers, p_encode=None):
  inputs = []
  captions = list(examples[caption_column])
  for _tokenizer in tokenizers:
     text_inputs = _tokenizer(
        captions,
        padding="max_length",
        max_length=_tokenizer.model_max_length,
        truncation=True,
        return_tensors="np"
     )
     inputs.append(text_inputs.input_ids)
  inputs = np.stack(inputs,axis=1)

  if p_encode:
    prompt_embeds, text_embeds = p_encode(inputs)
    examples["prompt_embeds"] = prompt_embeds
    examples["text_embeds"] = text_embeds
  examples["input_ids"] = inputs

  return examples

def train_step(unet_state, batch, train_rng, noise_scheduler, noise_scheduler_state, config):
  _, gen_dummy_rng = jax.random.split(train_rng)
  sample_rng, timestep_bias_rng, new_train_rng = jax.random.split(gen_dummy_rng, 3)
  def compute_loss(unet_params):
    if config.cache_latents_text_encoder_outputs:
      latents = batch["pixel_values"]
      prompt_embeds = batch["prompt_embeds"]
      text_embeds = batch["text_embeds"]
    else:
      raise ValueError("cache_latents_text_encoder_outputs = False currently not supported!")

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

    # TODO : @jfacevedo - support cropping.
    add_time_ids = get_add_time_ids(
      (config.resolution, config.resolution),
      (0, 0),
      (config.resolution, config.resolution),
      prompt_embeds.shape[0],
      dtype=prompt_embeds.dtype
    )
    unet_added_conditions = {"time_ids" : add_time_ids, "text_embeds" : text_embeds}

    # Predict the noise residual and compute loss
    model_pred = unet_state.apply_fn(
      {"params": unet_params},
      noisy_latents,
      timesteps,
      prompt_embeds,
      added_cond_kwargs=unet_added_conditions,
      train=True
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
    pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
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
    params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)

    noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(config.pretrained_model_name_or_path,
        revision=config.revision, subfolder="scheduler", dtype=jnp.float32)

    pipeline.scheduler = noise_scheduler
    params["scheduler"] = noise_scheduler_state

    sharding = PositionalSharding(devices_array).replicate()
    partial_device_put_replicated = partial(max_utils.device_put_replicated, sharding=sharding)
    params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])
    params["text_encoder_2"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder_2"])
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
    max_logging.log("Creating states...")
    s = time.time()
    (unet_state,
    unet_state_mesh_shardings,
    _, _) = max_utils.get_states(mesh,tx, rng, config,pipeline, params["unet"], params["vae"], training=True)
    max_logging.log(f"Create state time: {(time.time() - s)}")
    max_logging.log("Calculating training tflops...")
    s = time.time()
    per_device_tflops = calculate_unet_tflops(config, pipeline, rng, train=True)
    max_logging.log(f"Calculate training tflops time: {(time.time() - s)}")
    max_logging.log(f"Per train step, estimated total TFLOPs will be {per_device_tflops:.2f}")
    max_logging.log(f"Preparing dataset: {config.dataset_name}")
    s = time.time()
    if config.dataset_name == "diffusers/pokemon-gpt4-captions":
        p_encode = None
        p_vae_apply = None
        if config.cache_latents_text_encoder_outputs:
          p_encode = jax.jit(partial(encode_xl,
                              text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2],
                              text_encoder_params=[params["text_encoder"], params["text_encoder_2"]]))
          p_vae_apply = jax.jit(partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
        else:
           raise ValueError("cache_latents_text_encoder_outputs = False currently not supported!")

        tokenize_fn = partial(tokenize_captions_xl,
                          caption_column=config.caption_column,
                          tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2],
                          p_encode=p_encode)
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
        raise ValueError(f"{config.dataset_name} is currently not supported in this pipeline.")
    max_logging.log(f"Dataset prepare time: {time.time() - s}")
    # Initialize our training
    _, train_rngs = jax.random.split(rng)

    num_model_parameters = max_utils.calculate_num_params_from_pytree(unet_state.params)
    max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")

    my_data_sharding = {'input_ids': data_sharding,
                        'pixel_values': data_sharding,
                        'prompt_embeds' : data_sharding,
                        'text_embeds' : data_sharding}

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      p_train_step = jax.jit(
          partial(train_step, noise_scheduler=noise_scheduler, noise_scheduler_state=noise_scheduler_state, config=config),
          in_shardings=(unet_state_mesh_shardings, my_data_sharding, None),
          out_shardings=(unet_state_mesh_shardings, None, None),
          donate_argnums=(0,)
      )
      max_logging.log("Precompiling...")
      s = time.time()
      dummy_batch = get_shaped_batch(config, pipeline)
      p_train_step = p_train_step.lower(unet_state, dummy_batch, train_rngs)
      p_train_step = p_train_step.compile()
      max_logging.log(f"Compile time: {(time.time() - s )}")

    # clean up unused models in the training loop.
    if config.cache_latents_text_encoder_outputs:
       pipeline.vae = None
       params["vae"] = None
       pipeline.text_encoder = None
       params["text_encoder"] = None
       pipeline.text_encoder_2 = None
       params["text_encoder_2"] = None

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
           checkpoint_name = f"samples-{samples_count}"
           save_checkpoint(pipeline.unet.save_pretrained,
                           get_params_to_save(unet_state.params),
                           config,
                           os.path.join(config.checkpoint_dir, checkpoint_name))

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        checkpoint_name = "final"
        # Restore vae and text encoder if we cached latents and encoder outputs.
        if config.cache_latents_text_encoder_outputs:
            text_encoder = FlaxCLIPTextModel.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="text_encoder", dtype=weight_dtype, from_pt=config.from_pt
            )
            text_encoder_2 = FlaxCLIPTextModelWithProjection.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="text_encoder_2", dtype=weight_dtype, from_pt=config.from_pt
            )
            vae, vae_params = FlaxAutoencoderKL.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="vae", dtype=weight_dtype, from_pt=config.from_pt
            )
            pipeline.vae = vae
            pipeline.text_encoder = text_encoder
            pipeline.text_encoder_2 = text_encoder_2
            params["text_encoder"] = text_encoder.params
            params["text_encoder_2"] = text_encoder_2.params
            params["vae"] = vae_params

        pipeline = FlaxStableDiffusionXLPipeline(
            text_encoder=pipeline.text_encoder,
            text_encoder_2=pipeline.text_encoder_2,
            vae=pipeline.vae,
            unet=pipeline.unet,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            scheduler=noise_scheduler,
        )
        params={
            "text_encoder": get_params_to_save(params["text_encoder"]),
            "text_encoder_2" : get_params_to_save(params["text_encoder_2"]),
            "vae": get_params_to_save(params["vae"]),
            "unet": get_params_to_save(unet_state.params),
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
