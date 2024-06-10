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
import functools
import json
import os
import time
from typing import Sequence

import numpy as np

import jax
import jax.numpy as jnp
from maxdiffusion.max_utils import (
  create_device_mesh,
  get_dtype,
  get_states,
  device_put_replicated,
  get_flash_block_sizes
)
from maxdiffusion import pyconfig
from absl import app
from maxdiffusion import (
  FlaxStableDiffusionPipeline,
  FlaxDDIMScheduler
)

from maxdiffusion.maxdiffusion_utils import rescale_noise_cfg
from flax.linen import partitioning as nn_partitioning
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.sharding import Mesh, PositionalSharding
from maxdiffusion.image_processor import VaeImageProcessor

cc.set_cache_dir(os.path.expanduser("~/jax_cache"))

NUM_ITER = 5

def loop_body(step, args, model, pipeline, prompt_embeds, guidance_scale, guidance_rescale):
    latents, scheduler_state, state = args
    latents_input = jnp.concatenate([latents] * 2)

    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    timestep = jnp.broadcast_to(t, latents_input.shape[0])

    latents_input = pipeline.scheduler.scale_model_input(scheduler_state, latents_input, t)

    noise_pred = model.apply(
        {"params" : state.params},
        jnp.array(latents_input),
        jnp.array(timestep, dtype=jnp.int32),
        encoder_hidden_states=prompt_embeds
    ).sample

    noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
    noise_pred = rescale_noise_cfg(noise_pred, noise_prediction_text, guidance_rescale=guidance_rescale)

    latents, scheduler_state = pipeline.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()

    return latents, scheduler_state, state

def tokenize(prompt, tokenizer):
    """Tokenizes prompt."""
    return tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np"
    ).input_ids

def get_unet_inputs(rng, config, batch_size, pipeline, params):
    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    prompt_ids = [config.prompt] * batch_size
    prompt_ids = tokenize(prompt_ids, pipeline.tokenizer)
    negative_prompt_ids = [config.negative_prompt] * batch_size
    negative_prompt_ids = tokenize(negative_prompt_ids, pipeline.tokenizer)
    guidance_scale = config.guidance_scale
    guidance_rescale = config.guidance_rescale
    num_inference_steps = config.num_inference_steps

    prompt_embeds = pipeline.text_encoder(prompt_ids, params=params["text_encoder"])[0]
    negative_prompt_embeds = pipeline.text_encoder(negative_prompt_ids, params=params["text_encoder"])[0]
    context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])
    guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)
    guidance_rescale = jnp.array([guidance_rescale], dtype=jnp.float32)

    batch_size = prompt_ids.shape[0]
    latents_shape = (
        batch_size,
        pipeline.unet.config.in_channels,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor,
    )
    latents = jax.random.normal(rng, shape=latents_shape, dtype=jnp.float32)

    scheduler_state = pipeline.scheduler.set_timesteps(
        params["scheduler"],
        num_inference_steps=num_inference_steps,
        shape=latents.shape
    )
    latents = latents * params["scheduler"].init_noise_sigma

    return latents, context, guidance_scale, guidance_rescale, scheduler_state

def vae_decode(latents, state, pipeline):
    latents = 1 / pipeline.vae.config.scaling_factor * latents
    image = pipeline.vae.apply(
        {"params" : state.params},
        latents,
        method=pipeline.vae.decode
    ).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image

def run(config):
    rng = jax.random.PRNGKey(config.seed)
    # Setup Mesh
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    batch_size = jax.device_count() * config.per_device_batch_size

    weight_dtype = get_dtype(config)
    flash_block_sizes = get_flash_block_sizes(config)
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
        safety_checker=None,
        feature_extractor=None,
        split_head_dim=config.split_head_dim,
        norm_num_groups=config.norm_num_groups,
        from_pt=config.from_pt,
        attention_kernel=config.attention,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh
    )
    scheduler, scheduler_state = FlaxDDIMScheduler.from_pretrained(
        config.pretrained_model_name_or_path, revision=config.revision, subfolder="scheduler", dtype=jnp.float32
    )
    pipeline.scheduler = scheduler
    params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)
    params["scheduler"] = scheduler_state

    # Text encoder params
    sharding = PositionalSharding(mesh.devices).replicate()
    partial_device_put_replicated = functools.partial(device_put_replicated, sharding=sharding)
    params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])

    (unet_state,
     unet_state_mesh_shardings,
     vae_state,
     vae_state_mesh_shardings) = get_states(mesh, None, rng, config, pipeline, params["unet"], params["vae"], training=False)
    del params["vae"]
    del params["unet"]

    def run_inference(unet_state, vae_state, params, rng, config, batch_size, pipeline):

        (latents,
        context,
        guidance_scale,
        guidance_rescale,
        scheduler_state) = get_unet_inputs(rng, config, batch_size, pipeline, params)

        loop_body_p = functools.partial(loop_body, model=pipeline.unet,
                                        pipeline=pipeline,
                                        prompt_embeds=context,
                                        guidance_scale=guidance_scale,
                                        guidance_rescale=guidance_rescale)

        vae_decode_p = functools.partial(vae_decode, pipeline=pipeline)

        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            latents, _, _ = jax.lax.fori_loop(0, config.num_inference_steps,
                                            loop_body_p, (latents, scheduler_state, unet_state))
            image = vae_decode_p(latents, vae_state)
            return image

    p_run_inference = jax.jit(
        functools.partial(run_inference, rng=rng, config=config, batch_size=batch_size, pipeline=pipeline),
        in_shardings=(unet_state_mesh_shardings, vae_state_mesh_shardings, None),
        out_shardings=None
    )

    metrics_dict = {}
    for iter in range(NUM_ITER):
        if iter == 0:
            s = time.time()
            p_run_inference(unet_state, vae_state, params).block_until_ready()
            metrics_dict["compile_time"] = time.time() - s
        else:
            s = time.time()
            images = p_run_inference(unet_state, vae_state, params).block_until_ready()
            images.block_until_ready()
            inference_time = time.time() - s
            metrics_dict[f"inference_time_{iter}"] = inference_time

    dimensions_dict = {}
    current_dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dimensions_dict["date"] = current_dt

    for dim in config.get_keys():
        val = config.get(dim)
        if isinstance(val, str):
            dimensions_dict[str(dim)] = str(config.get(dim))
        elif isinstance(val, int) or isinstance(val, float):
            metrics_dict[dim] = val
        else:
            dimensions_dict[str(dim)] = str(val)

    final_dict = {}
    final_dict["metrics"] = metrics_dict
    final_dict["dimensions"] = dimensions_dict

    with open("metrics.json", 'w') as f:
        f.write(json.dumps(final_dict))

    numpy_images = np.array(images)
    images = VaeImageProcessor.numpy_to_pil(numpy_images)
    for i, image in enumerate(images):
        image.save(f"image_{i}.png")

    return images

def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    run(pyconfig.config)

if __name__ == "__main__":
    app.run(main)
