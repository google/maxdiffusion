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
import json
import os
import time
from typing import Sequence
from absl import app

import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax.experimental.compilation_cache import compilation_cache as cc
from maxdiffusion.utils import load_image
from PIL import Image
from maxdiffusion import pyconfig
from maxdiffusion import FlaxStableDiffusionXLControlNetPipeline, FlaxControlNetModel
from maxdiffusion.max_utils import (
  get_dtype
)
import cv2

NUM_ITER = 5

cc.set_cache_dir(os.path.expanduser("~/jax_cache"))

def create_key(seed=0):
  return jax.random.PRNGKey(seed)

def run(config):
  rng = jax.random.PRNGKey(config.seed)
  weight_dtype = get_dtype(config)

  prompts = config.prompt
  negative_prompts = config.negative_prompt
  controlnet_conditioning_scale = config.controlnet_conditioning_scale

  image = load_image(config.controlnet_image)

  image = np.array(image)
  image = cv2.Canny(image, 100, 200)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  image = Image.fromarray(image)

  controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
    config.controlnet_model_name_or_path,
    from_pt=config.controlnet_from_pt,
    dtype=weight_dtype
  )

  pipe, params = FlaxStableDiffusionXLControlNetPipeline.from_pretrained(
    config.pretrained_model_name_or_path,
    controlnet=controlnet,
    revision=config.revision,
    dtype=weight_dtype
  )

  scheduler_state = params.pop("scheduler")
  params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
  params["scheduler"] = scheduler_state

  params["controlnet"] = controlnet_params

  num_samples = jax.device_count() * config.per_device_batch_size
  rng = jax.random.split(rng, jax.device_count())

  prompt_ids = pipe.prepare_text_inputs([prompts] * num_samples)
  negative_prompt_ids = pipe.prepare_text_inputs([negative_prompts] * num_samples)
  processed_image = pipe.prepare_image_inputs([image] * num_samples)
  p_params = replicate(params)
  prompt_ids = shard(prompt_ids)
  negative_prompt_ids = shard(negative_prompt_ids)
  processed_image = shard(processed_image)

  metrics_dict = {}
  for iter in range(NUM_ITER):
    if iter == 0:
      s = time.time()
      output = pipe(
      prompt_ids=prompt_ids,
      image=processed_image,
      params=p_params,
      prng_seed=rng,
      num_inference_steps=config.num_inference_steps,
      neg_prompt_ids=negative_prompt_ids,
      controlnet_conditioning_scale=controlnet_conditioning_scale,
      jit=True,
      ).images

      metrics_dict["compile_time"] = time.time() - s
    else:
      s = time.time()
      output = pipe(
      prompt_ids=prompt_ids,
      image=processed_image,
      params=p_params,
      prng_seed=rng,
      num_inference_steps=config.num_inference_steps,
      neg_prompt_ids=negative_prompt_ids,
      controlnet_conditioning_scale=controlnet_conditioning_scale,
      jit=True,
      ).images

      inference_time = time.time() - s
      metrics_dict[f"inference_time_{iter}"] = inference_time

  dimensions_dict = {}
  current_dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  dimensions_dict["date"] = current_dt

  for dim in config.get_keys():
    val = config.get(dim)
    if isinstance(val, str):
      if dim == "model_name":
        dimensions_dict[dim] = "ControlNet" + str(config.get(dim))
      else:
        dimensions_dict[dim] = str(config.get(dim))
    elif isinstance(val, int) or isinstance(val, float):
      metrics_dict[dim] = val
    else:
      dimensions_dict[dim] = str(val)

  final_dict = {}
  final_dict["metrics"] = metrics_dict
  final_dict["dimensions"] = dimensions_dict

  with open("metrics.json", 'w') as f:
    f.write(json.dumps(final_dict))

  output_images = pipe.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
  output_images[0].save("generated_image.png")
  return output_images[0]

def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    run(pyconfig.config)

if __name__ == "__main__":
    app.run(main)
