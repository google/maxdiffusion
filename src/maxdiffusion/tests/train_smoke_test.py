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

""" Smoke test """
import os
import pathlib
import shutil
import unittest
from maxdiffusion.models.train import main as train_main
from maxdiffusion.train_sdxl import main as train_sdxl_main
from ..import pyconfig
from maxdiffusion.generate import run as generate_run
from maxdiffusion.generate_sdxl import run as generate_run_xl
from absl.testing import absltest

from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

HOME_DIR = pathlib.Path.home()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def cleanup(output_dir):
  shutil.rmtree(output_dir)

class Train(unittest.TestCase):
  """Smoke test."""
  def setUp(self):
    Train.dummy_data = {}

  def test_sdxl_config(self):
    output_dir="train-smoke-test"
    run_name="sdxl_train_smoke_test"
    train_sdxl_main([None,os.path.join(THIS_DIR,'..','configs','base_xl.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
      "revision=refs/pr/95","dtype=bfloat16",f"run_name={run_name}",
      "max_train_steps=21","dataset_name=diffusers/pokemon-gpt4-captions",
      "resolution=1024","per_device_batch_size=1","snr_gamma=5.0",
      'timestep_bias={"strategy" : "later", "multiplier" : 2.0, "portion" : 0.25}',
      "base_output_directory=gs://maxdiffusion-tests", f"output_dir={output_dir}",
      "cache_dir=gs://jfacevedo-maxdiffusion/cache_dir"])

    img_url = os.path.join(THIS_DIR,'images','test_sdxl.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_xl.yml'),
      f"pretrained_model_name_or_path={output_dir}/{run_name}/checkpoints/final",
      f"run_name={run_name}",
      "revision=main","dtype=bfloat16","resolution=1024",
      "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=9",
      "num_inference_steps=20","seed=47","per_device_batch_size=1",
      "split_head_dim=False", "cache_dir=gs://jfacevedo-maxdiffusion/cache_dir"])

    images = generate_run_xl(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

    cleanup(output_dir)

  def test_sd21_config(self):
    output_dir="train-smoke-test"
    run_name="sd2.1_smoke_test"
    train_main([None,os.path.join(THIS_DIR,'..','configs','base21.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
      "revision=bf16","dtype=bfloat16",f"run_name={run_name}",
      "max_train_steps=21","dataset_name=diffusers/pokemon-gpt4-captions",
      "resolution=768","per_device_batch_size=1",
      "base_output_directory=gs://maxdiffusion-tests", f"output_dir={output_dir}",
      "cache_dir=gs://jfacevedo-maxdiffusion/cache_dir"])

    img_url = os.path.join(THIS_DIR,'images','test.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base21.yml'),
      f"pretrained_model_name_or_path={output_dir}/{run_name}/checkpoints/final",
      "revision=bf16","dtype=bfloat16","resolution=768",
      "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=7.5",
      "num_inference_steps=30","seed=47",])

    images = generate_run(pyconfig.config)
    test_image = np.array(images[1]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

    cleanup(output_dir)

  def test_sd15_config(self):
    output_dir="train-smoke-test"
    run_name="sd15_smoke_test"
    train_main([None,os.path.join(THIS_DIR,'..','configs','base15.yml'),
      f"run_name={run_name}", "checkpoint_every=256","upload_ckpts_to_gcs=True",
      "max_train_steps=21","per_device_batch_size=8",
      "base_output_directory=gs://maxdiffusion-github-runner-test-assets/training_results/", f"output_dir={output_dir}"])

    img_url = os.path.join(THIS_DIR,'images','test_sd15.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)

    # here we test the unet saving works.
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base15.yml'),
      "dtype=bfloat16",
      "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=7.5",
      "num_inference_steps=30","seed=47", "cache_dir=gs://jfacevedo-maxdiffusion/cache_dir",
      f"unet_checkpoint=gs://maxdiffusion-github-runner-test-assets/training_results/{output_dir}/{run_name}/checkpoints/UNET-samples-512"])

    images = generate_run(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

    cleanup(output_dir)

  # @jfacevedo TODO - tf_records were processed with an extra dim,
  # which mess up the aot compilation. Instead of hacking the training script
  # to pass data dims that are incorrect, comment out this code to fix at a
  # later time when tfrecords are re-run and processed correctly.
  # def test_sd_2_base_config(self):
  #   output_dir="train-smoke-test"
  #   train_main([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
  #     "run_name=sd2_base_smoke_test","max_train_steps=21",
  #     "dataset_name=",
  #     "train_data_dir=gs://jfacevedo-maxdiffusion/laion400m/tf_records",
  #     "base_output_directory=gs://maxdiffusion-tests", f"output_dir={output_dir}",
  #     "attention=dot_product"])

  #   img_url = os.path.join(THIS_DIR,'images','test_2_base.png')
  #   base_image = np.array(Image.open(img_url)).astype(np.uint8)

  #   pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
  #     f"pretrained_model_name_or_path={output_dir}",
  #     "prompt=A magical castle in the middle of a forest, artistic drawing",
  #     "negative_prompt=purple, red","guidance_scale=7.5", "from_pt=False",
  #     "num_inference_steps=30","seed=47", "attention=dot_product"])

  #   images = generate_run(pyconfig.config)
  #   test_image = np.array(images[0]).astype(np.uint8)
  #   ssim_compare = ssim(base_image, test_image,
  #     multichannel=True, channel_axis=-1, data_range=255
  #   )
  #   assert base_image.shape == test_image.shape
  #   assert ssim_compare >=0.70

  #   cleanup(output_dir)

  # def test_sd_2_base_new_unet(self):
  #   output_dir="train-smoke-test"
  #   train_main([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
  #     "run_name=sd2_base_smoke_test","max_train_steps=21",
  #     "dataset_name=",
  #     "train_data_dir=gs://jfacevedo-maxdiffusion/laion400m/tf_records",
  #     "base_output_directory=gs://maxdiffusion-tests", f"output_dir={output_dir}",
  #     "attention=dot_product","train_new_unet=True"])

  #   img_url = os.path.join(THIS_DIR,'images','test_2_base.png')
  #   base_image = np.array(Image.open(img_url)).astype(np.uint8)

  #   pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
  #     f"pretrained_model_name_or_path={output_dir}", "from_pt=False",
  #     "prompt=A magical castle in the middle of a forest, artistic drawing",
  #     "negative_prompt=purple, red","guidance_scale=7.5",
  #     "num_inference_steps=30","seed=47", "attention=dot_product"])

  #   images = generate_run(pyconfig.config)
  #   test_image = np.array(images[0]).astype(np.uint8)
  #   ssim_compare = ssim(base_image, test_image,
  #     multichannel=True, channel_axis=-1, data_range=255
  #   )
  #   assert base_image.shape == test_image.shape
  #   assert ssim_compare <=0.40

  #   cleanup(output_dir)

if __name__ == '__main__':
  absltest.main()
