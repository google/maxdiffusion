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
from ..import pyconfig
from maxdiffusion.generate import run as generate_run
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
  def test_sd21_config(self):
    output_dir="train-smoke-test"
    train_main([None,os.path.join(THIS_DIR,'..','configs','base21.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
      "revision=bf16","dtype=bfloat16","run_name=sd2.1_smoke_test",
      "max_train_steps=21","train_data_dir=gs://jfacevedo-maxdiffusion/laion400m/tf_records",
      "resolution=768","per_device_batch_size=1",
      "base_output_directory=gs://maxdiffusion-tests", f"output_dir={output_dir}"])

    img_url = os.path.join(THIS_DIR,'images','test.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base21.yml'),
      f"pretrained_model_name_or_path={output_dir}",
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

  def test_sd_2_base_config(self):
    output_dir="train-smoke-test"
    train_main([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
      "run_name=sd2_base_smoke_test","max_train_steps=21","train_data_dir=gs://jfacevedo-maxdiffusion/laion400m/tf_records",
      "base_output_directory=gs://maxdiffusion-tests", f"output_dir={output_dir}",
      "attention=dot_product"])

    img_url = os.path.join(THIS_DIR,'images','test_2_base.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base_inference.yml'),
      f"pretrained_model_name_or_path={output_dir}",
      "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=7.5",
      "num_inference_steps=30","seed=47", "attention=dot_product"])

    images = generate_run(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

    cleanup(output_dir)

  def test_sd_2_base_new_unet(self):
    output_dir="train-smoke-test"
    train_main([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'),
      "run_name=sd2_base_smoke_test","max_train_steps=21","train_data_dir=gs://jfacevedo-maxdiffusion/laion400m/tf_records",
      "base_output_directory=gs://maxdiffusion-tests", f"output_dir={output_dir}",
      "attention=dot_product","train_new_unet=True"])

    img_url = os.path.join(THIS_DIR,'images','test_2_base.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)

    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base_inference.yml'),
      f"pretrained_model_name_or_path={output_dir}",
      "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=7.5",
      "num_inference_steps=30","seed=47", "attention=dot_product"])

    images = generate_run(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare <=0.40

    cleanup(output_dir)

if __name__ == '__main__':
  absltest.main()
