import os
import unittest

import numpy as np

from ..import pyconfig
from absl.testing import absltest
from maxdiffusion.generate import run as generate_run
from PIL import Image
from skimage.metrics import structural_similarity as ssim


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Generate(unittest.TestCase):
  """Smoke test."""
  def test_sd21_config(self):
    img_url = os.path.join(THIS_DIR,'images','test.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base21.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
      "revision=bf16","dtype=bfloat16","resolution=768",
      "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=7.5",
      "num_inference_steps=30","seed=47","split_head_dim=False"])
    images = generate_run(pyconfig.config)
    test_image = np.array(images[1]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

  def test_sd_2_base_config(self):
    img_url = os.path.join(THIS_DIR,'images','test_2_base.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_2_base.yml'), "seed=47"])
    images = generate_run(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

if __name__ == '__main__':
  absltest.main()
