import os
import unittest

import numpy as np

from ..import pyconfig
from absl.testing import absltest
from maxdiffusion.benchmarks.inference.sdxlbase.generate_sdxl import run as generate_run_xl
from PIL import Image
from skimage.metrics import structural_similarity as ssim


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Generate(unittest.TestCase):
  """Smoke test."""
  def test_sdxl_config(self):
    img_url = os.path.join(THIS_DIR,'images','test_sdxl.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_xl.yml'),
      "pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
      "revision=refs/pr/95","dtype=bfloat16","resolution=1024",
      "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=9",
      "num_inference_steps=20","seed=47","per_device_batch_size=1",
      "run_name=sdxl-inference-test","split_head_dim=False"])
    images = generate_run_xl(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.80

  def test_sdxl_from_gcs(self):
    """Verify load weights from gcs."""
    img_url = os.path.join(THIS_DIR,'images','test_sdxl.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_xl.yml'),
      "pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0",
      "revision=main","dtype=bfloat16","resolution=1024",
      "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=9",
      "num_inference_steps=20","seed=47","per_device_batch_size=1",
      "run_name=sdxl-inference-test","split_head_dim=False"])
    images = generate_run_xl(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.80

if __name__ == '__main__':
  absltest.main()
