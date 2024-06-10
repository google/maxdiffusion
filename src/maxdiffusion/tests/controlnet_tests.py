import os
import unittest

import numpy as np

from ..import pyconfig
from absl.testing import absltest
from maxdiffusion.benchmarks.inference.controlnet.generate_controlnet_replicated import run as generate_run
from maxdiffusion.benchmarks.inference.controlnet.generate_controlnet_sdxl_replicated import run as generate_run_sdxl
from PIL import Image
from skimage.metrics import structural_similarity as ssim


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Generate(unittest.TestCase):
  """Smoke test."""
  def test_controlnet(self):
    img_url = os.path.join(THIS_DIR,'images','cnet_test.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base15.yml'),
      "prompt=best quality, extremely detailed",
      "negative_prompt=monochrome, lowres, bad anatomy, worst quality, low quality",
      "num_inference_steps=50","seed=0","split_head_dim=False"])

    images = generate_run(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70

  def test_controlnet_sdxl(self):
    img_url = os.path.join(THIS_DIR,'images','cnet_test_sdxl.png')
    base_image = np.array(Image.open(img_url)).astype(np.uint8)
    pyconfig.initialize([None,os.path.join(THIS_DIR,'..','configs','base_xl.yml')])
    images = generate_run_sdxl(pyconfig.config)
    test_image = np.array(images[0]).astype(np.uint8)
    ssim_compare = ssim(base_image, test_image,
      multichannel=True, channel_axis=-1, data_range=255
    )
    assert base_image.shape == test_image.shape
    assert ssim_compare >=0.70


if __name__ == '__main__':
  absltest.main()
