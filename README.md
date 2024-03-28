<!--
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
 -->

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/UnitTests.yml/badge.svg)](https://github.com/google/maxdiffusion/actions/workflows/UnitTests.yml)

# Overview 

MaxDiffusion is a Latent Diffusion model written in pure Python/Jax and targeting Google Cloud TPUs. MaxDiffusion aims to be a launching off point for ambitious Diffusion projects both in research and production. 
We encourage users to start by experimenting with MaxDiffusion out of the box and then fork and modify MaxDiffusion to meet their needs.

MaxDiffusion supports 
* Stable Diffusion 2 base (training and inference)
* Stable Diffusion 2.1 (training and inference) 
* Stable Diffusion XL (inference).

WARNING: The training code is purely experimental and is under development.

# Table of Contents

* [Getting Started](#getting-started)
* [Comparison To Alternatives](#comparison-to-alternatives)
* [Development](#development)

# Getting Started

We recommend starting with single host first and then moving to multihost.

## Getting Started: Local Development for single host
Local development is a convenient way to run MaxDiffusion on a single host. 

1. [Create and SSH to the single-host TPU of your choice.](https://cloud.google.com/tpu/docs/users-guide-tpu-vm#creating_a_cloud_tpu_vm_with_gcloud) We recommend a `v4-8`.
2. Clone MaxDiffusion onto that TPUVM.
3. Within the root directory of that `git` repo, install dependencies by running:
```bash
pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -r requirements.txt
pip3 install -e .
```
4. After installation completes, run training with the command:
```bash
python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name="my_run" base_output_directory="gs://your-bucket/" train_data_dir=gs://jfacevedo-maxdiffusion/laion400m/tf_records
```
5. If you want to generate images, you can do it as follows.
- Stable Diffusion 2.1
  ```bash
  python -m src.maxdiffusion.generate src/maxdiffusion/configs/base.yml run_name="my_run"
  ```

- Stable Diffusion XL Lightning

  Multi host supported with sharding annotations:

  ```bash
  python -m src.maxdiffusion.generate_sdxl src/maxdiffusion/configs/base_xl_lightning.yml run_name="my_run"
  ```
- Stable Diffusion XL

  Multi host supported with sharding annotations:

  ```bash
  python -m src.maxdiffusion.generate_sdxl src/maxdiffusion/configs/base_xl.yml run_name="my_run"
  ```

  Single host pmap version:
  ```bash
  python -m src.maxdiffusion.generate_sdxl_replicated
  ```

## Getting Started: Multihost development
Multihost training can be ran as follows.
```bash
TPU_NAME=<your-tpu-name>
ZONE=<your-zone>
PROJECT_ID=<your-project-id>
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project $PROJECT_ID --worker=all --command="
git clone https://github.com/google/maxdiffusion
pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -r requirements.txt
pip3 install .
python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base.yml run_name=my_run base_output_directory=gs://your-bucket/"
```

# Comparison to Alternatives

MaxDiffusion started as a fork of [Diffusers](https://github.com/huggingface/diffusers), a Hugging Face diffusion library written in Python, Pytorch and Jax. MaxDiffusion is compatible with Hugging Face Jax models. MaxDiffusion is more complex with the aim to run distributed across TPU Pods. 

# Development

Whether you are forking MaxDiffusion for your own needs or intending to contribute back to the community, we offer simple testing recipes.

To run unit tests and lint, simply run:
```
python -m pytest
ruff check --fix .
```

The full suite of -end-to end tests is in `tests` and `src/maxdiffusion/tests`. We run them with a nightly cadance.
