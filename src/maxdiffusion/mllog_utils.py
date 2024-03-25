"""
 Copyright 2023 Google LLC
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

"""Utils that relevant to mllog for mlperf submission compliance."""
import jax
from mlperf_logging import mllog

mllogger = mllog.get_mllogger()

def train_init_start():
  if jax.process_index() == 0:
    mllogger.event(mllog.constants.CACHE_CLEAR)
    mllogger.start(mllog.constants.INIT_START)

def train_init_stop():
  if jax.process_index() == 0:
    mllogger.end(mllog.constants.INIT_STOP)

def train_run_start():
  if jax.process_index() == 0:
    mllogger.start(mllog.constants.RUN_START)

def train_run_end():
  if jax.process_index() == 0:
    mllogger.end(mllog.constants.RUN_STOP, metadata={'status': 'success'})

def train_init_print(config, device: str = 'tpu-v5p'):
  """an initial mllog for mlperf sumbission compliance check."""
  if jax.process_index() == 0:
    mllogger.event(mllog.constants.SUBMISSION_ORG, 'Google')
    mllogger.event(mllog.constants.SUBMISSION_PLATFORM, device)
    mllogger.event(mllog.constants.SUBMISSION_STATUS, mllog.constants.CLOUD)
    mllogger.event(mllog.constants.SUBMISSION_DIVISION, mllog.constants.CLOSED)
    mllogger.event(mllog.constants.SUBMISSION_BENCHMARK, mllog.constants.STABLE_DIFFUSION)
    mllogger.event(mllog.constants.GRADIENT_ACCUMULATION_STEPS, 1)
    mllogger.event(mllog.constants.GLOBAL_BATCH_SIZE,
                   config.per_device_batch_size * jax.device_count())

    mllogger.event(mllog.constants.OPT_NAME, mllog.constants.ADAMW)
    mllogger.event(mllog.constants.OPT_ADAMW_BETA_1, config.adam_b1)
    mllogger.event(mllog.constants.OPT_ADAMW_BETA_2, config.adam_b2)
    mllogger.event(mllog.constants.OPT_ADAMW_EPSILON, config.adam_eps)
    mllogger.event(mllog.constants.OPT_ADAMW_WEIGHT_DECAY, config.adam_weight_decay)

    mllogger.event(mllog.constants.OPT_BASE_LR, config.learning_rate)
    mllogger.event(mllog.constants.OPT_LR_WARMUP_STEPS,
                   int(config.learning_rate_schedule_steps * config.warmup_steps_fraction))

    # Training: a subset of laion-400m
    # Validation: a subset of coco-2014 validation
    mllogger.event(mllog.constants.TRAIN_SAMPLES, 6513144)
    mllogger.event(mllog.constants.EVAL_SAMPLES, 30000)

    mllogger.event(mllog.constants.SEED, config.seed)

def train_step_start(step):
  if jax.process_index() == 0:
    mllogger.start(
      mllog.constants.BLOCK_START,
      value="training_step",
      metadata={
        'step_num': step,
      },
    )

def train_step_end(step, loss, lr):
  if jax.process_index() == 0:
    mllogger.end(
      mllog.constants.BLOCK_STOP,
      value="training_step",
      metadata={
        'step_num': step,
        'loss': loss,
        'lr': lr,
      },
    )

def maybe_train_step_log(config, start_step, step, metric, train_log_interval: int = 100):
  if step > start_step and step % train_log_interval == 0 or step == config.max_train_steps - 1:
    train_step_end(
      step,
      loss=metric['scalar']['learning/loss'],
      lr=metric['scalar']['learning/current_learning_rate'],
    )
    # start new tracking except the last step
    if step < config.max_train_steps - 1:
      train_step_start(step)
