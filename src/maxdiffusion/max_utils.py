# ruff: noqa
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

# pylint: disable=bare-except, consider-using-generator
""" Common Max Utils needed by multiple modules"""
import functools
from functools import reduce
import json
import yaml
import os
from pathlib import Path
import subprocess

import numpy as np

import flax
import jax
import jax.numpy as jnp
import optax
from maxdiffusion import checkpointing, max_logging
from maxdiffusion.models.attention_flax import AttentionOp
from flax import linen as nn
import flax.linen as nn
import flax.linen.module as module_lib
from flax.linen.summary import _process_inputs
from flax.typing import (
  PRNGKey,
  RNGSequences,
)
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from flax import struct
from typing import (
  Callable,
  Any,
  Tuple,
  Union,
  Set,
)
from flax import core
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

from tensorboardX import writer

from google.cloud import storage

FrozenDict = core.frozen_dict.FrozenDict

class InferenceState(struct.PyTreeNode):
  # pylint: disable=g-bare-generic
  apply_fn: Callable = struct.field(pytree_node=False)
  params: FrozenDict[str, Any] | None = struct.field(pytree_node=True)

def l2norm_pytree(x):
  """L2 norm of a pytree of arrays."""
  return jax.tree_util.tree_reduce(
      lambda x, y: x + jax.numpy.sum(y ** 2), x, initializer=0.0
  ) ** 0.5

def activate_profiler(config):
  if jax.process_index() == 0 and config.enable_profiler:
    jax.profiler.start_trace(config.tensorboard_dir)

def deactivate_profiler(config):
  if jax.process_index() == 0 and config.enable_profiler:
    jax.profiler.stop_trace()

def initialize_summary_writer(config):
  return writer.SummaryWriter(config.tensorboard_dir) if jax.process_index() == 0 else None

def close_summary_writer(summary_writer):
  if jax.process_index() == 0:
    summary_writer.close()

def _prepare_metrics_for_json(metrics, step, run_name):
  """Converts metric dictionary into json supported types (e.g. float)"""
  metrics_dict = {}
  for val in metrics['scalar']:
    metrics_dict[val] = float(metrics['scalar'][val])
  metrics_dict['step'] = float(step)
  metrics_dict['run_name'] = run_name
  return metrics_dict

def write_metrics_locally(metrics, step, config, file):
  """Writes metrics locally for testing"""
  if step == 0:
    file.truncate(0)

  metrics_dict = _prepare_metrics_for_json(metrics, step, config.run_name)
  file.write(str(json.dumps(metrics_dict))+'\n')

  if step == config.max_train_steps - 1:
    file.close()

def write_metrics_for_gcs(metrics, step, config, running_metrics):
  """Writes metrics to gcs"""
  metrics_dict_step = _prepare_metrics_for_json(metrics, step, config.run_name)
  running_metrics.append(metrics_dict_step)
  if (step + 1) % config.log_period == 0 or step == config.steps - 1:
    start_step = (step // config.log_period) * config.log_period
    metrics_filename = f"metrics_step_{start_step:06}_to_step_{step:06}.txt"
    with open(metrics_filename, 'w', encoding="utf8") as metrics_for_gcs:
      for metrics_step in running_metrics:
        metrics_for_gcs.write(str(json.dumps(metrics_step))+'\n')

    metrics_for_gcs.close()
    gcs_filename=os.path.join(config.metrics_dir, metrics_filename)
    max_logging.log(f"Moving file {metrics_filename} to GCS...")
    upload_blob(gcs_filename, metrics_filename)
    max_logging.log(f"File {metrics_filename} moved successfully!")
    running_metrics = [] # reset running_metrics to empty list
  return running_metrics

def add_config_to_summary_writer(config, summary_writer):
  """Writes config params to tensorboard"""
  if jax.process_index() == 0:
    for key, value in config.get_keys().items():
      add_text_to_summary_writer(key, str(value), summary_writer)

def add_text_to_summary_writer(key, value, summary_writer):
  """Writes given key-value pair to tensorboard as text/summary"""
  if jax.process_index() == 0:
    summary_writer.add_text(key, value)

def write_metrics_for_gcs(metrics, step, config, running_metrics):
  """Writes metrics to gcs"""
  metrics_dict_step = _prepare_metrics_for_json(metrics, step, config.run_name)
  running_metrics.append(metrics_dict_step)
  if (step + 1) % config.log_period == 0 or step == config.max_train_steps - 1:
    start_step = (step // config.log_period) * config.log_period
    metrics_filename = f"metrics_step_{start_step:06}_to_step_{step:06}.txt"
    with open(metrics_filename, 'w', encoding="utf8") as metrics_for_gcs:
      for metrics_step in running_metrics:
        metrics_for_gcs.write(str(json.dumps(metrics_step))+'\n')

    metrics_for_gcs.close()
    gcs_filename=os.path.join(config.metrics_dir, metrics_filename)
    command = ["gsutil", "mv", metrics_filename, gcs_filename]
    max_logging.log(f"Moving file {metrics_filename} to GCS...")
    subprocess.run(command, check=True, capture_output=True)
    max_logging.log(f"File {metrics_filename} moved successfully!")
    running_metrics = [] # reset running_metrics to empty list
  return running_metrics

def write_config_raw_keys_for_gcs(raw_keys):
  """Writes config raw keys to GCS"""
  if not raw_keys["save_config_to_gcs"] or jax.process_index() != 0:
    return
  max_logging.log("Writing config to GCS...")

  raw_keys_dict = dict(raw_keys)
  filename = "config.yml"
  with open(filename, 'w', encoding="utf8") as config_for_gcs:
    yaml.dump(raw_keys_dict, config_for_gcs)
  config_for_gcs.close()

  gcs_filename=os.path.join(raw_keys["base_output_directory"], raw_keys["run_name"], filename)
  max_logging.log(f"Moving file {filename} to GCS...")
  upload_blob(gcs_filename, filename)
  max_logging.log(f"File {filename} moved successfully!")

def parse_gcs_bucket_and_prefix(destination_gcs_name):
  path_parts = destination_gcs_name.replace("gs://", "").split("/")
  bucket = path_parts.pop(0)
  key = "/".join(path_parts)
  return bucket, key

def download_blobs(source_gcs_folder, local_destination):
  """Downloads a folder to a local location"""
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(source_gcs_folder)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blobs = bucket.list_blobs(prefix=prefix_name)
  for blob in blobs:
    file_split = blob.name.split("/")
    directory = os.path.join(local_destination, "/".join(file_split[0:-1]))
    Path(directory).mkdir(parents=True, exist_ok=True)
    if len(file_split[-1]) <=0:
      continue
    download_to_filename = os.path.join(directory, file_split[-1])
    if not os.path.isfile(download_to_filename):
      blob.download_to_filename(download_to_filename)
  return os.path.join(local_destination, prefix_name)

def upload_blob(destination_gcs_name, source_file_name):
  """Uploads a file to a GCS location"""
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(destination_gcs_name)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(prefix_name)
  blob.upload_from_filename(source_file_name)

def walk_and_upload_blobs(config, output_dir):
  user_dir = os.path.expanduser('~')
  uploaded_files = set()
  for root, _, files in os.walk(os.path.abspath(output_dir)):
    for file in files:
      file_to_upload = os.path.join(root, file)
      if file_to_upload in uploaded_files:
        continue
      gcs_file_name = os.path.join(config.base_output_directory,
                                file_to_upload.replace(user_dir,"").strip("/").
                                replace("maxdiffusion","").strip("/"))
      max_logging.log(f"Moving file {file_to_upload} to {gcs_file_name}")
      upload_blob(gcs_file_name, file_to_upload)
      uploaded_files.add(file_to_upload)
      max_logging.log(f"File {file_to_upload} moved successfully!")

def initialize_jax_distributed_system():
  """ The best recipe to initialize the Jax Distributed System has varied over time. We keep a layer of
      indirection in MaxText to avoid breaking the call sites unnecessarily.

      Currently jax.distributed.initialize() fully works as expected!
  """
  max_logging.log("Attempting to initialize the jax distributed system...")
  jax.distributed.initialize()
  max_logging.log("Jax distributed system initialized!")

def device_put_replicated(x, sharding):
  return jax.make_array_from_callback(x.shape, sharding, lambda index: x[index])

def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert parallelism_vals.count(-1) == 1, f"Found unspecified values (-1) for more than one {parallelism_type}\
      parallelism axis. At most one axis can be unspecified."

    determined_val = target_product/np.prod(parallelism_vals)*-1

    assert determined_val >= 1 and determined_val.is_integer, f"Unspecified value unable to be determined with the given\
      {parallelism_type} parallelism values"

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = "slices" if parallelism_type == 'DCN' else "devices per slice"

  assert np.prod(parallelism_vals) == target_product, f"Number of {target_type} {target_product} does not match\
    the product of the {parallelism_type} parallelism {np.prod(parallelism_vals)}"

  return parallelism_vals

def create_device_mesh(config, devices=None, logging=True):
  """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas """
  if devices is None:
    devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max([d.slice_index for d in devices])
  except:
    num_slices = 1
  num_devices_per_slice = num_devices//num_slices
  max_logging.log(f"Devices: {devices} (num_devices: {num_devices})")
  assert len(devices) > 1, "You must have at least two devices"


  ici_parallelism = [config.ici_data_parallelism, config.ici_fsdp_parallelism, config.ici_tensor_parallelism]

  # Find possible unspecified parallelisms
  ici_parallelism = fill_unspecified_mesh_axes(ici_parallelism, num_devices_per_slice, 'ICI')
  mesh = mesh_utils.create_device_mesh(ici_parallelism, devices)

  if logging:
    max_logging.log(f"Decided on mesh: {mesh}")

  return mesh

def unbox_logicallypartioned_trainstate(
    boxed_train_state: train_state.TrainState):
  """ Unboxes the flax.LogicallyPartitioned pieces in a train state.

    Args:
      boxed_train_state: a train state that includes LogicallyPartitioned
        leaves.
    Returns:
      a TrainState where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(lambda x: x.unbox() if \
        isinstance(x, flax.linen.spmd.LogicallyPartitioned) \
        else x, boxed_train_state, \
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned))

def init_train_state(model_params, model, tx, training=True):
  """
  We pass in "static" objects like model, tx, config, as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model_params, model, tx, training
  """
  if training:
    state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=model_params,
      tx=tx)
  else:
    state = InferenceState(apply_fn=model.apply, params=model_params)
  return state

def get_abstract_state(model, tx, config, mesh, model_params, training=True):
  """ Get a shaped abstraction of the state (including optimizer)"""
  abstract_state = jax.eval_shape(functools.partial(init_train_state, model=model, tx=tx, training=training), model_params=model_params)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  unboxed_abstract_state = unbox_logicallypartioned_trainstate(abstract_state)

  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return unboxed_abstract_state, state_mesh_annotations

def setup_initial_state(model, tx, config, mesh, model_params, unboxed_abstract_state, state_mesh_annotations, checkpoint_manager=None, training=True):
  """ We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    mesh: jax.devices() mesh
    model_params: model parameters
    unboxed_abstract_state: abstract state from get_abstract_state()
    state_mesh_annotations: state mesh annotations from get_abstract_state()
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object
    training: boolean True when initial state is used in training

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """

  # Initialization
  state = None
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    if checkpoint_manager:
      state, raw_params = checkpointing.load_state_if_possible(checkpoint_manager,
                                                  config.load_parameters_path,
                                                  config.load_from_other_directory,
                                                  config.load_from_other_directory_step,
                                                  unboxed_abstract_state,
                                                  mesh,
                                                  state_mesh_annotations)

    state_mesh_shardings = jax.tree_util.tree_map(
        lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  if not state:
    init_train_state_partial = functools.partial(init_train_state, model=model, tx=tx, training=training)

    sharding = PositionalSharding(mesh.devices).replicate()
    partial_device_put_replicated = functools.partial(device_put_replicated, sharding=sharding)
    model_params = jax.tree_util.tree_map(partial_device_put_replicated, model_params)

    with jax.transfer_guard("disallow"):
      state = jax.jit(
          init_train_state_partial,
          in_shardings=None,
          out_shardings=state_mesh_shardings
      )(model_params=model_params)

  state = unbox_logicallypartioned_trainstate(state)

  state_mesh_shardings = jax.tree_util.tree_map(
    lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  return state, state_mesh_shardings

def get_states(mesh, tx, rng, config, pipeline, unet_params, vae_params, training=True):
  
  # Needed to initialize weights on multi-host with addressable devices.
  if config.train_new_unet:
    unet_variables = jax.jit(pipeline.unet.init_weights, static_argnames=["eval_only"])(rng, eval_only=False)
  else:
    unet_variables = pipeline.unet.init_weights(rng, eval_only=True)

  unboxed_abstract_state, state_mesh_annotations = get_abstract_state(pipeline.unet, tx, config, mesh, unet_variables, training=training)
  if config.train_new_unet:
    unet_params = unet_variables
  else:
    del unet_variables
  unet_state, unet_state_mesh_shardings = setup_initial_state(
  pipeline.unet,
  tx,
  config,
  mesh,
  unet_params,
  unboxed_abstract_state,
  state_mesh_annotations,
  training=training)

  vae_state = None
  vae_state_mesh_shardings = None
  if vae_params:
    vae_variables = jax.jit(pipeline.vae.init_weights)(rng)
    unboxed_abstract_state, state_mesh_annotations = get_abstract_state(pipeline.vae, tx, config, mesh, vae_variables, training=training)
    del vae_variables
    vae_state, vae_state_mesh_shardings = setup_initial_state(
        pipeline.vae,
        tx,
        config,
        mesh,
        vae_params,
        unboxed_abstract_state,
        state_mesh_annotations,
        training=training
    )

  return unet_state, unet_state_mesh_shardings, vae_state, vae_state_mesh_shardings

# Learning Rate Schedule
# -----------------------------------------------------------------------------

def create_learning_rate_schedule(config):
  """Creates a warmup to constant learning rate schedule:
  We take inspiration from WarmupHoldPolicy used in stable diffusion
    see https://github.com/NVIDIA/NeMo/blob/dbc8a6ee490355bfa0cb1e10b8d199dcc47482e0/nemo/core/optim/lr_scheduler.py#L142
  Learning rate schedule has either two parts:
  1) Linear warmup from 0 to [learning_rate] over steps 0 to [learning_rate_schedule_steps * warmup_steps_fraction]
  2) Constant learning rate of 0 afterwards.
  """
  lr = config.learning_rate

  warmup_steps = int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
  constant_zero_steps = config.max_train_steps - warmup_steps

  warmup_schedule = optax.linear_schedule(
      init_value=0.0,
      end_value=lr,
      transition_steps=warmup_steps
  )
  constant_schedule = optax.constant_schedule(lr)

  pieces = [warmup_schedule, constant_schedule]
  boundaries=[
   warmup_steps,
   warmup_steps + constant_zero_steps,
   ]

  return optax.join_schedules(pieces, boundaries)

def get_dtype(config):
  """Get dtype from config."""
  dtype_str = config.dtype
  retval = jnp.bfloat16
  if dtype_str == "float32":
    retval = jnp.float32
  if dtype_str == "float16":
    retval = jnp.float16
  return retval

def get_flash_block_sizes(config):
  """Create custom flash attention BlockSizes."""
  flash_block_sizes = None
  if len(config.flash_block_sizes.keys()) > 0:
    flash_block_sizes = splash_attention_kernel.BlockSizes(
      block_q=config.flash_block_sizes["block_q"],
      block_kv_compute=config.flash_block_sizes["block_kv_compute"],
      block_kv=config.flash_block_sizes["block_kv"],
      block_q_dkv=config.flash_block_sizes["block_q_dkv"],
      block_kv_dkv=config.flash_block_sizes["block_kv_dkv"],
      block_kv_dkv_compute=config.flash_block_sizes["block_kv_dkv_compute"],
      block_q_dq=config.flash_block_sizes["block_q_dq"],
      block_kv_dq=config.flash_block_sizes["block_kv_dq"],
    )
  return flash_block_sizes

def delete_pytree(to_delete):
  jax.tree_util.tree_map(lambda x: x.delete(), to_delete)

def get_memory_allocations():
  devices = jax.local_devices()
  gb = 10**9
  for device in devices:
    m_stats = device.memory_stats()
    max_logging.log(f'device : {device.process_index},'
                    f'bytes in use: {m_stats["bytes_in_use"] / gb} / {m_stats["bytes_limit"] / gb} GB')


# Taking inspiration from flax's https://flax.readthedocs.io/en/v0.5.3/_modules/flax/linen/summary.html#tabulate
# to retrieve layer parameters and calculate 
def calculate_model_tflops(
    module: module_lib.Module,
    rngs: Union[PRNGKey, RNGSequences],
    train,
    **kwargs
):
  """Calculates model tflops by passing a module."""
  with module_lib._tabulate_context():
    _ = jax.eval_shape(module.init,
                       rngs,
                       **kwargs)
    calls = module_lib._context.call_info_stack[-1].calls
    calls.sort(key=lambda c: c.index)

  visited_paths: Set[Tuple[str, ...]] = set()
  total_flops = 0
  for c in calls:
    inputs = _process_inputs(c.args, c.kwargs)
    if c.path in visited_paths:
      continue
    else:
      # Multiply 2 * input shapes * features
      # For simple Dense layer input_shape = batch, in_features = (16, 10) and features = 20
      # Then 2 * 16 * 10 * 20.
      # In case of attention, an example if input shape batch,seq,hidden = (16, 4096, 320) and features = 320
      # Then 2 * 16 * 4096 * 320^2.
      if isinstance(c.module, nn.Dense):
        total_flops += 2 * (
          reduce(lambda x, y: x * y, inputs.shape)
          * c.module.features
        )
      # Here we capture qk einsum, scaling, softmax and attention_values * v
      # qk einsum : 2 * batch_size * seq_length_1 * seq_length_2 * heads * head_dim where (heads * head_dim) == hidden_dim
      # Note that in qk einsum and diffusion, seq_length_1 and seq_length_2 can be different due to cross attention.
      # scaling : division of the attn scores matrix by sqrt of hidden dimension batch_size * seq_length^2)
      # softmax : rough estimate is batch_size * seq_length_1 * log(seq_length_2)
      # attention_values * v : 2 * batch_size * seq_length_1 * seq_length_2 * heads * head_dim where (heads * head_dim) == hidden_dim
      elif isinstance(c.module, AttentionOp):
        qk_einsum = 2 * (reduce(lambda x, y: x * y, inputs[0].shape)) * inputs[1].shape[1]
        scaling = inputs[0].shape[0] * inputs[0].shape[1] * inputs[1].shape[1]
        softmax = reduce(lambda x, y: x * y, inputs[0].shape) * np.log(inputs[1].shape[1])
        att_v = 2 * (reduce(lambda x, y: x * y, inputs[0].shape)) * inputs[2].shape[1]
        # When seq_length_1 == seq_length_2 then,
        # qk_einsum + scaling + softmax + att_v == 4 * batch_size * hidden_dim * seq_length ^ 2
        total_flops += qk_einsum + scaling + softmax + att_v
      elif isinstance(c.module, nn.Conv):
        total_flops += 2 * (
          reduce(lambda x, y: x * y, inputs.shape)
          * c.module.features
          * reduce(lambda x, y: x * y, c.module.kernel_size)) / reduce(lambda x, y: x * y, c.module.strides)
    visited_paths.add(c.path)
  
  total_flops = (total_flops * 3 if train else total_flops) / 10**12
  return total_flops

def calculate_num_params_from_pytree(params):
  """Calculates number of parameters from a pytree"""
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  return total_parameters