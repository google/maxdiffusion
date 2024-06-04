#!/bin/bash
#
export PLATFORM=GKE

set -e

set -ex

echo "Adjust Network settings and apply non cache copy"

# Disable slow start after idle
sysctl net.ipv4.tcp_slow_start_after_idle=0

# Disable metrics cache
sysctl net.ipv4.tcp_no_metrics_save=1

# Address rto_min issue with two default routing entries: screen/7RGgkiXkGXSeYF2
route=$(ip route show | sed -n 1p)
second_route=$(ip route show | sed -n 2p)
if [[ "${second_route}" =~ ^default.* ]]; then
  modified_route=${route//" lock"/}
  ip route delete ${modified_route}
fi
route=$(ip route show | sed -n 1p)
echo "route rto before change: $route"
if [[ "${route}" =~ .*lock.*5ms.* ]]; then
  echo "${route}"
else
  # shellcheck disable=SC2086
  ip route change $route rto_min 5ms
fi
route=$(ip route show | sed -n 1p)
echo "route rto after change: $route"

# Disable Cubic Hystart Ack-Train
echo 2 > /sys/module/tcp_cubic/parameters/hystart_detect

# Improve handling SYN burst
echo 4096 > /proc/sys/net/core/somaxconn
echo 4096 > /proc/sys/net/ipv4/tcp_max_syn_backlog

# Disable MTU Discovery
echo 0 > /proc/sys/net/ipv4/tcp_mtu_probing

# Increase TCP Zerocopy control memory
sysctl -w net.core.optmem_max=131072

# Printing output of `ip route show`
echo -e "\nPrinting output of 'ip route show':"
ip route show

first_line_res=$(ip route show | head -n 1)
dev_name=$(echo "$first_line_res" | awk -F'[[:space:]]' '{ print $5 }')
echo "dev_name=${dev_name}"
ethtool -K "${dev_name}" tx-nocache-copy on

echo "rto_setup.sh finished"
OUT_DIR=$1
export JAX_TRACEBACK_FILTERING=off
export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true'


DIR=/tmp/xla_dump

git clone -b mlperf_4 https://github.com/google/maxdiffusion.git
cd maxdiffusion
pip install .

TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name=v5p-data base_output_directory=${OUT_DIR}  \
train_data_dir=gs://mlperf-llm-data-e5-cp0/laion400m/tf_records_512_encoder_state per_device_batch_size=2 split_head_dim=True  attention=flash  train_new_unet=True norm_num_groups=16 \
start_step_to_checkpoint=5120000 \
enable_profiler=true skip_first_n_steps_for_profiler=495 reuse_example_batch=false max_train_steps=500  2>&1 | tee /tmp/SD_test_log


if [[ ${TPU_WORKER_ID} == "0" ]]; then
  gsutil -m cp /tmp/SD_test_log "${OUT_DIR}/SD_test_log"
fi
