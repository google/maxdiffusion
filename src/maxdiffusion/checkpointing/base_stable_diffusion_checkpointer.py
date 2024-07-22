
from abc import ABC, abstractmethod
import os
import json
import jax
from jax.sharding import Mesh
import orbax.checkpoint as ocp
from maxdiffusion import (
    max_utils,
    FlaxStableDiffusionPipeline,
    FlaxStableDiffusionXLPipeline,
    FlaxUNet2DConditionModel,
    FlaxAutoencoderKL,
)

from transformers import (
  CLIPTokenizer,
  FlaxCLIPTextModel,
  PretrainedConfig
)

from maxdiffusion.checkpointing.checkpointing_utils import (
    create_orbax_checkpoint_manager,
    load_stable_diffusion_configs,
)

STABLE_DIFFUSION_CHECKPOINT = "STABLE_DIFFUSION_CHECKPOINT"
STABLE_DIFFUSION_XL_CHECKPOINT = "STABLE_DIFUSSION_XL_CHECKPOINT"
_CHECKPOINT_FORMAT_DIFFUSERS = "CHECKPOINT_FORMAT_DIFFUSERS"
_CHECKPOINT_FORMAT_ORBAX = "CHECKPOINT_FORMAT_ORBAX"

class BaseStableDiffusionCheckpointer(ABC):
    def __init__(self, config, checkpoint_type):
        self.config = config
        self.checkpoint_type = checkpoint_type
        self.checkpoint_format = None
        if len(config.cache_dir) > 0:
            jax.config.update("jax_compilation_cache_dir", config.cache_dir)

        self.rng = jax.random.PRNGKey(self.config.seed)
        devices_array = max_utils.create_device_mesh(config)
        self.mesh = Mesh(devices_array, self.config.mesh_axes)
        self.total_train_batch_size = max_utils.get_global_batch_size(self.config)

        self.pipeline = None
        self.params = {}
        self.train_states = {}
        self.state_shardings = {}

        self.checkpoint_manager = create_orbax_checkpoint_manager(
            self.config.checkpoint_dir,
            enable_checkpointing=True,
            save_interval_steps=1,
            checkpoint_type=checkpoint_type
        )

    @abstractmethod
    def post_create_states_and_shard(self):
        """
        Hook to create any other states or additional shardings.
        Called last in create_states().
        """
        pass

    def _create_unet_state(self):
        if self.config.train_new_unet:
            unet_variables = jax.jit(self.pipeline.unet.init_weights, static_argnames=["eval_only"])(self.rng, eval_only=False)
            self.params["unet"] = unet_variables
        else:
            # required to initilaize the model
            _ = self.pipeline.unet.init_weights(self.rng, eval_only=True)

        unet_state, unet_state_mesh_shardings = max_utils.setup_initial_state(
            model=self.pipeline.unet,
            tx=self.get_optimizer(),
            config=self.config,
            mesh=self.mesh,
            model_params=self.params.get("unet", self.pipeline.unet.init_weights(self.rng, eval_only=True)),
            checkpoint_manager=self.checkpoint_manager,
            checkpoint_item="unet_state",
            training=True
        )
        self.train_states["unet_state"] = unet_state
        self.state_shardings["unet_state_shardings"] = unet_state_mesh_shardings

    def _create_vae_state(self):
        vae_state, vae_state_mesh_shardings = max_utils.setup_initial_state(
            model=self.pipeline.vae,
            tx=self.get_optimizer(),
            config=self.config,
            mesh=self.mesh,
            model_params=self.params.get("vae", self.pipeline.vae.init_weights(self.rng, eval_only=True)),
            checkpoint_manager=self.checkpoint_manager,
            checkpoint_item="vae_state",
            training=True
        )
        self.train_states["vae_state"] = vae_state
        self.state_shardings["vae_state_shardings"] = vae_state_mesh_shardings

    def _create_text_encoder_state(self):
        text_encoder_state, text_encoder_mesh_shardings = max_utils.setup_initial_state(
            model=self.pipeline.text_encoder,
            tx=self.get_optimizer(),
            config=self.config,
            mesh=self.mesh,
            model_params=self.params.get("text_encoder", self.pipeline.text_encoder.init_weights(self.rng, (self.total_train_batch_size, self.pipeline.tokenizer.model_max_length))),
            checkpoint_manager=self.checkpoint_manager,
            checkpoint_item="text_encoder_state",
            training=True
        )
        self.train_states["text_encoder_state"] = text_encoder_state
        self.state_shardings["text_encoder_state_shardings"] = text_encoder_mesh_shardings

    def create_states_and_shard(self):
        """
        Creates train states and shards models accordingly.
        """
        # pipeline should already be initialized here
        # but check anyway.
        if self.pipeline is None:
            self.load_checkpoint()

        self._create_unet_state()
        self._create_vae_state()
        self._create_text_encoder_state()

        # Hook
        self.post_create_states_and_shard()

    def _get_pipeline_class(self):
        if self.checkpoint_type == STABLE_DIFFUSION_CHECKPOINT:
            pipeline_class = FlaxStableDiffusionPipeline
        else:
            pipeline_class = FlaxStableDiffusionXLPipeline

        return pipeline_class

    def _set_checkpoint_format(self, checkpoint_format):
        self.checkpoint_format = checkpoint_format

    def load_diffusers_checkpoint(self):
        pipeline_class = self._get_pipeline_class()

        precision = max_utils.get_precision(self.config)
        flash_block_sizes = max_utils.get_flash_block_sizes(self.config)

        pipeline, params = pipeline_class.from_pretrained(
            self.config.pretrained_model_name_or_path,
            revision=self.config.revision,
            dtype=self.config.activations_dtype,
            safety_checker=None,
            feature_extractor=None,
            from_pt=self.config.from_pt,
            split_head_dim=self.config.split_head_dim,
            norm_num_groups=self.config.norm_num_groups,
            attention_kernel=self.config.attention,
            flash_block_sizes=flash_block_sizes,
            mesh=self.mesh,
            precision=precision
        )


        if len(self.config.unet_checkpoint) > 0:
            unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
                    self.config.unet_checkpoint,
                    split_head_dim=self.config.split_head_dim,
                    norm_num_groups=self.config.norm_num_groups,
                    attention_kernel=self.config.attention,
                    flash_block_sizes=flash_block_sizes,
                    mesh=self.mesh
                )
            params["unet"] = unet_params
            pipeline.unet = unet

        params = jax.tree_util.tree_map(lambda x: x.astype(self.config.weights_dtype), params)

        self.pipeline = pipeline
        self.params = params

    def save_checkpoint(self, train_step):
        def config_to_json(model_or_config):
            return json.loads(model_or_config.to_json_string())

        items = {
            'unet_config' : ocp.args.JsonSave(config_to_json(self.pipeline.unet)),
            'vae_config' : ocp.args.JsonSave(config_to_json(self.pipeline.vae)),
            'text_encoder_config' : ocp.args.JsonSave(config_to_json(self.pipeline.text_encoder.config)),
            "scheduler_config" : ocp.args.JsonSave(config_to_json(self.pipeline.scheduler))
        }

        items['unet_state'] = ocp.args.StandardSave(self.train_states["unet_state"])
        items['vae_state'] = ocp.args.StandardSave(self.train_states["vae_state"])
        items["text_encoder_state"] = ocp.args.StandardSave(self.train_states["text_encoder_state"])

        if hasattr(self.pipeline, "text_encoder_2"):
            items["text_encoder_state_2"] = ocp.args.StandardSave(self.train_states["text_encoder_2_state"])

        tokenizer_config = {"path" : self.config.tokenizer_model_name_or_path}
        items["tokenizer_config"] = ocp.args.JsonSave(tokenizer_config)
        self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))

    def load_checkpoint(self, step = None, scheduler_class = None):

        pipeline_class = self._get_pipeline_class()

        self.checkpoint_format = _CHECKPOINT_FORMAT_ORBAX

        precision = max_utils.get_precision(self.config)
        flash_block_sizes = max_utils.get_flash_block_sizes(self.config)
        # try loading using orbax, if not, use diffusers loading
        model_configs = load_stable_diffusion_configs(
            self.config,
            self.checkpoint_manager,
            self.checkpoint_type, step)

        if model_configs:
            unet = FlaxUNet2DConditionModel.from_config(
                model_configs[0]["unet_config"],
                dtype=self.config.activations_dtype,
                from_pt=self.config.from_pt,
                split_head_dim=self.config.split_head_dim,
                norm_num_groups=self.config.norm_num_groups,
                attention_kernel=self.config.attention,
                flash_block_sizes=flash_block_sizes,
                mesh=self.mesh,
                precision=precision
            )

            vae = FlaxAutoencoderKL.from_config(
                model_configs[0]["vae_config"],
                dtype=self.config.activations_dtype,
                from_pt=self.config.from_pt
            )

            te_pretrained_config = PretrainedConfig.from_dict(model_configs[0]["text_encoder_config"])
            text_encoder = FlaxCLIPTextModel(
                te_pretrained_config,
                seed=self.config.seed,
                dtype=self.config.activations_dtype
            )
            tokenizer_path = model_configs[0]["tokenizer_config"]["path"]
            if "gs://" in tokenizer_path:
                if "tokenizer" not in tokenizer_path:
                    tokenizer_path = os.path.join(tokenizer_path, "tokenizer")
                tokenizer_path = max_utils.download_blobs(tokenizer_path, "/tmp")
            tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_path,
                subfolder="tokenizer",
                dtype=self.config.activations_dtype,
            )

            scheduler = None
            if scheduler_class:
                scheduler = scheduler_class.from_config(model_configs[0]["scheduler_config"])

            pipeline_kwargs = {
            "unet" : unet,
            "vae" : vae,
            "text_encoder" : text_encoder,
            "scheduler" : scheduler,
            "tokenizer" : tokenizer,
            }

            if self.checkpoint_type == STABLE_DIFFUSION_CHECKPOINT:
                pipeline_kwargs["safety_checker"] = None
                pipeline_kwargs["feature_extractor"] = None
            else:
                te_pretrained_2_config = PretrainedConfig.from_dict(model_configs[0]["text_encoder_2_config"])
                text_encoder_2 = FlaxCLIPTextModel(
                    te_pretrained_2_config,
                    seed=self.config.seed,
                    dtype=self.config.activations_dtype
                )
                pipeline_kwargs["text_encoder_2"] = text_encoder_2
                pipeline_kwargs["tokenizer_2"] = tokenizer

            pipeline = pipeline_class(
            **pipeline_kwargs
            )
            self.pipeline = pipeline
        else:
            self.checkpoint_format = _CHECKPOINT_FORMAT_DIFFUSERS
            self.load_diffusers_checkpoint()

        self.create_states_and_shard()

        if self.checkpoint_format == _CHECKPOINT_FORMAT_DIFFUSERS:
            # save the original checkpoint using orbax
            self.save_checkpoint(0)

