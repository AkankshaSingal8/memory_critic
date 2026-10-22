"""Write a tiny BC image config for tool_hang/ph using robomimic's paper image settings (2 cams, 240px, 216 crops)."""
import json, sys
from robomimic.config import config_factory
from robomimic.scripts.generate_paper_configs import modify_config_for_default_image_exp, modify_config_for_dataset

dataset, out_dir = sys.argv[1], sys.argv[2]
config = config_factory("bc")
config = modify_config_for_default_image_exp(config)
config = modify_config_for_dataset(config, "tool_hang", "ph", "image", base_dataset_dir="/unused")
with config.values_unlocked():
    config.observation.modalities.obs.rgb = ["sideview_image", "robot0_eye_in_hand_image"]
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 216
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 216
    # smoke-test overrides: tiny training, short rollout, no checkpoints/logging
    config.experiment.name = "smoke_image"
    config.experiment.validate = False
    config.train.hdf5_validation_filter_key = None
    config.experiment.render_video = False
    config.experiment.save.enabled = False
    config.experiment.logging.log_tb = False
    config.experiment.logging.log_wandb = False
    config.experiment.epoch_every_n_steps = 20
    config.experiment.rollout.update(dict(n=2, horizon=50, rate=2))
    config.train.data = [{"path": dataset}]
    config.train.output_dir = out_dir
    config.train.num_epochs = 2
    config.train.cuda = True
print(json.dumps(config, indent=2))
