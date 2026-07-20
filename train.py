import os, sys
import argparse
import math
import shutil
import subprocess
import torch
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

from src.utils.train_util import instantiate_from_config


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="only resume model weights",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="base_config.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="experiment name",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes to use",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,",
        help="gpu ids to use",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging data",
    )
    return parser


class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "project.yaml"))


class CodeSnapshot(Callback):
    """
    Modified from https://github.com/threestudio-project/threestudio/blob/main/threestudio/utils/callbacks.py#L60
    """
    def __init__(self, savedir):
        self.savedir = savedir

    def get_file_list(self):
        return [
            b.decode()
            for b in set(
                subprocess.check_output(
                    'git ls-files -- ":!:configs/*"', shell=True
                ).splitlines()
            )
            | set(  # hard code, TODO: use config to exclude folders or files
                subprocess.check_output(
                    "git ls-files --others --exclude-standard", shell=True
                ).splitlines()
            )
        ]

    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except:
            rank_zero_warn(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


class AdapterOnlyCheckpoint(Callback):
    """Save compact RAG adapter weights without the frozen Zero123++ pipeline."""

    def __init__(self, dirpath, save_steps=None, save_last=True):
        super().__init__()
        self.dirpath = dirpath
        self.save_steps = {int(step) for step in (save_steps or [])}
        self.save_last = bool(save_last)
        self._saved_steps = set()

    def _save(self, trainer, pl_module, filename):
        if trainer.global_rank != 0:
            return
        if getattr(pl_module, "rag_adapter", None) is None:
            raise RuntimeError("Adapter-only checkpointing requires model.rag_adapter.")
        os.makedirs(self.dirpath, exist_ok=True)
        payload = {
            "state_dict": {
                f"rag_adapter.{name}": tensor.detach().cpu()
                for name, tensor in pl_module.rag_adapter.state_dict().items()
            },
            "global_step": int(trainer.global_step),
            "adapter_type": "view-aware reference-token adapter with spatial gating",
            "rag_spatial_gating": bool(getattr(pl_module, "rag_spatial_gating", False)),
            "rag_spatial_gate_scale": float(getattr(pl_module, "rag_spatial_gate_scale", 1.0)),
            "rag_token_scale": float(getattr(pl_module, "rag_token_scale", 0.1)),
            "rag_global_scale": float(getattr(pl_module, "rag_global_scale", 0.05)),
        }
        path = os.path.join(self.dirpath, filename)
        torch.save(payload, path)
        pl_module.rag_last_adapter_checkpoint_path = os.path.abspath(path)
        rank_zero_print(f"[RAG-ADAPTER] saved adapter-only checkpoint: {path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step in self.save_steps and step not in self._saved_steps:
            self._save(trainer, pl_module, f"adapter_step_{step:06d}.pt")
            self._saved_steps.add(step)

    def on_train_end(self, trainer, pl_module):
        if self.save_last:
            self._save(trainer, pl_module, "adapter_last.pt")


if __name__ == "__main__":
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    cfg_fname = os.path.split(opt.base)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    exp_name = "-" + opt.name if opt.name != "" else ""
    logdir = os.path.join(opt.logdir, cfg_name+exp_name)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    codedir = os.path.join(logdir, "code")
    seed_everything(opt.seed)

    # init configs
    config = OmegaConf.load(opt.base)
    lightning_config = config.lightning
    trainer_config = lightning_config.trainer
    if trainer_config.get("limit_val_batches", None) == 0:
        rank_zero_print("[RAG-ADAPTER] validation disabled for fast overfit")
    
    trainer_config["accelerator"] = "gpu"
    rank_zero_print(f"Running on GPUs {opt.gpus}")
    ngpu = len(opt.gpus.strip(",").split(','))
    trainer_config['devices'] = ngpu

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)
    if opt.resume and opt.resume_weights_only:
        model = model.__class__.load_from_checkpoint(opt.resume, **config.model.params)
    
    model.logdir = logdir

    # trainer and callbacks
    trainer_kwargs = dict()

    # logger
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": "tensorboard",
            "save_dir": logdir, 
            "version": "0",
        }
    }
    logger_cfg = OmegaConf.merge(default_logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # model checkpoint
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:08}",
            "verbose": True,
            "save_last": True,
            "every_n_train_steps": 5000,
            "save_top_k": -1,   # save all checkpoints
        }
    }

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # callbacks
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train.SetupCallback",
            "params": {
                "resume": opt.resume,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
    }
    if not bool(lightning_config.get("disable_code_snapshot", False)):
        default_callbacks_cfg["code_snapshot"] = {
            "target": "train.CodeSnapshot",
            "params": {
                "savedir": codedir,
            },
        }
    disable_full_checkpoints = bool(lightning_config.get("disable_full_checkpoints", False))
    if not disable_full_checkpoints:
        default_callbacks_cfg["checkpoint_callback"] = modelckpt_cfg
    else:
        trainer_config["enable_checkpointing"] = False

    adapter_checkpoint_cfg = lightning_config.get("adapter_only_checkpoint", None)
    if adapter_checkpoint_cfg and adapter_checkpoint_cfg.get("enabled", False):
        default_callbacks_cfg["adapter_only_checkpoint"] = {
            "target": "train.AdapterOnlyCheckpoint",
            "params": {
                "dirpath": os.path.join(logdir, "adapter_checkpoints"),
                "save_steps": list(adapter_checkpoint_cfg.get("save_steps", [500])),
                "save_last": bool(adapter_checkpoint_cfg.get("save_last", True)),
            },
        }

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    trainer_kwargs['precision'] = '32-true'
    if opt.num_nodes > 1 or ngpu > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)

    # trainer
    trainer = Trainer(**trainer_config, **trainer_kwargs, num_nodes=opt.num_nodes)
    trainer.logdir = logdir

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup("fit")

    train_samples = len(data.datasets["train"])
    batch_size = int(config.data.params.batch_size)
    accumulate = int(trainer_config.get("accumulate_grad_batches", 1))
    steps_per_epoch = math.ceil(train_samples / max(1, batch_size * ngpu * accumulate))
    max_steps = int(trainer_config.get("max_steps", -1))
    validation_disabled = trainer_config.get("limit_val_batches", None) == 0
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    checkpoint_steps = [] if not adapter_checkpoint_cfg else list(adapter_checkpoint_cfg.get("save_steps", []))
    rank_zero_print(f"[RAG-ADAPTER] training samples: {train_samples}")
    rank_zero_print(f"[RAG-ADAPTER] estimated steps per epoch: {steps_per_epoch}")
    rank_zero_print(f"[RAG-ADAPTER] max_steps: {max_steps}")
    rank_zero_print(f"[RAG-ADAPTER] validation disabled: {validation_disabled}")
    rank_zero_print(f"[RAG-ADAPTER] adapter checkpoint steps: {checkpoint_steps}; save_last={bool(adapter_checkpoint_cfg and adapter_checkpoint_cfg.get('save_last', True))}")
    rank_zero_print(f"[RAG-ADAPTER] adapter-only checkpoint saving: {bool(adapter_checkpoint_cfg and adapter_checkpoint_cfg.get('enabled', False))}")
    rank_zero_print(f"[RAG-ADAPTER] full Lightning checkpoints disabled: {disable_full_checkpoints}")
    rank_zero_print(f"[RAG-ADAPTER] trainable parameter count: {trainable_count}")
    rank_zero_print(f"[RAG-ADAPTER] spatial_gating: {bool(getattr(model, 'rag_spatial_gating', False))}")
    rank_zero_print(f"[RAG-ADAPTER] periodic tensor metrics: {bool(getattr(model, 'rag_debug_metrics_enabled', False))}; interval={int(getattr(model, 'rag_debug_metrics_interval', 0))}")
    rank_zero_print(f"[RAG-ADAPTER] post-train smoke test: {bool(getattr(model, 'rag_post_train_smoke_test', False))}")
    rank_zero_print("[RAG-ADAPTER] auto_view_assignment: disabled")

    # configure learning rate
    base_lr = config.model.base_learning_rate
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    model.learning_rate = base_lr
    rank_zero_print("++++ NOT USING LR SCALING ++++")
    rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")

    # run training loop
    if opt.resume and not opt.resume_weights_only:
        trainer.fit(model, data, ckpt_path=opt.resume)
    else:
        trainer.fit(model, data)

    if (
        trainer.global_rank == 0
        and bool(getattr(model, 'rag_post_train_smoke_test', False))
        and int(trainer.global_step) >= int(getattr(model, 'rag_post_train_smoke_steps', 0))
    ):
        from zero123plus.rag_smoke import run_post_train_rag_smoke_test

        checkpoint_path = getattr(model, 'rag_last_adapter_checkpoint_path', None)
        if checkpoint_path is None:
            candidate = os.path.join(logdir, 'adapter_checkpoints', 'adapter_last.pt')
            checkpoint_path = candidate if os.path.exists(candidate) else None
        try:
            run_post_train_rag_smoke_test(model, checkpoint_path=checkpoint_path)
        except Exception as error:
            smoke_dir = os.path.join(logdir, 'rag_smoke_test')
            os.makedirs(smoke_dir, exist_ok=True)
            failure_report = {
                'verdict': 'FAIL',
                'error': f'{type(error).__name__}: {error}',
                'adapter_checkpoint_used': checkpoint_path,
            }
            with open(os.path.join(smoke_dir, 'difference_report.json'), 'w', encoding='utf-8') as handle:
                import json
                json.dump(failure_report, handle, indent=2, sort_keys=True)
            rank_zero_warn(f"RAG post-train smoke test failed: {type(error).__name__}: {error}")
