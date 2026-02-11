import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode, init_deepspeed_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file")
    parser.add_argument("--use-deepspeed", default=False, action="store_true")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead."
    )

    args = parser.parse_args()

    return args

def setup_seeds(config):
    # allocate different seed to different processes
    seed = config.run_cfg.seed + get_rank()  # get_rank() returns present rank id

    # allocate seed to python, numpy and torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use determinism algorithms
    cudnn.benchmark = False  # compare and choose the best algorithm if True
    cudnn.deterministic = True

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    # the inner get_runner_class() is registry's function, used to return class according to string.
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()  # now() returns present time stamp

    # initialize config
    args = parse_args()
    cfg = Config(args)  # Config() function in lavis

    # use deepspeed or not
    if not args.use_deepspeed:
        init_distributed_mode(cfg.run_cfg)
    else:
        init_deepspeed_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    # print your config
    cfg.pretty_print()

    # instantiate a task
    task = tasks.setup_task(cfg)  # tasks in lavis
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()

if __name__ == '__main__':
    main()