import os
import os.path as osp

import contextlib

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import transformers

from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec

from .minigpt4qwen import Minigpt4Qwen

