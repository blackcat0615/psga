import torch
from torch.utils.data import DataLoader

# SmolVLM dataset imports
from .dataset_smolvlm import (
    SmolVLMDataReader,
    SmolVLMDataReaderWithPadding,
    create_smolvlm_dataloader,
)


def worker_init_fn(worker_id: int):
    """Worker process initialization: set random seeds and configure TensorFlow."""
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)
    
    # Configure TensorFlow environment to avoid GPU contention
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass


__all__ = [
    # SmolVLM dataset
    "SmolVLMDataReader",
    "SmolVLMDataReaderWithPadding",
    "create_smolvlm_dataloader",
    "worker_init_fn",
]
