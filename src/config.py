""" config.py """
from easydict import EasyDict as ed

# config for ResNetv2, cifar10
config1 = ed({
    "class_num": 10,
    "batch_size": 32,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "epoch_size": 200,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./checkpoint",
    "low_memory": False,
    "warmup_epochs": 5,
    "lr_decay_mode": "cosine",
    "lr_init": 0.1,
    "lr_end": 0.0000000005,
    "lr_max": 0.1,
})

# config for ResNetv2, cifar100
config2 = ed({
    "class_num": 100,
    "batch_size": 32,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "epoch_size": 100,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./checkpoint",
    "low_memory": False,
    "warmup_epochs": 5,
    "lr_decay_mode": "cosine",
    "lr_init": 0.1,
    "lr_end": 0.0000000005,
    "lr_max": 0.1,
})
