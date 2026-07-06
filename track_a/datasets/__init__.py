"""
track_a/datasets
================
One module per Track A dataset (gastrovision, ham10000, pathmnist), each
exposing the same interface so track_a/main.py never has to special-case
a specific dataset:

    CLASS_NAMES     : dict[int, str]           contiguous label -> display name
    CLASS_COUNTS    : dict[int, int]            contiguous label -> natural
                                                 total image count (pre-split)
    CLASS_PROMPTS   : dict[int, str]            per-class SD text prompt
    DOMAIN_PREFIX   : str                       shared prefix for all prompts
    NEGATIVE_PROMPT : str
    NATURALLY_RARE  : list[int]                 classes with real scarcity
    SWEEP_CLASSES   : list[int]                 classes swept across N_GRID
                                                 (naturally rare classes, plus
                                                 any artificially-designated
                                                 classes for datasets with no
                                                 natural scarcity, e.g. PathMNIST)
    get_splits(data_dir) -> (train_df, val_df, test_df)
                                                 each with columns
                                                 [image_path, label, class_name]

Registry lookup mirrors gastrovision/models.py's MODEL_REGISTRY pattern.
"""

from . import gastrovision, ham10000, pathmnist

DATASET_REGISTRY = {
    "gastrovision": gastrovision,
    "ham10000":     ham10000,
    "pathmnist":    pathmnist,
}


def get_dataset(name: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]
