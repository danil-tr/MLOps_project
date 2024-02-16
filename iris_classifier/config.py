from dataclasses import dataclass
from typing import Any


@dataclass
class IrisData:
    name: str
    path: str


@dataclass
class TrackingServer:
    experiment_name: str
    uri: str


@dataclass
class Model:
    name: str
    random_seed: int
    iterations: int
    learning_rate: float
    l2_leaf_reg: float
    bagging_temperature: float
    random_strength: float
    one_hot_max_size: float
    leaf_estimation_method: str
    silent: bool
    allow_writing_files: bool


@dataclass
class Params:
    data: Any
    model: Model
