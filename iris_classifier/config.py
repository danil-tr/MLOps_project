from dataclasses import dataclass
from typing import Any


@dataclass
class IrisData:
    name: str
    path: str


@dataclass
class Model:
    name: str
    penalty: str
    dual: bool
    tol: float
    C: float
    fit_intercept: bool
    intercept_scaling: float
    solver: str
    max_iter: int


@dataclass
class Params:
    data: Any
    model: Model
