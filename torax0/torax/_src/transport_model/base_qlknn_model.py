import abc
from typing import TypeAlias
import jax
from torax._src.transport_model import qualikiz_based_transport_model

ModelOutput: TypeAlias = dict[str, jax.Array]
InputsAndRanges: TypeAlias = dict[str, dict[str, float]]


class BaseQLKNNModel(abc.ABC):

    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
