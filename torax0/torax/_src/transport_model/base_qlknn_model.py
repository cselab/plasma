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
  @property
  @abc.abstractmethod
  def inputs_and_ranges(self) -> InputsAndRanges:
    raise NotImplementedError()
  @abc.abstractmethod
  def predict(
      self,
      inputs: jax.Array,
  ) -> ModelOutput:
    raise NotImplementedError()
  @abc.abstractmethod
  def get_model_inputs_from_qualikiz_inputs(
      self, qualikiz_inputs: qualikiz_based_transport_model.QualikizInputs
  ) -> jax.Array:
    raise NotImplementedError()
