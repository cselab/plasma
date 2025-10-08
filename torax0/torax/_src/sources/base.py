import abc
from typing import Annotated
import chex
from torax._src.sources import runtime_params
from torax._src.sources import source as source_lib
from torax._src.torax_pydantic import torax_pydantic


class SourceModelBase(torax_pydantic.BaseModelFrozen, abc.ABC):
    mode: Annotated[runtime_params.Mode,
                    torax_pydantic.JAX_STATIC] = (runtime_params.Mode.ZERO)
    is_explicit: Annotated[bool, torax_pydantic.JAX_STATIC] = False
    prescribed_values: tuple[torax_pydantic.TimeVaryingArray,
                             ...] = (torax_pydantic.ValidatedDefault(({
                                 0: {
                                     0: 0,
                                     1: 0
                                 }
                             }, )))

    @abc.abstractmethod
    def build_source(self) -> source_lib.Source:
        pass

    @property
    @abc.abstractmethod
    def model_func(self) -> source_lib.SourceProfileFunction:
        pass

    @abc.abstractmethod
    def build_runtime_params(self,
                             t: chex.Numeric) -> runtime_params.RuntimeParams:
        pass
