from typing import Annotated, Literal
from torax._src.neoclassical.bootstrap_current import base
from torax._src.neoclassical.bootstrap_current import runtime_params as bootstrap_runtime_params
from torax._src.torax_pydantic import torax_pydantic


class ZerosModel(base.BootstrapCurrentModel):
    pass


class ZerosModelConfig(base.BootstrapCurrentModelConfig):
    model_name: Annotated[Literal['zeros'],
                          torax_pydantic.JAX_STATIC] = 'zeros'

    def build_runtime_params(self):
      pass

    def build_model(self):
      pass
