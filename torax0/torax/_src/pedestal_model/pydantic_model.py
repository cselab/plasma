import abc
from typing import Annotated, Literal
import chex
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params
from torax._src.pedestal_model import set_tped_nped
from torax._src.torax_pydantic import torax_pydantic


class BasePedestal(torax_pydantic.BaseModelFrozen, abc.ABC):
    set_pedestal: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(False))

    @abc.abstractmethod
    def build_pedestal_model(self) -> pedestal_model.PedestalModel:
        pass

    @abc.abstractmethod
    def build_runtime_params(self,
                             t: chex.Numeric) -> runtime_params.RuntimeParams:
        pass


class SetTpedNped(BasePedestal):
    model_name: Annotated[Literal['set_T_ped_n_ped'],
                          torax_pydantic.JAX_STATIC] = 'set_T_ped_n_ped'
    n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        0.7e20)
    n_e_ped_is_fGW: bool = False
    T_i_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        5.0)
    T_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        5.0)
    rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.91))

    def build_pedestal_model(
        self, ) -> set_tped_nped.SetTemperatureDensityPedestalModel:
        return set_tped_nped.SetTemperatureDensityPedestalModel()

    def build_runtime_params(self,
                             t: chex.Numeric) -> set_tped_nped.RuntimeParams:
        return set_tped_nped.RuntimeParams(
            set_pedestal=self.set_pedestal.get_value(t),
            n_e_ped=self.n_e_ped.get_value(t),
            n_e_ped_is_fGW=self.n_e_ped_is_fGW,
            T_i_ped=self.T_i_ped.get_value(t),
            T_e_ped=self.T_e_ped.get_value(t),
            rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
        )


PedestalConfig = SetTpedNped
