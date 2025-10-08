import abc
import chex
import pydantic
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import runtime_params as runtime_params0
from torax._src.transport_model import transport_model


class TransportBase(torax_pydantic.BaseModelFrozen, abc.ABC):
    chi_min: torax_pydantic.MeterSquaredPerSecond = 0.05
    chi_max: torax_pydantic.MeterSquaredPerSecond = 100.0
    D_e_min: torax_pydantic.MeterSquaredPerSecond = 0.05
    D_e_max: torax_pydantic.MeterSquaredPerSecond = 100.0
    V_e_min: torax_pydantic.MeterPerSecond = -50.0
    V_e_max: torax_pydantic.MeterPerSecond = 50.0
    rho_min: torax_pydantic.UnitIntervalTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.0))
    rho_max: torax_pydantic.UnitIntervalTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(1.0))
    apply_inner_patch: interpolated_param_1d.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(False))
    D_e_inner: torax_pydantic.PositiveTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.2))
    V_e_inner: interpolated_param_1d.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.0))
    chi_i_inner: torax_pydantic.PositiveTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(1.0))
    chi_e_inner: torax_pydantic.PositiveTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(1.0))
    rho_inner: torax_pydantic.UnitIntervalTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.3))
    apply_outer_patch: interpolated_param_1d.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(False))
    D_e_outer: interpolated_param_1d.PositiveTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.2))
    V_e_outer: interpolated_param_1d.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.0))
    chi_i_outer: interpolated_param_1d.PositiveTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(1.0))
    chi_e_outer: interpolated_param_1d.PositiveTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(1.0))
    rho_outer: torax_pydantic.UnitIntervalTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.9))
    smoothing_width: pydantic.NonNegativeFloat = 0.0
    smooth_everywhere: bool = False

    def build_runtime_params(self, t):
        return runtime_params0.RuntimeParams(
            chi_min=self.chi_min,
            chi_max=self.chi_max,
            D_e_min=self.D_e_min,
            D_e_max=self.D_e_max,
            V_e_min=self.V_e_min,
            V_e_max=self.V_e_max,
            rho_min=self.rho_min.get_value(t),
            rho_max=self.rho_max.get_value(t),
            apply_inner_patch=self.apply_inner_patch.get_value(t),
            D_e_inner=self.D_e_inner.get_value(t),
            V_e_inner=self.V_e_inner.get_value(t),
            chi_i_inner=self.chi_i_inner.get_value(t),
            chi_e_inner=self.chi_e_inner.get_value(t),
            rho_inner=self.rho_inner.get_value(t),
            apply_outer_patch=self.apply_outer_patch.get_value(t),
            D_e_outer=self.D_e_outer.get_value(t),
            V_e_outer=self.V_e_outer.get_value(t),
            chi_i_outer=self.chi_i_outer.get_value(t),
            chi_e_outer=self.chi_e_outer.get_value(t),
            rho_outer=self.rho_outer.get_value(t),
            smoothing_width=self.smoothing_width,
            smooth_everywhere=self.smooth_everywhere,
        )

    @abc.abstractmethod
    def build_transport_model(self) -> transport_model.TransportModel:
        pass
