import dataclasses
import enum
from typing import Annotated, Final
import jax
import pydantic
from torax._src import array_typing
from torax._src.torax_pydantic import torax_pydantic

_MIN_IP_AMPS: Final[float] = 1e3
_MIN_DENSITY_M3: Final[float] = 1e10
_MAX_DENSITY_GW: Final[float] = 1e2
_MAX_TEMPERATURE_KEV: Final[float] = 1e3
_MAX_TEMPERATURE_BC_KEV: Final[float] = 5e1


class InitialPsiMode(enum.StrEnum):
    PROFILE_CONDITIONS = 'profile_conditions'
    GEOMETRY = 'geometry'
    J = 'j'


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParams:
    Ip: array_typing.FloatScalar
    v_loop_lcfs: array_typing.FloatScalar
    T_i_right_bc: array_typing.FloatScalar
    T_e_right_bc: array_typing.FloatScalar
    T_e: array_typing.FloatVector
    T_i: array_typing.FloatVector
    psi: array_typing.FloatVector | None
    psidot: array_typing.FloatVector | None
    n_e: array_typing.FloatVector
    nbar: array_typing.FloatScalar
    n_e_nbar_is_fGW: bool
    n_e_right_bc: array_typing.FloatScalar
    n_e_right_bc_is_fGW: bool
    current_profile_nu: float
    initial_j_is_total_current: bool
    initial_psi_from_j: bool
    normalize_n_e_to_nbar: bool = dataclasses.field(metadata={'static': True})
    use_v_loop_lcfs_boundary_condition: bool = dataclasses.field(
        metadata={'static': True})
    n_e_right_bc_is_absolute: bool = dataclasses.field(
        metadata={'static': True})
    initial_psi_mode: InitialPsiMode = dataclasses.field(
        metadata={'static': True})


class ProfileConditions(torax_pydantic.BaseModelFrozen):
    Ip: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        15e6)
    use_v_loop_lcfs_boundary_condition: Annotated[
        bool, torax_pydantic.JAX_STATIC] = False
    v_loop_lcfs: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.0))
    T_i_right_bc: torax_pydantic.PositiveTimeVaryingScalar | None = None
    T_e_right_bc: torax_pydantic.PositiveTimeVaryingScalar | None = None
    T_i: torax_pydantic.PositiveTimeVaryingArray = (
        torax_pydantic.ValidatedDefault({0: {
            0: 15.0,
            1: 1.0
        }}))
    T_e: torax_pydantic.PositiveTimeVaryingArray = (
        torax_pydantic.ValidatedDefault({0: {
            0: 15.0,
            1: 1.0
        }}))
    psi: torax_pydantic.TimeVaryingArray | None = None
    psidot: torax_pydantic.TimeVaryingArray | None = None
    n_e: torax_pydantic.PositiveTimeVaryingArray = (
        torax_pydantic.ValidatedDefault({0: {
            0: 1.2e20,
            1: 0.8e20
        }}))
    normalize_n_e_to_nbar: Annotated[bool, torax_pydantic.JAX_STATIC] = False
    nbar: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        0.85e20)
    n_e_nbar_is_fGW: bool = False
    n_e_right_bc: torax_pydantic.TimeVaryingScalar | None = None
    n_e_right_bc_is_fGW: bool = False
    current_profile_nu: float = 1.0
    initial_j_is_total_current: bool = False
    initial_psi_from_j: bool = False
    initial_psi_mode: Annotated[InitialPsiMode, torax_pydantic.JAX_STATIC] = (
        InitialPsiMode.PROFILE_CONDITIONS)

    @pydantic.model_validator(mode='after')
    def after_validator(self):
        return self

    def build_runtime_params(self, t):
        runtime_params = {
            x.name: getattr(self, x.name)
            for x in dataclasses.fields(RuntimeParams)
            if x.name != 'n_e_right_bc_is_absolute'
        }
        runtime_params['n_e_right_bc_is_absolute'] = True

        def _get_value(x):
            if isinstance(x, (torax_pydantic.TimeVaryingScalar,
                              torax_pydantic.TimeVaryingArray)):
                return x.get_value(t)
            else:
                return x

        runtime_params = {k: _get_value(v) for k, v in runtime_params.items()}
        return RuntimeParams(**runtime_params)
