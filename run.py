from absl import logging
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from fusion_surrogates.qlknn import qlknn_model
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import constants as constants_module
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src import xnp
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.config import runtime_validation_utils
from torax._src.fvm import cell_variable
from torax._src.fvm import convection_terms
from torax._src.fvm import diffusion_terms
from torax._src.fvm import enums
from torax._src.geometry import geometry
from torax._src.geometry import geometry as geometry_lib
from torax._src.geometry import geometry_provider
from torax._src.geometry import standard_geometry
from torax._src.neoclassical import neoclassical_models
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.neoclassical import runtime_params
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.bootstrap_current import sauter as sauter_current
from torax._src.neoclassical.bootstrap_current import zeros as bootstrap_current_zeros
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.conductivity import sauter as sauter_conductivity
from torax._src.neoclassical.transport import zeros as transport_zeros
from torax._src.physics import charge_states
from torax._src.physics import collisions
from torax._src.physics import formulas as formulas_ph
from torax._src.physics import psi_calculations
from torax._src.physics import scaling_laws
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import torax_pydantic
from typing import Annotated
from typing import Annotated, Any, Literal, TypeAlias, TypeVar, ClassVar, Final, Mapping, Protocol, Callable
from typing import ClassVar, Protocol
from typing import Literal
from typing_extensions import Annotated, Final, override
import abc
import chex
import copy
import dataclasses
import enum
import functools
import immutabledict
import inspect
import itertools
import jax
import jax.numpy as jnp
import jaxtyping as jt
import logging
import numpy as np
import pydantic
import typing
import typing_extensions
import xarray as xr


def exponential_profile(
    geo: geometry.Geometry,
    *,
    decay_start: float,
    width: float,
    total: float,
) -> jax.Array:
    r = geo.rho_norm
    S = jnp.exp(-(decay_start - r) / width)
    C = total / math_utils.volume_integration(S, geo)
    return C * S


def gaussian_profile(
    geo: geometry.Geometry,
    *,
    center: float,
    width: float,
    total: float,
) -> jax.Array:
    r = geo.rho_norm
    S = jnp.exp(-((r - center)**2) / (2 * width**2))
    C = total / math_utils.volume_integration(S, geo)
    return C * S


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QeiInfo:
    qei_coef: jax.Array
    implicit_ii: jax.Array
    explicit_i: jax.Array
    implicit_ee: jax.Array
    explicit_e: jax.Array
    implicit_ie: jax.Array
    implicit_ei: jax.Array

    @classmethod
    def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
        return QeiInfo(
            qei_coef=jnp.zeros_like(geo.rho),
            implicit_ii=jnp.zeros_like(geo.rho),
            explicit_i=jnp.zeros_like(geo.rho),
            implicit_ee=jnp.zeros_like(geo.rho),
            explicit_e=jnp.zeros_like(geo.rho),
            implicit_ie=jnp.zeros_like(geo.rho),
            implicit_ei=jnp.zeros_like(geo.rho),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SourceProfiles:
    bootstrap_current: bootstrap_current_base.BootstrapCurrent
    qei: QeiInfo
    T_e: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    T_i: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    n_e: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    psi: dict[str, jax.Array] = dataclasses.field(default_factory=dict)

    @classmethod
    def merge(
        cls,
        explicit_source_profiles: typing_extensions.Self,
        implicit_source_profiles: typing_extensions.Self,
    ) -> typing_extensions.Self:
        sum_profiles = lambda a, b: a + b
        return jax.tree_util.tree_map(sum_profiles, explicit_source_profiles,
                                      implicit_source_profiles)

    def total_psi_sources(self, geo: geometry.Geometry) -> jax.Array:
        total = self.bootstrap_current.j_bootstrap
        total += sum(self.psi.values())
        mu0 = constants.CONSTANTS.mu_0
        prefactor = 8 * geo.vpr * jnp.pi**2 * geo.B_0 * mu0 * geo.Phi_b / geo.F**2
        return -total * prefactor

    def total_sources(
        self,
        source_type: Literal['n_e', 'T_i', 'T_e'],
        geo: geometry.Geometry,
    ) -> jax.Array:
        source: dict[str, jax.Array] = getattr(self, source_type)
        total = sum(source.values())
        return total * geo.vpr


@enum.unique
class Mode(enum.Enum):
    ZERO = "ZERO"
    MODEL_BASED = "MODEL_BASED"
    PRESCRIBED = "PRESCRIBED"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsSrc:
    prescribed_values: tuple[array_typing.FloatVector, ...]
    mode: Mode = dataclasses.field(metadata={"static": True})
    is_explicit: bool = dataclasses.field(metadata={"static": True})


class SourceModelBase(torax_pydantic.BaseModelFrozen, abc.ABC):
    mode: Annotated[Mode, torax_pydantic.JAX_STATIC] = (Mode.ZERO)
    is_explicit: Annotated[bool, torax_pydantic.JAX_STATIC] = False
    prescribed_values: tuple[torax_pydantic.TimeVaryingArray,
                             ...] = (torax_pydantic.ValidatedDefault(({
                                 0: {
                                     0: 0,
                                     1: 0
                                 }
                             }, )))

    @abc.abstractmethod
    def build_source(self):
        pass

    @property
    @abc.abstractmethod
    def model_func(self):
        pass

    @abc.abstractmethod
    def build_runtime_params(self, t):
        pass


@typing.runtime_checkable
class SourceProfileFunction(Protocol):

    def __call__(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        source_name: str,
        core_profiles: state.CoreProfiles,
        calculated_source_profiles: SourceProfiles | None,
        unused_conductivity: conductivity_base.Conductivity | None,
    ) -> tuple[array_typing.FloatVectorCell, ...]:
        ...


@enum.unique
class AffectedCoreProfile(enum.IntEnum):
    PSI = 1
    NE = 2
    TEMP_ION = 3
    TEMP_EL = 4


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Source(abc.ABC):
    SOURCE_NAME: ClassVar[str] = 'source'
    model_func: SourceProfileFunction | None = None

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def affected_core_profiles(self) -> tuple[AffectedCoreProfile, ...]:
        pass

    def get_value(self, runtime_params, geo, core_profiles,
                  calculated_source_profiles, conductivity):
        source_params = runtime_params.sources[self.source_name]
        mode = source_params.mode
        match mode:
            case Mode.MODEL_BASED:
                return self.model_func(
                    runtime_params,
                    geo,
                    self.source_name,
                    core_profiles,
                    calculated_source_profiles,
                    conductivity,
                )
            case _:
                raise ValueError(f'Unknown mode: {mode}')


@typing.runtime_checkable
class SourceProfileFunction(Protocol):

    def __call__(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        source_name: str,
        core_profiles: state.CoreProfiles,
        calculated_source_profiles: SourceProfiles | None,
        unused_conductivity: conductivity_base.Conductivity | None,
    ) -> tuple[array_typing.FloatVectorCell, ...]:
        ...


@enum.unique
class AffectedCoreProfile(enum.IntEnum):
    PSI = 1
    NE = 2
    TEMP_ION = 3
    TEMP_EL = 4


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Source(abc.ABC):
    SOURCE_NAME: ClassVar[str] = 'source'
    model_func: SourceProfileFunction | None = None

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def affected_core_profiles(self) -> tuple[AffectedCoreProfile, ...]:
        pass

    def get_value(self, runtime_params, geo, core_profiles,
                  calculated_source_profiles, conductivity):
        source_params = runtime_params.sources[self.source_name]
        mode = source_params.mode
        match mode:
            case Mode.MODEL_BASED:
                return self.model_func(
                    runtime_params,
                    geo,
                    self.source_name,
                    core_profiles,
                    calculated_source_profiles,
                    conductivity,
                )
            case _:
                raise ValueError(f'Unknown mode: {mode}')


@typing.runtime_checkable
class SourceProfileFunction(Protocol):

    def __call__(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        source_name: str,
        core_profiles: state.CoreProfiles,
        calculated_source_profiles: SourceProfiles | None,
        unused_conductivity: conductivity_base.Conductivity | None,
    ) -> tuple[array_typing.FloatVectorCell, ...]:
        ...


@enum.unique
class AffectedCoreProfile(enum.IntEnum):
    PSI = 1
    NE = 2
    TEMP_ION = 3
    TEMP_EL = 4


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Source(abc.ABC):
    SOURCE_NAME: ClassVar[str] = 'source'
    model_func: SourceProfileFunction | None = None

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def affected_core_profiles(self) -> tuple[AffectedCoreProfile, ...]:
        pass

    def get_value(self, runtime_params, geo, core_profiles,
                  calculated_source_profiles, conductivity):
        source_params = runtime_params.sources[self.source_name]
        mode = source_params.mode
        match mode:
            case Mode.MODEL_BASED:
                return self.model_func(
                    runtime_params,
                    geo,
                    self.source_name,
                    core_profiles,
                    calculated_source_profiles,
                    conductivity,
                )
            case _:
                raise ValueError(f'Unknown mode: {mode}')


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsGcS(RuntimeParamsSrc):
    I_generic: array_typing.FloatScalar
    fraction_of_total_current: array_typing.FloatScalar
    gaussian_width: array_typing.FloatScalar
    gaussian_location: array_typing.FloatScalar
    use_absolute_current: bool


def calculate_generic_current(runtime_params, geo, source_name, unused_state,
                              unused_calculated_source_profiles,
                              unused_conductivity):
    source_params = runtime_params.sources[source_name]
    I_generic = _calculate_I_generic(
        runtime_params,
        source_params,
    )
    generic_current_form = jnp.exp(
        -((geo.rho_norm - source_params.gaussian_location)**2) /
        (2 * source_params.gaussian_width**2))
    Cext = I_generic / math_utils.area_integration(generic_current_form, geo)
    generic_current_profile = Cext * generic_current_form
    return (generic_current_profile, )


def _calculate_I_generic(runtime_params, source_params):
    return jnp.where(
        source_params.use_absolute_current,
        source_params.I_generic,
        (runtime_params.profile_conditions.Ip *
         source_params.fraction_of_total_current),
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericCurrentSource(Source):
    SOURCE_NAME: ClassVar[str] = 'generic_current'
    model_func: SourceProfileFunction = calculate_generic_current

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (AffectedCoreProfile.PSI, )


class GenericCurrentSourceConfig(SourceModelBase):
    model_name: Annotated[Literal['gaussian'],
                          torax_pydantic.JAX_STATIC] = ('gaussian')
    I_generic: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        3.0e6)
    fraction_of_total_current: torax_pydantic.UnitIntervalTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.2))
    gaussian_width: torax_pydantic.PositiveTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.05))
    gaussian_location: torax_pydantic.UnitIntervalTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.4))
    use_absolute_current: bool = False
    mode: Annotated[Mode, torax_pydantic.JAX_STATIC] = (Mode.MODEL_BASED)

    @property
    def model_func(self):
        return calculate_generic_current

    def build_runtime_params(
        self,
        t,
    ):
        return RuntimeParamsGcS(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            I_generic=self.I_generic.get_value(t),
            fraction_of_total_current=self.fraction_of_total_current.get_value(
                t),
            gaussian_width=self.gaussian_width.get_value(t),
            gaussian_location=self.gaussian_location.get_value(t),
            use_absolute_current=self.use_absolute_current,
        )

    def build_source(self) -> GenericCurrentSource:
        return GenericCurrentSource(model_func=self.model_func)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsGeIO(RuntimeParamsSrc):
    gaussian_width: array_typing.FloatScalar
    gaussian_location: array_typing.FloatScalar
    P_total: array_typing.FloatScalar
    electron_heat_fraction: array_typing.FloatScalar
    absorption_fraction: array_typing.FloatScalar


def calc_generic_heat_source(
    geo: geometry.Geometry,
    gaussian_location: float,
    gaussian_width: float,
    P_total: float,
    electron_heat_fraction: float,
    absorption_fraction: float,
) -> tuple[array_typing.FloatVectorCell, array_typing.FloatVectorCell]:
    absorbed_power = P_total * absorption_fraction
    profile = gaussian_profile(geo,
                               center=gaussian_location,
                               width=gaussian_width,
                               total=absorbed_power)
    source_ion = profile * (1 - electron_heat_fraction)
    source_el = profile * electron_heat_fraction
    return source_ion, source_el


def default_formula(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    unused_core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
):
    source_params = runtime_params.sources[source_name]
    ion, el = calc_generic_heat_source(
        geo,
        source_params.gaussian_location,
        source_params.gaussian_width,
        source_params.P_total,
        source_params.electron_heat_fraction,
        source_params.absorption_fraction,
    )
    return (ion, el)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericIonElectronHeatSource(Source):
    SOURCE_NAME: ClassVar[str] = 'generic_heat'
    model_func: SourceProfileFunction = default_formula

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (
            AffectedCoreProfile.TEMP_ION,
            AffectedCoreProfile.TEMP_EL,
        )


class GenericIonElHeatSourceConfig(SourceModelBase):
    model_name: Annotated[Literal['gaussian'],
                          torax_pydantic.JAX_STATIC] = ('gaussian')
    gaussian_width: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.25))
    gaussian_location: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.0))
    P_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        120e6)
    electron_heat_fraction: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.66666))
    absorption_fraction: torax_pydantic.PositiveTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(1.0))
    mode: Annotated[Mode, torax_pydantic.JAX_STATIC] = (Mode.MODEL_BASED)

    @property
    def model_func(self):
        return default_formula

    def build_runtime_params(
        self,
        t: chex.Numeric,
    ):
        return RuntimeParamsGeIO(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            gaussian_width=self.gaussian_width.get_value(t),
            gaussian_location=self.gaussian_location.get_value(t),
            P_total=self.P_total.get_value(t),
            electron_heat_fraction=self.electron_heat_fraction.get_value(t),
            absorption_fraction=self.absorption_fraction.get_value(t),
        )

    def build_source(self) -> GenericIonElectronHeatSource:
        return GenericIonElectronHeatSource(model_func=self.model_func)


def calc_generic_particle_source(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, ...]:
    source_params = runtime_params.sources[source_name]
    return (gaussian_profile(
        center=source_params.deposition_location,
        width=source_params.particle_width,
        total=source_params.S_total,
        geo=geo,
    ), )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericParticleSource(Source):
    SOURCE_NAME: ClassVar[str] = 'generic_particle'
    model_func: SourceProfileFunction = calc_generic_particle_source

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (AffectedCoreProfile.NE, )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsPaSo(RuntimeParamsSrc):
    particle_width: array_typing.FloatScalar
    deposition_location: array_typing.FloatScalar
    S_total: array_typing.FloatScalar


class GenericParticleSourceConfig(SourceModelBase):
    model_name: Annotated[Literal['gaussian'],
                          torax_pydantic.JAX_STATIC] = ('gaussian')
    particle_width: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.25))
    deposition_location: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.0))
    S_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        1e22)
    mode: Annotated[Mode, torax_pydantic.JAX_STATIC] = (Mode.MODEL_BASED)

    @property
    def model_func(self):
        return calc_generic_particle_source

    def build_runtime_params(
        self,
        t: chex.Numeric,
    ):
        return RuntimeParamsPaSo(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            particle_width=self.particle_width.get_value(t),
            deposition_location=self.deposition_location.get_value(t),
            S_total=self.S_total.get_value(t),
        )

    def build_source(self) -> GenericParticleSource:
        return GenericParticleSource(model_func=self.model_func)


def calc_pellet_source(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, ...]:
    source_params = runtime_params.sources[source_name]
    return (gaussian_profile(
        center=source_params.pellet_deposition_location,
        width=source_params.pellet_width,
        total=source_params.S_total,
        geo=geo,
    ), )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PelletSource(Source):
    SOURCE_NAME: ClassVar[str] = 'pellet'
    model_func: SourceProfileFunction = calc_pellet_source

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (AffectedCoreProfile.NE, )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsPE(RuntimeParamsSrc):
    pellet_width: array_typing.FloatScalar
    pellet_deposition_location: array_typing.FloatScalar
    S_total: array_typing.FloatScalar


class PelletSourceConfig(SourceModelBase):
    model_name: Annotated[Literal['gaussian'],
                          torax_pydantic.JAX_STATIC] = ('gaussian')
    pellet_width: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.1))
    pellet_deposition_location: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.85))
    S_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        2e22)
    mode: Annotated[Mode, torax_pydantic.JAX_STATIC] = (Mode.MODEL_BASED)

    @property
    def model_func(self):
        return calc_pellet_source

    def build_runtime_params(self, t):
        return RuntimeParamsPE(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            pellet_width=self.pellet_width.get_value(t),
            pellet_deposition_location=self.pellet_deposition_location.
            get_value(t),
            S_total=self.S_total.get_value(t),
        )

    def build_source(self) -> PelletSource:
        return PelletSource(model_func=self.model_func)


def calc_fusion(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    runtime_params: runtime_params_slice.RuntimeParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if not {'D', 'T'}.issubset(
            runtime_params.plasma_composition.main_ion_names):
        return (
            jnp.array(0.0, dtype=jax_utils.get_dtype()),
            jnp.zeros_like(core_profiles.T_i.value),
            jnp.zeros_like(core_profiles.T_i.value),
        )
    else:
        product = 1.0
        for fraction, symbol in zip(
                runtime_params.plasma_composition.main_ion.fractions,
                runtime_params.plasma_composition.main_ion_names,
        ):
            if symbol == 'D' or symbol == 'T':
                product *= fraction
        DT_fraction_product = product
    t_face = core_profiles.T_i.face_value()
    Efus = 17.6 * 1e3 * constants.CONSTANTS.keV_to_J
    mrc2 = 1124656
    BG = 34.3827
    C1 = 1.17302e-9
    C2 = 1.51361e-2
    C3 = 7.51886e-2
    C4 = 4.60643e-3
    C5 = 1.35e-2
    C6 = -1.0675e-4
    C7 = 1.366e-5
    theta = t_face / (1.0 - (t_face * (C2 + t_face * (C4 + t_face * C6))) /
                      (1.0 + t_face * (C3 + t_face * (C5 + t_face * C7))))
    xi = (BG**2 / (4 * theta))**(1 / 3)
    logsigmav = (jnp.log(C1 * theta) + 0.5 * jnp.log(xi / (mrc2 * t_face**3)) -
                 3 * xi - jnp.log(1e6))
    logPfus = (jnp.log(DT_fraction_product * Efus) +
               2 * jnp.log(core_profiles.n_i.face_value()) + logsigmav)
    Pfus_face = jnp.exp(logPfus)
    Pfus_cell = 0.5 * (Pfus_face[:-1] + Pfus_face[1:])
    P_total = (jax.scipy.integrate.trapezoid(Pfus_face * geo.vpr_face,
                                             geo.rho_face_norm) / 1e6)
    alpha_fraction = 3.5 / 17.6
    birth_energy = 3520
    alpha_mass = 4.002602
    frac_i = collisions.fast_ion_fractional_heating_formula(
        birth_energy,
        core_profiles.T_e.value,
        alpha_mass,
    )
    frac_e = 1.0 - frac_i
    Pfus_i = Pfus_cell * frac_i * alpha_fraction
    Pfus_e = Pfus_cell * frac_e * alpha_fraction
    return P_total, Pfus_i, Pfus_e


def fusion_heat_model_func(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    unused_source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, array_typing.FloatVectorCell]:
    _, Pfus_i, Pfus_e = calc_fusion(
        geo,
        core_profiles,
        runtime_params,
    )
    return (Pfus_i, Pfus_e)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class FusionHeatSource(Source):
    SOURCE_NAME: ClassVar[str] = 'fusion'
    model_func: SourceProfileFunction = fusion_heat_model_func

    @property
    def source_name(self) -> str:
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (
            AffectedCoreProfile.TEMP_ION,
            AffectedCoreProfile.TEMP_EL,
        )


class FusionHeatSourceConfig(SourceModelBase):
    model_name: Annotated[Literal['bosch_hale'],
                          torax_pydantic.JAX_STATIC] = ('bosch_hale')
    mode: Annotated[Mode, torax_pydantic.JAX_STATIC] = (Mode.MODEL_BASED)

    @property
    def model_func(self):
        return fusion_heat_model_func

    def build_runtime_params(
        self,
        t: chex.Numeric,
    ) -> RuntimeParamsSrc:
        return RuntimeParamsSrc(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
        )

    def build_source(self) -> FusionHeatSource:
        return FusionHeatSource(model_func=self.model_func)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsPS(RuntimeParamsSrc):
    puff_decay_length: array_typing.FloatScalar
    S_total: array_typing.FloatScalar


def calc_puff_source(
    runtime_params,
    geo,
    source_name,
    unused_state,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    source_params = runtime_params.sources[source_name]
    return (exponential_profile(
        decay_start=1.0,
        width=source_params.puff_decay_length,
        total=source_params.S_total,
        geo=geo,
    ), )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GasPuffSource(Source):
    SOURCE_NAME: ClassVar[str] = 'gas_puff'
    model_func: SourceProfileFunction = calc_puff_source

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (AffectedCoreProfile.NE, )


class GasPuffSourceConfig(SourceModelBase):
    model_name: Annotated[Literal['exponential'],
                          torax_pydantic.JAX_STATIC] = ('exponential')
    puff_decay_length: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.05))
    S_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        1e22)
    mode: Annotated[Mode, torax_pydantic.JAX_STATIC] = (Mode.MODEL_BASED)

    @property
    def model_func(self):
        return calc_puff_source

    def build_runtime_params(self, t):
        return RuntimeParamsPS(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            puff_decay_length=self.puff_decay_length.get_value(t),
            S_total=self.S_total.get_value(t),
        )

    def build_source(self):
        return GasPuffSource(model_func=self.model_func)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsQ(RuntimeParamsSrc):
    Qei_multiplier: float


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class QeiSource(Source):
    SOURCE_NAME: ClassVar[str] = 'ei_exchange'

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (
            source.AffectedCoreProfile.TEMP_ION,
            source.AffectedCoreProfile.TEMP_EL,
        )

    def get_qei(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ):
        return jax.lax.cond(
            runtime_params.sources[self.source_name].mode == Mode.MODEL_BASED,
            lambda: _model_based_qei(
                runtime_params,
                geo,
                core_profiles,
            ),
            lambda: QeiInfo.zeros(geo),
        )

    def get_value(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
        calculated_source_profiles: SourceProfiles | None,
        conductivity: conductivity_base.Conductivity | None,
    ):
        raise NotImplementedError('Call get_qei() instead.')

    def get_source_profile_for_affected_core_profile(
        self,
        profile: tuple[array_typing.Array, ...],
        affected_mesh_state: int,
        geo: geometry.Geometry,
    ):
        raise NotImplementedError('This method is not valid for QeiSource.')


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SourceModels:
    qei_source: QeiSource
    standard_sources: immutabledict.immutabledict[str, Source]

    @functools.cached_property
    def psi_sources(self):
        return immutabledict.immutabledict({
            name: source
            for name, source in self.standard_sources.items()
            if AffectedCoreProfile.PSI in source.affected_core_profiles
        })

    def __hash__(self):
        hashes = [hash(self.standard_sources)]
        hashes.append(hash(self.qei_source))
        return hash(tuple(hashes))


def _model_based_qei(runtime_params, geo, core_profiles):
    source_params = runtime_params.sources[QeiSource.SOURCE_NAME]
    zeros = jnp.zeros_like(geo.rho_norm)
    qei_coef = collisions.coll_exchange(
        core_profiles=core_profiles,
        Qei_multiplier=source_params.Qei_multiplier,
    )
    implicit_ii = -qei_coef
    implicit_ee = -qei_coef
    if ((runtime_params.numerics.evolve_ion_heat
         and not runtime_params.numerics.evolve_electron_heat)
            or (runtime_params.numerics.evolve_electron_heat
                and not runtime_params.numerics.evolve_ion_heat)):
        explicit_i = qei_coef * core_profiles.T_e.value
        explicit_e = qei_coef * core_profiles.T_i.value
        implicit_ie = zeros
        implicit_ei = zeros
    else:
        explicit_i = zeros
        explicit_e = zeros
        implicit_ie = qei_coef
        implicit_ei = qei_coef
    return QeiInfo(
        qei_coef=qei_coef,
        implicit_ii=implicit_ii,
        explicit_i=explicit_i,
        implicit_ee=implicit_ee,
        explicit_e=explicit_e,
        implicit_ie=implicit_ie,
        implicit_ei=implicit_ei,
    )


class QeiSourceConfig(SourceModelBase):
    Qei_multiplier: float = 1.0
    mode: Annotated[Mode, torax_pydantic.JAX_STATIC] = (Mode.MODEL_BASED)

    @property
    def model_func(self):
        return None

    def build_runtime_params(
        self,
        t: chex.Numeric,
    ):
        return RuntimeParamsQ(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            Qei_multiplier=self.Qei_multiplier,
        )

    def build_source(self):
        return QeiSource(model_func=self.model_func)


class Sources(torax_pydantic.BaseModelFrozen):
    ei_exchange: QeiSourceConfig = torax_pydantic.ValidatedDefault(
        {'mode': 'ZERO'})
    cyclotron_radiation: (None) = pydantic.Field(
        discriminator='model_name',
        default=None,
    )
    fusion: FusionHeatSourceConfig | None = pydantic.Field(
        discriminator='model_name',
        default=None,
    )
    gas_puff: GasPuffSourceConfig | None = pydantic.Field(
        discriminator='model_name',
        default=None,
    )
    generic_current: GenericCurrentSourceConfig = (
        torax_pydantic.ValidatedDefault({'mode': 'ZERO'}))
    generic_heat: (GenericIonElHeatSourceConfig
                   | None) = pydantic.Field(
                       discriminator='model_name',
                       default=None,
                   )
    generic_particle: (GenericParticleSourceConfig
                       | None) = pydantic.Field(
                           discriminator='model_name',
                           default=None,
                       )
    impurity_radiation: (None) = pydantic.Field(
        discriminator='model_name',
        default=None,
    )
    pellet: PelletSourceConfig | None = pydantic.Field(
        discriminator='model_name',
        default=None,
    )

    @pydantic.model_validator(mode='before')
    @classmethod
    def _set_default_model_functions(cls, x):
        constructor_data = copy.deepcopy(x)
        for k, v in x.items():
            match k:
                case 'gas_puff':
                    if 'model_name' not in v:
                        constructor_data[k]['model_name'] = 'exponential'
                case 'generic_particle':
                    if 'model_name' not in v:
                        constructor_data[k]['model_name'] = 'gaussian'
                case 'pellet':
                    if 'model_name' not in v:
                        constructor_data[k]['model_name'] = 'gaussian'
                case 'fusion':
                    if 'model_name' not in v:
                        constructor_data[k]['model_name'] = 'bosch_hale'
                case 'generic_heat':
                    if 'model_name' not in v:
                        constructor_data[k]['model_name'] = 'gaussian'
        return constructor_data

    def build_models(self):
        standard_sources = {}
        for k, v in dict(self).items():
            if k == 'ei_exchange':
                continue
            else:
                if v is not None:
                    source = v.build_source()
                    standard_sources[k] = source
        qei_source_model = self.ei_exchange.build_source()
        return SourceModels(
            qei_source=qei_source_model,
            standard_sources=immutabledict.immutabledict(standard_sources),
        )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ImpurityRadiationHeatSink(Source):
    SOURCE_NAME = "impurity_radiation"
    model_func: SourceProfileFunction


_FINAL_SOURCES = frozenset([ImpurityRadiationHeatSink.SOURCE_NAME])


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'source_models',
        'neoclassical_models',
        'explicit',
    ],
)
def build_source_profiles(runtime_params,
                          geo,
                          core_profiles,
                          source_models,
                          neoclassical_models,
                          explicit,
                          explicit_source_profiles=None,
                          conductivity=None):
    if not explicit and explicit_source_profiles is None:
        raise ValueError(
            '`explicit_source_profiles` must be provided if explicit is False.'
        )
    if explicit:
        qei = QeiInfo.zeros(geo)
        bootstrap_current = bootstrap_current_base.BootstrapCurrent.zeros(geo)
    else:
        qei = source_models.qei_source.get_qei(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
        )
        bootstrap_current = (
            neoclassical_models.bootstrap_current.calculate_bootstrap_current(
                runtime_params, geo, core_profiles))
    profiles = SourceProfiles(
        bootstrap_current=bootstrap_current,
        qei=qei,
        T_e=explicit_source_profiles.T_e if explicit_source_profiles else {},
        T_i=explicit_source_profiles.T_i if explicit_source_profiles else {},
        n_e=explicit_source_profiles.n_e if explicit_source_profiles else {},
        psi=explicit_source_profiles.psi if explicit_source_profiles else {},
    )
    build_standard_source_profiles(
        calculated_source_profiles=profiles,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        explicit=explicit,
        conductivity=conductivity,
    )
    return profiles


def build_standard_source_profiles(
    *,
    calculated_source_profiles: SourceProfiles,
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
    explicit: bool = True,
    conductivity: conductivity_base.Conductivity | None = None,
    calculate_anyway: bool = False,
    psi_only: bool = False,
):

    def calculate_source(source_name, source):
        source_params = runtime_params.sources[source_name]
        if (explicit == source_params.is_explicit) | calculate_anyway:
            value = source.get_value(
                runtime_params,
                geo,
                core_profiles,
                calculated_source_profiles,
                conductivity,
            )
            _update_standard_source_profiles(
                calculated_source_profiles,
                source_name,
                source.affected_core_profiles,
                value,
            )

    for source_name, source in source_models.psi_sources.items():
        calculate_source(source_name, source)
    if psi_only:
        return
    to_calculate = {}
    for source_name, source in source_models.standard_sources.items():
        if source_name in _FINAL_SOURCES:
            to_calculate[source_name] = source
            continue
        if source_name not in source_models.psi_sources:
            calculate_source(source_name, source)
    for source_name, source in to_calculate.items():
        calculate_source(source_name, source)


def _update_standard_source_profiles(
    calculated_source_profiles: SourceProfiles,
    source_name: str,
    affected_core_profiles: tuple[AffectedCoreProfile, ...],
    profile: tuple[array_typing.FloatVectorCell, ...],
):
    for profile, affected_core_profile in zip(profile,
                                              affected_core_profiles,
                                              strict=True):
        match affected_core_profile:
            case AffectedCoreProfile.PSI:
                calculated_source_profiles.psi[source_name] = profile
            case AffectedCoreProfile.NE:
                calculated_source_profiles.n_e[source_name] = profile
            case AffectedCoreProfile.TEMP_ION:
                calculated_source_profiles.T_i[source_name] = profile
            case AffectedCoreProfile.TEMP_EL:
                calculated_source_profiles.T_e[source_name] = profile


def build_all_zero_profiles(geo: geometry.Geometry, ):
    return SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=QeiInfo.zeros(geo),
    )


def get_all_source_profiles(runtime_params, geo, core_profiles, source_models,
                            neoclassical_models, conductivity):
    explicit_source_profiles = build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        explicit=True,
    )
    return build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
        conductivity=conductivity,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalModelOutput:
    rho_norm_ped_top: array_typing.FloatScalar
    rho_norm_ped_top_idx: array_typing.IntScalar
    T_i_ped: array_typing.FloatScalar
    T_e_ped: array_typing.FloatScalar
    n_e_ped: array_typing.FloatScalar


class PedestalModel(abc.ABC):

    def __setattr__(self, attr, value):
        return super().__setattr__(attr, value)

    def __call__(
        self,
        runtime_params,
        geo,
        core_profiles,
    ):
        return jax.lax.cond(
            runtime_params.pedestal.set_pedestal,
            lambda: self._call_implementation(runtime_params, geo,
                                              core_profiles),
            lambda: PedestalModelOutput(
                rho_norm_ped_top=jnp.inf,
                T_i_ped=0.0,
                T_e_ped=0.0,
                n_e_ped=0.0,
                rho_norm_ped_top_idx=geo.torax_mesh.nx,
            ),
        )

    @abc.abstractmethod
    def _call_implementation(
        self,
        runtime_params,
        geo,
        core_profiles,
    ):
        pass

    @abc.abstractmethod
    def __hash__(self):
        ...

    @abc.abstractmethod
    def __eq__(self, other):
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsPED:
    set_pedestal: array_typing.BoolScalar
    n_e_ped: array_typing.FloatScalar
    T_i_ped: array_typing.FloatScalar
    T_e_ped: array_typing.FloatScalar
    rho_norm_ped_top: array_typing.FloatScalar
    n_e_ped_is_fGW: array_typing.BoolScalar


class SetTemperatureDensityPedestalModel(PedestalModel):

    def __init__(self, ):
        super().__init__()
        self._frozen = True

    @override
    def _call_implementation(self, runtime_params, geo, core_profiles):
        pedestal_params = runtime_params.pedestal
        nGW = (runtime_params.profile_conditions.Ip / 1e6 /
               (jnp.pi * geo.a_minor**2) * 1e20)
        n_e_ped = jnp.where(
            pedestal_params.n_e_ped_is_fGW,
            pedestal_params.n_e_ped * nGW,
            pedestal_params.n_e_ped,
        )
        return PedestalModelOutput(
            n_e_ped=n_e_ped,
            T_i_ped=pedestal_params.T_i_ped,
            T_e_ped=pedestal_params.T_e_ped,
            rho_norm_ped_top=pedestal_params.rho_norm_ped_top,
            rho_norm_ped_top_idx=jnp.abs(
                geo.rho_norm - pedestal_params.rho_norm_ped_top).argmin(),
        )

    def __hash__(self):
        return hash('SetTemperatureDensityPedestalModel')

    def __eq__(self, other):
        return isinstance(other, SetTemperatureDensityPedestalModel)


class BasePedestal(torax_pydantic.BaseModelFrozen, abc.ABC):
    set_pedestal: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(False))

    @abc.abstractmethod
    def build_pedestal_model(self):
        pass

    @abc.abstractmethod
    def build_runtime_params(self, t):
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

    def build_pedestal_model(self):
        return SetTemperatureDensityPedestalModel()

    def build_runtime_params(self, t):
        return RuntimeParamsPED(
            set_pedestal=self.set_pedestal.get_value(t),
            n_e_ped=self.n_e_ped.get_value(t),
            n_e_ped_is_fGW=self.n_e_ped_is_fGW,
            T_i_ped=self.T_i_ped.get_value(t),
            T_e_ped=self.T_e_ped.get_value(t),
            rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
        )


PedestalConfig = SetTpedNped

_IMPURITY_MODE_FRACTIONS: Final[str] = 'fractions'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsIM:
    fractions: array_typing.FloatVector
    A_avg: array_typing.FloatScalar | array_typing.FloatVectorCell
    Z_override: array_typing.FloatScalar | None = None


class IonMixture(torax_pydantic.BaseModelFrozen):
    species: runtime_validation_utils.IonMapping
    Z_override: torax_pydantic.TimeVaryingScalar | None = None
    A_override: torax_pydantic.TimeVaryingScalar | None = None

    def build_runtime_params(self, t):
        ions = self.species.keys()
        fractions = jnp.array([self.species[ion].get_value(t) for ion in ions])
        Z_override = None if not self.Z_override else self.Z_override.get_value(
            t)
        if not self.A_override:
            As = jnp.array(
                [constants.ION_PROPERTIES_DICT[ion].A for ion in ions])
            A_avg = jnp.sum(As * fractions)
        else:
            A_avg = self.A_override.get_value(t)
        return RuntimeParamsIM(
            fractions=fractions,
            A_avg=A_avg,
            Z_override=Z_override,
        )


def _impurity_before_validator(value):
    return {value: 1.0}


def _impurity_after_validator(value):
    first_key = next(iter(value))
    first_tva = value[first_key]
    reference_times = first_tva.value.keys()
    for t in reference_times:
        reference_rho_norm, _ = first_tva.value[t]
        values_at_t = [tva.value[t][1] for tva in value.values()]
        reference_shape = values_at_t[0].shape
        sum_of_values = np.sum(np.stack(values_at_t, axis=0), axis=0)
    return value


ImpurityMapping: TypeAlias = Annotated[
    Mapping[str, torax_pydantic.NonNegativeTimeVaryingArray],
    pydantic.BeforeValidator(_impurity_before_validator),
    pydantic.AfterValidator(_impurity_after_validator),
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsIF:
    fractions: jt.Float[array_typing.Array, 'ion_symbol rhon']
    fractions_face: jt.Float[array_typing.Array, 'ion_symbol rhon+1']
    A_avg: array_typing.FloatVectorCell
    A_avg_face: array_typing.FloatVectorFace
    Z_override: array_typing.FloatScalar | None = None


class ImpurityFractions(torax_pydantic.BaseModelFrozen):
    impurity_mode: Annotated[Literal['fractions'],
                             torax_pydantic.JAX_STATIC] = ('fractions')
    species: ImpurityMapping = torax_pydantic.ValidatedDefault({'Ne': 1.0})
    Z_override: torax_pydantic.TimeVaryingScalar | None = None
    A_override: torax_pydantic.TimeVaryingScalar | None = None

    def build_runtime_params(self, t):
        ions = self.species.keys()
        fractions = jnp.array([self.species[ion].get_value(t) for ion in ions])
        fractions_face = jnp.array(
            [self.species[ion].get_value(t, grid_type='face') for ion in ions])
        Z_override = None if not self.Z_override else self.Z_override.get_value(
            t)
        As = jnp.array([constants.ION_PROPERTIES_DICT[ion].A for ion in ions])
        A_avg = jnp.sum(As[..., jnp.newaxis] * fractions, axis=0)
        A_avg_face = jnp.sum(As[..., jnp.newaxis] * fractions_face, axis=0)
        return RuntimeParamsIF(
            fractions=fractions,
            fractions_face=fractions_face,
            A_avg=A_avg,
            A_avg_face=A_avg_face,
            Z_override=Z_override,
        )

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_impurity_data(cls, data):
        if 'legacy' in data:
            del data['legacy']
        return data


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
class RuntimeParamsPC:
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
            for x in dataclasses.fields(RuntimeParamsPC)
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
        return RuntimeParamsPC(**runtime_params)


_IMPURITY_MODE_NE_RATIOS: Final[str] = 'n_e_ratios'
_IMPURITY_MODE_NE_RATIOS_ZEFF: Final[str] = 'n_e_ratios_Z_eff'


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParamsP:
    main_ion_names: tuple[str,
                          ...] = dataclasses.field(metadata={'static': True})
    impurity_names: tuple[str,
                          ...] = dataclasses.field(metadata={'static': True})
    main_ion: RuntimeParamsIM
    impurity: RuntimeParamsIM
    Z_eff: array_typing.FloatVectorCell
    Z_eff_face: array_typing.FloatVectorFace


@jax.tree_util.register_pytree_node_class
class PlasmaComposition(torax_pydantic.BaseModelFrozen):
    impurity: Annotated[
        ImpurityFractions,
        pydantic.Field(discriminator='impurity_mode'),
    ]
    main_ion: runtime_validation_utils.IonMapping = (
        torax_pydantic.ValidatedDefault({
            'D': 0.5,
            'T': 0.5
        }))
    Z_eff: (runtime_validation_utils.
            TimeVaryingArrayDefinedAtRightBoundaryAndBounded
            ) = torax_pydantic.ValidatedDefault(1.0)
    Z_i_override: torax_pydantic.TimeVaryingScalar | None = None
    A_i_override: torax_pydantic.TimeVaryingScalar | None = None
    Z_impurity_override: torax_pydantic.TimeVaryingScalar | None = None
    A_impurity_override: torax_pydantic.TimeVaryingScalar | None = None

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_impurity_data(cls, data):
        configurable_data = copy.deepcopy(data)
        Z_impurity_override = configurable_data.get('Z_impurity_override')
        A_impurity_override = configurable_data.get('A_impurity_override')
        impurity_data = configurable_data['impurity']
        configurable_data['impurity'] = {
            'impurity_mode': _IMPURITY_MODE_FRACTIONS,
            'species': impurity_data,
            'Z_override': Z_impurity_override,
            'A_override': A_impurity_override,
            'legacy': True,
        }
        return configurable_data

    def tree_flatten(self):
        children = (
            self.main_ion,
            self.impurity,
            self.Z_eff,
            self.Z_i_override,
            self.A_i_override,
            self.Z_impurity_override,
            self.A_impurity_override,
            self._main_ion_mixture,
        )
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.model_construct(
            main_ion=children[0],
            impurity=children[1],
            Z_eff=children[2],
            Z_i_override=children[3],
            A_i_override=children[4],
            Z_impurity_override=children[5],
            A_impurity_override=children[6],
        )
        obj._main_ion_mixture = children[7]
        return obj

    @functools.cached_property
    def _main_ion_mixture(self):
        return IonMixture.model_construct(
            species=self.main_ion,
            Z_override=self.Z_i_override,
            A_override=self.A_i_override,
        )

    def get_main_ion_names(self):
        return tuple(self._main_ion_mixture.species.keys())

    def get_impurity_names(self):
        return tuple(self.impurity.species.keys())

    def build_runtime_params(self, t):
        return RuntimeParamsP(
            main_ion_names=self.get_main_ion_names(),
            impurity_names=self.get_impurity_names(),
            main_ion=self._main_ion_mixture.build_runtime_params(t),
            impurity=self.impurity.build_runtime_params(t),
            Z_eff=self.Z_eff.get_value(t),
            Z_eff_face=self.Z_eff.get_value(t, grid_type='face'),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsX:
    chi_min: float
    chi_max: float
    D_e_min: float
    D_e_max: float
    V_e_min: float
    V_e_max: float
    rho_min: array_typing.FloatScalar
    rho_max: array_typing.FloatScalar
    apply_inner_patch: array_typing.BoolScalar
    D_e_inner: array_typing.FloatScalar
    V_e_inner: array_typing.FloatScalar
    chi_i_inner: array_typing.FloatScalar
    chi_e_inner: array_typing.FloatScalar
    rho_inner: array_typing.FloatScalar
    apply_outer_patch: array_typing.BoolScalar
    D_e_outer: array_typing.FloatScalar
    V_e_outer: array_typing.FloatScalar
    chi_i_outer: array_typing.FloatScalar
    chi_e_outer: array_typing.FloatScalar
    rho_outer: array_typing.FloatScalar
    smoothing_width: float
    smooth_everywhere: bool


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TurbulentTransport:
    chi_face_ion: jax.Array
    chi_face_el: jax.Array
    d_face_el: jax.Array
    v_face_el: jax.Array
    chi_face_el_bohm: jax.Array | None = None
    chi_face_el_gyrobohm: jax.Array | None = None
    chi_face_ion_bohm: jax.Array | None = None
    chi_face_ion_gyrobohm: jax.Array | None = None


class TransportModel(abc.ABC):

    def __setattr__(self, attr, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("TransportModels are immutable.")
        return super().__setattr__(attr, value)

    def __call__(self, runtime_params, geo, core_profiles,
                 pedestal_model_output):
        if not getattr(self, "_frozen", False):
            raise RuntimeError(
                f"Subclass implementation {type(self)} forgot to "
                "freeze at the end of __init__.")
        transport_runtime_params = runtime_params.transport
        transport_coeffs = self._call_implementation(
            transport_runtime_params,
            runtime_params,
            geo,
            core_profiles,
            pedestal_model_output,
        )
        transport_coeffs = self._apply_domain_restriction(
            transport_runtime_params,
            geo,
            transport_coeffs,
            pedestal_model_output,
        )
        transport_coeffs = self._apply_clipping(
            transport_runtime_params,
            transport_coeffs,
        )
        transport_coeffs = self._apply_transport_patches(
            transport_runtime_params,
            runtime_params,
            geo,
            transport_coeffs,
        )
        return self._smooth_coeffs(
            transport_runtime_params,
            runtime_params,
            geo,
            transport_coeffs,
            pedestal_model_output,
        )

    @abc.abstractmethod
    def _call_implementation(self, transport_runtime_params, runtime_params,
                             geo, core_profiles, pedestal_model_output):
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    def _apply_domain_restriction(self, transport_runtime_params, geo,
                                  transport_coeffs, pedestal_model_output):
        active_mask = (
            (geo.rho_face_norm > transport_runtime_params.rho_min)
            & (geo.rho_face_norm <= transport_runtime_params.rho_max)
            & (geo.rho_face_norm <= pedestal_model_output.rho_norm_ped_top))
        active_mask = (jnp.asarray(active_mask).at[0].set(
            transport_runtime_params.rho_min == 0))
        chi_face_ion = jnp.where(active_mask, transport_coeffs.chi_face_ion,
                                 0.0)
        chi_face_el = jnp.where(active_mask, transport_coeffs.chi_face_el, 0.0)
        d_face_el = jnp.where(active_mask, transport_coeffs.d_face_el, 0.0)
        v_face_el = jnp.where(active_mask, transport_coeffs.v_face_el, 0.0)
        return dataclasses.replace(
            transport_coeffs,
            chi_face_ion=chi_face_ion,
            chi_face_el=chi_face_el,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )

    def _apply_clipping(self, transport_runtime_params, transport_coeffs):
        chi_face_ion = jnp.clip(
            transport_coeffs.chi_face_ion,
            transport_runtime_params.chi_min,
            transport_runtime_params.chi_max,
        )
        chi_face_el = jnp.clip(
            transport_coeffs.chi_face_el,
            transport_runtime_params.chi_min,
            transport_runtime_params.chi_max,
        )
        d_face_el = jnp.clip(
            transport_coeffs.d_face_el,
            transport_runtime_params.D_e_min,
            transport_runtime_params.D_e_max,
        )
        v_face_el = jnp.clip(
            transport_coeffs.v_face_el,
            transport_runtime_params.V_e_min,
            transport_runtime_params.V_e_max,
        )
        return dataclasses.replace(
            transport_coeffs,
            chi_face_ion=chi_face_ion,
            chi_face_el=chi_face_el,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )

    def _apply_transport_patches(self, transport_runtime_params,
                                 runtime_params, geo, transport_coeffs):
        consts = constants.CONSTANTS
        chi_face_ion = jnp.where(
            jnp.logical_and(
                transport_runtime_params.apply_inner_patch,
                geo.rho_face_norm
                < transport_runtime_params.rho_inner + consts.eps,
            ),
            transport_runtime_params.chi_i_inner,
            transport_coeffs.chi_face_ion,
        )
        chi_face_el = jnp.where(
            jnp.logical_and(
                transport_runtime_params.apply_inner_patch,
                geo.rho_face_norm
                < transport_runtime_params.rho_inner + consts.eps,
            ),
            transport_runtime_params.chi_e_inner,
            transport_coeffs.chi_face_el,
        )
        d_face_el = jnp.where(
            jnp.logical_and(
                transport_runtime_params.apply_inner_patch,
                geo.rho_face_norm
                < transport_runtime_params.rho_inner + consts.eps,
            ),
            transport_runtime_params.D_e_inner,
            transport_coeffs.d_face_el,
        )
        v_face_el = jnp.where(
            jnp.logical_and(
                transport_runtime_params.apply_inner_patch,
                geo.rho_face_norm
                < transport_runtime_params.rho_inner + consts.eps,
            ),
            transport_runtime_params.V_e_inner,
            transport_coeffs.v_face_el,
        )
        chi_face_ion = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    transport_runtime_params.apply_outer_patch,
                    jnp.logical_not(runtime_params.pedestal.set_pedestal),
                ),
                geo.rho_face_norm
                > transport_runtime_params.rho_outer - consts.eps,
            ),
            transport_runtime_params.chi_i_outer,
            chi_face_ion,
        )
        chi_face_el = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    transport_runtime_params.apply_outer_patch,
                    jnp.logical_not(runtime_params.pedestal.set_pedestal),
                ),
                geo.rho_face_norm
                > transport_runtime_params.rho_outer - consts.eps,
            ),
            transport_runtime_params.chi_e_outer,
            chi_face_el,
        )
        d_face_el = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    transport_runtime_params.apply_outer_patch,
                    jnp.logical_not(runtime_params.pedestal.set_pedestal),
                ),
                geo.rho_face_norm
                > transport_runtime_params.rho_outer - consts.eps,
            ),
            transport_runtime_params.D_e_outer,
            d_face_el,
        )
        v_face_el = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    transport_runtime_params.apply_outer_patch,
                    jnp.logical_not(runtime_params.pedestal.set_pedestal),
                ),
                geo.rho_face_norm
                > transport_runtime_params.rho_outer - consts.eps,
            ),
            transport_runtime_params.V_e_outer,
            v_face_el,
        )
        return dataclasses.replace(
            transport_coeffs,
            chi_face_ion=chi_face_ion,
            chi_face_el=chi_face_el,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )

    def _smooth_coeffs(self, transport_runtime_params, runtime_params, geo,
                       transport_coeffs, pedestal_model_output):
        smoothing_matrix = _build_smoothing_matrix(
            transport_runtime_params,
            runtime_params,
            geo,
            pedestal_model_output,
        )

        def smooth_single_coeff(coeff):
            return jax.lax.cond(
                jnp.all(coeff == 0.0),
                lambda: coeff,
                lambda: jnp.dot(smoothing_matrix, coeff),
            )

        return jax.tree_util.tree_map(smooth_single_coeff, transport_coeffs)


def _build_smoothing_matrix(transport_runtime_params, runtime_params, geo,
                            pedestal_model_output):
    lower_cutoff = 0.01
    consts = constants.CONSTANTS
    kernel = jnp.exp(
        -jnp.log(2) *
        (geo.rho_face_norm[:, jnp.newaxis] - geo.rho_face_norm)**2 /
        (transport_runtime_params.smoothing_width**2 + consts.eps))
    mask_outer_edge = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_not(runtime_params.pedestal.set_pedestal),
            transport_runtime_params.apply_outer_patch,
        ),
        lambda: transport_runtime_params.rho_outer - consts.eps,
        lambda: pedestal_model_output.rho_norm_ped_top - consts.eps,
    )
    mask_inner_edge = jax.lax.cond(
        transport_runtime_params.apply_inner_patch,
        lambda: transport_runtime_params.rho_inner + consts.eps,
        lambda: 0.0,
    )
    mask = jnp.where(
        jnp.logical_or(
            transport_runtime_params.smooth_everywhere,
            jnp.logical_and(
                geo.rho_face_norm > mask_inner_edge,
                geo.rho_face_norm < mask_outer_edge,
            ),
        ),
        1.0,
        0.0,
    )
    diag_mask = jnp.diag(mask)
    kernel = jnp.dot(diag_mask, kernel)
    num_rows = len(mask)
    mask_mat = jnp.tile(mask, (num_rows, 1))
    kernel *= mask_mat
    zero_row_mask = jnp.all(kernel == 0, axis=1)
    kernel = jnp.where(zero_row_mask[:, jnp.newaxis], jnp.eye(kernel.shape[0]),
                       kernel)
    row_sums = jnp.sum(kernel, axis=1)
    kernel /= row_sums[:, jnp.newaxis]
    kernel = jnp.where(kernel < lower_cutoff, 0.0, kernel)
    row_sums = jnp.sum(kernel, axis=1)
    kernel /= row_sums[:, jnp.newaxis]
    return kernel


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NormalizedLogarithmicGradients:
    lref_over_lti: array_typing.FloatVectorFace
    lref_over_lte: array_typing.FloatVectorFace
    lref_over_lne: array_typing.FloatVectorFace
    lref_over_lni0: array_typing.FloatVectorFace
    lref_over_lni1: array_typing.FloatVectorFace

    @classmethod
    def from_profiles(
        cls,
        core_profiles: state.CoreProfiles,
        radial_coordinate: jnp.ndarray,
        reference_length: jnp.ndarray,
    ):
        gradients = {}
        for name, profile in {
                "lref_over_lti": core_profiles.T_i,
                "lref_over_lte": core_profiles.T_e,
                "lref_over_lne": core_profiles.n_e,
                "lref_over_lni0": core_profiles.n_i,
                "lref_over_lni1": core_profiles.n_impurity,
        }.items():
            gradients[name] = calculate_normalized_logarithmic_gradient(
                var=profile,
                radial_coordinate=radial_coordinate,
                reference_length=reference_length,
            )
        return cls(**gradients)


def calculate_chiGB(
    reference_temperature: array_typing.Array,
    reference_magnetic_field: chex.Numeric,
    reference_mass: chex.Numeric,
    reference_length: chex.Numeric,
):
    constants = constants_module.CONSTANTS
    return ((reference_mass * constants.m_amu)**0.5 /
            (reference_magnetic_field * constants.q_e)**2 *
            (reference_temperature * constants.keV_to_J)**1.5 /
            reference_length)


def calculate_alpha(core_profiles, q, reference_magnetic_field,
                    normalized_logarithmic_gradients):
    constants = constants_module.CONSTANTS
    factor_0 = (2 * constants.keV_to_J / reference_magnetic_field**2 *
                constants.mu_0 * q**2)
    alpha = factor_0 * (
        core_profiles.T_e.face_value() * core_profiles.n_e.face_value() *
        (normalized_logarithmic_gradients.lref_over_lte +
         normalized_logarithmic_gradients.lref_over_lne) +
        core_profiles.n_i.face_value() * core_profiles.T_i.face_value() *
        (normalized_logarithmic_gradients.lref_over_lti +
         normalized_logarithmic_gradients.lref_over_lni0) +
        core_profiles.n_impurity.face_value() *
        core_profiles.T_i.face_value() *
        (normalized_logarithmic_gradients.lref_over_lti +
         normalized_logarithmic_gradients.lref_over_lni1))
    return alpha


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams00(RuntimeParamsX):
    DV_effective: bool
    An_min: float


def calculate_normalized_logarithmic_gradient(
    var: cell_variable.CellVariable,
    radial_coordinate: jax.Array,
    reference_length: jax.Array,
):
    result = jnp.where(
        jnp.abs(var.face_value()) < constants_module.CONSTANTS.eps,
        constants_module.CONSTANTS.eps,
        -reference_length * var.face_grad(radial_coordinate) /
        var.face_value(),
    )
    result = jnp.where(
        jnp.abs(result) < constants_module.CONSTANTS.eps,
        constants_module.CONSTANTS.eps,
        result,
    )
    return result


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QuasilinearInputs:
    chiGB: (array_typing.FloatVectorFace)
    Rmin: array_typing.FloatScalar
    Rmaj: array_typing.FloatScalar
    lref_over_lti: array_typing.FloatVectorFace
    lref_over_lte: array_typing.FloatVectorFace
    lref_over_lne: array_typing.FloatVectorFace
    lref_over_lni0: array_typing.FloatVectorFace
    lref_over_lni1: array_typing.FloatVectorFace


class QuasilinearTransportModel(TransportModel):

    def _make_core_transport(
        self,
        qi: jax.Array,
        qe: jax.Array,
        pfe: jax.Array,
        quasilinear_inputs: QuasilinearInputs,
        transport,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
        gradient_reference_length: chex.Numeric,
        gyrobohm_flux_reference_length: chex.Numeric,
    ):
        constants = constants_module.CONSTANTS
        pfe_SI = (pfe * core_profiles.n_e.face_value() *
                  quasilinear_inputs.chiGB / gyrobohm_flux_reference_length)
        chi_face_ion = (
            ((gradient_reference_length / gyrobohm_flux_reference_length) * qi)
            / quasilinear_inputs.lref_over_lti) * quasilinear_inputs.chiGB
        chi_face_el = (
            ((gradient_reference_length / gyrobohm_flux_reference_length) * qe)
            / quasilinear_inputs.lref_over_lte) * quasilinear_inputs.chiGB

        def DV_effective_approach():
            Deff = -pfe_SI / (core_profiles.n_e.face_grad() *
                              geo.g1_over_vpr2_face * geo.rho_b +
                              constants.eps)
            Veff = pfe_SI / (core_profiles.n_e.face_value() *
                             geo.g0_over_vpr_face * geo.rho_b)
            Deff_mask = (
                ((pfe >= 0) & (quasilinear_inputs.lref_over_lne >= 0))
                | ((pfe < 0) & (quasilinear_inputs.lref_over_lne < 0))) & (abs(
                    quasilinear_inputs.lref_over_lne) >= transport.An_min)
            Veff_mask = jnp.invert(Deff_mask)
            d_face_el = jnp.where(Veff_mask, 0.0, Deff)
            v_face_el = jnp.where(Deff_mask, 0.0, Veff)
            return d_face_el, v_face_el

        def Dscaled_approach():
            d_face_el = chi_face_el
            v_face_el = (pfe_SI / core_profiles.n_e.face_value() -
                         quasilinear_inputs.lref_over_lne * d_face_el /
                         gradient_reference_length * geo.g1_over_vpr2_face *
                         geo.rho_b**2) / (geo.g0_over_vpr_face * geo.rho_b)
            return d_face_el, v_face_el

        d_face_el, v_face_el = jax.lax.cond(
            transport.DV_effective,
            DV_effective_approach,
            Dscaled_approach,
        )
        return TurbulentTransport(
            chi_face_ion=chi_face_ion,
            chi_face_el=chi_face_el,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(RuntimeParams00):
    collisionality_multiplier: float
    avoid_big_negative_s: bool
    smag_alpha_correction: bool
    q_sawtooth_proxy: bool


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QualikizInputs(QuasilinearInputs):
    Z_eff_face: array_typing.FloatVectorFace
    q: array_typing.FloatVectorFace
    smag: array_typing.FloatVectorFace
    x: array_typing.FloatVectorFace
    Ti_Te: array_typing.FloatVectorFace
    log_nu_star_face: array_typing.FloatVectorFace
    normni: array_typing.FloatVectorFace
    alpha: array_typing.FloatVectorFace
    epsilon_lcfs: array_typing.FloatScalar

    @property
    def Ati(self):
        return self.lref_over_lti

    @property
    def Ate(self):
        return self.lref_over_lte

    @property
    def Ane(self):
        return self.lref_over_lne

    @property
    def Ani0(self):
        return self.lref_over_lni0

    @property
    def Ani1(self):
        return self.lref_over_lni1


class QualikizBasedTransportModel(QuasilinearTransportModel):

    def _prepare_qualikiz_inputs(self, transport, geo, core_profiles):
        constants = constants_module.CONSTANTS
        rmid = (geo.R_out - geo.R_in) * 0.5
        rmid_face = (geo.R_out_face - geo.R_in_face) * 0.5
        chiGB = calculate_chiGB(
            reference_temperature=core_profiles.T_i.face_value(),
            reference_magnetic_field=geo.B_0,
            reference_mass=core_profiles.A_i,
            reference_length=geo.a_minor,
        )
        normalized_logarithmic_gradients = NormalizedLogarithmicGradients.from_profiles(
            core_profiles=core_profiles,
            radial_coordinate=rmid,
            reference_length=geo.R_major,
        )
        q = core_profiles.q_face
        smag = psi_calculations.calc_s_rmid(
            geo,
            core_profiles.psi,
        )
        epsilon_lcfs = rmid_face[-1] / geo.R_major
        x = rmid_face / rmid_face[-1]
        x = jnp.where(jnp.abs(x) < constants.eps, constants.eps, x)
        Ti_Te = core_profiles.T_i.face_value() / core_profiles.T_e.face_value()
        nu_star = collisions.calc_nu_star(
            geo=geo,
            core_profiles=core_profiles,
            collisionality_multiplier=transport.collisionality_multiplier,
        )
        log_nu_star_face = jnp.log10(nu_star)
        alpha = calculate_alpha(
            core_profiles=core_profiles,
            q=q,
            reference_magnetic_field=geo.B_0,
            normalized_logarithmic_gradients=normalized_logarithmic_gradients,
        )
        smag = jnp.where(
            transport.smag_alpha_correction,
            smag - alpha / 2,
            smag,
        )
        smag = jnp.where(
            jnp.logical_and(
                transport.q_sawtooth_proxy,
                q < 1,
            ),
            0.1,
            smag,
        )
        q = jnp.where(
            jnp.logical_and(
                transport.q_sawtooth_proxy,
                q < 1,
            ),
            1,
            q,
        )
        smag = jnp.where(
            jnp.logical_and(
                transport.avoid_big_negative_s,
                smag - alpha < -0.2,
            ),
            alpha - 0.2,
            smag,
        )
        normni = core_profiles.n_i.face_value() / core_profiles.n_e.face_value(
        )
        return QualikizInputs(
            Z_eff_face=core_profiles.Z_eff_face,
            lref_over_lti=normalized_logarithmic_gradients.lref_over_lti,
            lref_over_lte=normalized_logarithmic_gradients.lref_over_lte,
            lref_over_lne=normalized_logarithmic_gradients.lref_over_lne,
            lref_over_lni0=normalized_logarithmic_gradients.lref_over_lni0,
            lref_over_lni1=normalized_logarithmic_gradients.lref_over_lni1,
            q=q,
            smag=smag,
            x=x,
            Ti_Te=Ti_Te,
            log_nu_star_face=log_nu_star_face,
            normni=normni,
            chiGB=chiGB,
            Rmaj=geo.R_major,
            Rmin=geo.a_minor,
            alpha=alpha,
            epsilon_lcfs=epsilon_lcfs,
        )


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
        return RuntimeParamsX(
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
    def build_transport_model(self):
        pass


_FLUX_NAME_MAP: Final[Mapping[str, str]] = immutabledict.immutabledict({
    'efiITG':
    'qi_itg',
    'efeITG':
    'qe_itg',
    'pfeITG':
    'pfe_itg',
    'efeTEM':
    'qe_tem',
    'efiTEM':
    'qi_tem',
    'pfeTEM':
    'pfe_tem',
    'efeETG':
    'qe_etg',
})


class QLKNNModelWrapper:

    def __init__(
        self,
        path: str,
        name: str = '',
        flux_name_map: Mapping[str, str] | None = None,
    ):
        self.path = path
        self.name = name
        if flux_name_map is None:
            flux_name_map = _FLUX_NAME_MAP
        self._flux_name_map = flux_name_map
        self._model = qlknn_model.QLKNNModel.load_default_model()

    @property
    def inputs_and_ranges(self):
        return self._model.inputs_and_ranges

    def get_model_inputs_from_qualikiz_inputs(self, qualikiz_inputs):
        input_map = {
            'Ani': lambda x: x.Ani0,
            'LogNuStar': lambda x: x.log_nu_star_face,
        }

        def _get_input(key):
            return jnp.array(
                input_map.get(key, lambda x: getattr(x, key))(qualikiz_inputs),
                dtype=jax_utils.get_dtype(),
            )

        return jnp.array(
            [_get_input(key) for key in self.inputs_and_ranges.keys()],
            dtype=jax_utils.get_dtype(),
        ).T

    def predict(self, inputs: jax.Array):
        model_predictions = self._model.predict(inputs)
        return {
            self._flux_name_map.get(flux_name, flux_name): flux_value
            for flux_name, flux_value in model_predictions.items()
        }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams0(RuntimeParams):
    include_ITG: bool
    include_TEM: bool
    include_ETG: bool
    ITG_flux_ratio_correction: float
    ETG_correction_factor: float
    clip_inputs: bool
    clip_margin: float


_EPSILON_NN: Final[float] = (1 / 3)


@functools.lru_cache(maxsize=1)
def get_model(path: str, name: str):
    return QLKNNModelWrapper(path, name)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QLKNNRuntimeConfigInputs:
    transport: RuntimeParams0
    Ped_top: float
    set_pedestal: bool

    @staticmethod
    def from_runtime_params_slice(transport_runtime_params, runtime_params,
                                  pedestal_model_output):
        assert isinstance(transport_runtime_params, RuntimeParams0)
        return QLKNNRuntimeConfigInputs(
            transport=transport_runtime_params,
            Ped_top=pedestal_model_output.rho_norm_ped_top,
            set_pedestal=runtime_params.pedestal.set_pedestal,
        )


def _filter_model_output(
    model_output: None,
    include_ITG: bool,
    include_TEM: bool,
    include_ETG: bool,
):
    filter_map = {
        'qi_itg': include_ITG,
        'qe_itg': include_ITG,
        'pfe_itg': include_ITG,
        'qe_tem': include_TEM,
        'qi_tem': include_TEM,
        'pfe_tem': include_TEM,
        'qe_etg': include_ETG,
    }

    def filter_flux(flux_name: str, value: jax.Array):
        return jax.lax.cond(
            filter_map.get(flux_name, True),
            lambda: value,
            lambda: jnp.zeros_like(value),
        )

    return {k: filter_flux(k, v) for k, v in model_output.items()}


def clip_inputs(feature_scan, clip_margin, inputs_and_ranges):
    for i, key in enumerate(inputs_and_ranges.keys()):
        bounds = inputs_and_ranges[key]
        min_val = bounds.get('min', -jnp.inf)
        max_val = bounds.get('max', jnp.inf)
        min_val += jnp.where(jnp.isfinite(min_val),
                             jnp.abs(min_val) * (1 - clip_margin), 0.0)
        max_val -= jnp.where(jnp.isfinite(max_val),
                             jnp.abs(max_val) * (1 - clip_margin), 0.0)
        feature_scan = feature_scan.at[:, i].set(
            jnp.clip(
                feature_scan[:, i],
                min_val,
                max_val,
            ))
    return feature_scan


class QLKNNTransportModel0(QualikizBasedTransportModel):

    def __init__(
        self,
        path: str,
        name: str,
    ):
        super().__init__()
        self._path = path
        self._name = name
        self._frozen = True

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    def _call_implementation(self, transport_runtime_params, runtime_params,
                             geo, core_profiles, pedestal_model_output):
        runtime_config_inputs = QLKNNRuntimeConfigInputs.from_runtime_params_slice(
            transport_runtime_params,
            runtime_params,
            pedestal_model_output,
        )
        return self._combined(runtime_config_inputs, geo, core_profiles)

    def _combined(
        self,
        runtime_config_inputs: QLKNNRuntimeConfigInputs,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ):
        qualikiz_inputs = self._prepare_qualikiz_inputs(
            transport=runtime_config_inputs.transport,
            geo=geo,
            core_profiles=core_profiles,
        )
        model = get_model(self.path, self.name)
        qualikiz_inputs = dataclasses.replace(
            qualikiz_inputs,
            x=qualikiz_inputs.x * qualikiz_inputs.epsilon_lcfs / _EPSILON_NN,
        )
        feature_scan = model.get_model_inputs_from_qualikiz_inputs(
            qualikiz_inputs)
        feature_scan = jax.lax.cond(
            runtime_config_inputs.transport.clip_inputs,
            lambda: clip_inputs(
                feature_scan,
                runtime_config_inputs.transport.clip_margin,
                model.inputs_and_ranges,
            ),
            lambda: feature_scan,
        )
        model_output = model.predict(feature_scan)
        model_output = _filter_model_output(
            model_output=model_output,
            include_ITG=runtime_config_inputs.transport.include_ITG,
            include_TEM=runtime_config_inputs.transport.include_TEM,
            include_ETG=runtime_config_inputs.transport.include_ETG,
        )
        qi_itg_squeezed = model_output['qi_itg'].squeeze()
        qi = qi_itg_squeezed + model_output['qi_tem'].squeeze()
        qe = (model_output['qe_itg'].squeeze() *
              runtime_config_inputs.transport.ITG_flux_ratio_correction +
              model_output['qe_tem'].squeeze() +
              model_output['qe_etg'].squeeze() *
              runtime_config_inputs.transport.ETG_correction_factor)
        pfe = model_output['pfe_itg'].squeeze(
        ) + model_output['pfe_tem'].squeeze()
        return self._make_core_transport(
            qi=qi,
            qe=qe,
            pfe=pfe,
            quasilinear_inputs=qualikiz_inputs,
            transport=runtime_config_inputs.transport,
            geo=geo,
            core_profiles=core_profiles,
            gradient_reference_length=geo.R_major,
            gyrobohm_flux_reference_length=geo.a_minor,
        )

    def __hash__(self):
        return hash(('QLKNNTransportModel' + self.path + self.name))

    def __eq__(self, other):
        return (isinstance(other, QLKNNTransportModel)
                and self.path == other.path and self.name == other.name)


class QLKNNTransportModel(TransportBase):
    model_name: Annotated[Literal['qlknn'],
                          torax_pydantic.JAX_STATIC] = 'qlknn'
    model_path: Annotated[str, torax_pydantic.JAX_STATIC] = ''
    qlknn_model_name: Annotated[str, torax_pydantic.JAX_STATIC] = ''
    include_ITG: bool = True
    include_TEM: bool = True
    include_ETG: bool = True
    ITG_flux_ratio_correction: float = 1.0
    ETG_correction_factor: float = 1.0 / 3.0
    clip_inputs: bool = False
    clip_margin: float = 0.95
    collisionality_multiplier: float = 1.0
    avoid_big_negative_s: bool = True
    smag_alpha_correction: bool = True
    q_sawtooth_proxy: bool = True
    DV_effective: bool = False
    An_min: pydantic.PositiveFloat = 0.05

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_data(cls, data):
        data = copy.deepcopy(data)
        data['qlknn_model_name'] = data.get('qlknn_model_name', '')
        if 'smoothing_width' not in data:
            data['smoothing_width'] = 0.1
        return data

    def build_transport_model(self):
        return QLKNNTransportModel0(path=self.model_path,
                                    name=self.qlknn_model_name)

    def build_runtime_params(self, t: chex.Numeric):
        base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
        return RuntimeParams0(
            include_ITG=self.include_ITG,
            include_TEM=self.include_TEM,
            include_ETG=self.include_ETG,
            ITG_flux_ratio_correction=self.ITG_flux_ratio_correction,
            ETG_correction_factor=self.ETG_correction_factor,
            clip_inputs=self.clip_inputs,
            clip_margin=self.clip_margin,
            collisionality_multiplier=self.collisionality_multiplier,
            avoid_big_negative_s=self.avoid_big_negative_s,
            smag_alpha_correction=self.smag_alpha_correction,
            q_sawtooth_proxy=self.q_sawtooth_proxy,
            DV_effective=self.DV_effective,
            An_min=self.An_min,
            **base_kwargs,
        )


@functools.partial(jax_utils.jit, static_argnums=(0, 1, 2))
def calculate_total_transport_coeffs(pedestal_model, transport_model,
                                     neoclassical_models, runtime_params, geo,
                                     core_profiles):
    pedestal_model_output = pedestal_model(runtime_params, geo, core_profiles)
    turbulent_transport = transport_model(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        pedestal_model_output=pedestal_model_output,
    )
    neoclassical_transport_coeffs = neoclassical_models.transport(
        runtime_params,
        geo,
        core_profiles,
    )
    return state.CoreTransport(
        **dataclasses.asdict(turbulent_transport),
        **dataclasses.asdict(neoclassical_transport_coeffs),
    )


class Neoclassical0(torax_pydantic.BaseModelFrozen):
    bootstrap_current: (bootstrap_current_zeros.ZerosModelConfig
                        | sauter_current.SauterModelConfig) = pydantic.Field(
                            discriminator="model_name")
    conductivity: sauter_conductivity.SauterModelConfig = (
        torax_pydantic.ValidatedDefault(
            sauter_conductivity.SauterModelConfig()))
    transport: (transport_zeros.ZerosModelConfig) = pydantic.Field(
        discriminator="model_name")

    @pydantic.model_validator(mode="before")
    @classmethod
    def _defaults(cls, data):
        configurable_data = copy.deepcopy(data)
        if "bootstrap_current" not in configurable_data:
            configurable_data["bootstrap_current"] = {"model_name": "zeros"}
        if "transport" not in configurable_data:
            configurable_data["transport"] = {"model_name": "zeros"}
        if "model_name" not in configurable_data["bootstrap_current"]:
            configurable_data["bootstrap_current"]["model_name"] = "sauter"
        return configurable_data

    def build_runtime_params(self):
        return runtime_params.RuntimeParams(
            bootstrap_current=self.bootstrap_current.build_runtime_params(),
            conductivity=self.conductivity.build_runtime_params(),
            transport=self.transport.build_runtime_params(),
        )

    def build_models(self):
        return neoclassical_models.NeoclassicalModels(
            conductivity=self.conductivity.build_model(),
            bootstrap_current=self.bootstrap_current.build_model(),
            transport=self.transport.build_model(),
        )


T = TypeVar('T')
LY_OBJECT_TYPE: TypeAlias = (str
                             | Mapping[str, torax_pydantic.NumpyArray | float])
TIME_INVARIANT = torax_pydantic.TIME_INVARIANT


class CheaseConfig(torax_pydantic.BaseModelFrozen):
    geometry_type: Annotated[Literal['chease'], TIME_INVARIANT] = 'chease'
    n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
    hires_factor: pydantic.PositiveInt = 4
    geometry_directory: Annotated[str | None, TIME_INVARIANT] = None
    Ip_from_parameters: Annotated[bool, TIME_INVARIANT] = True
    geometry_file: str = 'ITER_hybrid_citrin_equil_cheasedata.mat2cols'
    R_major: torax_pydantic.Meter = 6.2
    a_minor: torax_pydantic.Meter = 2.0
    B_0: torax_pydantic.Tesla = 5.3

    @pydantic.model_validator(mode='after')
    def _check_fields(self):
        return self

    def build_geometry(self):
        return standard_geometry.build_standard_geometry(
            _apply_relevant_kwargs(
                standard_geometry.StandardGeometryIntermediates.from_chease,
                self.__dict__,
            ))


class GeometryConfig(torax_pydantic.BaseModelFrozen):
    config: (CheaseConfig) = pydantic.Field(discriminator='geometry_type')


class Geometry0(torax_pydantic.BaseModelFrozen):
    geometry_type: geometry.GeometryType
    geometry_configs: GeometryConfig | dict[torax_pydantic.Second,
                                            GeometryConfig]

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_data(cls, data):
        geometry_type = data['geometry_type']
        return _conform_user_data(data)

    @functools.cached_property
    def build_provider(self):
        geometries = self.geometry_configs.config.build_geometry()
        provider = geometry_provider.ConstantGeometryProvider
        return provider(geometries)


def _conform_user_data(data):
    data_copy = data.copy()
    data_copy['geometry_type'] = data['geometry_type'].lower()
    geometry_type = getattr(geometry.GeometryType,
                            data['geometry_type'].upper())
    constructor_args = {'geometry_type': geometry_type}
    configs_time_dependent = data_copy.pop('geometry_configs', None)
    constructor_args['geometry_configs'] = {'config': data_copy}
    return constructor_args


def _apply_relevant_kwargs(f, kwargs):
    relevant_kwargs = [
        i.name for i in inspect.signature(f).parameters.values()
    ]
    kwargs = {k: kwargs[k] for k in relevant_kwargs}
    return f(**kwargs)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SafetyFactorFit:
    rho_q_min: array_typing.FloatScalar
    q_min: array_typing.FloatScalar
    rho_q_3_2_first: array_typing.FloatScalar
    rho_q_2_1_first: array_typing.FloatScalar
    rho_q_3_1_first: array_typing.FloatScalar
    rho_q_3_2_second: array_typing.FloatScalar
    rho_q_2_1_second: array_typing.FloatScalar
    rho_q_3_1_second: array_typing.FloatScalar


def _sliding_window_of_three(flat_array):
    window_size = 3
    starts = jnp.arange(len(flat_array) - window_size + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(
        flat_array, (start, ), (window_size, )))(starts)


def _fit_polynomial_to_intervals_of_three(rho_norm: jax.Array,
                                          q_face: jax.Array):
    q_face_intervals = _sliding_window_of_three(q_face, )
    rho_norm_intervals = _sliding_window_of_three(rho_norm, )

    @jax.vmap
    def batch_polyfit(q_face_interval: jax.Array,
                      rho_norm_interval: jax.Array):
        chex.assert_shape(q_face_interval, (3, ))
        chex.assert_shape(rho_norm_interval, (3, ))
        rho_norm_squared = rho_norm_interval**2
        A = jnp.array([
            [rho_norm_squared[0], rho_norm_interval[0], 1],
            [rho_norm_squared[1], rho_norm_interval[1], 1],
            [rho_norm_squared[2], rho_norm_interval[2], 1],
        ])
        b = jnp.array(
            [q_face_interval[0], q_face_interval[1], q_face_interval[2]])
        coeffs = jnp.linalg.solve(A, b)
        return coeffs

    return (
        batch_polyfit(q_face_intervals, rho_norm_intervals),
        rho_norm_intervals,
        q_face_intervals,
    )


@jax.vmap
def _minimum_location_value_in_interval(coeffs: jax.Array,
                                        rho_norm_interval: jax.Array,
                                        q_interval: jax.Array):
    min_interval, max_interval = rho_norm_interval[0], rho_norm_interval[1]
    q_min_interval, q_max_interval = (
        q_interval[0],
        q_interval[1],
    )
    a, b = coeffs[0], coeffs[1]
    extremum_location = -b / (2 * a)
    extremum_in_interval = jnp.greater(extremum_location,
                                       min_interval) & jnp.less(
                                           extremum_location, max_interval)
    extremum_value = jax.lax.cond(
        extremum_in_interval,
        lambda x: jnp.polyval(coeffs, x),
        lambda x: jnp.inf,
        extremum_location,
    )
    interval_minimum_location, interval_minimum_value = jax.lax.cond(
        jnp.less(q_min_interval, q_max_interval),
        lambda: (min_interval, q_min_interval),
        lambda: (max_interval, q_max_interval),
    )
    overall_minimum_location, overall_minimum_value = jax.lax.cond(
        jnp.less(interval_minimum_value, extremum_value),
        lambda: (interval_minimum_location, interval_minimum_value),
        lambda: (extremum_location, extremum_value),
    )
    return overall_minimum_location, overall_minimum_value


def _find_roots_quadratic(coeffs: jax.Array):
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    determinant = b**2 - 4.0 * a * c
    roots_exist = jnp.greater(determinant, 0)
    plus_root = jax.lax.cond(
        roots_exist,
        lambda: (-b + jnp.sqrt(determinant)) / (2.0 * a),
        lambda: -jnp.inf,
    )
    minus_root = jax.lax.cond(
        roots_exist,
        lambda: (-b - jnp.sqrt(determinant)) / (2.0 * a),
        lambda: -jnp.inf,
    )
    return jnp.array([plus_root, minus_root])


@functools.partial(jax.vmap, in_axes=(0, 0, None))
def _root_in_interval(coeffs: jax.Array, interval: jax.Array,
                      q_surface: float):
    intercept_coeffs = coeffs - jnp.array([0.0, 0.0, q_surface])
    min_interval, max_interval = interval[0], interval[1]
    root_values = _find_roots_quadratic(intercept_coeffs)
    in_interval = jnp.greater(root_values, min_interval) & jnp.less(
        root_values, max_interval)
    return jnp.where(in_interval, root_values, -jnp.inf)


@jax_utils.jit
def find_min_q_and_q_surface_intercepts(rho_norm, q_face):
    sorted_indices = jnp.argsort(rho_norm)
    rho_norm = rho_norm[sorted_indices]
    q_face = q_face[sorted_indices]
    poly_coeffs, rho_norm_3, q_face_3 = _fit_polynomial_to_intervals_of_three(
        rho_norm, q_face)
    first_rho_norm = jnp.expand_dims(jnp.array([rho_norm[0], rho_norm[2]]),
                                     axis=0)
    first_q_face = jnp.expand_dims(jnp.array([q_face[0], q_face[2]]), axis=0)
    rho_norms = jnp.concat([first_rho_norm, rho_norm_3[1:, 1:]], axis=0)
    q_faces = jnp.concat([first_q_face, q_face_3[1:, 1:]], axis=0)
    rho_q_min_intervals, q_min_intervals = _minimum_location_value_in_interval(
        poly_coeffs, rho_norms, q_faces)
    arg_q_min = jnp.argmin(q_min_intervals)
    rho_q_min = rho_q_min_intervals[arg_q_min]
    q_min = q_min_intervals[arg_q_min]
    rho_q_3_2 = _root_in_interval(poly_coeffs, rho_norms, 1.5).flatten()
    outermost_rho_q_3_2 = rho_q_3_2[jnp.argsort(rho_q_3_2)[-2:]]
    rho_q_2_1 = _root_in_interval(poly_coeffs, rho_norms, 2.0).flatten()
    outermost_rho_q_2_1 = rho_q_2_1[jnp.argsort(rho_q_2_1)[-2:]]
    rho_q_3_1 = _root_in_interval(poly_coeffs, rho_norms, 3.0).flatten()
    outermost_rho_q_3_1 = rho_q_3_1[jnp.argsort(rho_q_3_1)[-2:]]
    return SafetyFactorFit(
        rho_q_min=rho_q_min,
        q_min=q_min,
        rho_q_3_2_first=outermost_rho_q_3_2[0],
        rho_q_2_1_first=outermost_rho_q_2_1[0],
        rho_q_3_1_first=outermost_rho_q_3_1[0],
        rho_q_3_2_second=outermost_rho_q_3_2[1],
        rho_q_2_1_second=outermost_rho_q_2_1[1],
        rho_q_3_1_second=outermost_rho_q_3_1[1],
    )


RADIATION_OUTPUT_NAME = "radiation_impurity_species"
DENSITY_OUTPUT_NAME = "n_impurity_species"
Z_OUTPUT_NAME = "Z_impurity_species"
IMPURITY_DIM = "impurity_symbol"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ImpuritySpeciesOutput:
    radiation: array_typing.FloatVectorCell
    n_impurity: array_typing.FloatVectorCell
    Z_impurity: array_typing.FloatVectorCell


def calculate_impurity_species_output(sim_state, runtime_params):
    impurity_species_output = {}
    mavrin_active = False
    impurity_fractions = sim_state.core_profiles.impurity_fractions
    impurity_names = runtime_params.plasma_composition.impurity_names
    charge_state_info = charge_states.get_average_charge_state(
        ion_symbols=impurity_names,
        T_e=sim_state.core_profiles.T_e.value,
        fractions=jnp.stack(
            [impurity_fractions[symbol] for symbol in impurity_names]),
        Z_override=runtime_params.plasma_composition.impurity.Z_override,
    )
    for i, symbol in enumerate(impurity_names):
        core_profiles = sim_state.core_profiles
        impurity_density_scaling = (core_profiles.Z_impurity /
                                    charge_state_info.Z_avg)
        n_imp = (impurity_fractions[symbol] * core_profiles.n_impurity.value *
                 impurity_density_scaling)
        Z_imp = charge_state_info.Z_per_species[i]
        radiation = jnp.zeros_like(n_imp)
        impurity_species_output[symbol] = ImpuritySpeciesOutput(
            radiation=radiation, n_impurity=n_imp, Z_impurity=Z_imp)
    return impurity_species_output


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PostProcessedOutputs:
    pressure_thermal_i: cell_variable.CellVariable
    pressure_thermal_e: cell_variable.CellVariable
    pressure_thermal_total: cell_variable.CellVariable
    pprime: array_typing.FloatVector
    W_thermal_i: array_typing.FloatScalar
    W_thermal_e: array_typing.FloatScalar
    W_thermal_total: array_typing.FloatScalar
    tau_E: array_typing.FloatScalar
    H89P: array_typing.FloatScalar
    H98: array_typing.FloatScalar
    H97L: array_typing.FloatScalar
    H20: array_typing.FloatScalar
    FFprime: array_typing.FloatVector
    psi_norm: array_typing.FloatVector
    P_SOL_i: array_typing.FloatScalar
    P_SOL_e: array_typing.FloatScalar
    P_SOL_total: array_typing.FloatScalar
    P_aux_i: array_typing.FloatScalar
    P_aux_e: array_typing.FloatScalar
    P_aux_total: array_typing.FloatScalar
    P_external_injected: array_typing.FloatScalar
    P_external_total: array_typing.FloatScalar
    P_ei_exchange_i: array_typing.FloatScalar
    P_ei_exchange_e: array_typing.FloatScalar
    P_aux_generic_i: array_typing.FloatScalar
    P_aux_generic_e: array_typing.FloatScalar
    P_aux_generic_total: array_typing.FloatScalar
    P_alpha_i: array_typing.FloatScalar
    P_alpha_e: array_typing.FloatScalar
    P_alpha_total: array_typing.FloatScalar
    P_ohmic_e: array_typing.FloatScalar
    P_bremsstrahlung_e: array_typing.FloatScalar
    P_cyclotron_e: array_typing.FloatScalar
    P_ecrh_e: array_typing.FloatScalar
    P_radiation_e: array_typing.FloatScalar
    I_ecrh: array_typing.FloatScalar
    I_aux_generic: array_typing.FloatScalar
    P_fusion: array_typing.FloatScalar
    Q_fusion: array_typing.FloatScalar
    P_icrh_e: array_typing.FloatScalar
    P_icrh_i: array_typing.FloatScalar
    P_icrh_total: array_typing.FloatScalar
    P_LH_high_density: array_typing.FloatScalar
    P_LH_min: array_typing.FloatScalar
    P_LH: array_typing.FloatScalar
    n_e_min_P_LH: array_typing.FloatScalar
    E_fusion: array_typing.FloatScalar
    E_aux_total: array_typing.FloatScalar
    E_ohmic_e: array_typing.FloatScalar
    E_external_injected: array_typing.FloatScalar
    E_external_total: array_typing.FloatScalar
    T_e_volume_avg: array_typing.FloatScalar
    T_i_volume_avg: array_typing.FloatScalar
    n_e_volume_avg: array_typing.FloatScalar
    n_i_volume_avg: array_typing.FloatScalar
    n_e_line_avg: array_typing.FloatScalar
    n_i_line_avg: array_typing.FloatScalar
    fgw_n_e_volume_avg: array_typing.FloatScalar
    fgw_n_e_line_avg: array_typing.FloatScalar
    q95: array_typing.FloatScalar
    W_pol: array_typing.FloatScalar
    li3: array_typing.FloatScalar
    dW_thermal_dt: array_typing.FloatScalar
    rho_q_min: array_typing.FloatScalar
    q_min: array_typing.FloatScalar
    rho_q_3_2_first: array_typing.FloatScalar
    rho_q_3_2_second: array_typing.FloatScalar
    rho_q_2_1_first: array_typing.FloatScalar
    rho_q_2_1_second: array_typing.FloatScalar
    rho_q_3_1_first: array_typing.FloatScalar
    rho_q_3_1_second: array_typing.FloatScalar
    I_bootstrap: array_typing.FloatScalar
    j_external: array_typing.FloatVector
    j_ohmic: array_typing.FloatVector
    S_gas_puff: array_typing.FloatScalar
    S_pellet: array_typing.FloatScalar
    S_generic_particle: array_typing.FloatScalar
    beta_tor: array_typing.FloatScalar
    beta_pol: array_typing.FloatScalar
    beta_N: array_typing.FloatScalar
    S_total: array_typing.FloatScalar
    impurity_species: dict[str, ImpuritySpeciesOutput]


ION_EL_HEAT_SOURCE_TRANSFORMATIONS = {
    'generic_heat': 'P_aux_generic',
    'fusion': 'P_alpha',
    'icrh': 'P_icrh',
}
EL_HEAT_SOURCE_TRANSFORMATIONS = {
    'ohmic': 'P_ohmic_e',
    'bremsstrahlung': 'P_bremsstrahlung_e',
    'cyclotron_radiation': 'P_cyclotron_e',
    'ecrh': 'P_ecrh_e',
    'impurity_radiation': 'P_radiation_e',
}
EXTERNAL_HEATING_SOURCES = [
    'generic_heat',
    'ecrh',
    'icrh',
]
CURRENT_SOURCE_TRANSFORMATIONS = {
    'generic_current': 'I_aux_generic',
    'ecrh': 'I_ecrh',
}
PARTICLE_SOURCE_TRANSFORMATIONS = {
    'gas_puff': 'S_gas_puff',
    'pellet': 'S_pellet',
    'generic_particle': 'S_generic_particle',
}


def _get_integrated_source_value(
    source_profiles_dict: dict[str, array_typing.FloatVector],
    internal_source_name: str,
    geo: geometry.Geometry,
    integration_fn: Callable[[array_typing.FloatVector, geometry.Geometry],
                             jax.Array],
):
    if internal_source_name in source_profiles_dict:
        return integration_fn(source_profiles_dict[internal_source_name], geo)
    else:
        return jnp.array(0.0, dtype=jax_utils.get_dtype())


def _calculate_integrated_sources(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    core_sources: SourceProfiles,
    runtime_params: runtime_params_slice.RuntimeParams,
):
    integrated = {}
    integrated['P_alpha_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    integrated['S_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    qei = core_sources.qei.qei_coef * (core_profiles.T_e.value -
                                       core_profiles.T_i.value)
    integrated['P_ei_exchange_i'] = math_utils.volume_integration(qei, geo)
    integrated['P_ei_exchange_e'] = -integrated['P_ei_exchange_i']
    integrated['P_SOL_i'] = integrated['P_ei_exchange_i']
    integrated['P_SOL_e'] = integrated['P_ei_exchange_e']
    integrated['P_aux_i'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    integrated['P_aux_e'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    integrated['P_external_injected'] = jnp.array(0.0,
                                                  dtype=jax_utils.get_dtype())
    for key, value in ION_EL_HEAT_SOURCE_TRANSFORMATIONS.items():
        is_in_T_i = key in core_sources.T_i
        is_in_T_e = key in core_sources.T_e
        integrated[f'{value}_i'] = _get_integrated_source_value(
            core_sources.T_i, key, geo, math_utils.volume_integration)
        integrated[f'{value}_e'] = _get_integrated_source_value(
            core_sources.T_e, key, geo, math_utils.volume_integration)
        integrated[f'{value}_total'] = (integrated[f'{value}_i'] +
                                        integrated[f'{value}_e'])
        integrated['P_SOL_i'] += integrated[f'{value}_i']
        integrated['P_SOL_e'] += integrated[f'{value}_e']
        if key in EXTERNAL_HEATING_SOURCES:
            integrated['P_aux_i'] += integrated[f'{value}_i']
            integrated['P_aux_e'] += integrated[f'{value}_e']
            source_params = runtime_params.sources.get(key)
            if source_params is not None and hasattr(source_params,
                                                     'absorption_fraction'):
                total_absorbed = integrated[f'{value}_total']
                injected_power = total_absorbed / source_params.absorption_fraction
                integrated['P_external_injected'] += injected_power
            else:
                integrated['P_external_injected'] += integrated[
                    f'{value}_total']
    for key, value in EL_HEAT_SOURCE_TRANSFORMATIONS.items():
        integrated[f'{value}'] = _get_integrated_source_value(
            core_sources.T_e, key, geo, math_utils.volume_integration)
        integrated['P_SOL_e'] += integrated[f'{value}']
        if key in EXTERNAL_HEATING_SOURCES:
            integrated['P_aux_e'] += integrated[f'{value}']
            integrated['P_external_injected'] += integrated[f'{value}']
    for key, value in CURRENT_SOURCE_TRANSFORMATIONS.items():
        integrated[f'{value}'] = _get_integrated_source_value(
            core_sources.psi, key, geo, math_utils.area_integration)
    for key, value in PARTICLE_SOURCE_TRANSFORMATIONS.items():
        integrated[f'{value}'] = _get_integrated_source_value(
            core_sources.n_e, key, geo, math_utils.volume_integration)
        integrated['S_total'] += integrated[f'{value}']
    integrated['P_SOL_total'] = integrated['P_SOL_i'] + integrated['P_SOL_e']
    integrated['P_aux_total'] = integrated['P_aux_i'] + integrated['P_aux_e']
    integrated['P_fusion'] = 5 * integrated['P_alpha_total']
    integrated['P_external_total'] = (integrated['P_external_injected'] +
                                      integrated['P_ohmic_e'])
    return integrated


@jax_utils.jit
def make_post_processed_outputs(
    sim_state,
    runtime_params: runtime_params_slice.RuntimeParams,
    previous_post_processed_outputs: PostProcessedOutputs | None = None,
):
    impurity_radiation_outputs = (calculate_impurity_species_output(
        sim_state, runtime_params))
    (
        pressure_thermal_el,
        pressure_thermal_ion,
        pressure_thermal_tot,
    ) = formulas_ph.calculate_pressure(sim_state.core_profiles)
    pprime_face = formulas_ph.calc_pprime(sim_state.core_profiles)
    W_thermal_el, W_thermal_ion, W_thermal_tot = (
        formulas_ph.calculate_stored_thermal_energy(
            pressure_thermal_el,
            pressure_thermal_ion,
            pressure_thermal_tot,
            sim_state.geometry,
        ))
    FFprime_face = formulas_ph.calc_FFprime(sim_state.core_profiles,
                                            sim_state.geometry)
    psi_face = sim_state.core_profiles.psi.face_value()
    psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])
    integrated_sources = _calculate_integrated_sources(
        sim_state.geometry,
        sim_state.core_profiles,
        sim_state.core_sources,
        runtime_params,
    )
    Q_fusion = (
        integrated_sources['P_fusion'] /
        (integrated_sources['P_external_total'] + constants.CONSTANTS.eps))
    P_LH_hi_dens, P_LH_min, P_LH, n_e_min_P_LH = (
        scaling_laws.calculate_plh_scaling_factor(sim_state.geometry,
                                                  sim_state.core_profiles))
    Ploss = (integrated_sources['P_alpha_total'] +
             integrated_sources['P_aux_total'] +
             integrated_sources['P_ohmic_e'] + constants.CONSTANTS.eps)
    if previous_post_processed_outputs is not None:
        dW_th_dt = (
            W_thermal_tot -
            previous_post_processed_outputs.W_thermal_total) / sim_state.dt
    else:
        dW_th_dt = 0.0
    tauE = W_thermal_tot / Ploss
    tauH89P = scaling_laws.calculate_scaling_law_confinement_time(
        sim_state.geometry, sim_state.core_profiles, Ploss, 'H89P')
    tauH98 = scaling_laws.calculate_scaling_law_confinement_time(
        sim_state.geometry, sim_state.core_profiles, Ploss, 'H98')
    tauH97L = scaling_laws.calculate_scaling_law_confinement_time(
        sim_state.geometry, sim_state.core_profiles, Ploss, 'H97L')
    tauH20 = scaling_laws.calculate_scaling_law_confinement_time(
        sim_state.geometry, sim_state.core_profiles, Ploss, 'H20')
    H89P = tauE / tauH89P
    H98 = tauE / tauH98
    H97L = tauE / tauH97L
    H20 = tauE / tauH20
    if previous_post_processed_outputs is not None:
        E_fusion = (previous_post_processed_outputs.E_fusion + sim_state.dt *
                    (integrated_sources['P_fusion'] +
                     previous_post_processed_outputs.P_fusion) / 2.0)
        E_aux_total = (previous_post_processed_outputs.E_aux_total +
                       sim_state.dt *
                       (integrated_sources['P_aux_total'] +
                        previous_post_processed_outputs.P_aux_total) / 2.0)
        E_ohmic_e = (previous_post_processed_outputs.E_ohmic_e + sim_state.dt *
                     (integrated_sources['P_ohmic_e'] +
                      previous_post_processed_outputs.P_ohmic_e) / 2.0)
        E_external_injected = (
            previous_post_processed_outputs.E_external_injected +
            sim_state.dt *
            (integrated_sources['P_external_injected'] +
             previous_post_processed_outputs.P_external_injected) / 2.0)
        E_external_total = (
            previous_post_processed_outputs.E_external_total + sim_state.dt *
            (integrated_sources['P_external_total'] +
             previous_post_processed_outputs.P_external_total) / 2.0)
    else:
        E_fusion = 0.0
        E_aux_total = 0.0
        E_ohmic_e = 0.0
        E_external_injected = 0.0
        E_external_total = 0.0
    q95 = psi_calculations.calc_q95(psi_norm_face,
                                    sim_state.core_profiles.q_face)
    te_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.T_e.value, sim_state.geometry)
    ti_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.T_i.value, sim_state.geometry)
    n_e_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.n_e.value, sim_state.geometry)
    n_i_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.n_i.value, sim_state.geometry)
    n_e_line_avg = math_utils.line_average(sim_state.core_profiles.n_e.value,
                                           sim_state.geometry)
    n_i_line_avg = math_utils.line_average(sim_state.core_profiles.n_i.value,
                                           sim_state.geometry)
    fgw_n_e_volume_avg = formulas_ph.calculate_greenwald_fraction(
        n_e_volume_avg, sim_state.core_profiles, sim_state.geometry)
    fgw_n_e_line_avg = formulas_ph.calculate_greenwald_fraction(
        n_e_line_avg, sim_state.core_profiles, sim_state.geometry)
    Wpol = psi_calculations.calc_Wpol(sim_state.geometry,
                                      sim_state.core_profiles.psi)
    li3 = psi_calculations.calc_li3(
        sim_state.geometry.R_major,
        Wpol,
        sim_state.core_profiles.Ip_profile_face[-1],
    )
    safety_factor_fit_outputs = (find_min_q_and_q_surface_intercepts(
        sim_state.geometry.rho_face_norm,
        sim_state.core_profiles.q_face,
    ))
    I_bootstrap = math_utils.area_integration(
        sim_state.core_sources.bootstrap_current.j_bootstrap,
        sim_state.geometry)
    j_external = sum(sim_state.core_sources.psi.values())
    psi_current = (j_external +
                   sim_state.core_sources.bootstrap_current.j_bootstrap)
    j_ohmic = sim_state.core_profiles.j_total - psi_current
    beta_tor, beta_pol, beta_N = formulas_ph.calculate_betas(
        sim_state.core_profiles, sim_state.geometry)
    return PostProcessedOutputs(
        pressure_thermal_i=pressure_thermal_ion,
        pressure_thermal_e=pressure_thermal_el,
        pressure_thermal_total=pressure_thermal_tot,
        pprime=pprime_face,
        W_thermal_i=W_thermal_ion,
        W_thermal_e=W_thermal_el,
        W_thermal_total=W_thermal_tot,
        tau_E=tauE,
        H89P=H89P,
        H98=H98,
        H97L=H97L,
        H20=H20,
        FFprime=FFprime_face,
        psi_norm=psi_norm_face,
        **integrated_sources,
        Q_fusion=Q_fusion,
        P_LH=P_LH,
        P_LH_min=P_LH_min,
        P_LH_high_density=P_LH_hi_dens,
        n_e_min_P_LH=n_e_min_P_LH,
        E_fusion=E_fusion,
        E_aux_total=E_aux_total,
        E_ohmic_e=E_ohmic_e,
        E_external_injected=E_external_injected,
        E_external_total=E_external_total,
        T_e_volume_avg=te_volume_avg,
        T_i_volume_avg=ti_volume_avg,
        n_e_volume_avg=n_e_volume_avg,
        n_i_volume_avg=n_i_volume_avg,
        n_e_line_avg=n_e_line_avg,
        n_i_line_avg=n_i_line_avg,
        fgw_n_e_volume_avg=fgw_n_e_volume_avg,
        fgw_n_e_line_avg=fgw_n_e_line_avg,
        q95=q95,
        W_pol=Wpol,
        li3=li3,
        dW_thermal_dt=dW_th_dt,
        rho_q_min=safety_factor_fit_outputs.rho_q_min,
        q_min=safety_factor_fit_outputs.q_min,
        rho_q_3_2_first=safety_factor_fit_outputs.rho_q_3_2_first,
        rho_q_2_1_first=safety_factor_fit_outputs.rho_q_2_1_first,
        rho_q_3_1_first=safety_factor_fit_outputs.rho_q_3_1_first,
        rho_q_3_2_second=safety_factor_fit_outputs.rho_q_3_2_second,
        rho_q_2_1_second=safety_factor_fit_outputs.rho_q_2_1_second,
        rho_q_3_1_second=safety_factor_fit_outputs.rho_q_3_1_second,
        I_bootstrap=I_bootstrap,
        j_external=j_external,
        j_ohmic=j_ohmic,
        beta_tor=beta_tor,
        beta_pol=beta_pol,
        beta_N=beta_N,
        impurity_species=impurity_radiation_outputs,
    )


def construct_xarray_for_radiation_output(impurity_radiation_outputs, times,
                                          rho_cell_norm, time_coord,
                                          rho_cell_norm_coord):
    radiation_data = []
    n_impurity_data = []
    Z_impurity_data = []
    impurity_symbols = []
    xr_dict = {}
    for impurity_symbol in impurity_radiation_outputs:
        radiation_data.append(
            impurity_radiation_outputs[impurity_symbol].radiation)
        n_impurity_data.append(
            impurity_radiation_outputs[impurity_symbol].n_impurity)
        Z_impurity_data.append(
            impurity_radiation_outputs[impurity_symbol].Z_impurity)
        impurity_symbols.append(impurity_symbol)
    radiation_data = np.stack(radiation_data, axis=0)
    n_impurity_data = np.stack(n_impurity_data, axis=0)
    Z_impurity_data = np.stack(Z_impurity_data, axis=0)
    xr_dict[RADIATION_OUTPUT_NAME] = xr.DataArray(
        radiation_data,
        dims=[IMPURITY_DIM, time_coord, rho_cell_norm_coord],
        coords={
            IMPURITY_DIM: impurity_symbols,
            time_coord: times,
            rho_cell_norm_coord: rho_cell_norm,
        },
        name=RADIATION_OUTPUT_NAME,
    )
    xr_dict[DENSITY_OUTPUT_NAME] = xr.DataArray(
        n_impurity_data,
        dims=[IMPURITY_DIM, time_coord, rho_cell_norm_coord],
        coords={
            IMPURITY_DIM: impurity_symbols,
            time_coord: times,
            rho_cell_norm_coord: rho_cell_norm,
        },
        name=DENSITY_OUTPUT_NAME,
    )
    xr_dict[Z_OUTPUT_NAME] = xr.DataArray(
        Z_impurity_data,
        dims=[IMPURITY_DIM, time_coord, rho_cell_norm_coord],
        coords={
            IMPURITY_DIM: impurity_symbols,
            time_coord: times,
            rho_cell_norm_coord: rho_cell_norm,
        },
        name=Z_OUTPUT_NAME,
    )
    return xr_dict


SCALING_FACTORS: Final[Mapping[str, float]] = immutabledict.immutabledict({
    'T_i':
    1.0,
    'T_e':
    1.0,
    'n_e':
    1e20,
    'psi':
    1.0,
})
_trapz = jax.scipy.integrate.trapezoid


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Ions:
    n_i: cell_variable.CellVariable
    n_impurity: cell_variable.CellVariable
    impurity_fractions: Mapping[str, array_typing.FloatVectorCell]
    Z_i: array_typing.FloatVectorCell
    Z_i_face: array_typing.FloatVectorFace
    Z_impurity: array_typing.FloatVectorCell
    Z_impurity_face: array_typing.FloatVectorFace
    A_i: array_typing.FloatScalar
    A_impurity: array_typing.FloatVectorCell
    A_impurity_face: array_typing.FloatVectorFace
    Z_eff: array_typing.FloatVectorCell
    Z_eff_face: array_typing.FloatVectorFace


def get_updated_ion_temperature(profile_conditions_params, geo):
    T_i = cell_variable.CellVariable(
        value=profile_conditions_params.T_i,
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=profile_conditions_params.T_i_right_bc,
        dr=geo.drho_norm,
    )
    return T_i


def get_updated_electron_temperature(profile_conditions_params, geo):
    T_e = cell_variable.CellVariable(
        value=profile_conditions_params.T_e,
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=profile_conditions_params.T_e_right_bc,
        dr=geo.drho_norm,
    )
    return T_e


def get_updated_electron_density(profile_conditions_params, geo):
    nGW = (profile_conditions_params.Ip / 1e6 / (jnp.pi * geo.a_minor**2) *
           1e20)
    n_e_value = jnp.where(
        profile_conditions_params.n_e_nbar_is_fGW,
        profile_conditions_params.n_e * nGW,
        profile_conditions_params.n_e,
    )
    n_e_right_bc = jnp.where(
        profile_conditions_params.n_e_right_bc_is_fGW,
        profile_conditions_params.n_e_right_bc * nGW,
        profile_conditions_params.n_e_right_bc,
    )
    face_left = n_e_value[0]
    face_right = n_e_right_bc
    face_inner = (n_e_value[..., :-1] + n_e_value[..., 1:]) / 2.0
    n_e_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]], )
    a_minor_out = geo.R_out_face[-1] - geo.R_out_face[0]
    target_nbar = jnp.where(
        profile_conditions_params.n_e_nbar_is_fGW,
        profile_conditions_params.nbar * nGW,
        profile_conditions_params.nbar,
    )
    nbar_from_n_e_face_inner = (_trapz(n_e_face[:-1], geo.R_out_face[:-1]) /
                                a_minor_out)
    dr_edge = geo.R_out_face[-1] - geo.R_out_face[-2]
    C = (target_nbar - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
        nbar_from_n_e_face_inner + 0.5 * n_e_face[-2] * dr_edge / a_minor_out)
    n_e_value = C * n_e_value
    n_e = cell_variable.CellVariable(
        value=n_e_value,
        dr=geo.drho_norm,
        right_face_grad_constraint=None,
        right_face_constraint=n_e_right_bc,
    )
    return n_e


@dataclasses.dataclass(frozen=True)
class _IonProperties:
    A_impurity: array_typing.FloatVectorCell
    A_impurity_face: array_typing.FloatVectorFace
    Z_impurity: array_typing.FloatVectorCell
    Z_impurity_face: array_typing.FloatVectorFace
    Z_eff: array_typing.FloatVectorCell
    dilution_factor: array_typing.FloatVectorCell
    dilution_factor_edge: array_typing.FloatScalar
    impurity_fractions: array_typing.FloatVector


def _get_ion_properties_from_fractions(impurity_symbols, impurity_params, T_e,
                                       Z_i, Z_i_face, Z_eff_from_config,
                                       Z_eff_face_from_config):
    Z_impurity = charge_states.get_average_charge_state(
        ion_symbols=impurity_symbols,
        T_e=T_e.value,
        fractions=impurity_params.fractions,
        Z_override=impurity_params.Z_override,
    ).Z_mixture
    Z_impurity_face = charge_states.get_average_charge_state(
        ion_symbols=impurity_symbols,
        T_e=T_e.face_value(),
        fractions=impurity_params.fractions_face,
        Z_override=impurity_params.Z_override,
    ).Z_mixture
    Z_eff = Z_eff_from_config
    Z_eff_edge = Z_eff_face_from_config[-1]
    dilution_factor = jnp.where(
        Z_eff == 1.0,
        1.0,
        formulas_ph.calculate_main_ion_dilution_factor(Z_i, Z_impurity, Z_eff),
    )
    dilution_factor_edge = jnp.where(
        Z_eff_edge == 1.0,
        1.0,
        formulas_ph.calculate_main_ion_dilution_factor(Z_i_face[-1],
                                                       Z_impurity_face[-1],
                                                       Z_eff_edge),
    )
    return _IonProperties(
        A_impurity=impurity_params.A_avg,
        A_impurity_face=impurity_params.A_avg_face,
        Z_impurity=Z_impurity,
        Z_impurity_face=Z_impurity_face,
        Z_eff=Z_eff,
        dilution_factor=dilution_factor,
        dilution_factor_edge=dilution_factor_edge,
        impurity_fractions=impurity_params.fractions,
    )


@jax_utils.jit
def get_updated_ions(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
):
    Z_i = charge_states.get_average_charge_state(
        ion_symbols=runtime_params.plasma_composition.main_ion_names,
        T_e=T_e.value,
        fractions=runtime_params.plasma_composition.main_ion.fractions,
        Z_override=runtime_params.plasma_composition.main_ion.Z_override,
    ).Z_mixture
    Z_i_face = charge_states.get_average_charge_state(
        ion_symbols=runtime_params.plasma_composition.main_ion_names,
        T_e=T_e.face_value(),
        fractions=runtime_params.plasma_composition.main_ion.fractions,
        Z_override=runtime_params.plasma_composition.main_ion.Z_override,
    ).Z_mixture
    impurity_params = runtime_params.plasma_composition.impurity
    match impurity_params:
        case RuntimeParamsIF():
            ion_properties = _get_ion_properties_from_fractions(
                runtime_params.plasma_composition.impurity_names,
                impurity_params,
                T_e,
                Z_i,
                Z_i_face,
                runtime_params.plasma_composition.Z_eff,
                runtime_params.plasma_composition.Z_eff_face,
            )
        case _:
            raise ValueError("Unknown impurity mode.")
    n_i = cell_variable.CellVariable(
        value=n_e.value * ion_properties.dilution_factor,
        dr=geo.drho_norm,
        right_face_grad_constraint=None,
        right_face_constraint=n_e.right_face_constraint *
        ion_properties.dilution_factor_edge,
    )
    n_impurity_value = jnp.where(
        ion_properties.dilution_factor == 1.0,
        0.0,
        (n_e.value - n_i.value * Z_i) / ion_properties.Z_impurity,
    )
    n_impurity_right_face_constraint = jnp.where(
        ion_properties.dilution_factor_edge == 1.0,
        0.0,
        (n_e.right_face_constraint - n_i.right_face_constraint * Z_i_face[-1])
        / ion_properties.Z_impurity_face[-1],
    )
    n_impurity = cell_variable.CellVariable(
        value=n_impurity_value,
        dr=geo.drho_norm,
        right_face_grad_constraint=None,
        right_face_constraint=n_impurity_right_face_constraint,
    )
    Z_eff_face = _calculate_Z_eff(
        Z_i_face,
        ion_properties.Z_impurity_face,
        n_i.face_value(),
        n_impurity.face_value(),
        n_e.face_value(),
    )
    impurity_fractions_dict = {}
    for i, symbol in enumerate(
            runtime_params.plasma_composition.impurity_names):
        fraction = ion_properties.impurity_fractions[i]
        impurity_fractions_dict[symbol] = fraction
    return Ions(
        n_i=n_i,
        n_impurity=n_impurity,
        impurity_fractions=impurity_fractions_dict,
        Z_i=Z_i,
        Z_i_face=Z_i_face,
        Z_impurity=ion_properties.Z_impurity,
        Z_impurity_face=ion_properties.Z_impurity_face,
        A_i=runtime_params.plasma_composition.main_ion.A_avg,
        A_impurity=ion_properties.A_impurity,
        A_impurity_face=ion_properties.A_impurity_face,
        Z_eff=ion_properties.Z_eff,
        Z_eff_face=Z_eff_face,
    )


def _calculate_Z_eff(Z_i, Z_impurity, n_i, n_impurity, n_e):
    return (Z_i**2 * n_i + Z_impurity**2 * n_impurity) / n_e


def initial_core_profiles0(runtime_params, geo, source_models,
                           neoclassical_models):
    T_i = get_updated_ion_temperature(runtime_params.profile_conditions, geo)
    T_e = get_updated_electron_temperature(runtime_params.profile_conditions,
                                           geo)
    n_e = get_updated_electron_density(runtime_params.profile_conditions, geo)
    ions = get_updated_ions(runtime_params, geo, n_e, T_e)
    v_loop_lcfs = (
        np.array(runtime_params.profile_conditions.v_loop_lcfs)
        if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
        else np.array(0.0, dtype=jax_utils.get_dtype()))
    psidot = cell_variable.CellVariable(
        value=np.zeros_like(geo.rho),
        dr=geo.drho_norm,
    )
    psi = cell_variable.CellVariable(value=np.zeros_like(geo.rho),
                                     dr=geo.drho_norm)
    core_profiles = state.CoreProfiles(
        T_i=T_i,
        T_e=T_e,
        n_e=n_e,
        n_i=ions.n_i,
        Z_i=ions.Z_i,
        Z_i_face=ions.Z_i_face,
        A_i=ions.A_i,
        n_impurity=ions.n_impurity,
        impurity_fractions=ions.impurity_fractions,
        Z_impurity=ions.Z_impurity,
        Z_impurity_face=ions.Z_impurity_face,
        A_impurity=ions.A_impurity,
        A_impurity_face=ions.A_impurity_face,
        Z_eff=ions.Z_eff,
        Z_eff_face=ions.Z_eff_face,
        psi=psi,
        psidot=psidot,
        q_face=np.zeros_like(geo.rho_face),
        s_face=np.zeros_like(geo.rho_face),
        v_loop_lcfs=v_loop_lcfs,
        sigma=np.zeros_like(geo.rho),
        sigma_face=np.zeros_like(geo.rho_face),
        j_total=np.zeros_like(geo.rho),
        j_total_face=np.zeros_like(geo.rho_face),
        Ip_profile_face=np.zeros_like(geo.rho_face),
    )
    return _init_psi_and_psi_derived(
        runtime_params,
        geo,
        core_profiles,
        source_models,
        neoclassical_models,
    )


def _get_initial_psi_mode(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
):
    psi_mode = runtime_params.profile_conditions.initial_psi_mode
    if psi_mode == InitialPsiMode.PROFILE_CONDITIONS:
        if runtime_params.profile_conditions.psi is None:
            logging.warning(
                'Falling back to legacy behavior as `profile_conditions.psi` is '
                'None. Future versions of TORAX will require `psi` to be provided '
                'if `initial_psi_mode` is PROFILE_CONDITIONS. Use '
                '`initial_psi_mode` to initialize psi from `j` or `geometry` and '
                'avoid this warning.')
            if (isinstance(geo, standard_geometry.StandardGeometry) and
                    not runtime_params.profile_conditions.initial_psi_from_j):
                psi_mode = InitialPsiMode.GEOMETRY
            else:
                psi_mode = profile_conditions_lib.InitialPsiMode.J
    return psi_mode


def _init_psi_and_psi_derived(runtime_params, geo, core_profiles,
                              source_models, neoclassical_models):
    sources_are_calculated = False
    source_profiles = build_all_zero_profiles(geo)
    initial_psi_mode = _get_initial_psi_mode(runtime_params, geo)
    match initial_psi_mode:
        case InitialPsiMode.PROFILE_CONDITIONS:
            if runtime_params.profile_conditions.psi is None:
                raise ValueError(
                    'psi is None, but initial_psi_mode is PROFILE_CONDITIONS.')
            dpsi_drhonorm_edge = (
                psi_calculations.calculate_psi_grad_constraint_from_Ip(
                    runtime_params.profile_conditions.Ip,
                    geo,
                ))
            if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition:
                right_face_grad_constraint = None
                right_face_constraint = (
                    runtime_params.profile_conditions.psi[-1] +
                    dpsi_drhonorm_edge * geo.drho_norm / 2)
            else:
                right_face_grad_constraint = dpsi_drhonorm_edge
                right_face_constraint = None
            psi = cell_variable.CellVariable(
                value=runtime_params.profile_conditions.psi,
                right_face_grad_constraint=right_face_grad_constraint,
                right_face_constraint=right_face_constraint,
                dr=geo.drho_norm,
            )
        case InitialPsiMode.GEOMETRY:
            if not isinstance(geo, standard_geometry.StandardGeometry):
                raise ValueError(
                    'GEOMETRY initial_psi_source is only supported for standard'
                    ' geometry.')
            dpsi_drhonorm_edge = (
                psi_calculations.calculate_psi_grad_constraint_from_Ip(
                    runtime_params.profile_conditions.Ip,
                    geo,
                ))
            psi = cell_variable.CellVariable(
                value=geo.psi_from_Ip,
                right_face_grad_constraint=None
                if runtime_params.profile_conditions.
                use_v_loop_lcfs_boundary_condition else dpsi_drhonorm_edge,
                right_face_constraint=geo.psi_from_Ip_face[-1]
                if runtime_params.profile_conditions.
                use_v_loop_lcfs_boundary_condition else None,
                dr=geo.drho_norm,
            )
        case profile_conditions_lib.InitialPsiMode.J:
            j_total_hires = _get_j_total_hires_with_no_external_sources(
                runtime_params, geo)
            psi = update_psi_from_j(
                runtime_params.profile_conditions.Ip,
                geo,
                j_total_hires,
                use_v_loop_lcfs_boundary_condition=runtime_params.
                profile_conditions.use_v_loop_lcfs_boundary_condition,
            )
            if not (runtime_params.profile_conditions.
                    initial_j_is_total_current):
                core_profiles_initial = dataclasses.replace(
                    core_profiles,
                    psi=psi,
                    q_face=psi_calculations.calc_q_face(geo, psi),
                    s_face=psi_calculations.calc_s_face(geo, psi),
                )
                psi, source_profiles = _iterate_psi_and_sources(
                    runtime_params=runtime_params,
                    geo=geo,
                    core_profiles=core_profiles_initial,
                    neoclassical_models=neoclassical_models,
                    source_models=source_models,
                    source_profiles=source_profiles,
                    iterations=2,
                )
                sources_are_calculated = True
    core_profiles = _calculate_all_psi_dependent_profiles(
        runtime_params=runtime_params,
        geo=geo,
        psi=psi,
        core_profiles=core_profiles,
        source_profiles=source_profiles,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        sources_are_calculated=sources_are_calculated,
    )
    return core_profiles


def _calculate_all_psi_dependent_profiles(runtime_params, geo, psi,
                                          core_profiles, source_profiles,
                                          source_models, neoclassical_models,
                                          sources_are_calculated):
    j_total, j_total_face, Ip_profile_face = psi_calculations.calc_j_total(
        geo, psi)
    core_profiles = dataclasses.replace(
        core_profiles,
        psi=psi,
        q_face=psi_calculations.calc_q_face(geo, psi),
        s_face=psi_calculations.calc_s_face(geo, psi),
        j_total=j_total,
        j_total_face=j_total_face,
        Ip_profile_face=Ip_profile_face,
    )
    conductivity = neoclassical_models.conductivity.calculate_conductivity(
        geo,
        core_profiles,
    )
    if not sources_are_calculated:
        source_profiles = _get_bootstrap_and_standard_source_profiles(
            runtime_params,
            geo,
            core_profiles,
            neoclassical_models,
            source_models,
            source_profiles,
        )
    if (not runtime_params.numerics.evolve_current
            and runtime_params.profile_conditions.psidot is not None):
        psidot_value = runtime_params.profile_conditions.psidot
    else:
        psi_sources = source_profiles.total_psi_sources(geo)
        psidot_value = psi_calculations.calculate_psidot_from_psi_sources(
            psi_sources=psi_sources,
            sigma=conductivity.sigma,
            resistivity_multiplier=runtime_params.numerics.
            resistivity_multiplier,
            psi=psi,
            geo=geo,
        )
    v_loop_lcfs = (
        runtime_params.profile_conditions.v_loop_lcfs
        if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
        else psidot_value[-1])
    psidot = dataclasses.replace(
        core_profiles.psidot,
        value=psidot_value,
        right_face_constraint=v_loop_lcfs,
        right_face_grad_constraint=None,
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        psidot=psidot,
        sigma=conductivity.sigma,
        sigma_face=conductivity.sigma_face,
    )
    return core_profiles


def _get_bootstrap_and_standard_source_profiles(runtime_params, geo,
                                                core_profiles,
                                                neoclassical_models,
                                                source_models,
                                                source_profiles):
    build_standard_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        psi_only=True,
        calculate_anyway=True,
        calculated_source_profiles=source_profiles,
    )
    bootstrap_current = (
        neoclassical_models.bootstrap_current.calculate_bootstrap_current(
            runtime_params, geo, core_profiles))
    source_profiles = dataclasses.replace(source_profiles,
                                          bootstrap_current=bootstrap_current)
    return source_profiles


def core_profiles_to_solver_x_tuple(
    core_profiles,
    evolving_names,
):
    x_tuple_for_solver_list = []
    for name in evolving_names:
        original_units_cv = getattr(core_profiles, name)
        solver_x_tuple_cv = scale_cell_variable(
            cv=original_units_cv,
            scaling_factor=1 / SCALING_FACTORS[name],
        )
        x_tuple_for_solver_list.append(solver_x_tuple_cv)
    return tuple(x_tuple_for_solver_list)


def solver_x_tuple_to_core_profiles(x_new, evolving_names, core_profiles):
    updated_vars = {}
    for i, var_name in enumerate(evolving_names):
        solver_x_tuple_cv = x_new[i]
        original_units_cv = scale_cell_variable(
            cv=solver_x_tuple_cv,
            scaling_factor=SCALING_FACTORS[var_name],
        )
        updated_vars[var_name] = original_units_cv
    return dataclasses.replace(core_profiles, **updated_vars)


def scale_cell_variable(cv, scaling_factor):
    operation = lambda x, factor: x * factor if x is not None else None
    scaled_value = operation(cv.value, scaling_factor)
    scaled_left_face_constraint = operation(cv.left_face_constraint,
                                            scaling_factor)
    scaled_left_face_grad_constraint = operation(cv.left_face_grad_constraint,
                                                 scaling_factor)
    scaled_right_face_constraint = operation(cv.right_face_constraint,
                                             scaling_factor)
    scaled_right_face_grad_constraint = operation(
        cv.right_face_grad_constraint, scaling_factor)
    return cell_variable.CellVariable(
        value=scaled_value,
        left_face_constraint=scaled_left_face_constraint,
        left_face_grad_constraint=scaled_left_face_grad_constraint,
        right_face_constraint=scaled_right_face_constraint,
        right_face_grad_constraint=scaled_right_face_grad_constraint,
        dr=cv.dr,
    )


OptionalTupleMatrix: TypeAlias = tuple[tuple[jax.Array | None, ...],
                                       ...] | None
AuxiliaryOutput: TypeAlias = Any


def _calculate_psi_value_constraint_from_v_loop(dt, theta, v_loop_lcfs_t,
                                                v_loop_lcfs_t_plus_dt,
                                                psi_lcfs_t):
    theta_weighted_v_loop_lcfs = (
        1 - theta) * v_loop_lcfs_t + theta * v_loop_lcfs_t_plus_dt
    return psi_lcfs_t + theta_weighted_v_loop_lcfs * dt


@jax_utils.jit
def get_prescribed_core_profile_values(runtime_params, geo, core_profiles):
    if not runtime_params.numerics.evolve_ion_heat:
        T_i = get_updated_ion_temperature(runtime_params.profile_conditions,
                                          geo).value
    else:
        T_i = core_profiles.T_i.value
    if not runtime_params.numerics.evolve_electron_heat:
        T_e_cell_variable = get_updated_electron_temperature(
            runtime_params.profile_conditions, geo)
        T_e = T_e_cell_variable.value
    else:
        T_e_cell_variable = core_profiles.T_e
        T_e = T_e_cell_variable.value
    if not runtime_params.numerics.evolve_density:
        n_e_cell_variable = get_updated_electron_density(
            runtime_params.profile_conditions, geo)
    else:
        n_e_cell_variable = core_profiles.n_e
    ions = get_updated_ions(
        runtime_params,
        geo,
        n_e_cell_variable,
        T_e_cell_variable,
    )
    n_e = n_e_cell_variable.value
    n_i = ions.n_i.value
    n_impurity = ions.n_impurity.value
    impurity_fractions = ions.impurity_fractions
    return {
        'T_i': T_i,
        'T_e': T_e,
        'n_e': n_e,
        'n_i': n_i,
        'n_impurity': n_impurity,
        'impurity_fractions': impurity_fractions,
        'Z_i': ions.Z_i,
        'Z_i_face': ions.Z_i_face,
        'Z_impurity': ions.Z_impurity,
        'Z_impurity_face': ions.Z_impurity_face,
        'A_i': ions.A_i,
        'A_impurity': ions.A_impurity,
        'A_impurity_face': ions.A_impurity_face,
        'Z_eff': ions.Z_eff,
        'Z_eff_face': ions.Z_eff_face,
    }


@functools.partial(jax_utils.jit, static_argnames=['evolving_names'])
def update_core_profiles_during_step(x_new, runtime_params, geo, core_profiles,
                                     evolving_names):
    updated_core_profiles = solver_x_tuple_to_core_profiles(
        x_new, evolving_names, core_profiles)
    ions = get_updated_ions(
        runtime_params,
        geo,
        updated_core_profiles.n_e,
        updated_core_profiles.T_e,
    )
    return dataclasses.replace(
        updated_core_profiles,
        n_i=ions.n_i,
        n_impurity=ions.n_impurity,
        impurity_fractions=ions.impurity_fractions,
        Z_i=ions.Z_i,
        Z_i_face=ions.Z_i_face,
        Z_impurity=ions.Z_impurity,
        Z_impurity_face=ions.Z_impurity_face,
        A_i=ions.A_i,
        A_impurity=ions.A_impurity,
        A_impurity_face=ions.A_impurity_face,
        Z_eff=ions.Z_eff,
        Z_eff_face=ions.Z_eff_face,
        q_face=psi_calculations.calc_q_face(geo, updated_core_profiles.psi),
        s_face=psi_calculations.calc_s_face(geo, updated_core_profiles.psi),
    )


def update_core_and_source_profiles_after_step(
        dt, x_new, runtime_params_t_plus_dt, geo, core_profiles_t,
        core_profiles_t_plus_dt, explicit_source_profiles, source_models,
        neoclassical_models, evolving_names):
    updated_core_profiles_t_plus_dt = solver_x_tuple_to_core_profiles(
        x_new, evolving_names, core_profiles_t_plus_dt)
    ions = get_updated_ions(
        runtime_params_t_plus_dt,
        geo,
        updated_core_profiles_t_plus_dt.n_e,
        updated_core_profiles_t_plus_dt.T_e,
    )
    v_loop_lcfs = (runtime_params_t_plus_dt.profile_conditions.v_loop_lcfs
                   if runtime_params_t_plus_dt.profile_conditions.
                   use_v_loop_lcfs_boundary_condition else
                   _update_v_loop_lcfs_from_psi(
                       core_profiles_t.psi,
                       updated_core_profiles_t_plus_dt.psi,
                       dt,
                   ))
    j_total, j_total_face, Ip_profile_face = psi_calculations.calc_j_total(
        geo,
        updated_core_profiles_t_plus_dt.psi,
    )
    intermediate_core_profiles = state.CoreProfiles(
        T_i=updated_core_profiles_t_plus_dt.T_i,
        T_e=updated_core_profiles_t_plus_dt.T_e,
        psi=updated_core_profiles_t_plus_dt.psi,
        n_e=updated_core_profiles_t_plus_dt.n_e,
        n_i=ions.n_i,
        n_impurity=ions.n_impurity,
        impurity_fractions=ions.impurity_fractions,
        Z_i=ions.Z_i,
        Z_i_face=ions.Z_i_face,
        Z_impurity=ions.Z_impurity,
        Z_impurity_face=ions.Z_impurity_face,
        psidot=core_profiles_t_plus_dt.psidot,
        q_face=psi_calculations.calc_q_face(
            geo, updated_core_profiles_t_plus_dt.psi),
        s_face=psi_calculations.calc_s_face(
            geo, updated_core_profiles_t_plus_dt.psi),
        A_i=ions.A_i,
        A_impurity=ions.A_impurity,
        A_impurity_face=ions.A_impurity_face,
        Z_eff=ions.Z_eff,
        Z_eff_face=ions.Z_eff_face,
        v_loop_lcfs=v_loop_lcfs,
        sigma=core_profiles_t_plus_dt.sigma,
        sigma_face=core_profiles_t_plus_dt.sigma_face,
        j_total=j_total,
        j_total_face=j_total_face,
        Ip_profile_face=Ip_profile_face,
    )
    conductivity = neoclassical_models.conductivity.calculate_conductivity(
        geo, intermediate_core_profiles)
    intermediate_core_profiles = dataclasses.replace(
        intermediate_core_profiles,
        sigma=conductivity.sigma,
        sigma_face=conductivity.sigma_face,
    )
    total_source_profiles = build_source_profiles(
        runtime_params=runtime_params_t_plus_dt,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        core_profiles=intermediate_core_profiles,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
        conductivity=conductivity,
    )
    if (not runtime_params_t_plus_dt.numerics.evolve_current and
            runtime_params_t_plus_dt.profile_conditions.psidot is not None):
        psidot_value = (runtime_params_t_plus_dt.profile_conditions.psidot)
    else:
        psi_sources = total_source_profiles.total_psi_sources(geo)
        psidot_value = psi_calculations.calculate_psidot_from_psi_sources(
            psi_sources=psi_sources,
            sigma=intermediate_core_profiles.sigma,
            resistivity_multiplier=runtime_params_t_plus_dt.numerics.
            resistivity_multiplier,
            psi=intermediate_core_profiles.psi,
            geo=geo,
        )
    psidot = dataclasses.replace(
        core_profiles_t_plus_dt.psidot,
        value=psidot_value,
        right_face_constraint=v_loop_lcfs,
        right_face_grad_constraint=None,
    )
    core_profiles_t_plus_dt = dataclasses.replace(
        intermediate_core_profiles,
        psidot=psidot,
    )
    return core_profiles_t_plus_dt, total_source_profiles


def compute_boundary_conditions_for_t_plus_dt(dt, runtime_params_t,
                                              runtime_params_t_plus_dt,
                                              geo_t_plus_dt, core_profiles_t):
    profile_conditions_t_plus_dt = (
        runtime_params_t_plus_dt.profile_conditions)
    n_e = get_updated_electron_density(profile_conditions_t_plus_dt,
                                       geo_t_plus_dt)
    n_e_right_bc = n_e.right_face_constraint
    ions_edge = get_updated_ions(
        runtime_params_t_plus_dt,
        geo_t_plus_dt,
        dataclasses.replace(
            core_profiles_t.n_e,
            right_face_constraint=profile_conditions_t_plus_dt.n_e_right_bc,
        ),
        dataclasses.replace(
            core_profiles_t.T_e,
            right_face_constraint=profile_conditions_t_plus_dt.T_e_right_bc,
        ),
    )
    Z_i_edge = ions_edge.Z_i_face[-1]
    Z_impurity_edge = ions_edge.Z_impurity_face[-1]
    dilution_factor_edge = formulas_ph.calculate_main_ion_dilution_factor(
        Z_i_edge,
        Z_impurity_edge,
        runtime_params_t_plus_dt.plasma_composition.Z_eff_face[-1],
    )
    n_i_bound_right = n_e_right_bc * dilution_factor_edge
    n_impurity_bound_right = (n_e_right_bc -
                              n_i_bound_right * Z_i_edge) / Z_impurity_edge
    return {
        'T_i':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=profile_conditions_t_plus_dt.T_i_right_bc,
        ),
        'T_e':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=profile_conditions_t_plus_dt.T_e_right_bc,
        ),
        'n_e':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(n_e_right_bc),
        ),
        'n_i':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(n_i_bound_right),
        ),
        'n_impurity':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(n_impurity_bound_right),
        ),
        'psi':
        dict(
            right_face_grad_constraint=(
                psi_calculations.calculate_psi_grad_constraint_from_Ip(
                    Ip=profile_conditions_t_plus_dt.Ip,
                    geo=geo_t_plus_dt,
                ) if not runtime_params_t.profile_conditions.
                use_v_loop_lcfs_boundary_condition else None),
            right_face_constraint=(_calculate_psi_value_constraint_from_v_loop(
                dt=dt,
                v_loop_lcfs_t=runtime_params_t.profile_conditions.v_loop_lcfs,
                v_loop_lcfs_t_plus_dt=profile_conditions_t_plus_dt.v_loop_lcfs,
                psi_lcfs_t=core_profiles_t.psi.right_face_constraint,
                theta=runtime_params_t.solver.theta_implicit,
            ) if runtime_params_t.profile_conditions.
                                   use_v_loop_lcfs_boundary_condition else
                                   None),
        ),
        'Z_i_edge':
        Z_i_edge,
        'Z_impurity_edge':
        Z_impurity_edge,
    }


def provide_core_profiles_t_plus_dt(dt, runtime_params_t,
                                    runtime_params_t_plus_dt, geo_t_plus_dt,
                                    core_profiles_t):
    updated_boundary_conditions = compute_boundary_conditions_for_t_plus_dt(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
    )
    updated_values = get_prescribed_core_profile_values(
        runtime_params=runtime_params_t_plus_dt,
        geo=geo_t_plus_dt,
        core_profiles=core_profiles_t,
    )
    T_i = dataclasses.replace(
        core_profiles_t.T_i,
        value=updated_values['T_i'],
        **updated_boundary_conditions['T_i'],
    )
    T_e = dataclasses.replace(
        core_profiles_t.T_e,
        value=updated_values['T_e'],
        **updated_boundary_conditions['T_e'],
    )
    psi = dataclasses.replace(core_profiles_t.psi,
                              **updated_boundary_conditions['psi'])
    n_e = dataclasses.replace(
        core_profiles_t.n_e,
        value=updated_values['n_e'],
        **updated_boundary_conditions['n_e'],
    )
    n_i = dataclasses.replace(
        core_profiles_t.n_i,
        value=updated_values['n_i'],
        **updated_boundary_conditions['n_i'],
    )
    n_impurity = dataclasses.replace(
        core_profiles_t.n_impurity,
        value=updated_values['n_impurity'],
        **updated_boundary_conditions['n_impurity'],
    )
    Z_i_face = jnp.concatenate([
        updated_values['Z_i_face'][:-1],
        jnp.array([updated_boundary_conditions['Z_i_edge']]),
    ], )
    Z_impurity_face = jnp.concatenate([
        updated_values['Z_impurity_face'][:-1],
        jnp.array([updated_boundary_conditions['Z_impurity_edge']]),
    ], )
    core_profiles_t_plus_dt = dataclasses.replace(
        core_profiles_t,
        T_i=T_i,
        T_e=T_e,
        psi=psi,
        n_e=n_e,
        n_i=n_i,
        n_impurity=n_impurity,
        impurity_fractions=updated_values['impurity_fractions'],
        Z_i=updated_values['Z_i'],
        Z_i_face=Z_i_face,
        Z_impurity=updated_values['Z_impurity'],
        Z_impurity_face=Z_impurity_face,
        A_i=updated_values['A_i'],
        A_impurity=updated_values['A_impurity'],
        A_impurity_face=updated_values['A_impurity_face'],
        Z_eff=updated_values['Z_eff'],
        Z_eff_face=updated_values['Z_eff_face'],
    )
    return core_profiles_t_plus_dt


def _update_v_loop_lcfs_from_psi(psi_t, psi_t_plus_dt, dt):
    psi_lcfs_t = psi_t.face_value()[-1]
    psi_lcfs_t_plus_dt = psi_t_plus_dt.face_value()[-1]
    v_loop_lcfs_t_plus_dt = (psi_lcfs_t_plus_dt - psi_lcfs_t) / dt
    return v_loop_lcfs_t_plus_dt


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Block1DCoeffs:
    transient_in_cell: tuple[jax.Array, ...]
    transient_out_cell: tuple[jax.Array, ...] | None = None
    d_face: tuple[jax.Array, ...] | None = None
    v_face: tuple[jax.Array, ...] | None = None
    source_mat_cell: OptionalTupleMatrix = None
    source_cell: tuple[jax.Array | None, ...] | None = None
    auxiliary_outputs: AuxiliaryOutput | None = None


Block1DCoeffs: TypeAlias = Block1DCoeffs
AuxiliaryOutput: TypeAlias = AuxiliaryOutput


def cell_variable_tuple_to_vec(
    x_tuple: tuple[cell_variable.CellVariable, ...], ) -> jax.Array:
    x_vec = jnp.concatenate([x.value for x in x_tuple])
    return x_vec


class CoeffsCallback:

    def __init__(self, physics_models, evolving_names):
        self.physics_models = physics_models
        self.evolving_names = evolving_names

    def __hash__(self) -> int:
        return hash((
            self.physics_models,
            self.evolving_names,
        ))

    def __eq__(self, other: typing_extensions.Self) -> bool:
        return (self.physics_models == other.physics_models
                and self.evolving_names == other.evolving_names)

    def __call__(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
        x: tuple[cell_variable.CellVariable, ...],
        explicit_source_profiles: SourceProfiles,
        allow_pereverzev: bool = False,
        explicit_call: bool = False,
    ):
        core_profiles = update_core_profiles_during_step(
            x,
            runtime_params,
            geo,
            core_profiles,
            self.evolving_names,
        )
        if allow_pereverzev:
            use_pereverzev = runtime_params.solver.use_pereverzev
        else:
            use_pereverzev = False
        return calc_coeffs(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            explicit_source_profiles=explicit_source_profiles,
            physics_models=self.physics_models,
            evolving_names=self.evolving_names,
            use_pereverzev=use_pereverzev,
            explicit_call=explicit_call,
        )


def _calculate_pereverzev_flux(runtime_params, geo, core_profiles,
                               pedestal_model_output):
    consts = constants.CONSTANTS
    geo_factor = jnp.concatenate(
        [jnp.ones(1), geo.g1_over_vpr_face[1:] / geo.g0_face[1:]])
    chi_face_per_ion = (geo.g1_over_vpr_face * core_profiles.n_i.face_value() *
                        consts.keV_to_J * runtime_params.solver.chi_pereverzev)
    chi_face_per_el = (geo.g1_over_vpr_face * core_profiles.n_e.face_value() *
                       consts.keV_to_J * runtime_params.solver.chi_pereverzev)
    d_face_per_el = runtime_params.solver.D_pereverzev
    v_face_per_el = (core_profiles.n_e.face_grad() /
                     core_profiles.n_e.face_value() * d_face_per_el *
                     geo_factor)
    chi_face_per_ion = jnp.where(
        geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
        0.0,
        chi_face_per_ion,
    )
    chi_face_per_el = jnp.where(
        geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
        0.0,
        chi_face_per_el,
    )
    v_heat_face_ion = (core_profiles.T_i.face_grad() /
                       core_profiles.T_i.face_value() * chi_face_per_ion)
    v_heat_face_el = (core_profiles.T_e.face_grad() /
                      core_profiles.T_e.face_value() * chi_face_per_el)
    d_face_per_el = jnp.where(
        geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
        0.0,
        d_face_per_el * geo.g1_over_vpr_face,
    )
    v_face_per_el = jnp.where(
        geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
        0.0,
        v_face_per_el * geo.g0_face,
    )
    chi_face_per_ion = chi_face_per_ion.at[0].set(chi_face_per_ion[1])
    chi_face_per_el = chi_face_per_el.at[0].set(chi_face_per_el[1])
    return (
        chi_face_per_ion,
        chi_face_per_el,
        v_heat_face_ion,
        v_heat_face_el,
        d_face_per_el,
        v_face_per_el,
    )


def calc_coeffs(runtime_params,
                geo,
                core_profiles,
                explicit_source_profiles,
                physics_models,
                evolving_names,
                use_pereverzev=False,
                explicit_call=False):
    if explicit_call and runtime_params.solver.theta_implicit == 1.0:
        return _calc_coeffs_reduced(
            geo,
            core_profiles,
            evolving_names,
        )
    else:
        return _calc_coeffs_full(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            explicit_source_profiles=explicit_source_profiles,
            physics_models=physics_models,
            evolving_names=evolving_names,
            use_pereverzev=use_pereverzev,
        )


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _calc_coeffs_full(runtime_params,
                      geo,
                      core_profiles,
                      explicit_source_profiles,
                      physics_models,
                      evolving_names,
                      use_pereverzev=False):
    consts = constants.CONSTANTS
    pedestal_model_output = physics_models.pedestal_model(
        runtime_params, geo, core_profiles)
    mask = (jnp.zeros_like(
        geo.rho,
        dtype=bool).at[pedestal_model_output.rho_norm_ped_top_idx].set(True))
    conductivity = (
        physics_models.neoclassical_models.conductivity.calculate_conductivity(
            geo, core_profiles))
    merged_source_profiles = build_source_profiles(
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
        conductivity=conductivity,
    )
    source_mat_psi = jnp.zeros_like(geo.rho)
    source_psi = merged_source_profiles.total_psi_sources(geo)
    toc_T_i = 1.5 * geo.vpr**(-2.0 / 3.0) * consts.keV_to_J
    tic_T_i = core_profiles.n_i.value * geo.vpr**(5.0 / 3.0)
    toc_T_e = 1.5 * geo.vpr**(-2.0 / 3.0) * consts.keV_to_J
    tic_T_e = core_profiles.n_e.value * geo.vpr**(5.0 / 3.0)
    toc_psi = (1.0 / runtime_params.numerics.resistivity_multiplier *
               geo.rho_norm * conductivity.sigma * consts.mu_0 * 16 *
               jnp.pi**2 * geo.Phi_b**2 / geo.F**2)
    tic_psi = jnp.ones_like(toc_psi)
    toc_dens_el = jnp.ones_like(geo.vpr)
    tic_dens_el = geo.vpr
    turbulent_transport = physics_models.transport_model(
        runtime_params, geo, core_profiles, pedestal_model_output)
    neoclassical_transport = physics_models.neoclassical_models.transport(
        runtime_params, geo, core_profiles)
    chi_face_ion_total = (turbulent_transport.chi_face_ion +
                          neoclassical_transport.chi_neo_i)
    chi_face_el_total = (turbulent_transport.chi_face_el +
                         neoclassical_transport.chi_neo_e)
    d_face_el_total = (turbulent_transport.d_face_el +
                       neoclassical_transport.D_neo_e)
    v_face_el_total = (turbulent_transport.v_face_el +
                       neoclassical_transport.V_neo_e +
                       neoclassical_transport.V_neo_ware_e)
    d_face_psi = geo.g2g3_over_rhon_face
    v_face_psi = jnp.zeros_like(d_face_psi)
    full_chi_face_ion = (geo.g1_over_vpr_face *
                         core_profiles.n_i.face_value() * consts.keV_to_J *
                         chi_face_ion_total)
    full_chi_face_el = (geo.g1_over_vpr_face * core_profiles.n_e.face_value() *
                        consts.keV_to_J * chi_face_el_total)
    full_d_face_el = geo.g1_over_vpr_face * d_face_el_total
    full_v_face_el = geo.g0_face * v_face_el_total
    source_mat_nn = jnp.zeros_like(geo.rho)
    source_n_e = merged_source_profiles.total_sources('n_e', geo)
    source_n_e += (mask * runtime_params.numerics.adaptive_n_source_prefactor *
                   pedestal_model_output.n_e_ped)
    source_mat_nn += -(mask *
                       runtime_params.numerics.adaptive_n_source_prefactor)
    (
        chi_face_per_ion,
        chi_face_per_el,
        v_heat_face_ion,
        v_heat_face_el,
        d_face_per_el,
        v_face_per_el,
    ) = jax.lax.cond(
        use_pereverzev,
        lambda: _calculate_pereverzev_flux(
            runtime_params,
            geo,
            core_profiles,
            pedestal_model_output,
        ),
        lambda: tuple([jnp.zeros_like(geo.rho_face)] * 6),
    )
    full_chi_face_ion += chi_face_per_ion
    full_chi_face_el += chi_face_per_el
    full_d_face_el += d_face_per_el
    full_v_face_el += v_face_per_el
    v_heat_face_ion += (-3.0 / 4.0 * geo.Phi_b_dot / geo.Phi_b *
                        geo.rho_face_norm * geo.vpr_face *
                        core_profiles.n_i.face_value() * consts.keV_to_J)
    v_heat_face_el += (-3.0 / 4.0 * geo.Phi_b_dot / geo.Phi_b *
                       geo.rho_face_norm * geo.vpr_face *
                       core_profiles.n_e.face_value() * consts.keV_to_J)
    full_v_face_el += (-1.0 / 2.0 * geo.Phi_b_dot / geo.Phi_b *
                       geo.rho_face_norm * geo.vpr_face)
    source_i = merged_source_profiles.total_sources('T_i', geo)
    source_e = merged_source_profiles.total_sources('T_e', geo)
    qei = merged_source_profiles.qei
    source_mat_ii = qei.implicit_ii * geo.vpr
    source_i += qei.explicit_i * geo.vpr
    source_mat_ee = qei.implicit_ee * geo.vpr
    source_e += qei.explicit_e * geo.vpr
    source_mat_ie = qei.implicit_ie * geo.vpr
    source_mat_ei = qei.implicit_ei * geo.vpr
    source_i += (mask * runtime_params.numerics.adaptive_T_source_prefactor *
                 pedestal_model_output.T_i_ped)
    source_e += (mask * runtime_params.numerics.adaptive_T_source_prefactor *
                 pedestal_model_output.T_e_ped)
    source_mat_ii -= mask * runtime_params.numerics.adaptive_T_source_prefactor
    source_mat_ee -= mask * runtime_params.numerics.adaptive_T_source_prefactor
    d_vpr53_rhon_n_e_drhon = jnp.gradient(
        geo.vpr**(5.0 / 3.0) * geo.rho_norm * core_profiles.n_e.value,
        geo.rho_norm,
    )
    d_vpr53_rhon_n_i_drhon = jnp.gradient(
        geo.vpr**(5.0 / 3.0) * geo.rho_norm * core_profiles.n_i.value,
        geo.rho_norm,
    )
    source_i += (3.0 / 4.0 * geo.vpr**(-2.0 / 3.0) * d_vpr53_rhon_n_i_drhon *
                 geo.Phi_b_dot / geo.Phi_b * core_profiles.T_i.value *
                 consts.keV_to_J)
    source_e += (3.0 / 4.0 * geo.vpr**(-2.0 / 3.0) * d_vpr53_rhon_n_e_drhon *
                 geo.Phi_b_dot / geo.Phi_b * core_profiles.T_e.value *
                 consts.keV_to_J)
    d_vpr_rhon_drhon = jnp.gradient(geo.vpr * geo.rho_norm, geo.rho_norm)
    source_n_e += (1.0 / 2.0 * d_vpr_rhon_drhon * geo.Phi_b_dot / geo.Phi_b *
                   core_profiles.n_e.value)
    source_psi += (8.0 * jnp.pi**2 * consts.mu_0 * geo.Phi_b_dot * geo.Phi_b *
                   geo.rho_norm**2 * conductivity.sigma / geo.F**2 *
                   core_profiles.psi.grad())
    var_to_toc = {
        'T_i': toc_T_i,
        'T_e': toc_T_e,
        'psi': toc_psi,
        'n_e': toc_dens_el,
    }
    var_to_tic = {
        'T_i': tic_T_i,
        'T_e': tic_T_e,
        'psi': tic_psi,
        'n_e': tic_dens_el,
    }
    transient_out_cell = tuple(var_to_toc[var] for var in evolving_names)
    transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)
    var_to_d_face = {
        'T_i': full_chi_face_ion,
        'T_e': full_chi_face_el,
        'psi': d_face_psi,
        'n_e': full_d_face_el,
    }
    d_face = tuple(var_to_d_face[var] for var in evolving_names)
    var_to_v_face = {
        'T_i': v_heat_face_ion,
        'T_e': v_heat_face_el,
        'psi': v_face_psi,
        'n_e': full_v_face_el,
    }
    v_face = tuple(var_to_v_face.get(var) for var in evolving_names)
    d = {
        ('T_i', 'T_i'): source_mat_ii,
        ('T_i', 'T_e'): source_mat_ie,
        ('T_e', 'T_i'): source_mat_ei,
        ('T_e', 'T_e'): source_mat_ee,
        ('n_e', 'n_e'): source_mat_nn,
        ('psi', 'psi'): source_mat_psi,
    }
    source_mat_cell = tuple(
        tuple(d.get((row_block, col_block)) for col_block in evolving_names)
        for row_block in evolving_names)
    var_to_source = {
        'T_i': source_i / SCALING_FACTORS['T_i'],
        'T_e': source_e / SCALING_FACTORS['T_e'],
        'psi': source_psi / SCALING_FACTORS['psi'],
        'n_e': source_n_e / SCALING_FACTORS['n_e'],
    }
    source_cell = tuple(var_to_source.get(var) for var in evolving_names)
    coeffs = Block1DCoeffs(
        transient_out_cell=transient_out_cell,
        transient_in_cell=transient_in_cell,
        d_face=d_face,
        v_face=v_face,
        source_mat_cell=source_mat_cell,
        source_cell=source_cell,
        auxiliary_outputs=(
            merged_source_profiles,
            conductivity,
            state.CoreTransport(**dataclasses.asdict(turbulent_transport),
                                **dataclasses.asdict(neoclassical_transport)),
        ),
    )
    return coeffs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _calc_coeffs_reduced(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
):
    tic_T_i = core_profiles.n_i.value * geo.vpr**(5.0 / 3.0)
    tic_T_e = core_profiles.n_e.value * geo.vpr**(5.0 / 3.0)
    tic_psi = jnp.ones_like(geo.vpr)
    tic_dens_el = geo.vpr
    var_to_tic = {
        'T_i': tic_T_i,
        'T_e': tic_T_e,
        'psi': tic_psi,
        'n_e': tic_dens_el,
    }
    transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)
    coeffs = Block1DCoeffs(transient_in_cell=transient_in_cell, )
    return coeffs


def calc_c(
    x: tuple[cell_variable.CellVariable, ...],
    coeffs: Block1DCoeffs,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[jax.Array, jax.Array]:
    d_face = coeffs.d_face
    v_face = coeffs.v_face
    source_mat_cell = coeffs.source_mat_cell
    source_cell = coeffs.source_cell
    num_cells = x[0].value.shape[0]
    num_channels = len(x)
    for _, x_i in enumerate(x):
        if x_i.value.shape != (num_cells, ):
            raise ValueError(
                f'Expected each x channel to have shape ({num_cells},) '
                f'but got {x_i.value.shape}.')
    zero_block = jnp.zeros((num_cells, num_cells))
    zero_row_of_blocks = [zero_block] * num_channels
    zero_vec = jnp.zeros((num_cells))
    zero_block_vec = [zero_vec] * num_channels
    c_mat = [zero_row_of_blocks.copy() for _ in range(num_channels)]
    c = zero_block_vec.copy()
    if d_face is not None:
        for i in range(num_channels):
            (
                diffusion_mat,
                diffusion_vec,
            ) = diffusion_terms.make_diffusion_terms(
                d_face[i],
                x[i],
            )
            c_mat[i][i] += diffusion_mat
            c[i] += diffusion_vec
    if v_face is not None:
        for i in range(num_channels):
            d_face_i = d_face[i] if d_face is not None else None
            d_face_i = jnp.zeros_like(
                v_face[i]) if d_face_i is None else d_face_i
            (
                conv_mat,
                conv_vec,
            ) = convection_terms.make_convection_terms(
                v_face[i],
                d_face_i,
                x[i],
                dirichlet_mode=convection_dirichlet_mode,
                neumann_mode=convection_neumann_mode,
            )
            c_mat[i][i] += conv_mat
            c[i] += conv_vec
    if source_mat_cell is not None:
        for i in range(num_channels):
            for j in range(num_channels):
                source = source_mat_cell[i][j]
                if source is not None:
                    c_mat[i][j] += jnp.diag(source)

    def add(left: jax.Array, right: jax.Array | None):
        if right is not None:
            return left + right
        return left

    if source_cell is not None:
        c = [add(c_i, source_i) for c_i, source_i in zip(c, source_cell)]
    c_mat = jnp.block(c_mat)
    c = jnp.block(c)
    return c_mat, c


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'convection_dirichlet_mode',
        'convection_neumann_mode',
        'theta_implicit',
    ],
)
def theta_method_matrix_equation(
    dt: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    coeffs_old: Block1DCoeffs,
    coeffs_new: Block1DCoeffs,
    theta_implicit: float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
):
    x_new_guess_vec = cell_variable_tuple_to_vec(x_new_guess)
    theta_exp = 1.0 - theta_implicit
    tc_in_old = jnp.concatenate(coeffs_old.transient_in_cell)
    tc_out_new = jnp.concatenate(coeffs_new.transient_out_cell)
    tc_in_new = jnp.concatenate(coeffs_new.transient_in_cell)
    eps = 1e-7
    left_transient = jnp.identity(len(x_new_guess_vec))
    right_transient = jnp.diag(jnp.squeeze(tc_in_old / tc_in_new))
    c_mat_new, c_new = calc_c(
        x_new_guess,
        coeffs_new,
        convection_dirichlet_mode,
        convection_neumann_mode,
    )
    broadcasted = jnp.expand_dims(1 / (tc_out_new * tc_in_new), 1)
    lhs_mat = left_transient - dt * theta_implicit * broadcasted * c_mat_new
    lhs_vec = -theta_implicit * dt * (1 / (tc_out_new * tc_in_new)) * c_new
    if theta_exp > 0.0:
        tc_out_old = jnp.concatenate(coeffs_old.transient_out_cell)
        tc_in_new = jax_utils.error_if(
            tc_in_new,
            jnp.any(jnp.abs(tc_out_old * tc_in_new) < eps),
            msg='|tc_out_old*tc_in_new| unexpectedly < eps',
        )
        c_mat_old, c_old = discrete_system.calc_c(
            x_old,
            coeffs_old,
            convection_dirichlet_mode,
            convection_neumann_mode,
        )
        broadcasted = jnp.expand_dims(1 / (tc_out_old * tc_in_new), 1)
        rhs_mat = right_transient + dt * theta_exp * broadcasted * c_mat_old
        rhs_vec = dt * theta_exp * (1 / (tc_out_old * tc_in_new)) * c_old
    else:
        rhs_mat = right_transient
        rhs_vec = jnp.zeros_like(x_new_guess_vec)
    return lhs_mat, lhs_vec, rhs_mat, rhs_vec


MIN_DELTA: Final[float] = 1e-7


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'convection_dirichlet_mode',
        'convection_neumann_mode',
        'theta_implicit',
    ],
)
def implicit_solve_block(
    dt: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    coeffs_old,
    coeffs_new,
    theta_implicit: float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[cell_variable.CellVariable, ...]:
    x_old_vec = cell_variable_tuple_to_vec(x_old)
    lhs_mat, lhs_vec, rhs_mat, rhs_vec = (theta_method_matrix_equation(
        dt=dt,
        x_old=x_old,
        x_new_guess=x_new_guess,
        coeffs_old=coeffs_old,
        coeffs_new=coeffs_new,
        theta_implicit=theta_implicit,
        convection_dirichlet_mode=convection_dirichlet_mode,
        convection_neumann_mode=convection_neumann_mode,
    ))
    rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec - lhs_vec
    x_new = jnp.linalg.solve(lhs_mat, rhs)
    x_new = jnp.split(x_new, len(x_old))
    out = [
        dataclasses.replace(var, value=value)
        for var, value in zip(x_new_guess, x_new)
    ]
    out = tuple(out)
    return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    theta_implicit: float = dataclasses.field(metadata={'static': True})
    use_predictor_corrector: bool = dataclasses.field(
        metadata={'static': True})
    n_corrector_steps: int = dataclasses.field(metadata={'static': True})
    convection_dirichlet_mode: str = dataclasses.field(
        metadata={'static': True})
    convection_neumann_mode: str = dataclasses.field(metadata={'static': True})
    use_pereverzev: bool = dataclasses.field(metadata={'static': True})
    chi_pereverzev: float
    D_pereverzev: float


class Solver(abc.ABC):

    def __init__(self, physics_models):
        self.physics_models = physics_models

    def __hash__(self) -> int:
        return hash(self.physics_models)

    def __eq__(self, other: typing_extensions.Self) -> bool:
        return self.physics_models == other.physics_models

    @functools.partial(
        jax_utils.jit,
        static_argnames=[
            'self',
        ],
    )
    def __call__(
        self,
        t: jax.Array,
        dt: jax.Array,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        geo_t_plus_dt: geometry.Geometry,
        core_profiles_t: state.CoreProfiles,
        core_profiles_t_plus_dt: state.CoreProfiles,
        explicit_source_profiles: SourceProfiles,
    ) -> tuple[
            tuple[cell_variable.CellVariable, ...],
            state.SolverNumericOutputs,
    ]:
        if runtime_params_t.numerics.evolving_names:
            (
                x_new,
                solver_numeric_output,
            ) = self._x_new(
                dt=dt,
                runtime_params_t=runtime_params_t,
                runtime_params_t_plus_dt=runtime_params_t_plus_dt,
                geo_t=geo_t,
                geo_t_plus_dt=geo_t_plus_dt,
                core_profiles_t=core_profiles_t,
                core_profiles_t_plus_dt=core_profiles_t_plus_dt,
                explicit_source_profiles=explicit_source_profiles,
                evolving_names=runtime_params_t.numerics.evolving_names,
            )
        else:
            x_new = tuple()
            solver_numeric_output = state.SolverNumericOutputs()
        return (
            x_new,
            solver_numeric_output,
        )

    def _x_new(
        self,
        dt: jax.Array,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        geo_t_plus_dt: geometry.Geometry,
        core_profiles_t: state.CoreProfiles,
        core_profiles_t_plus_dt: state.CoreProfiles,
        explicit_source_profiles: SourceProfiles,
        evolving_names: tuple[str, ...],
    ) -> tuple[
            tuple[cell_variable.CellVariable, ...],
            state.SolverNumericOutputs,
    ]:
        raise NotImplementedError(
            f'{type(self)} must implement `_x_new` or '
            'implement a different `__call__` that does not'
            ' need `_x_new`.')


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'coeffs_callback',
    ],
)
def predictor_corrector_method(
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    coeffs_exp,
    explicit_source_profiles: SourceProfiles,
    coeffs_callback: CoeffsCallback,
) -> tuple[cell_variable.CellVariable, ...]:
    solver_params = runtime_params_t_plus_dt.solver

    def loop_body(i, x_new_guess):
        coeffs_new = coeffs_callback(
            runtime_params_t_plus_dt,
            geo_t_plus_dt,
            core_profiles_t_plus_dt,
            x_new_guess,
            explicit_source_profiles=explicit_source_profiles,
            allow_pereverzev=True,
        )
        return implicit_solve_block(
            dt=dt,
            x_old=x_old,
            x_new_guess=x_new_guess,
            coeffs_old=coeffs_exp,
            coeffs_new=coeffs_new,
            theta_implicit=solver_params.theta_implicit,
            convection_dirichlet_mode=(
                solver_params.convection_dirichlet_mode),
            convection_neumann_mode=(solver_params.convection_neumann_mode),
        )

    if solver_params.use_predictor_corrector:
        x_new = xnp.fori_loop(
            0,
            runtime_params_t_plus_dt.solver.n_corrector_steps + 1,
            loop_body,
            x_new_guess,
        )
    else:
        x_new = loop_body(0, x_new_guess)
    return x_new


class LinearThetaMethod0(Solver):

    @functools.partial(
        jax_utils.jit,
        static_argnames=[
            'self',
            'evolving_names',
        ],
    )
    def _x_new(
        self,
        dt: jax.Array,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        geo_t_plus_dt: geometry.Geometry,
        core_profiles_t: state.CoreProfiles,
        core_profiles_t_plus_dt: state.CoreProfiles,
        explicit_source_profiles: SourceProfiles,
        evolving_names: tuple[str, ...],
    ) -> tuple[
            tuple[cell_variable.CellVariable, ...],
            state.SolverNumericOutputs,
    ]:
        x_old = core_profiles_to_solver_x_tuple(core_profiles_t,
                                                evolving_names)
        x_new_guess = core_profiles_to_solver_x_tuple(core_profiles_t_plus_dt,
                                                      evolving_names)
        coeffs_callback = CoeffsCallback(
            physics_models=self.physics_models,
            evolving_names=evolving_names,
        )
        coeffs_exp = coeffs_callback(
            runtime_params_t,
            geo_t,
            core_profiles_t,
            x_old,
            explicit_source_profiles=explicit_source_profiles,
            allow_pereverzev=True,
            explicit_call=True,
        )
        x_new = predictor_corrector_method(
            dt=dt,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo_t_plus_dt=geo_t_plus_dt,
            x_old=x_old,
            x_new_guess=x_new_guess,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            coeffs_exp=coeffs_exp,
            coeffs_callback=coeffs_callback,
            explicit_source_profiles=explicit_source_profiles,
        )
        if runtime_params_t_plus_dt.solver.use_predictor_corrector:
            inner_solver_iterations = (
                1 + runtime_params_t_plus_dt.solver.n_corrector_steps)
        else:
            inner_solver_iterations = 1
        solver_numeric_outputs = state.SolverNumericOutputs(
            inner_solver_iterations=inner_solver_iterations,
            outer_solver_iterations=1,
            solver_error_state=0,
        )
        return (
            x_new,
            solver_numeric_outputs,
        )


class BaseSolver(torax_pydantic.BaseModelFrozen, abc.ABC):
    theta_implicit: Annotated[torax_pydantic.UnitInterval,
                              torax_pydantic.JAX_STATIC] = 1.0
    use_predictor_corrector: Annotated[bool, torax_pydantic.JAX_STATIC] = False
    n_corrector_steps: Annotated[pydantic.PositiveInt,
                                 torax_pydantic.JAX_STATIC] = 10
    convection_dirichlet_mode: Annotated[Literal['ghost', 'direct',
                                                 'semi-implicit'],
                                         torax_pydantic.JAX_STATIC] = 'ghost'
    convection_neumann_mode: Annotated[Literal['ghost', 'semi-implicit'],
                                       torax_pydantic.JAX_STATIC] = 'ghost'
    use_pereverzev: Annotated[bool, torax_pydantic.JAX_STATIC] = False
    chi_pereverzev: pydantic.PositiveFloat = 30.0
    D_pereverzev: pydantic.NonNegativeFloat = 15.0

    @property
    @abc.abstractmethod
    def build_runtime_params(self):
        pass

    @abc.abstractmethod
    def build_solver(self, physics_models):
        pass


class LinearThetaMethod(BaseSolver):
    solver_type: Annotated[Literal['linear'],
                           torax_pydantic.JAX_STATIC] = ('linear')

    @pydantic.model_validator(mode='before')
    @classmethod
    def scrub_log_iterations(cls, x: dict[str, Any]) -> dict[str, Any]:
        if 'log_iterations' in x:
            del x['log_iterations']
        return x

    @functools.cached_property
    def build_runtime_params(self):
        return RuntimeParams(
            theta_implicit=self.theta_implicit,
            convection_dirichlet_mode=self.convection_dirichlet_mode,
            convection_neumann_mode=self.convection_neumann_mode,
            use_pereverzev=self.use_pereverzev,
            use_predictor_corrector=self.use_predictor_corrector,
            chi_pereverzev=self.chi_pereverzev,
            D_pereverzev=self.D_pereverzev,
            n_corrector_steps=self.n_corrector_steps,
        )

    def build_solver(self, physics_models):
        return LinearThetaMethod0(physics_models=physics_models, )


class NewtonRaphsonThetaMethod(BaseSolver):
    solver_type: Annotated[Literal['newton_raphson'],
                           torax_pydantic.JAX_STATIC] = 'newton_raphson'
    log_iterations: Annotated[bool, torax_pydantic.JAX_STATIC] = False
    initial_guess_mode: Annotated[
        enums.InitialGuessMode,
        torax_pydantic.JAX_STATIC] = enums.InitialGuessMode.LINEAR
    n_max_iterations: pydantic.NonNegativeInt = 30
    residual_tol: float = 1e-5
    residual_coarse_tol: float = 1e-2
    delta_reduction_factor: float = 0.5
    tau_min: float = 0.01


SolverConfig = (LinearThetaMethod | NewtonRaphsonThetaMethod)


class g:
    pass


def not_done(t, t_final):
    return t < (t_final - g.tolerance)


def next_dt(t, runtime_params, geo, core_profiles, core_transport):
    chi_max = core_transport.chi_max(geo)
    basic_dt = (3.0 / 4.0) * (geo.drho_norm**2) / chi_max
    dt = jnp.minimum(
        runtime_params.numerics.chi_timestep_prefactor * basic_dt,
        runtime_params.numerics.max_dt,
    )
    crosses_t_final = (t < runtime_params.numerics.t_final) * (
        t + dt > runtime_params.numerics.t_final)
    dt = jax.lax.select(
        jnp.logical_and(
            runtime_params.numerics.exact_t_final,
            crosses_t_final,
        ),
        runtime_params.numerics.t_final - t,
        dt,
    )
    return dt


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsProvider:
    sources: Any
    numerics: Any
    profile_conditions: Any
    plasma_composition: Any
    transport_model: Any
    solver: Any
    pedestal: Any
    neoclassical: Any

    @classmethod
    def from_config(cls, config):
        return cls(
            sources=config.sources,
            numerics=config.numerics,
            profile_conditions=config.profile_conditions,
            plasma_composition=config.plasma_composition,
            transport_model=config.transport,
            solver=config.solver,
            pedestal=config.pedestal,
            neoclassical=config.neoclassical,
        )

    @jax_utils.jit
    def __call__(
        self,
        t: chex.Numeric,
    ) -> runtime_params_slice.RuntimeParams:
        return runtime_params_slice.RuntimeParams(
            transport=self.transport_model.build_runtime_params(t),
            solver=self.solver.build_runtime_params,
            sources={
                source_name: source_config.build_runtime_params(t)
                for source_name, source_config in dict(self.sources).items()
                if source_config is not None
            },
            plasma_composition=self.plasma_composition.build_runtime_params(t),
            profile_conditions=self.profile_conditions.build_runtime_params(t),
            numerics=self.numerics.build_runtime_params(t),
            neoclassical=self.neoclassical.build_runtime_params(),
            pedestal=self.pedestal.build_runtime_params(t),
        )


def get_consistent_runtime_params_and_geometry(*, t, runtime_params_provider,
                                               geometry_provider):
    geo = geometry_provider(t)
    runtime_params = runtime_params_provider(t=t)
    return runtime_params_slice.make_ip_consistent(runtime_params, geo)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PhysicsModels:
    source_models: SourceModels = dataclasses.field(metadata=dict(static=True))
    transport_model: TransportModel = dataclasses.field(metadata=dict(
        static=True))
    pedestal_model: PedestalModel = dataclasses.field(metadata=dict(
        static=True))
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels = (
        dataclasses.field(metadata=dict(static=True)))


StaticKwargs: TypeAlias = dict[str, Any]
DynamicArgs: TypeAlias = list[Any]
PROFILES = "profiles"
SCALARS = "scalars"
NUMERICS = "numerics"
T_E = "T_e"
T_I = "T_i"
PSI = "psi"
V_LOOP = "v_loop"
N_E = "n_e"
N_I = "n_i"
Q = "q"
MAGNETIC_SHEAR = "magnetic_shear"
N_IMPURITY = "n_impurity"
Z_IMPURITY = "Z_impurity"
Z_EFF = "Z_eff"
SIGMA_PARALLEL = "sigma_parallel"
V_LOOP_LCFS = "v_loop_lcfs"
J_TOTAL = "j_total"
IP_PROFILE = "Ip_profile"
IP = "Ip"
J_OHMIC = "j_ohmic"
J_EXTERNAL = "j_external"
J_BOOTSTRAP = "j_bootstrap"
I_BOOTSTRAP = "I_bootstrap"
CHI_TURB_I = "chi_turb_i"
CHI_TURB_E = "chi_turb_e"
D_TURB_E = "D_turb_e"
V_TURB_E = "V_turb_e"
CHI_NEO_I = "chi_neo_i"
CHI_NEO_E = "chi_neo_e"
D_NEO_E = "D_neo_e"
V_NEO_E = "V_neo_e"
V_NEO_WARE_E = "V_neo_ware_e"
CHI_BOHM_E = "chi_bohm_e"
CHI_GYROBOHM_E = "chi_gyrobohm_e"
CHI_BOHM_I = "chi_bohm_i"
CHI_GYROBOHM_I = "chi_gyrobohm_i"
RHO_FACE_NORM = "rho_face_norm"
RHO_CELL_NORM = "rho_cell_norm"
RHO_NORM = "rho_norm"
RHO_FACE = "rho_face"
RHO_CELL = "rho_cell"
TIME = "time"
Q_FUSION = "Q_fusion"
SIM_ERROR = "sim_error"
OUTER_SOLVER_ITERATIONS = "outer_solver_iterations"
INNER_SOLVER_ITERATIONS = "inner_solver_iterations"
SAWTOOTH_CRASH = "sawtooth_crash"
EXCLUDED_GEOMETRY_NAMES = frozenset({
    RHO_FACE,
    RHO_CELL,
    RHO_CELL_NORM,
    RHO_FACE_NORM,
    "rho",
    "rho_norm",
    "q_correction_factor",
})


def _extend_cell_grid_to_boundaries(
    cell_var: array_typing.FloatVectorCell,
    face_var: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorCellPlusBoundaries:
    left_value = np.expand_dims(face_var[:, 0], axis=-1)
    right_value = np.expand_dims(face_var[:, -1], axis=-1)
    return np.concatenate([left_value, cell_var, right_value], axis=-1)


class StateHistory:

    def __init__(self, state_history, post_processed_outputs_history,
                 torax_config):
        state_history[0].core_profiles = dataclasses.replace(
            state_history[0].core_profiles,
            v_loop_lcfs=state_history[1].core_profiles.v_loop_lcfs,
        )
        self._torax_config = torax_config
        self._post_processed_outputs = post_processed_outputs_history
        self._solver_numeric_outputs = [
            state.solver_numeric_outputs for state in state_history
        ]
        self._core_profiles = [state.core_profiles for state in state_history]
        self._core_sources = [state.core_sources for state in state_history]
        self._transport = [state.core_transport for state in state_history]
        self._geometries = [state.geometry for state in state_history]
        self._stacked_geometry = geometry_lib.stack_geometries(self.geometries)
        stack = lambda *ys: np.stack(ys)
        self._stacked_core_profiles: state.CoreProfiles = jax.tree_util.tree_map(
            stack, *self._core_profiles)
        self._stacked_core_sources: SourceProfiles = (jax.tree_util.tree_map(
            stack, *self._core_sources))
        self._stacked_core_transport: state.CoreTransport = jax.tree_util.tree_map(
            stack, *self._transport)
        self._stacked_post_processed_outputs: (
            PostProcessedOutputs) = jax.tree_util.tree_map(
                stack, *post_processed_outputs_history)
        self._stacked_solver_numeric_outputs: state.SolverNumericOutputs = (
            jax.tree_util.tree_map(stack, *self._solver_numeric_outputs))
        self._times = np.array([state.t for state in state_history])
        self._rho_cell_norm = state_history[0].geometry.rho_norm
        self._rho_face_norm = state_history[0].geometry.rho_face_norm
        self._rho_norm = np.concatenate([[0.0], self.rho_cell_norm, [1.0]])

    @property
    def torax_config(self):
        return self._torax_config

    @property
    def times(self) -> array_typing.Array:
        return self._times

    @property
    def rho_cell_norm(self) -> array_typing.FloatVectorCell:
        return self._rho_cell_norm

    @property
    def rho_face_norm(self) -> array_typing.FloatVectorFace:
        return self._rho_face_norm

    @property
    def rho_norm(self) -> array_typing.FloatVectorCellPlusBoundaries:
        return self._rho_norm

    @property
    def geometries(self) -> Sequence[geometry_lib.Geometry]:
        return self._geometries

    def simulation_output_to_xr(self) -> xr.DataTree:
        time = xr.DataArray(self.times, dims=[TIME], name=TIME)
        rho_face_norm = xr.DataArray(self.rho_face_norm,
                                     dims=[RHO_FACE_NORM],
                                     name=RHO_FACE_NORM)
        rho_cell_norm = xr.DataArray(self.rho_cell_norm,
                                     dims=[RHO_CELL_NORM],
                                     name=RHO_CELL_NORM)
        rho_norm = xr.DataArray(
            self.rho_norm,
            dims=[RHO_NORM],
            name=RHO_NORM,
        )
        coords = {
            TIME: time,
            RHO_FACE_NORM: rho_face_norm,
            RHO_CELL_NORM: rho_cell_norm,
            RHO_NORM: rho_norm,
        }
        all_dicts = [
            self._save_core_profiles(),
            self._save_core_transport(),
            self._save_core_sources(),
            self._save_post_processed_outputs(),
            self._save_geometry(),
        ]
        flat_dict = {}
        for key, value in itertools.chain(*(d.items() for d in all_dicts)):
            if key not in flat_dict:
                flat_dict[key] = value
            else:
                raise ValueError(f"Duplicate key: {key}")
        numerics_dict = {
            SAWTOOTH_CRASH:
            xr.DataArray(
                self._stacked_solver_numeric_outputs.sawtooth_crash,
                dims=[TIME],
                name=SAWTOOTH_CRASH,
            ),
            OUTER_SOLVER_ITERATIONS:
            xr.DataArray(
                self._stacked_solver_numeric_outputs.outer_solver_iterations,
                dims=[TIME],
                name=OUTER_SOLVER_ITERATIONS,
            ),
            INNER_SOLVER_ITERATIONS:
            xr.DataArray(
                self._stacked_solver_numeric_outputs.inner_solver_iterations,
                dims=[TIME],
                name=INNER_SOLVER_ITERATIONS,
            ),
        }
        numerics = xr.Dataset(numerics_dict)
        profiles_dict = {
            k: v
            for k, v in flat_dict.items()
            if v is not None and v.values.ndim > 1
        }
        profiles = xr.Dataset(profiles_dict)
        scalars_dict = {
            k: v
            for k, v in flat_dict.items()
            if v is not None and v.values.ndim in [0, 1]
        }
        scalars = xr.Dataset(scalars_dict)
        data_tree = xr.DataTree(
            children={
                NUMERICS: xr.DataTree(dataset=numerics),
                PROFILES: xr.DataTree(dataset=profiles),
                SCALARS: xr.DataTree(dataset=scalars),
            },
            dataset=xr.Dataset(
                data_vars=None,
                coords=coords,
            ),
        )
        return data_tree

    def _pack_into_data_array(self, name, data):
        if data is None:
            return None
        is_face_var = lambda x: x.ndim == 2 and x.shape == (
            len(self.times),
            len(self.rho_face_norm),
        )
        is_cell_var = lambda x: x.ndim == 2 and x.shape == (
            len(self.times),
            len(self.rho_cell_norm),
        )
        is_cell_plus_boundaries_var = lambda x: x.ndim == 2 and x.shape == (
            len(self.times),
            len(self.rho_norm),
        )
        is_scalar = lambda x: x.ndim == 1 and x.shape == (len(self.times), )
        is_constant = lambda x: x.ndim == 0
        match data:
            case data if is_face_var(data):
                dims = [TIME, RHO_FACE_NORM]
            case data if is_cell_var(data):
                dims = [TIME, RHO_CELL_NORM]
            case data if is_scalar(data):
                dims = [TIME]
            case data if is_constant(data):
                dims = []
            case data if is_cell_plus_boundaries_var(data):
                dims = [TIME, RHO_NORM]
            case _:
                logging.warning(
                    "Unsupported data shape for %s: %s. Skipping persisting.",
                    name,
                    data.shape,
                )
                return None
        return xr.DataArray(data, dims=dims, name=name)

    def _save_core_profiles(self):
        xr_dict = {}
        stacked_core_profiles = self._stacked_core_profiles
        output_name_map = {
            "psidot": V_LOOP,
            "sigma": SIGMA_PARALLEL,
            "Ip_profile_face": IP_PROFILE,
            "q_face": Q,
            "s_face": MAGNETIC_SHEAR,
        }
        core_profile_field_names = {
            f.name
            for f in dataclasses.fields(stacked_core_profiles)
        }
        for field in dataclasses.fields(stacked_core_profiles):
            attr_name = field.name
            if attr_name == "impurity_fractions":
                continue
            attr_value = getattr(stacked_core_profiles, attr_name)
            output_key = output_name_map.get(attr_name, attr_name)
            if attr_name.endswith("_face") and (attr_name.removesuffix("_face")
                                                in core_profile_field_names):
                continue
            if attr_name == "A_impurity":
                is_constant = np.all(attr_value == attr_value[..., 0:1],
                                     axis=-1)
                if np.all(is_constant):
                    data_to_save = attr_value[..., 0]
                else:
                    face_value = getattr(stacked_core_profiles,
                                         "A_impurity_face")
                    data_to_save = _extend_cell_grid_to_boundaries(
                        attr_value, face_value)
                xr_dict[output_key] = self._pack_into_data_array(
                    output_key, data_to_save)
                continue
            if hasattr(attr_value, "cell_plus_boundaries"):
                data_to_save = attr_value.cell_plus_boundaries()
            else:
                face_attr_name = f"{attr_name}_face"
                if face_attr_name in core_profile_field_names:
                    face_value = getattr(stacked_core_profiles, face_attr_name)
                    data_to_save = _extend_cell_grid_to_boundaries(
                        attr_value, face_value)
                else:
                    data_to_save = attr_value
            xr_dict[output_key] = self._pack_into_data_array(
                output_key, data_to_save)
        Ip_data = stacked_core_profiles.Ip_profile_face[..., -1]
        xr_dict[IP] = self._pack_into_data_array(IP, Ip_data)
        return xr_dict

    def _save_core_transport(self):
        xr_dict = {}
        core_transport = self._stacked_core_transport
        xr_dict[CHI_TURB_I] = core_transport.chi_face_ion
        xr_dict[CHI_TURB_E] = core_transport.chi_face_el
        xr_dict[D_TURB_E] = core_transport.d_face_el
        xr_dict[V_TURB_E] = core_transport.v_face_el
        xr_dict[CHI_NEO_I] = core_transport.chi_neo_i
        xr_dict[CHI_NEO_E] = core_transport.chi_neo_e
        xr_dict[D_NEO_E] = core_transport.D_neo_e
        xr_dict[V_NEO_E] = core_transport.V_neo_e
        xr_dict[V_NEO_WARE_E] = core_transport.V_neo_ware_e
        xr_dict = {
            name: self._pack_into_data_array(
                name,
                data,
            )
            for name, data in xr_dict.items()
        }
        return xr_dict

    def _save_core_sources(self):
        xr_dict = {}
        xr_dict[QeiSource.SOURCE_NAME] = (
            self._stacked_core_sources.qei.qei_coef *
            (self._stacked_core_profiles.T_e.value -
             self._stacked_core_profiles.T_i.value))
        xr_dict[J_BOOTSTRAP] = _extend_cell_grid_to_boundaries(
            self._stacked_core_sources.bootstrap_current.j_bootstrap,
            self._stacked_core_sources.bootstrap_current.j_bootstrap_face,
        )
        for profile in self._stacked_core_sources.T_i:
            if profile == "fusion":
                xr_dict["p_alpha_i"] = self._stacked_core_sources.T_i[profile]
            else:
                xr_dict[f"p_{profile}_i"] = self._stacked_core_sources.T_i[
                    profile]
        for profile in self._stacked_core_sources.T_e:
            if profile == "fusion":
                xr_dict["p_alpha_e"] = self._stacked_core_sources.T_e[profile]
            else:
                xr_dict[f"p_{profile}_e"] = self._stacked_core_sources.T_e[
                    profile]
        for profile in self._stacked_core_sources.psi:
            xr_dict[f"j_{profile}"] = self._stacked_core_sources.psi[profile]
        for profile in self._stacked_core_sources.n_e:
            xr_dict[f"s_{profile}"] = self._stacked_core_sources.n_e[profile]
        xr_dict = {
            name: self._pack_into_data_array(name, data)
            for name, data in xr_dict.items()
        }
        return xr_dict

    def _save_post_processed_outputs(self):
        xr_dict = {}
        for field in dataclasses.fields(self._stacked_post_processed_outputs):
            attr_name = field.name
            if attr_name == "impurity_species":
                continue
            attr_value = getattr(self._stacked_post_processed_outputs,
                                 attr_name)
            if hasattr(attr_value, "cell_plus_boundaries"):
                data_to_save = attr_value.cell_plus_boundaries()
            else:
                data_to_save = attr_value
            xr_dict[attr_name] = self._pack_into_data_array(
                attr_name, data_to_save)
        if self._stacked_post_processed_outputs.impurity_species:
            radiation_outputs = (construct_xarray_for_radiation_output(
                self._stacked_post_processed_outputs.impurity_species,
                self.times,
                self.rho_cell_norm,
                TIME,
                RHO_CELL_NORM,
            ))
            for key, value in radiation_outputs.items():
                xr_dict[key] = value
        return xr_dict

    def _save_geometry(self):
        xr_dict = {}
        geometry_attributes = dataclasses.asdict(self._stacked_geometry)
        for field_name, data in geometry_attributes.items():
            if ("hires" in field_name or
                (field_name.endswith("_face")
                 and field_name.removesuffix("_face") in geometry_attributes)
                    or field_name == "geometry_type"
                    or field_name == "Ip_from_parameters"
                    or field_name == "j_total"
                    or not isinstance(data, array_typing.Array)):
                continue
            if f"{field_name}_face" in geometry_attributes:
                data = _extend_cell_grid_to_boundaries(
                    data, geometry_attributes[f"{field_name}_face"])
            if field_name.endswith("_face"):
                field_name = field_name.removesuffix("_face")
            if field_name == "Ip_profile":
                field_name = "Ip_profile_from_geo"
            if field_name == "psi":
                field_name = "psi_from_geo"
            if field_name == "_z_magnetic_axis":
                field_name = "z_magnetic_axis"
            data_array = self._pack_into_data_array(
                field_name,
                data,
            )
            if data_array is not None:
                xr_dict[field_name] = data_array
        geometry_properties = inspect.getmembers(type(self._stacked_geometry))
        property_names = set([name for name, _ in geometry_properties])
        for name, value in geometry_properties:
            if (name.endswith("_face")
                    and name.removesuffix("_face") in property_names):
                continue
            if name in EXCLUDED_GEOMETRY_NAMES:
                continue
            if isinstance(value, property):
                property_data = value.fget(self._stacked_geometry)
                if f"{name}_face" in property_names:
                    face_data = getattr(self._stacked_geometry, f"{name}_face")
                    property_data = _extend_cell_grid_to_boundaries(
                        property_data, face_data)
                data_array = self._pack_into_data_array(name, property_data)
                if data_array is not None:
                    if name.endswith("_face"):
                        name = name.removesuffix("_face")
                    xr_dict[name] = data_array
        return xr_dict


@enum.unique
class GeometryType(enum.IntEnum):
    CIRCULAR = 0
    CHEASE = 1
    FBT = 2
    EQDSK = 3
    IMAS = 4


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Geometry:
    geometry_type: GeometryType
    torax_mesh: Any
    Phi: array_typing.Array
    Phi_face: array_typing.Array
    R_major: array_typing.FloatScalar
    a_minor: array_typing.FloatScalar
    B_0: array_typing.FloatScalar
    volume: array_typing.Array
    volume_face: array_typing.Array
    area: array_typing.Array
    area_face: array_typing.Array
    vpr: array_typing.Array
    vpr_face: array_typing.Array
    spr: array_typing.Array
    spr_face: array_typing.Array
    delta_face: array_typing.Array
    elongation: array_typing.Array
    elongation_face: array_typing.Array
    g0: array_typing.Array
    g0_face: array_typing.Array
    g1: array_typing.Array
    g1_face: array_typing.Array
    g2: array_typing.Array
    g2_face: array_typing.Array
    g3: array_typing.Array
    g3_face: array_typing.Array
    gm4: array_typing.Array
    gm4_face: array_typing.Array
    gm5: array_typing.Array
    gm5_face: array_typing.Array
    g2g3_over_rhon: array_typing.Array
    g2g3_over_rhon_face: array_typing.Array
    g2g3_over_rhon_hires: array_typing.Array
    F: array_typing.Array
    F_face: array_typing.Array
    F_hires: array_typing.Array
    R_in: array_typing.Array
    R_in_face: array_typing.Array
    R_out: array_typing.Array
    R_out_face: array_typing.Array
    spr_hires: array_typing.Array
    rho_hires_norm: array_typing.Array
    rho_hires: array_typing.Array
    Phi_b_dot: array_typing.FloatScalar
    _z_magnetic_axis: array_typing.FloatScalar | None


def update_geometries_with_Phibdot(*, dt, geo_t, geo_t_plus_dt):
    Phibdot = (geo_t_plus_dt.Phi_b - geo_t.Phi_b) / dt
    geo_t = dataclasses.replace(geo_t, Phi_b_dot=Phibdot)
    geo_t_plus_dt = dataclasses.replace(geo_t_plus_dt, Phi_b_dot=Phibdot)
    return geo_t, geo_t_plus_dt


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ToraxSimState:
    t: array_typing.FloatScalar
    dt: array_typing.FloatScalar
    core_profiles: state.CoreProfiles
    core_transport: state.CoreTransport
    core_sources: SourceProfiles
    geometry: Any
    solver_numeric_outputs: state.SolverNumericOutputs

    def has_nan(self) -> bool:
        return any([np.any(np.isnan(x)) for x in jax.tree.leaves(self)])


def _get_initial_state(runtime_params, geo, step_fn):
    physics_models = g.solver.physics_models
    initial_core_profiles = initial_core_profiles0(
        runtime_params,
        geo,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
    )
    initial_core_sources = get_all_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=initial_core_profiles,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
        conductivity=conductivity_base.Conductivity(
            sigma=initial_core_profiles.sigma,
            sigma_face=initial_core_profiles.sigma_face,
        ),
    )
    transport_coeffs = (calculate_total_transport_coeffs(
        physics_models.pedestal_model,
        physics_models.transport_model,
        physics_models.neoclassical_models,
        runtime_params,
        geo,
        initial_core_profiles,
    ))
    return ToraxSimState(
        t=np.array(runtime_params.numerics.t_initial),
        dt=np.zeros(()),
        core_profiles=initial_core_profiles,
        core_sources=initial_core_sources,
        core_transport=transport_coeffs,
        solver_numeric_outputs=state.SolverNumericOutputs(
            solver_error_state=0,
            outer_solver_iterations=0,
            inner_solver_iterations=0,
        ),
        geometry=geo,
    )


class SimulationStepFn:

    def __init__(self, runtime_params_provider, geometry_provider):
        self._geometry_provider = geometry_provider
        self._runtime_params_provider = runtime_params_provider

    @xnp.jit
    def __call__(
        self,
        input_state,
        previous_post_processed_outputs,
    ):
        runtime_params_t, geo_t = (get_consistent_runtime_params_and_geometry(
            t=input_state.t,
            runtime_params_provider=self._runtime_params_provider,
            geometry_provider=self._geometry_provider,
        ))
        explicit_source_profiles = build_source_profiles(
            runtime_params=runtime_params_t,
            geo=geo_t,
            core_profiles=input_state.core_profiles,
            source_models=g.solver.physics_models.source_models,
            neoclassical_models=g.solver.physics_models.neoclassical_models,
            explicit=True,
        )

        def _step():
            return self._adaptive_step(
                runtime_params_t,
                geo_t,
                explicit_source_profiles,
                input_state,
                previous_post_processed_outputs,
            )

        output_state, post_processed_outputs = _step()
        return output_state, post_processed_outputs

    def _adaptive_step(
        self,
        runtime_params_t,
        geo_t,
        explicit_source_profiles,
        input_state,
        previous_post_processed_outputs,
    ):
        evolving_names = runtime_params_t.numerics.evolving_names
        initial_dt = next_dt(
            input_state.t,
            runtime_params_t,
            geo_t,
            input_state.core_profiles,
            input_state.core_transport,
        )

        def cond_fun(inputs):
            next_dt, output = inputs
            solver_outputs = output[2]
            is_nan_next_dt = xnp.isnan(next_dt)
            solver_did_not_converge = solver_outputs.solver_error_state == 1
            if runtime_params_t.numerics.exact_t_final:
                at_exact_t_final = xnp.allclose(
                    input_state.t + next_dt,
                    runtime_params_t.numerics.t_final,
                )
            else:
                at_exact_t_final = xnp.array(False)
            next_dt_too_small = next_dt < runtime_params_t.numerics.min_dt
            take_another_step = xnp.cond(
                solver_did_not_converge,
                lambda: xnp.cond(at_exact_t_final, lambda: True, lambda:
                                 ~next_dt_too_small),
                lambda: False,
            )
            return take_another_step & ~is_nan_next_dt

        def body_fun(inputs):
            dt, output = inputs
            old_solver_outputs = output[2]
            runtime_params_t_plus_dt, geo_t_with_phibdot, geo_t_plus_dt = (
                _get_geo_and_runtime_params_at_t_plus_dt_and_phibdot(
                    input_state.t,
                    dt,
                    self._runtime_params_provider,
                    geo_t,
                    self._geometry_provider,
                ))
            core_profiles_t_plus_dt = provide_core_profiles_t_plus_dt(
                dt=dt,
                runtime_params_t=runtime_params_t,
                runtime_params_t_plus_dt=runtime_params_t_plus_dt,
                geo_t_plus_dt=geo_t_plus_dt,
                core_profiles_t=input_state.core_profiles,
            )
            x_new, solver_numeric_outputs = g.solver(
                t=input_state.t,
                dt=dt,
                runtime_params_t=runtime_params_t,
                runtime_params_t_plus_dt=runtime_params_t_plus_dt,
                geo_t=geo_t_with_phibdot,
                geo_t_plus_dt=geo_t_plus_dt,
                core_profiles_t=input_state.core_profiles,
                core_profiles_t_plus_dt=core_profiles_t_plus_dt,
                explicit_source_profiles=explicit_source_profiles,
            )
            solver_numeric_outputs = state.SolverNumericOutputs(
                solver_error_state=solver_numeric_outputs.solver_error_state,
                outer_solver_iterations=old_solver_outputs.
                outer_solver_iterations + 1,
                inner_solver_iterations=old_solver_outputs.
                inner_solver_iterations +
                solver_numeric_outputs.inner_solver_iterations,
                sawtooth_crash=solver_numeric_outputs.sawtooth_crash,
            )
            next_dt = dt / runtime_params_t_plus_dt.numerics.dt_reduction_factor
            return next_dt, (
                x_new,
                dt,
                solver_numeric_outputs,
                runtime_params_t_plus_dt,
                geo_t_plus_dt,
                core_profiles_t_plus_dt,
            )

        _, result = xnp.while_loop(
            cond_fun,
            body_fun,
            (
                initial_dt,
                (
                    core_profiles_to_solver_x_tuple(input_state.core_profiles,
                                                    evolving_names),
                    initial_dt,
                    state.SolverNumericOutputs(
                        solver_error_state=1,
                        outer_solver_iterations=0,
                        inner_solver_iterations=0,
                        sawtooth_crash=False,
                    ),
                    runtime_params_t,
                    geo_t,
                    input_state.core_profiles,
                ),
            ),
        )
        output_state, post_processed_outputs = _finalize_outputs(
            t=input_state.t,
            dt=result[1],
            x_new=result[0],
            solver_numeric_outputs=result[2],
            runtime_params_t_plus_dt=result[3],
            geometry_t_plus_dt=result[4],
            core_profiles_t=input_state.core_profiles,
            core_profiles_t_plus_dt=result[5],
            explicit_source_profiles=explicit_source_profiles,
            physics_models=g.solver.physics_models,
            evolving_names=evolving_names,
            input_post_processed_outputs=previous_post_processed_outputs,
        )
        return output_state, post_processed_outputs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _finalize_outputs(t, dt, x_new, solver_numeric_outputs, geometry_t_plus_dt,
                      runtime_params_t_plus_dt, core_profiles_t,
                      core_profiles_t_plus_dt, explicit_source_profiles,
                      physics_models, evolving_names,
                      input_post_processed_outputs):
    final_core_profiles, final_source_profiles = (
        update_core_and_source_profiles_after_step(
            dt=dt,
            x_new=x_new,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo=geometry_t_plus_dt,
            core_profiles_t=core_profiles_t,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            explicit_source_profiles=explicit_source_profiles,
            source_models=physics_models.source_models,
            neoclassical_models=physics_models.neoclassical_models,
            evolving_names=evolving_names,
        ))
    final_total_transport = (calculate_total_transport_coeffs(
        physics_models.pedestal_model,
        physics_models.transport_model,
        physics_models.neoclassical_models,
        runtime_params_t_plus_dt,
        geometry_t_plus_dt,
        final_core_profiles,
    ))
    output_state = ToraxSimState(
        t=t + dt,
        dt=dt,
        core_profiles=final_core_profiles,
        core_sources=final_source_profiles,
        core_transport=final_total_transport,
        geometry=geometry_t_plus_dt,
        solver_numeric_outputs=solver_numeric_outputs,
    )
    post_processed_outputs = make_post_processed_outputs(
        sim_state=output_state,
        runtime_params=runtime_params_t_plus_dt,
        previous_post_processed_outputs=input_post_processed_outputs,
    )
    return output_state, post_processed_outputs


def _get_geo_and_runtime_params_at_t_plus_dt_and_phibdot(
    t,
    dt,
    runtime_params_provider,
    geo_t,
    geometry_provider,
):
    runtime_params_t_plus_dt, geo_t_plus_dt = (
        get_consistent_runtime_params_and_geometry(
            t=t + dt,
            runtime_params_provider=runtime_params_provider,
            geometry_provider=geometry_provider,
        ))
    if runtime_params_t_plus_dt.numerics.calcphibdot:
        geo_t, geo_t_plus_dt = update_geometries_with_Phibdot(
            dt=dt,
            geo_t=geo_t,
            geo_t_plus_dt=geo_t_plus_dt,
        )
    return (
        runtime_params_t_plus_dt,
        geo_t,
        geo_t_plus_dt,
    )


class ToraxConfig(model_base.BaseModelFrozen):
    profile_conditions: ProfileConditions
    numerics: numerics_lib.Numerics
    plasma_composition: PlasmaComposition
    geometry: Geometry0
    sources: Sources
    neoclassical: Neoclassical0 = Neoclassical0()
    solver: SolverConfig = pydantic.Field(discriminator='solver_type')
    transport: QLKNNTransportModel = pydantic.Field(discriminator='model_name')
    pedestal: PedestalConfig = pydantic.Field(discriminator='model_name')


CONFIG = {
    'plasma_composition': {
        'main_ion': {
            'D': 0.5,
            'T': 0.5
        },
        'impurity': 'Ne',
        'Z_eff': 1.6,
    },
    'profile_conditions': {
        'Ip': 10.5e6,
        'T_i': {
            0.0: {
                0.0: 15.0,
                1.0: 0.2
            }
        },
        'T_i_right_bc': 0.2,
        'T_e': {
            0.0: {
                0.0: 15.0,
                1.0: 0.2
            }
        },
        'T_e_right_bc': 0.2,
        'n_e_right_bc': 0.25e20,
        'n_e_nbar_is_fGW': True,
        'normalize_n_e_to_nbar': True,
        'nbar': 0.8,
        'n_e': {
            0: {
                0.0: 1.5,
                1.0: 1.0
            }
        },
    },
    'numerics': {
        't_final': 5,
        'resistivity_multiplier': 200,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
        'evolve_density': True,
        'max_dt': 0.5,
        'chi_timestep_prefactor': 50,
        'dt_reduction_factor': 3,
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
        'R_major': 6.2,
        'a_minor': 2.0,
        'B_0': 5.3,
    },
    'neoclassical': {
        'bootstrap_current': {
            'bootstrap_multiplier': 1.0,
        },
    },
    'sources': {
        'generic_current': {
            'fraction_of_total_current': 0.46,
            'gaussian_width': 0.075,
            'gaussian_location': 0.36,
        },
        'generic_particle': {
            'S_total': 2.05e20,
            'deposition_location': 0.3,
            'particle_width': 0.25,
        },
        'gas_puff': {
            'puff_decay_length': 0.3,
            'S_total': 6.0e21,
        },
        'pellet': {
            'S_total': 0.0e22,
            'pellet_width': 0.1,
            'pellet_deposition_location': 0.85,
        },
        'generic_heat': {
            'gaussian_location': 0.12741589640723575,
            'gaussian_width': 0.07280908366127758,
            'P_total': 51.0e6,
            'electron_heat_fraction': 0.68,
        },
        'fusion': {},
        'ei_exchange': {
            'Qei_multiplier': 1.0,
        },
    },
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'T_i_ped': 4.5,
        'T_e_ped': 4.5,
        'n_e_ped': 0.62e20,
        'rho_norm_ped_top': 0.9,
    },
    'transport': {
        'model_name': 'qlknn',
        'apply_inner_patch': True,
        'D_e_inner': 0.25,
        'V_e_inner': 0.0,
        'chi_i_inner': 1.0,
        'chi_e_inner': 1.0,
        'rho_inner': 0.2,
        'apply_outer_patch': True,
        'D_e_outer': 0.1,
        'V_e_outer': 0.0,
        'chi_i_outer': 2.0,
        'chi_e_outer': 2.0,
        'rho_outer': 0.9,
        'chi_min': 0.05,
        'chi_max': 100,
        'D_e_min': 0.05,
        'DV_effective': True,
        'include_ITG': True,
        'include_TEM': True,
        'include_ETG': True,
        'avoid_big_negative_s': True,
        'An_min': 0.05,
        'ITG_flux_ratio_correction': 1,
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': True,
        'n_corrector_steps': 1,
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
    },
}
g.tolerance = 1e-7
torax_config = ToraxConfig.from_dict(CONFIG)
mesh = torax_config.geometry.build_provider.torax_mesh
interpolated_param_2d.set_grid(torax_config, mesh, mode='relaxed')
geometry_provider = torax_config.geometry.build_provider
g.physics_models = PhysicsModels(
    pedestal_model=torax_config.pedestal.build_pedestal_model(),
    source_models=torax_config.sources.build_models(),
    transport_model=torax_config.transport.build_transport_model(),
    neoclassical_models=torax_config.neoclassical.build_models(),
)
g.solver = torax_config.solver.build_solver(physics_models=g.physics_models)
runtime_params_provider = (RuntimeParamsProvider.from_config(torax_config))
step_fn = SimulationStepFn(
    geometry_provider=geometry_provider,
    runtime_params_provider=runtime_params_provider,
)
runtime_params_for_init, geo_for_init = (
    get_consistent_runtime_params_and_geometry(
        t=torax_config.numerics.t_initial,
        runtime_params_provider=runtime_params_provider,
        geometry_provider=geometry_provider,
    ))
initial_state = _get_initial_state(
    runtime_params=runtime_params_for_init,
    geo=geo_for_init,
    step_fn=step_fn,
)
post_processed_outputs = make_post_processed_outputs(initial_state,
                                                     runtime_params_for_init)
initial_post_processed_outputs = post_processed_outputs
current_state = initial_state
state_history = [current_state]
post_processing_history = [initial_post_processed_outputs]
initial_runtime_params = runtime_params_provider(initial_state.t)
while not_done(current_state.t, runtime_params_provider.numerics.t_final):
    current_state, post_processed_outputs = step_fn(
        current_state,
        post_processing_history[-1],
    )
    state_history.append(current_state)
    post_processing_history.append(post_processed_outputs)
state_history = StateHistory(
    state_history=state_history,
    post_processed_outputs_history=post_processing_history,
    torax_config=torax_config,
)
data_tree = state_history.simulation_output_to_xr()
data_tree.to_netcdf("run.nc")
print(data_tree)
import matplotlib.pyplot as plt
t = data_tree.time.to_numpy()
rho = data_tree.rho_norm.to_numpy()
nt, = np.shape(t)
for key in 'T_i', 'T_e', 'psi':
    var = data_tree.profiles[key].to_numpy()
    lo = np.min(var).item()
    hi = np.max(var).item()
    for i, idx in enumerate([0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]):
        plt.title(f'time: {t[idx]:8.3e}')
        plt.axis([None, None, lo, hi])
        plt.plot(rho, var[idx], 'o-')
        plt.savefig(f'{key}.{i:04d}.png')
        plt.close()
