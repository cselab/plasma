import dataclasses
import functools
from typing import Annotated
import chex
import jax
import pydantic
from torax._src import array_typing
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Self
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParams:
  t_initial: float
  t_final: float
  max_dt: float
  min_dt: float
  chi_timestep_prefactor: float
  fixed_dt: float
  dt_reduction_factor: float
  resistivity_multiplier: array_typing.FloatScalar
  adaptive_T_source_prefactor: float
  adaptive_n_source_prefactor: float
  evolve_ion_heat: bool = dataclasses.field(metadata={'static': True})
  evolve_electron_heat: bool = dataclasses.field(metadata={'static': True})
  evolve_current: bool = dataclasses.field(metadata={'static': True})
  evolve_density: bool = dataclasses.field(metadata={'static': True})
  exact_t_final: bool = dataclasses.field(metadata={'static': True})
  adaptive_dt: bool = dataclasses.field(metadata={'static': True})
  calcphibdot: bool = dataclasses.field(metadata={'static': True})
  @functools.cached_property
  def evolving_names(self) -> tuple[str, ...]:
    evolving_names = []
    if self.evolve_ion_heat:
      evolving_names.append('T_i')
    if self.evolve_electron_heat:
      evolving_names.append('T_e')
    if self.evolve_current:
      evolving_names.append('psi')
    if self.evolve_density:
      evolving_names.append('n_e')
    return tuple(evolving_names)
class Numerics(torax_pydantic.BaseModelFrozen):
  t_initial: torax_pydantic.Second = 0.0
  t_final: torax_pydantic.Second = 5.0
  exact_t_final: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  max_dt: torax_pydantic.Second = 2.0
  min_dt: torax_pydantic.Second = 1e-8
  chi_timestep_prefactor: pydantic.PositiveFloat = 50.0
  fixed_dt: torax_pydantic.Second = 1e-1
  adaptive_dt: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  dt_reduction_factor: pydantic.PositiveFloat = 3.0
  evolve_ion_heat: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  evolve_electron_heat: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  evolve_current: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  evolve_density: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  calcphibdot: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  resistivity_multiplier: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  adaptive_T_source_prefactor: pydantic.PositiveFloat = 2.0e10
  adaptive_n_source_prefactor: pydantic.PositiveFloat = 2.0e8
  @pydantic.model_validator(mode='after')
  def model_validation(self) -> Self:
    if self.t_initial > self.t_final:
      raise ValueError(
          't_initial must be less than or equal to t_final. '
          f't_initial: {self.t_initial}, t_final: {self.t_final}'
      )
    if self.min_dt > self.max_dt:
      raise ValueError(
          'max_dt must be greater than or equal to min_dt. '
          f'max_dt: {self.max_dt}, min_dt: {self.min_dt}'
      )
    return self
  @property
  def evolving_names(self) -> tuple[str, ...]:
    evolving_names = []
    if self.evolve_ion_heat:
      evolving_names.append('T_i')
    if self.evolve_electron_heat:
      evolving_names.append('T_e')
    if self.evolve_current:
      evolving_names.append('psi')
    if self.evolve_density:
      evolving_names.append('n_e')
    return tuple(evolving_names)
  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    return RuntimeParams(
        t_initial=self.t_initial,
        t_final=self.t_final,
        max_dt=self.max_dt,
        min_dt=self.min_dt,
        chi_timestep_prefactor=self.chi_timestep_prefactor,
        fixed_dt=self.fixed_dt,
        dt_reduction_factor=self.dt_reduction_factor,
        resistivity_multiplier=self.resistivity_multiplier.get_value(t),
        adaptive_T_source_prefactor=self.adaptive_T_source_prefactor,
        adaptive_n_source_prefactor=self.adaptive_n_source_prefactor,
        evolve_ion_heat=self.evolve_ion_heat,
        evolve_electron_heat=self.evolve_electron_heat,
        evolve_current=self.evolve_current,
        evolve_density=self.evolve_density,
        exact_t_final=self.exact_t_final,
        adaptive_dt=self.adaptive_dt,
        calcphibdot=self.calcphibdot,
    )
