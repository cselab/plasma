import dataclasses
from typing import Literal
import jax
import jax.numpy as jnp
from torax._src import constants
from torax._src.geometry import geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
import typing_extensions
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
    return jax.tree_util.tree_map(
        sum_profiles, explicit_source_profiles, implicit_source_profiles
    )
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
