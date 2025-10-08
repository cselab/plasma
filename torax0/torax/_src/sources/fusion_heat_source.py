import dataclasses
from typing import Annotated, ClassVar, Literal
import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import collisions
from torax._src.sources import base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

DEFAULT_MODEL_FUNCTION_NAME: str = 'bosch_hale'


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
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, array_typing.FloatVectorCell]:
    _, Pfus_i, Pfus_e = calc_fusion(
        geo,
        core_profiles,
        runtime_params,
    )
    return (Pfus_i, Pfus_e)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class FusionHeatSource(source.Source):
    SOURCE_NAME: ClassVar[str] = 'fusion'
    model_func: source.SourceProfileFunction = fusion_heat_model_func

    @property
    def source_name(self) -> str:
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
        return (
            source.AffectedCoreProfile.TEMP_ION,
            source.AffectedCoreProfile.TEMP_EL,
        )


class FusionHeatSourceConfig(base.SourceModelBase):
    model_name: Annotated[Literal['bosch_hale'],
                          torax_pydantic.JAX_STATIC] = ('bosch_hale')
    mode: Annotated[runtime_params_lib.Mode, torax_pydantic.JAX_STATIC] = (
        runtime_params_lib.Mode.MODEL_BASED)

    @property
    def model_func(self) -> source.SourceProfileFunction:
        return fusion_heat_model_func

    def build_runtime_params(
        self,
        t: chex.Numeric,
    ) -> runtime_params_lib.RuntimeParams:
        return runtime_params_lib.RuntimeParams(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
        )

    def build_source(self) -> FusionHeatSource:
        return FusionHeatSource(model_func=self.model_func)
