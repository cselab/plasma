from collections.abc import Mapping
import dataclasses
import jax
from torax._src.config import numerics
from torax._src.core_profiles import profile_conditions
from torax._src.core_profiles.plasma_composition import plasma_composition
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.mhd import runtime_params as mhd_runtime_params
from torax._src.neoclassical import runtime_params as neoclassical_params
from torax._src.pedestal_model import runtime_params as pedestal_model_params
from torax._src.sources import runtime_params as sources_params
from torax._src.transport_model import runtime_params as transport_model_params
from typing import Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    mhd: mhd_runtime_params.RuntimeParams
    neoclassical: neoclassical_params.RuntimeParams
    numerics: numerics.RuntimeParams
    pedestal: pedestal_model_params.RuntimeParams
    plasma_composition: plasma_composition.RuntimeParams
    profile_conditions: profile_conditions.RuntimeParams
    solver: Any
    sources: Mapping[str, sources_params.RuntimeParams]
    transport: transport_model_params.RuntimeParams


def make_ip_consistent(
    runtime_params: RuntimeParams,
    geo: geometry.Geometry,
) -> tuple[RuntimeParams, geometry.Geometry]:
    if isinstance(geo, standard_geometry.StandardGeometry):
        if geo.Ip_from_parameters:
            param_Ip = runtime_params.profile_conditions.Ip
            Ip_scale_factor = param_Ip / geo.Ip_profile_face[-1]
            geo = dataclasses.replace(
                geo,
                Ip_profile_face=geo.Ip_profile_face * Ip_scale_factor,
                psi_from_Ip=geo.psi_from_Ip * Ip_scale_factor,
                psi_from_Ip_face=geo.psi_from_Ip_face * Ip_scale_factor,
                j_total=geo.j_total * Ip_scale_factor,
                j_total_face=geo.j_total_face * Ip_scale_factor,
            )
        else:
            runtime_params = dataclasses.replace(
                runtime_params,
                profile_conditions=dataclasses.replace(
                    runtime_params.profile_conditions,
                    Ip=geo.Ip_profile_face[-1],
                ),
            )
    return runtime_params, geo
