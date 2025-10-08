import dataclasses
import jax
from typing import Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    neoclassical: Any
    numerics: Any
    pedestal: Any
    plasma_composition: Any
    profile_conditions: Any
    solver: Any
    sources: Any
    transport: Any


def make_ip_consistent(runtime_params, geo):
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
    return runtime_params, geo
