import dataclasses
import functools
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import transport_model as transport_model_lib
@functools.partial(jax_utils.jit, static_argnums=(0, 1, 2))
def calculate_total_transport_coeffs(
    pedestal_model: pedestal_model_lib.PedestalModel,
    transport_model: transport_model_lib.TransportModel,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> state.CoreTransport:
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
