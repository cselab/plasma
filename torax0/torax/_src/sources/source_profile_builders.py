import functools
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import source as source_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profiles
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink

_FINAL_SOURCES = frozenset(
    [impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME])


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'source_models',
        'neoclassical_models',
        'explicit',
    ],
)
def build_source_profiles(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    explicit: bool,
    explicit_source_profiles: source_profiles.SourceProfiles | None = None,
    conductivity: conductivity_base.Conductivity | None = None,
) -> source_profiles.SourceProfiles:
    if not explicit and explicit_source_profiles is None:
        raise ValueError(
            '`explicit_source_profiles` must be provided if explicit is False.'
        )
    if explicit:
        qei = source_profiles.QeiInfo.zeros(geo)
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
    profiles = source_profiles.SourceProfiles(
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
    calculated_source_profiles: source_profiles.SourceProfiles,
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    explicit: bool = True,
    conductivity: conductivity_base.Conductivity | None = None,
    calculate_anyway: bool = False,
    psi_only: bool = False,
):

    def calculate_source(source_name: str, source: source_lib.Source):
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
    calculated_source_profiles: source_profiles.SourceProfiles,
    source_name: str,
    affected_core_profiles: tuple[source_lib.AffectedCoreProfile, ...],
    profile: tuple[array_typing.FloatVectorCell, ...],
):
    for profile, affected_core_profile in zip(profile,
                                              affected_core_profiles,
                                              strict=True):
        match affected_core_profile:
            case source_lib.AffectedCoreProfile.PSI:
                calculated_source_profiles.psi[source_name] = profile
            case source_lib.AffectedCoreProfile.NE:
                calculated_source_profiles.n_e[source_name] = profile
            case source_lib.AffectedCoreProfile.TEMP_ION:
                calculated_source_profiles.T_i[source_name] = profile
            case source_lib.AffectedCoreProfile.TEMP_EL:
                calculated_source_profiles.T_e[source_name] = profile


def build_all_zero_profiles(
    geo: geometry.Geometry, ) -> source_profiles.SourceProfiles:
    return source_profiles.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=source_profiles.QeiInfo.zeros(geo),
    )


def get_all_source_profiles(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    conductivity: conductivity_base.Conductivity,
) -> source_profiles.SourceProfiles:
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
