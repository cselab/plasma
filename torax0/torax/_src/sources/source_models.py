import dataclasses
import functools
import immutabledict
import jax
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source as source_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SourceModels:
    qei_source: qei_source_lib.QeiSource
    standard_sources: immutabledict.immutabledict[str, source_lib.Source]

    @functools.cached_property
    def psi_sources(
            self) -> immutabledict.immutabledict[str, source_lib.Source]:
        return immutabledict.immutabledict({
            name: source
            for name, source in self.standard_sources.items() if
            source_lib.AffectedCoreProfile.PSI in source.affected_core_profiles
        })

    def __hash__(self) -> int:
        hashes = [hash(self.standard_sources)]
        hashes.append(hash(self.qei_source))
        return hash(tuple(hashes))

