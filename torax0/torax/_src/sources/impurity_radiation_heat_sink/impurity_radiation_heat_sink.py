import dataclasses
from torax._src.sources import source as source_lib


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ImpurityRadiationHeatSink(source_lib.Source):
    SOURCE_NAME = "impurity_radiation"
    model_func: source_lib.SourceProfileFunction
