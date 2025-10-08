import dataclasses
from torax._src.mhd.sawtooth import redistribution_base
from torax._src.mhd.sawtooth import trigger_base


@dataclasses.dataclass
class SawtoothModels:
    trigger_model: trigger_base.TriggerModel
    redistribution_model: redistribution_base.RedistributionModel

    def __eq__(self, other: 'SawtoothModels') -> bool:
        return (self.trigger_model == other.trigger_model
                and self.redistribution_model == other.redistribution_model)

    def __hash__(self) -> int:
        return hash((
            self.trigger_model,
            self.redistribution_model,
        ))
