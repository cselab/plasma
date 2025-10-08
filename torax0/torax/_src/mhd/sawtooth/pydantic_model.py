import chex
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.mhd.sawtooth import sawtooth_models
from torax._src.torax_pydantic import torax_pydantic
class SawtoothConfig(torax_pydantic.BaseModelFrozen):
  crash_step_duration: torax_pydantic.Second = 1e-3
  def build_models(self):
    return sawtooth_models.SawtoothModels(
        trigger_model=self.trigger_model.build_trigger_model(),
        redistribution_model=self.redistribution_model.build_redistribution_model(),
    )
  def build_runtime_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.RuntimeParams:
    return sawtooth_runtime_params.RuntimeParams(
        crash_step_duration=self.crash_step_duration,
        trigger_params=self.trigger_model.build_runtime_params(t),
        redistribution_params=self.redistribution_model.build_runtime_params(t),
    )
