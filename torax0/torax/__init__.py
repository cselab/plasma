# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library functionality for TORAX."""

import logging
import os

# pylint: disable=g-importing-member
import jax
from torax._src.state import CoreProfiles
from torax._src.state import CoreTransport
from torax._src.state import SolverNumericOutputs

# pylint: enable=g-importing-member

os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '')
    + ' --xla_backend_extra_options=xla_cpu_flatten_after_fusion'
)

__all__ = [
    'build_torax_config_from_file',
    'import_module',
    'CoreProfiles',
    'CoreTransport',
    'SimError',
    'SolverNumericOutputs',
    'SourceProfiles',
    'StateHistory',
    'ToraxConfig',
]


def set_jax_precision():
  # Default TORAX JAX precision is f64
  precision = os.getenv('JAX_PRECISION', 'f64')
  assert precision == 'f64' or precision == 'f32', (
      'Unknown JAX precision environment variable: %s' % precision
  )
  if precision == 'f64':
    jax.config.update('jax_enable_x64', True)


def log_jax_backend():
  logging.info('JAX running on a default %s backend', jax.default_backend())


set_jax_precision()
