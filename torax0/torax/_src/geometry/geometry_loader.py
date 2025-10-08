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

"""File I/O for loading geometry files."""

import enum
import os

import numpy as np
import torax


@enum.unique
class GeometrySource(enum.Enum):
  """Integer enum for geometry source."""

  CHEASE = 0
  FBT = 1
  EQDSK = 2


def _load_CHEASE_data(  # pylint: disable=invalid-name
    file_path: str,
) -> dict[str, np.ndarray]:
  """Loads the data from a CHEASE file into a dictionary."""

  with open(file_path, 'r') as file:
    chease_data = {}
    var_labels = file.readline().strip().split()[1:]  # ignore % comment column

    for var_label in var_labels:
      chease_data[var_label] = []

    # store data in respective keys
    for line in file:
      values = line.strip().split()
      for var_label, value in zip(var_labels, values):
        chease_data[var_label].append(float(value))

  # Convert lists to jax arrays.
  return {
      var_label: np.asarray(chease_data[var_label]) for var_label in chease_data
  }

def get_geometry_dir(geometry_dir: str | None = None) -> str:
  """Gets the default geometry directory if no geometry_dir is provided."""
  if geometry_dir is None:
    geometry_dir = os.path.join(torax.__path__[0], 'data/third_party/geo')
  return geometry_dir


def load_geo_data(
    geometry_dir: str | None,
    geometry_file: str,
    geometry_source: GeometrySource,
) -> dict[str, np.ndarray]:
  """Loads the data from a geometry file into a dictionary."""
  geometry_dir = get_geometry_dir(geometry_dir)
  filepath = os.path.join(geometry_dir, geometry_file)

  # initialize geometry from file
  match geometry_source:
    case GeometrySource.CHEASE:
      return _load_CHEASE_data(file_path=filepath)
    case _:
      raise ValueError(f'Unknown geometry source: {geometry_source}')
