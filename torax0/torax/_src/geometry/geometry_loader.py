import enum
import os
import numpy as np
import torax


@enum.unique
class GeometrySource(enum.Enum):
    CHEASE = 0
    FBT = 1
    EQDSK = 2


def _load_CHEASE_data(file_path):
    with open(file_path, 'r') as file:
        chease_data = {}
        var_labels = file.readline().strip().split()[1:]
        for var_label in var_labels:
            chease_data[var_label] = []
        for line in file:
            values = line.strip().split()
            for var_label, value in zip(var_labels, values):
                chease_data[var_label].append(float(value))
    return {
        var_label: np.asarray(chease_data[var_label])
        for var_label in chease_data
    }


def get_geometry_dir(geometry_dir):
    if geometry_dir is None:
        geometry_dir = os.path.join(torax.__path__[0], 'data/third_party/geo')
    return geometry_dir


def load_geo_data(
    geometry_dir,
    geometry_file,
    geometry_source
):
    geometry_dir = get_geometry_dir(geometry_dir)
    filepath = os.path.join(geometry_dir, geometry_file)
    return _load_CHEASE_data(file_path=filepath)
