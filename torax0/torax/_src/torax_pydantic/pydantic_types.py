from typing import Annotated, TypeAlias
import numpy as np
import pydantic

DataTypes: TypeAlias = float | int | bool
DtypeName: TypeAlias = str
NestedList: TypeAlias = (DataTypes
                         | list[DataTypes]
                         | list[list[DataTypes]]
                         | list[list[list[DataTypes]]])
NumpySerialized: TypeAlias = tuple[DtypeName, NestedList]


def _numpy_array_before_validator(
    x: np.ndarray | NumpySerialized, ) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, tuple) or isinstance(x, list) and len(x) == 2:
        dtype, data = x
        return np.array(data, dtype=np.dtype(dtype))
    else:
        raise ValueError(
            'Expected NumPy or a tuple representing a serialized NumPy array, but'
            f' got a {type(x)}')


def _numpy_array_serializer(x: np.ndarray) -> NumpySerialized:
    return (x.dtype.name, x.tolist())


def _numpy_array_is_rank_1(x: np.ndarray) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError(f'NumPy array is not 1D, rather of rank {x.ndim}')
    return x


def _numpy_array_is_sorted(x: np.ndarray) -> np.ndarray:
    if not np.all(x[:-1] <= x[1:]):
        raise ValueError(f'NumPy array is not sorted: {x}')
    return x


NumpyArray = Annotated[
    np.ndarray,
    pydantic.BeforeValidator(_numpy_array_before_validator),
    pydantic.
    PlainSerializer(_numpy_array_serializer, return_type=NumpySerialized),
]
NumpyArray1D = Annotated[NumpyArray,
                         pydantic.AfterValidator(_numpy_array_is_rank_1)]
NumpyArray1DSorted = Annotated[NumpyArray,
                               pydantic.AfterValidator(_numpy_array_is_sorted)]


def _array_is_unit_interval(array: np.ndarray) -> np.ndarray:
    if not np.all((array >= 0.0) & (array <= 1.0)):
        raise ValueError(
            f'Some array elements are not in the unit interval: {array}')
    return array


NumpyArray1DUnitInterval = Annotated[
    NumpyArray1D,
    pydantic.AfterValidator(_array_is_unit_interval),
]
