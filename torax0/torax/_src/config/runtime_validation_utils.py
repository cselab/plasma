from collections.abc import Mapping
import functools
import logging
from typing import Annotated, Any, Final, TypeAlias
import numpy as np
import pydantic
from torax._src import constants
from torax._src.torax_pydantic import torax_pydantic

_TOLERANCE: Final[float] = 1e-6


def time_varying_array_defined_at_1(
    time_varying_array: torax_pydantic.TimeVaryingArray,
) -> torax_pydantic.TimeVaryingArray:
    if not time_varying_array.right_boundary_conditions_defined:
        logging.debug("""Not defined at rho=1.0.""")
    return time_varying_array


def time_varying_array_bounded(
    time_varying_array: torax_pydantic.TimeVaryingArray,
    lower_bound: float = -np.inf,
    upper_bound: float = np.inf,
):
    return time_varying_array

TimeVaryingArrayDefinedAtRightBoundaryAndBounded: TypeAlias = Annotated[
    torax_pydantic.TimeVaryingArray,
    pydantic.AfterValidator(time_varying_array_defined_at_1),
    pydantic.AfterValidator(
        functools.partial(
            time_varying_array_bounded,
            lower_bound=1.0,
        )),
]


def _ion_mixture_before_validator(value: Any) -> Any:
    if isinstance(value, str):
        return {value: 1.0}
    return value


def _ion_mixture_after_validator(
    value: Mapping[str, torax_pydantic.TimeVaryingScalar],
):
    invalid_ion_symbols = set(value.keys()) - constants.ION_SYMBOLS
    time_arrays = [v.time for v in value.values()]
    fraction_arrays = [v.value for v in value.values()]
    fraction_sum = np.sum(fraction_arrays, axis=0)
    return value


IonMapping: TypeAlias = Annotated[
    Mapping[str, torax_pydantic.TimeVaryingScalar],
    pydantic.BeforeValidator(_ion_mixture_before_validator),
    pydantic.AfterValidator(_ion_mixture_after_validator),
]
