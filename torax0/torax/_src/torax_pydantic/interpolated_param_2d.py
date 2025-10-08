from collections.abc import Mapping
import functools
from typing import Any, Literal, TypeAlias
import chex
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import interpolated_param
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import pydantic_types
import typing_extensions

ValueType: TypeAlias = dict[
    float,
    tuple[pydantic_types.NumpyArray1DUnitInterval,
          pydantic_types.NumpyArray1D],
]


class Grid1D(model_base.BaseModelFrozen):
    nx: typing_extensions.Annotated[pydantic.conint(ge=4),
                                    model_base.JAX_STATIC]

    @functools.cached_property
    def dx(self) -> float:
        return 1 / self.nx

    @property
    def face_centers(self) -> np.ndarray:
        return _get_face_centers(nx=self.nx, dx=self.dx)

    @property
    def cell_centers(self) -> np.ndarray:
        return _get_cell_centers(nx=self.nx, dx=self.dx)

    def __eq__(self, other: typing_extensions.Self) -> bool:
        return self.nx == other.nx and self.dx == other.dx


class TimeVaryingArray(model_base.BaseModelFrozen):
    value: ValueType
    rho_interpolation_mode: typing_extensions.Annotated[
        interpolated_param.InterpolationMode, model_base.
        JAX_STATIC] = interpolated_param.InterpolationMode.PIECEWISE_LINEAR
    time_interpolation_mode: typing_extensions.Annotated[
        interpolated_param.InterpolationMode, model_base.
        JAX_STATIC] = interpolated_param.InterpolationMode.PIECEWISE_LINEAR
    grid: Grid1D | None = None

    def tree_flatten(self):
        children = (
            self.value,
            self._get_cached_interpolated_param_cell,
            self._get_cached_interpolated_param_face,
            self._get_cached_interpolated_param_face_right,
        )
        aux_data = (
            self.rho_interpolation_mode,
            self.time_interpolation_mode,
            self.grid,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.model_construct(
            value=children[0],
            rho_interpolation_mode=aux_data[0],
            time_interpolation_mode=aux_data[1],
            grid=aux_data[2],
        )
        obj._get_cached_interpolated_param_cell = children[1]
        obj._get_cached_interpolated_param_face = children[2]
        obj._get_cached_interpolated_param_face_right = children[3]
        return obj

    @functools.cached_property
    def right_boundary_conditions_defined(self) -> bool:
        for rho_norm, _ in self.value.values():
            if 1.0 not in rho_norm:
                return False
        return True

    def get_value(
        self,
        t: chex.Numeric,
        grid_type: Literal['cell', 'face', 'face_right'] = 'cell',
    ) -> array_typing.Array:
        match grid_type:
            case 'cell':
                return self._get_cached_interpolated_param_cell.get_value(t)
            case 'face':
                return self._get_cached_interpolated_param_face.get_value(t)
            case 'face_right':
                return self._get_cached_interpolated_param_face_right.get_value(
                    t)
            case _:
                raise ValueError(f'Unknown grid type: {grid_type}')


    @pydantic.field_validator('value', mode='after')
    @classmethod
    def _valid_value(cls, value: ValueType) -> ValueType:
        value = dict(sorted(value.items()))
        return value

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_data(
        cls, data: interpolated_param.TimeRhoInterpolatedInput | dict[str, Any]
    ) -> dict[str, Any]:
        if isinstance(data, dict):
            data.pop('_get_cached_interpolated_param_cell_centers', None)
            data.pop('_get_cached_interpolated_param_face_centers', None)
            data.pop('_get_cached_interpolated_param_face_right_centers', None)
            if set(data.keys()).issubset(cls.model_fields.keys()):
                return data
        time_interpolation_mode = (
            interpolated_param.InterpolationMode.PIECEWISE_LINEAR)
        rho_interpolation_mode = (
            interpolated_param.InterpolationMode.PIECEWISE_LINEAR)
        value = _load_from_primitives(data)
        return dict(
            value=value,
            time_interpolation_mode=time_interpolation_mode,
            rho_interpolation_mode=rho_interpolation_mode,
        )

    @functools.cached_property
    def _get_cached_interpolated_param_cell(
        self, ) -> interpolated_param.InterpolatedVarTimeRho:
        return interpolated_param.InterpolatedVarTimeRho(
            self.value,
            rho_norm=self.grid.cell_centers,
            time_interpolation_mode=self.time_interpolation_mode,
            rho_interpolation_mode=self.rho_interpolation_mode,
        )

    @functools.cached_property
    def _get_cached_interpolated_param_face(
        self, ) -> interpolated_param.InterpolatedVarTimeRho:
        return interpolated_param.InterpolatedVarTimeRho(
            self.value,
            rho_norm=self.grid.face_centers,
            time_interpolation_mode=self.time_interpolation_mode,
            rho_interpolation_mode=self.rho_interpolation_mode,
        )

    @functools.cached_property
    def _get_cached_interpolated_param_face_right(
        self, ) -> interpolated_param.InterpolatedVarTimeRho:
        return interpolated_param.InterpolatedVarTimeRho(
            self.value,
            rho_norm=self.grid.face_centers[-1],
            time_interpolation_mode=self.time_interpolation_mode,
            rho_interpolation_mode=self.rho_interpolation_mode,
        )


def _is_positive(array: TimeVaryingArray) -> TimeVaryingArray:
    return array


PositiveTimeVaryingArray = typing_extensions.Annotated[
    TimeVaryingArray, pydantic.AfterValidator(_is_positive)]


def _load_from_primitives(
    primitive_values: (
        Mapping[float, interpolated_param.InterpolatedVarSingleAxisInput]
        | float),
) -> Mapping[float, tuple[array_typing.Array, array_typing.Array]]:
    if isinstance(primitive_values, (float, int)):
        primitive_values = {0.0: {0.0: primitive_values}}
    if isinstance(primitive_values, Mapping) and all(
            isinstance(v, float) for v in primitive_values.values()):
        primitive_values = {0.0: primitive_values}
    if len(set(primitive_values.keys())) != len(primitive_values):
        raise ValueError('Indicies in values mapping must be unique.')
    if not primitive_values:
        raise ValueError('Values mapping must not be empty.')
    loaded_values = {}
    for t, v in primitive_values.items():
        x, y, _, _ = interpolated_param.convert_input_to_xs_ys(v)
        loaded_values[t] = (x, y)
    return loaded_values


def set_grid(
    model: model_base.BaseModelFrozen,
    grid: Grid1D,
    mode: Literal['strict', 'force', 'relaxed'] = 'strict',
):

    def _update_rule(submodel):
        new_grid = Grid1D.model_construct(
            nx=grid.nx,
            face_centers=grid.face_centers,
            cell_centers=grid.cell_centers,
        )
        if submodel.grid is None:
            submodel.__dict__['grid'] = new_grid
        else:
            assert False

    for submodel in model.submodels:
        if isinstance(submodel, TimeVaryingArray):
            _update_rule(submodel)


def _is_non_negative(
    time_varying_array: TimeVaryingArray, ) -> TimeVaryingArray:
    for _, value in time_varying_array.value.values():
        if not np.all(value >= 0.0):
            raise ValueError('All values must be non-negative.')
    return time_varying_array


@functools.cache
def _get_face_centers(nx: int, dx: float) -> np.ndarray:
    return np.linspace(0, nx * dx, nx + 1)


@functools.cache
def _get_cell_centers(nx: int, dx: float) -> np.ndarray:
    return np.linspace(dx * 0.5, (nx - 0.5) * dx, nx)


NonNegativeTimeVaryingArray: TypeAlias = typing_extensions.Annotated[
    TimeVaryingArray,
    pydantic.AfterValidator(_is_non_negative)]
