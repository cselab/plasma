import abc
from collections.abc import Mapping
import enum
from typing import Final, Literal, TypeAlias
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import jax_utils
import xarray as xr

RHO_NORM: Final[str] = 'rho_norm'
_interp_fn = jax_utils.jit(jnp.interp)
_interp_fn_vmap = jax_utils.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1)))


@jax_utils.jit
def _step_interpolation(xs: array_typing.Array,
                        x: chex.Numeric) -> array_typing.Array:
    return jnp.clip(
        jnp.searchsorted(xs, x, side='left') - 1, 0, xs.shape[0] - 1)


@enum.unique
class InterpolationMode(enum.Enum):
    PIECEWISE_LINEAR = 'piecewise_linear'
    STEP = 'step'


InterpolationModeLiteral: TypeAlias = Literal['step', 'STEP',
                                              'piecewise_linear',
                                              'PIECEWISE_LINEAR']
_ArrayOrListOfFloats: TypeAlias = array_typing.Array | list[float]
InterpolatedVarSingleAxisInput: TypeAlias = (float
                                             | dict[float, float]
                                             | bool
                                             | dict[float, bool]
                                             | tuple[_ArrayOrListOfFloats,
                                                     _ArrayOrListOfFloats]
                                             | xr.DataArray)
InterpolatedVarTimeRhoInput: TypeAlias = (
    Mapping[float, InterpolatedVarSingleAxisInput]
    | float
    | xr.DataArray
    | tuple[_ArrayOrListOfFloats, _ArrayOrListOfFloats, _ArrayOrListOfFloats]
    | tuple[_ArrayOrListOfFloats, _ArrayOrListOfFloats])
TimeInterpolatedInput: TypeAlias = (InterpolatedVarSingleAxisInput
                                    | tuple[InterpolatedVarSingleAxisInput,
                                            InterpolationModeLiteral])
TimeRhoInterpolatedInput: TypeAlias = (
    InterpolatedVarTimeRhoInput
    | tuple[
        InterpolatedVarTimeRhoInput,
        Mapping[
            Literal['time_interpolation_mode', 'rho_interpolation_mode'],
            InterpolationModeLiteral,
        ],
    ])


class InterpolatedParamBase(abc.ABC):

    @abc.abstractmethod
    def get_value(self, x: chex.Numeric) -> array_typing.Array:
        pass


class _PiecewiseLinearInterpolatedParam(InterpolatedParamBase):

    def __init__(self, xs: array_typing.Array, ys: array_typing.Array):
        if not np.issubdtype(xs.dtype, np.floating):
            raise ValueError(f'xs must be a float array, but got {xs.dtype}.')
        if not np.issubdtype(ys.dtype, np.floating):
            raise ValueError(f'ys must be a float array, but got {ys.dtype}.')
        self._xs = xs
        self._ys = ys
        if self.xs.shape[0] != self.ys.shape[0]:
            raise ValueError(
                'xs and ys must have the same number of elements in the first '
                f'dimension. Given: {self.xs.shape} and {self.ys.shape}.')
        if ys.ndim not in (1, 2):
            raise ValueError(
                f'ys must be either 1D or 2D. Given: {self.ys.shape}.')

    @property
    def xs(self) -> array_typing.Array:
        return self._xs

    @property
    def ys(self) -> array_typing.Array:
        return self._ys

    def get_value(
        self,
        x: chex.Numeric,
    ) -> array_typing.Array:
        x_shape = getattr(x, 'shape', ())
        is_jax = isinstance(x, jax.Array)
        interp = _interp_fn if is_jax else np.interp
        full = jnp.full if is_jax else np.full
        match self.ys.ndim:
            case 1:
                if self.ys.size == 1:
                    if x_shape == ():
                        return self.ys[0]
                    else:
                        return full(x_shape, self.ys[0], dtype=self.ys.dtype)
                else:
                    return interp(x, self.xs, self.ys)
            case 2:
                if len(self.ys) == 1 and x_shape == ():
                    return self.ys[0]
                else:
                    return _interp_fn_vmap(x, self.xs, self.ys)
            case _:
                raise ValueError(
                    f'ys must be either 1D or 2D. Given: {self.ys.shape}.')


def _is_bool(interp_input: InterpolatedVarSingleAxisInput, ) -> bool:
    if isinstance(interp_input, dict):
        if not interp_input:
            raise ValueError(
                'InterpolatedVarSingleAxisInput must include values.')
        value = list(interp_input.values())[0]
        return isinstance(value, bool)
    return isinstance(interp_input, bool)


def _convert_value_to_floats(
    interp_input: InterpolatedVarSingleAxisInput,
) -> InterpolatedVarSingleAxisInput:
    if isinstance(interp_input, dict):
        return {key: float(value) for key, value in interp_input.items()}
    return float(interp_input)


def convert_input_to_xs_ys(
    interp_input: TimeInterpolatedInput,
) -> tuple[np.ndarray, np.ndarray, InterpolationMode, bool]:
    interpolation_mode = InterpolationMode.PIECEWISE_LINEAR
    if isinstance(interp_input, tuple):
        if len(interp_input) != 2:
            raise ValueError(
                'Single axis interpolated var tuple length must be 2. The first '
                'element are the values and the second element is the '
                'interpolation mode or both values should be arrays to be directly '
                f'interpolated. Given: {interp_input}.')
        if isinstance(interp_input[1], str):
            interpolation_mode = InterpolationMode[interp_input[1].upper()]
            interp_input = interp_input[0]
    if _is_bool(interp_input):
        interp_input = _convert_value_to_floats(interp_input)
        is_bool_param = True
    else:
        is_bool_param = False
    if isinstance(interp_input, xr.DataArray):
        if not isinstance(interp_input.coords, Mapping):
            raise ValueError(
                'The coords in the xr.DataArray must be a mapping.')
        if 'time' not in interp_input.coords:
            raise ValueError(
                'The coords in the xr.DataArray must include a "time" coordinate.'
            )
        return (
            np.asarray(interp_input.coords['time'],
                       dtype=jax_utils.get_np_dtype()),
            np.asarray(interp_input.values, dtype=jax_utils.get_np_dtype()),
            interpolation_mode,
            is_bool_param,
        )
    if isinstance(interp_input, tuple):
        if len(interp_input) != 2:
            raise ValueError(
                'The time interpolated input tuple must be length 2. Given: '
                f'{interp_input}.')
        xs, ys = interp_input
        xs = np.asarray(xs, dtype=jax_utils.get_np_dtype())
        ys = np.asarray(ys, dtype=jax_utils.get_np_dtype())
        return xs, ys, interpolation_mode, is_bool_param
    if isinstance(interp_input, dict):
        if not interp_input:
            raise ValueError(
                'The time interpolated input dict must be non-empty.')
        return (
            np.array(list(interp_input.keys()),
                     dtype=jax_utils.get_np_dtype()),
            np.array(list(interp_input.values()),
                     dtype=jax_utils.get_np_dtype()),
            interpolation_mode,
            is_bool_param,
        )
    else:
        return (
            np.array([0.0], dtype=jax_utils.get_np_dtype()),
            np.array([interp_input], dtype=jax_utils.get_np_dtype()),
            interpolation_mode,
            is_bool_param,
        )


@jax.tree_util.register_pytree_node_class
class InterpolatedVarSingleAxis(InterpolatedParamBase):

    def __init__(
        self,
        value: tuple[array_typing.Array, array_typing.Array],
        interpolation_mode: InterpolationMode = (
            InterpolationMode.PIECEWISE_LINEAR),
        is_bool_param: bool = False,
    ):
        self._value = value
        xs, ys = value
        if not np.issubdtype(xs.dtype, np.floating):
            raise ValueError(f'xs must be a float array, but got {xs.dtype}.')
        if not np.issubdtype(ys.dtype, np.floating):
            raise ValueError(f'ys must be a float array, but got {ys.dtype}.')
        self._is_bool_param = is_bool_param
        self._interpolation_mode = interpolation_mode
        match interpolation_mode:
            case InterpolationMode.PIECEWISE_LINEAR:
                self._param = _PiecewiseLinearInterpolatedParam(xs=xs, ys=ys)
            case _:
                raise ValueError('Unknown interpolation mode.')

    def tree_flatten(self):
        static_params = {
            'interpolation_mode': self.interpolation_mode,
            'is_bool_param': self.is_bool_param,
        }
        return (self._value, static_params)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children, **aux_data)

    @property
    def is_bool_param(self) -> bool:
        return self._is_bool_param

    @property
    def interpolation_mode(self) -> InterpolationMode:
        return self._interpolation_mode

    def get_value(
        self,
        x: chex.Numeric,
    ) -> array_typing.Array:
        value = self._param.get_value(x)
        if self._is_bool_param:
            return jnp.bool_(value > 0.5)
        return value

    @property
    def param(self) -> InterpolatedParamBase:
        return self._param

    def __eq__(self, other: 'InterpolatedVarSingleAxis') -> bool:
        try:
            chex.assert_trees_all_equal(self, other)
        except AssertionError:
            return False
        return True


@jax.tree_util.register_pytree_node_class
class InterpolatedVarTimeRho(InterpolatedParamBase):

    def __init__(
        self,
        values: Mapping[float, tuple[array_typing.Array, array_typing.Array]],
        rho_norm: array_typing.Array,
        time_interpolation_mode: InterpolationMode = (
            InterpolationMode.PIECEWISE_LINEAR),
        rho_interpolation_mode: InterpolationMode = (
            InterpolationMode.PIECEWISE_LINEAR),
    ):
        self._rho_interpolation_mode = rho_interpolation_mode
        self._time_interpolation_mode = time_interpolation_mode
        sorted_indices = np.array(sorted(values.keys()))
        rho_norm_interpolated_values = np.stack(
            [
                InterpolatedVarSingleAxis(
                    values[t], rho_interpolation_mode).get_value(rho_norm)
                for t in sorted_indices
            ],
            axis=0,
        )
        self._time_interpolated_var = InterpolatedVarSingleAxis(
            value=(sorted_indices, rho_norm_interpolated_values),
            interpolation_mode=time_interpolation_mode,
        )

    def tree_flatten(self):
        children = (self._time_interpolated_var, )
        aux_data = (self._rho_interpolation_mode,
                    self._time_interpolation_mode)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(InterpolatedVarTimeRho)
        obj._time_interpolated_var = children[0]
        obj._rho_interpolation_mode = aux_data[0]
        obj._time_interpolation_mode = aux_data[1]
        return obj

    @property
    def time_interpolation_mode(self) -> InterpolationMode:
        return self._time_interpolation_mode

    @property
    def rho_interpolation_mode(self) -> InterpolationMode:
        return self._rho_interpolation_mode

    def get_value(self, x: chex.Numeric) -> array_typing.Array:
        return self._time_interpolated_var.get_value(x)
