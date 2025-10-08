import dataclasses
import chex
import jax
from jax import numpy as jnp
import jaxtyping as jt
from torax._src import array_typing
import typing_extensions


def _zero() -> array_typing.FloatScalar:
    return jnp.zeros(())


@chex.dataclass(frozen=True)
class CellVariable:
    value: jt.Float[chex.Array, 't* cell']
    dr: jt.Float[chex.Array, 't*']
    left_face_constraint: jt.Float[chex.Array, 't*'] | None = None
    right_face_constraint: jt.Float[chex.Array, 't*'] | None = None
    left_face_grad_constraint: jt.Float[chex.Array, 't*'] | None = (
        dataclasses.field(default_factory=_zero))
    right_face_grad_constraint: jt.Float[chex.Array, 't*'] | None = (
        dataclasses.field(default_factory=_zero))

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            name = field.name
            if isinstance(value, jax.Array):
                if value.dtype != jnp.float64 and jax.config.read(
                        'jax_enable_x64'):
                    raise TypeError(
                        f'Expected dtype float64, got dtype {value.dtype} for `{name}`'
                    )
                if value.dtype != jnp.float32 and not jax.config.read(
                        'jax_enable_x64'):
                    raise TypeError(
                        f'Expected dtype float32, got dtype {value.dtype} for `{name}`'
                    )
        left_and = (self.left_face_constraint is not None
                    and self.left_face_grad_constraint is not None)
        left_or = (self.left_face_constraint is not None
                   or self.left_face_grad_constraint is not None)
        if left_and or not left_or:
            raise ValueError('Exactly one of left_face_constraint and '
                             'left_face_grad_constraint must be set.')
        right_and = (self.right_face_constraint is not None
                     and self.right_face_grad_constraint is not None)
        right_or = (self.right_face_constraint is not None
                    or self.right_face_grad_constraint is not None)
        if right_and or not right_or:
            raise ValueError('Exactly one of right_face_constraint and '
                             'right_face_grad_constraint must be set.')

    def _assert_unbatched(self):
        if len(self.value.shape) != 1:
            raise AssertionError(
                'CellVariable must be unbatched, but has `value` shape '
                f'{self.value.shape}. Consider using vmap to batch the function call.'
            )
        if self.dr.shape:
            raise AssertionError(
                'CellVariable must be unbatched, but has `dr` shape '
                f'{self.dr.shape}. Consider using vmap to batch the function call.'
            )

    def face_grad(
        self,
        x: jt.Float[chex.Array, 'cell'] | None = None
    ) -> jt.Float[chex.Array, 'face']:
        self._assert_unbatched()
        if x is None:
            forward_difference = jnp.diff(self.value) / self.dr
        else:
            forward_difference = jnp.diff(self.value) / jnp.diff(x)

        def constrained_grad(
            face: jax.Array | None,
            grad: jax.Array | None,
            cell: jax.Array,
            right: bool,
        ) -> jax.Array:
            if face is not None:
                if grad is not None:
                    raise ValueError(
                        'Cannot constraint both the value and gradient of '
                        'a face variable.')
                if x is None:
                    dx = self.dr
                else:
                    dx = x[-1] - x[-2] if right else x[1] - x[0]
                sign = -1 if right else 1
                return sign * (cell - face) / (0.5 * dx)
            else:
                if grad is None:
                    raise ValueError('Must specify one of value or gradient.')
                return grad

        left_grad = constrained_grad(
            self.left_face_constraint,
            self.left_face_grad_constraint,
            self.value[0],
            right=False,
        )
        right_grad = constrained_grad(
            self.right_face_constraint,
            self.right_face_grad_constraint,
            self.value[-1],
            right=True,
        )
        left = jnp.expand_dims(left_grad, axis=0)
        right = jnp.expand_dims(right_grad, axis=0)
        return jnp.concatenate([left, forward_difference, right])

    def _left_face_value(self) -> jt.Float[chex.Array, '#t']:
        if self.left_face_constraint is not None:
            value = self.left_face_constraint
            value = jnp.expand_dims(value, axis=-1)
        else:
            value = self.value[..., 0:1]
        return value

    def _right_face_value(self) -> jt.Float[chex.Array, '#t']:
        if self.right_face_constraint is not None:
            value = self.right_face_constraint
            value = jnp.expand_dims(value, axis=-1)
        else:
            value = (
                self.value[..., -1:] +
                jnp.expand_dims(self.right_face_grad_constraint, axis=-1) *
                jnp.expand_dims(self.dr, axis=-1) / 2)
        return value

    def face_value(self) -> jt.Float[jax.Array, 't* face']:
        inner = (self.value[..., :-1] + self.value[..., 1:]) / 2.0
        return jnp.concatenate(
            [self._left_face_value(), inner,
             self._right_face_value()],
            axis=-1)

    def grad(self) -> jt.Float[jax.Array, 't* face']:
        face = self.face_value()
        return jnp.diff(face) / jnp.expand_dims(self.dr, axis=-1)

    def __str__(self) -> str:
        output_string = f'CellVariable(value={self.value}'
        if self.left_face_constraint is not None:
            output_string += f', left_face_constraint={self.left_face_constraint}'
        if self.right_face_constraint is not None:
            output_string += f', right_face_constraint={self.right_face_constraint}'
        if self.left_face_grad_constraint is not None:
            output_string += (
                f', left_face_grad_constraint={self.left_face_grad_constraint}'
            )
        if self.right_face_grad_constraint is not None:
            output_string += (
                f', right_face_grad_constraint={self.right_face_grad_constraint}'
            )
        output_string += ')'
        return output_string

    def cell_plus_boundaries(self) -> jt.Float[jax.Array, 't* cell+2']:
        right_value = self._right_face_value()
        left_value = self._left_face_value()
        return jnp.concatenate(
            [left_value, self.value, right_value],
            axis=-1,
        )

    def __eq__(self, other: typing_extensions.Self) -> bool:
        try:
            chex.assert_trees_all_equal(self, other)
            return True
        except AssertionError:
            return False
