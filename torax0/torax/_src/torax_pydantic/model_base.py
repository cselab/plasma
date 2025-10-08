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

"""Pydantic utilities and base classes."""

from collections.abc import Set
import functools
from typing import Any, Final, Mapping, Sequence, TypeAlias

import jax
import pydantic
from typing_extensions import Self

TIME_INVARIANT: Final[str] = '_pydantic_time_invariant_field'
JAX_STATIC: Final[str] = '_pydantic_jax_static_field'

StaticKwargs: TypeAlias = dict[str, Any]
DynamicArgs: TypeAlias = list[Any]


class BaseModelFrozen(pydantic.BaseModel):
  """Base config with frozen fields.

  See https://docs.pydantic.dev/latest/ for documentation on pydantic.

  This class is compatible with JAX, so can be used as an argument to a JITted
  function. Static fields can be annotated via
  `typing.Annotated[dtype, torax_pydantic.JAX_STATIC`] to make them static in
  the JAX tree. These fields must be hashable.
  """

  model_config = pydantic.ConfigDict(
      frozen=True,
      # Do not allow attributes not defined in pydantic model.
      extra='forbid',
      arbitrary_types_allowed=True,
      validate_default=True,
  )

  def __new__(cls, *unused_args, **unused_kwargs):
    try:
      registered_cls = jax.tree_util.register_pytree_node_class(cls)
    except ValueError:
      registered_cls = cls  # Already registered.
    return super().__new__(registered_cls)

  @classmethod
  @functools.cache
  def _jit_dynamic_kwarg_names(cls) -> tuple[str, ...]:
    return tuple(
        name
        for name in cls.model_fields.keys()
        if JAX_STATIC not in cls.model_fields[name].metadata
    )

  @classmethod
  @functools.cache
  def _jit_static_kwarg_names(cls) -> tuple[str, ...]:
    return tuple(
        name
        for name in cls.model_fields.keys()
        if JAX_STATIC in cls.model_fields[name].metadata
    )

  def tree_flatten(self) -> tuple[DynamicArgs, StaticKwargs]:
    """Flattens the model into a JAX dynamic and static argument tuple.

    Static arguments are model fields annotated via
    `typing.Annotated[dtype, torax_pydantic.JAX_STATIC]`. Dynamic arguments are
    all other fields.

    Required by the use of `jax.tree_util.register_pytree_node_class`.

    Returns:
      A tuple of the dynamic and static arguments. Dynamic arguments are a list
      of numeric values compatible with `jax.jit`. Static arguments are a
      dictionary of hashable values considered `static_argnames` by `jax.jit`.
    """
    static_names = self._jit_static_kwarg_names()
    dynamic_names = self._jit_dynamic_kwarg_names()
    static_children = {name: getattr(self, name) for name in static_names}
    dynamic_children = [getattr(self, name) for name in dynamic_names]

    return (dynamic_children, static_children)

  @classmethod
  def tree_unflatten(
      cls, aux_data: StaticKwargs, children: DynamicArgs
  ) -> Self:
    """Reconstructs a model from a JAX dynamic and static argument tuple.

    Required by the use of `jax.tree_util.register_pytree_node_class`.

    Args:
      aux_data: A dictionary of static arguments.
      children: A list of dynamic arguments.

    Returns:
      A model instance.
    """
    dynamic_kwargs = {
        name: value
        for name, value in zip(
            cls._jit_dynamic_kwarg_names(), children, strict=True
        )
    }
    # The model needs to be reconstructed without validation, as init can
    # contain JAX tracers inside a JIT, which will fail Pydantic validation. In
    # addition, validation is unecessary overhead.
    return cls.model_construct(**(dynamic_kwargs | aux_data))

  @classmethod
  def from_dict(cls: type[Self], cfg: Mapping[str, Any]) -> Self:
    return cls.model_validate(cfg)

  @property
  def _direct_submodels(self) -> tuple[Self, ...]:
    """Direct submodels in the model."""

    def is_leaf(x):
      if isinstance(x, (Mapping, Sequence, Set)):
        return False
      return True

    # Exclude non-field values in __dict__, such as cached_properties.
    leaves = {k: self.__dict__[k] for k in self.__class__.model_fields.keys()}
    # Some Pydantic models are values of a dict. We flatten the tree to access
    # them.
    leaves = jax.tree.flatten(leaves, is_leaf=is_leaf)[0]
    return tuple(i for i in leaves if isinstance(i, BaseModelFrozen))

  @property
  def submodels(self) -> tuple[Self, ...]:
    """A tuple of the model and all submodels.

    This will return all Pydantic models directly inside model fields, and
    inside container types: mappings, sequences, and sets.

    Returns:
      A tuple of the model and all model submodels.
    """

    all_submodels = [self]
    new_submodels = self._direct_submodels
    while new_submodels:
      new_submodels_temp = []
      for model in new_submodels:
        all_submodels.append(model)
        new_submodels_temp += model._direct_submodels  # pylint: disable=protected-access
      new_submodels = new_submodels_temp
    return tuple(all_submodels)

