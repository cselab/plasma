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
    model_config = pydantic.ConfigDict(
        frozen=True,
        extra='forbid',
        arbitrary_types_allowed=True,
        validate_default=True,
    )

    def __new__(cls, *unused_args, **unused_kwargs):
        try:
            registered_cls = jax.tree_util.register_pytree_node_class(cls)
        except ValueError:
            registered_cls = cls
        return super().__new__(registered_cls)

    @classmethod
    @functools.cache
    def _jit_dynamic_kwarg_names(cls) -> tuple[str, ...]:
        return tuple(name for name in cls.model_fields.keys()
                     if JAX_STATIC not in cls.model_fields[name].metadata)

    @classmethod
    @functools.cache
    def _jit_static_kwarg_names(cls) -> tuple[str, ...]:
        return tuple(name for name in cls.model_fields.keys()
                     if JAX_STATIC in cls.model_fields[name].metadata)

    def tree_flatten(self) -> tuple[DynamicArgs, StaticKwargs]:
        static_names = self._jit_static_kwarg_names()
        dynamic_names = self._jit_dynamic_kwarg_names()
        static_children = {name: getattr(self, name) for name in static_names}
        dynamic_children = [getattr(self, name) for name in dynamic_names]
        return (dynamic_children, static_children)

    @classmethod
    def tree_unflatten(cls, aux_data: StaticKwargs,
                       children: DynamicArgs) -> Self:
        dynamic_kwargs = {
            name: value
            for name, value in zip(
                cls._jit_dynamic_kwarg_names(), children, strict=True)
        }
        return cls.model_construct(**(dynamic_kwargs | aux_data))

    @classmethod
    def from_dict(cls: type[Self], cfg: Mapping[str, Any]) -> Self:
        return cls.model_validate(cfg)

    @property
    def _direct_submodels(self) -> tuple[Self, ...]:

        def is_leaf(x):
            if isinstance(x, (Mapping, Sequence, Set)):
                return False
            return True

        leaves = {
            k: self.__dict__[k]
            for k in self.__class__.model_fields.keys()
        }
        leaves = jax.tree.flatten(leaves, is_leaf=is_leaf)[0]
        return tuple(i for i in leaves if isinstance(i, BaseModelFrozen))

    @property
    def submodels(self) -> tuple[Self, ...]:
        all_submodels = [self]
        new_submodels = self._direct_submodels
        while new_submodels:
            new_submodels_temp = []
            for model in new_submodels:
                all_submodels.append(model)
                new_submodels_temp += model._direct_submodels
            new_submodels = new_submodels_temp
        return tuple(all_submodels)
