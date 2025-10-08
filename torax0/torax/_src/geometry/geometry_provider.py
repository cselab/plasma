import dataclasses
from typing import Protocol
import chex
import jax
from torax._src.geometry import geometry


class GeometryProvider(Protocol):

    def __call__(
        self,
        t: chex.Numeric,
    ) -> geometry.Geometry:
        pass

    @property
    def torax_mesh(self):
        pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConstantGeometryProvider(GeometryProvider):
    geo: geometry.Geometry

    def __call__(self, t: chex.Numeric) -> geometry.Geometry:
        del t
        return self.geo

    @property
    def torax_mesh(self):
        return self.geo.torax_mesh
