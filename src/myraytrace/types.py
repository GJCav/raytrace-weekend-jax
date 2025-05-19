import jax.numpy as jnp
from jaxtyping import Float, Array, Int, Bool
import jax

Vec3 = Float[Array, "3"]
Vec3arr = Float[Array, "n 3"]
FloatArr = Float[Array, "n"]
Int3 = Int[Array, "3"]
Int3arr = Int[Array, "m 3"]