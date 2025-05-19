from myraytrace.utils import random_in_hemishpere, random_in_sphere
import pyvista as pv
import numpy as np
import jax.numpy as jnp
import jax.random


src = jnp.arange(0, 10, 2)  # shape (N,)
# src = jnp.array([1])
src = jnp.stack((jnp.zeros(src.shape), src, jnp.zeros(src.shape)), axis=-1)  # shape (N, 3)

randkey = jax.random.PRNGKey(0)
normal = random_in_sphere(shape=(src.shape[0],), randkey=randkey)  # shape (1, 3)
assert jnp.all(jnp.isclose(jnp.linalg.norm(normal, axis=-1), 1.0)), "Normal is not 1"

s = 30
randvec = random_in_hemishpere(normal=normal, sample_count=s, randkey=randkey)  # shape (s, N, 3)

length = jnp.linalg.norm(randvec, axis=-1)  # shape (s, N)
# check if length is 1
assert jnp.all(jnp.isclose(length, 1.0)), "Length is not 1"

plotter = pv.Plotter()

plotter.add_arrows(
    np.asarray(src + normal * 1),
    np.asarray(normal),
    color="red",
)

plotter.add_arrows(
    np.asarray(jnp.stack((src,) * s, axis=0)).reshape(-1, 3), # shape (s*N, 3)
    np.asarray(randvec).reshape(-1, 3), # shape (s*N, 3)
    color="blue",
)

plotter.show_axes()  # Show the axes
plotter.show()