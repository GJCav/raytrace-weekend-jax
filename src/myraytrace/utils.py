from typing import Iterable
import jax.numpy as jnp
from jaxtyping import Float, Array, Int, Bool
import jax
import numpy as np

from .types import Vec3, Vec3arr, Int3, Int3arr, FloatArr


def random_in_sphere(shape: tuple[int], randkey = jax.random.PRNGKey(0)):
    """
    Sample random points in a sphere.
    Input:
        shape: The batch dimensions of the sphere.
        randkey: JAX random key for reproducibility.
    Output:
        shape (*shape, 3) array
    """
    directions = jax.random.normal(randkey, (*shape, 3))  # shape (*shape, 3)
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)  # shape (*shape, 3)
    return directions


def random_in_hemishpere(
        normal: Vec3arr,   # shape (N, 3)
        sample_count: int, # S = sample_count
        randkey = jax.random.PRNGKey(0),
    ) -> Float[Array, "sample_count N 3"]:  # noqa: F722
    """
    Sample `sample_count` directions in N hemispheres identified by `normal`.

    Input:
        normal: Shape (N, 3) array, indicating the normal of each hemisphere.
        sample_count: Number of samples to generate in each hemisphere.
        randkey: JAX random key for reproducibility.
    """
    # Generate random points in the unit sphere
    directions = random_in_sphere(
        shape=(sample_count, normal.shape[0]), 
        randkey=randkey
    )  # shape (S, N, 3)

    # # Normalize the directions
    proj = jnp.sum(directions * normal[None, :, :], axis=-1) # shape (S, N)
    proj = jnp.where(proj > 0, 0, proj) # shape (S, N)
    directions = directions - 2 * normal * proj[:, :, None] # shape (S, N, 3)

    return directions


def random_in_disk(
        radius: FloatArr,  # shape (N,)
        normal: Vec3arr,   # shape (N, 3)
        sample_count: int, # S = sample_count
        randkey = jax.random.PRNGKey(0),
    ):
    
    disturb = jax.random.normal(randkey, (sample_count, normal.shape[0], 3)) # shape (S, N, 3)
    normal = jnp.repeat(normal[None, :, :], sample_count, axis=0) # shape (S, N, 3)
    result = normal + disturb * radius[None, :, None] # shape (S, N, 3)
    result = result / jnp.linalg.norm(result, axis=-1, keepdims=True) # shape (S, N, 3)
    return result


def unit_vector(v: Vec3 | Vec3arr) -> Vec3 | Vec3arr:
    """
    Normalize a vector or an array of vectors.
    """
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)


def iter_over_list_of_iterable(list_of_iterable: list[Iterable]) -> Iterable:
    """
    Iterate over a list of iterables.
    """
    for it in list_of_iterable:
        for item in it:
            yield item


def auto_split_randkey(randkey: jax.random.PRNGKey) -> Iterable[jax.random.PRNGKey]:
    """
    Split a random key into `num` keys.
    """
    while True:
        randkey, subkey = jax.random.split(randkey)
        yield subkey