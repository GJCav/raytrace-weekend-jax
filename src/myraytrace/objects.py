import importlib.resources
import jax.numpy as jnp
from jaxtyping import Float, Array, Int, Bool
import jax
import numpy as np
from flax.struct import dataclass
from loguru import logger
from PIL import Image

import importlib.resources

from .raytrace import Ray, HitRecord, Hitable
from .types import FloatArr, Vec3, Vec3arr, Int3, Int3arr

import pyvista as pv

class Sky(Hitable):
    def __init__(self):
        self.blue = jnp.array([0.5, 0.7, 1.0], dtype=jnp.float32)
        self.white = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

    def hit(self, record: HitRecord):
        """
        Hit test for the sky.

        If there is an intersection and the distance is less than ray.t,
        update ray.t to the distance.
        """
        # The sky is only hit at infinity.
        mask = (record.ray.t == jnp.inf)  # shape (N,)
        idx = jnp.where(mask)[0]  # shape (K,), K is the number of rays that hit the sky

        direction = record.ray.direction[idx]  # shape (K, 3)
        portion = (direction[:, 3] + 1) / 2  # shape (K,), assuming z is up and maps to [0, 1]
        mixed_color = (
            portion[:, None] * self.blue +     # (K, 1) * (3) => (K, 3)
            (1 - portion[:, None]) * self.white
        ) # shape (K, 3)

        # Because no more rays will be traced after reaching the sky, so no need
        # to set:
        #   - normal_vector
        #   - duffuse_strength
        record.intensity_modu = record.intensity_modu.at[idx].set(mixed_color)
        record.hit_light_src = record.hit_light_src.at[idx].set(True)


class Mesh(Hitable):
    def __init__(
            self, 
            vertices: Vec3arr, 
            faces: Int3arr, 
            point_normals: Vec3arr = None,
            point_color: Vec3arr = [0.7, 0, 0],
            light_source: bool = False,

            enable_diffuse: bool = True,
            diffuse_strength: float = 0.5,

            enable_specular: bool = False,
            specular_strength: float = 0.8,
            specular_fuzz: float = 0.01,
        ):
        if point_normals is None:
            raise ValueError("point_normals must be provided")
        
        self.vertices = vertices
        self.faces = faces

        self.point_normals: Vec3arr = point_normals # point normals are described in the .obj file
        self._point_color = None
        self.point_color = point_color
        self.light_source = light_source

        self._uv_img = None
        self._uv = None     # If enable, should be (N, 2)

        self.enable_diffuse = enable_diffuse
        self.diffuse_strength = diffuse_strength

        self.enable_specular = enable_specular
        self.specular_strength = specular_strength
        self.specular_fuzz = specular_fuzz

        self.face_normals: Vec3arr | None = None
        if self.face_normals is None:
            self.compute_normals()

    @property
    def point_color(self) -> Vec3arr:
        return self._point_color
    
    @point_color.setter
    def point_color(self, color: list | Vec3 | Vec3arr):
        if color is None:
            raise ValueError("point_color must be provided")

        color = jnp.asarray(color, dtype=jnp.float32)
        # broadcast to the shape of vertices
        color = jnp.broadcast_to(color, self.vertices.shape)
        self._point_color = color


    def compute_normals(self):
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        normals = jnp.cross(v1 - v0, v2 - v0)
        normals = jnp.where(
            jnp.linalg.norm(normals, axis=1, keepdims=True) < 1e-8, 
            jnp.array([1.0, 0.0, 0.0]), 
            normals
        )
        normals /= jnp.linalg.norm(normals, axis=1, keepdims=True)

        self.face_normals = normals

    @dataclass
    class _HitInput:
        # parameter for N rays
        O: Vec3arr # origins, shape (N, 3)
        D: Vec3arr # directions, shape (N, 3)
        ray_t: FloatArr # distance to the hit point, shape (N,)

        # parameters for M triangles
        P0: Vec3arr # shape (M, 3)
        P1: Vec3arr
        P2: Vec3arr
        face_normals: Vec3arr


    @dataclass
    class _HitOutput:
        mask: Bool[Array, "N"]   # shape (N,)  # noqa: F821
        tri_idx: Int[Array, "N"] # shape (N,)  # noqa: F821
        t: Vec3arr
        b1: FloatArr
        b2: FloatArr


    @staticmethod
    @jax.jit
    def _hit_jit(input: "_HitInput") -> _HitOutput:
        logger.warning("JIT cache miss, this is expected for the first call")

        O = input.O
        D = input.D
        ray_t = input.ray_t

        P0 = input.P0
        P1 = input.P1
        P2 = input.P2
        face_normals = input.face_normals

        E1 = P1 - P0            # shape (M, 3)
        E2 = P2 - P0

        # Now we need to test between each ray and each triangle
        # We have N rays and M triangles. Need to test each ray against each triangle.
        # O: (N, 3) -> (N, 1, 3)
        # D: (N, 3) -> (N, 1, 3)
        # P0, E1, E2: (M, 3) -> (1, M, 3)
        # Operations will broadcast to (N, M, 3) where needed, or (N, M) for scalars.
        O_b = O[:, None, :] # shape (N, 1, 3)
        D_b = D[:, None, :] # shape (N, 1, 3)
        P0_b = P0[None, :, :] # shape (1, M, 3)
        E1_b = E1[None, :, :] # shape (1, M, 3)
        E2_b = E2[None, :, :] # shape (1, M, 3)

        S = O_b - P0_b # shape (N, M, 3)
        S1 = jnp.cross(D_b, E2_b) # shape (N, M, 3)
        S2 = jnp.cross(S, E1_b) # shape (N, M, 3)
        S1_dot_E1 = jnp.sum(S1 * E1_b, axis=-1) # shape (N, M)

        t = jnp.sum(S2 * E2_b, axis=-1) / S1_dot_E1 # shape (N, M)
        b1 = jnp.sum(S1 * S, axis=-1) / S1_dot_E1 # shape (N, M)
        b2 = jnp.sum(S2 * D_b, axis=-1) / S1_dot_E1 # shape (N, M)

        # Replace invalid elements to inf so they won't be selected
        face_normals_b = face_normals[None, :, :]   # shape (1, M, 3)
        hit_dir = jnp.sum(D_b * face_normals_b, axis=-1) # shape (N, M)
        valid_hits = (
            (t > 0) & (t != jnp.inf) &                # the ray hits the triangle plane
            (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1) &  # the hit point is inside the triangle
            (hit_dir < 0)           # the ray hits the triangle from the normal side
        )
        t = jnp.where(valid_hits, t, jnp.inf) # shape (N, M)

        # For each ray, find the closest triangle (if the distance is infinite, it means no hit)
        tri_idx = jnp.argmin(t, axis=1) # shape (N,)
        
        t = jnp.take_along_axis(
            t, # shape (N, M)
            jnp.expand_dims(tri_idx, axis=-1), # shape (N, 1)
            axis=-1 # select along the last axis
        ) # -> shape (N, 1)
        t = jnp.squeeze(t, axis=-1) # shape (N,)
        
        b1 = jnp.take_along_axis(
            b1, # shape (N, M)
            jnp.expand_dims(tri_idx, axis=-1), # shape (N, 1)
            axis=-1 # select along the last axis
        ) # -> shape (N, 1)
        b1 = jnp.squeeze(b1, axis=-1) # shape (N,)

        b2 = jnp.take_along_axis(
            b2, # shape (N, M)
            jnp.expand_dims(tri_idx, axis=-1), # shape (N, 1)
            axis=-1 # select along the last axis
        ) # -> shape (N, 1)
        b2 = jnp.squeeze(b2, axis=-1) # shape (N,)

        # Filter out the rays that never hit any triangles, or that hit
        # triangles of other objects ahead of the current object. But due to the
        # limitation of jax.jit, we cannot filter them out now, and need to
        # write the mask to the output, and do the filtering in the caller
        # function.
        mask = (t != jnp.inf) & (t < ray_t)  # shape (N,)

        output = Mesh._HitOutput(
            mask=mask,
            tri_idx=tri_idx,
            t=t,
            b1=b1,
            b2=b2
        )
        return output
    
    @staticmethod
    @jax.jit
    def _hit_jit_face_first(input: "_HitInput") -> _HitOutput:
        """
        For (M triangles, N rays) case, if N >> M, this is slower than _hit_jit,
        the (N rays, M triangles) case.
        """
        logger.warning(f"JIT cache miss, compile for {input.P0.shape[0]} triangles x {input.O.shape[0]} rays")

        O = input.O      # shape (N, 3)
        D = input.D
        ray_t = input.ray_t

        P0 = input.P0
        P1 = input.P1
        P2 = input.P2
        face_normals = input.face_normals

        E1 = P1 - P0            # shape (M, 3)
        E2 = P2 - P0

        # Similar to _hit_jit, but broadcast in the way that the triangles are
        # first, and the rays are second.
        O_b = O[None, :, :] # shape (1, N, 3)
        D_b = D[None, :, :] # shape (1, N, 3)
        P0_b = P0[:, None, :] # shape (M, 1, 3)
        E1_b = E1[:, None, :] # shape (M, 1, 3)
        E2_b = E2[:, None, :] # shape (M, 1, 3)

        S = O_b - P0_b # shape (M, N, 3)
        S1 = jnp.cross(D_b, E2_b) # shape (M, N, 3)
        S2 = jnp.cross(S, E1_b) # shape (M, N, 3)
        S1_dot_E1 = jnp.sum(S1 * E1_b, axis=-1) # shape (M, N)

        t = jnp.sum(S2 * E2_b, axis=-1) / S1_dot_E1 # shape (M, N)
        b1 = jnp.sum(S1 * S, axis=-1) / S1_dot_E1 # shape (M, N)
        b2 = jnp.sum(S2 * D_b, axis=-1) / S1_dot_E1 # shape (M, N)

        # Replace invalid elements to inf so they won't be selected
        face_normals_b = face_normals[:, None, :]   # (M, 3) => (M, 1, 3)
        hit_dir = jnp.sum(D_b * face_normals_b, axis=-1) # shape (M, N)
        valid_hits = (
            (t > 0) & (t != jnp.inf) &                # the ray hits the triangle plane
            (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1) &  # the hit point is inside the triangle
            (hit_dir < 0)           # the ray hits the triangle from the normal side
        )
        t = jnp.where(valid_hits, t, jnp.inf) # shape (M, N)

        # For each ray, find the closest triangle (if the distance is infinite, it means no hit)
        tri_idx = jnp.argmin(t, axis=0) # shape (N,)

        t = jnp.take_along_axis(
            t, # shape (M, N)
            jnp.expand_dims(tri_idx, axis=0), # shape (1, N)
            axis=0 # select along the first axis
        ) # -> shape (1, N)
        t = jnp.squeeze(t, axis=0) # shape (N,)

        b1 = jnp.take_along_axis(
            b1, # shape (M, N)
            jnp.expand_dims(tri_idx, axis=0), # shape (1, N)
            axis=0 # select along the first axis
        ) # -> shape (1, N)
        b1 = jnp.squeeze(b1, axis=0) # shape (N,)

        b2 = jnp.take_along_axis(
            b2, # shape (M, N)
            jnp.expand_dims(tri_idx, axis=0), # shape (1, N)
            axis=0 # select along the first axis
        ) # -> shape (1, N)
        b2 = jnp.squeeze(b2, axis=0) # shape (N,)

        # Filter out the rays that never hit any triangles, or that hit
        # triangles of other objects ahead of the current object. But due to the
        # limitation of jax.jit, we cannot filter them out now, and need to
        # write the mask to the output, and do the filtering in the caller
        # function.
        mask = (t != jnp.inf) & (t < ray_t)  # shape (N,)

        output = Mesh._HitOutput(
            mask=mask,
            tri_idx=tri_idx,
            t=t,
            b1=b1,
            b2=b2
        )
        return output
    
    def hit(self, record: HitRecord):
        """
        Hit test for the mesh.

        If there is an intersection and the distance is less than ray.t,
        update ray.t to the distance.

        An intersection is defined as the ray intersecting the triangle
        from the normal side.
        """
        verts = self.vertices
        faces = self.faces
        ray = record.ray
        
        output = Mesh._hit_jit(Mesh._HitInput(
            O = ray.origin,           # shape (N, 3), N is the number of rays
            D = ray.direction,        # shape (N, 3)
            ray_t = ray.t,            # shape (N, 3)
            P0 = verts[faces[:, 0]],   # shape (M, 3), M is the number of triangles
            P1 = verts[faces[:, 1]],   # shape (M, 3)
            P2 = verts[faces[:, 2]],   # shape (M, 3)
            face_normals= self.face_normals, # shape (M, 3)
        ))

        mask = output.mask
        tri_idx = output.tri_idx
        t = output.t
        b1 = output.b1
        b2 = output.b2

        ray_idx = jnp.where(mask)[0]  # shape (K,), K is the number of rays that actually hit a triangle
        tri_idx = tri_idx[ray_idx]    # shape (K,), now the indices are valid
        t = t[ray_idx]                # shape (K,)
        b1 = b1[ray_idx]    # shape (K,)
        b2 = b2[ray_idx]    # shape (K,)

        hit_faces = faces[tri_idx]  # shape (K, 3)
        if self._uv_img is None:
            point_color = self.point_color       # shape (V, 3), V is the number of vertices
            hit_point_color = (
                point_color[hit_faces[:, 0]] * (1 - b1 - b2)[:, None] +
                point_color[hit_faces[:, 1]] * b1[:, None] +
                point_color[hit_faces[:, 2]] * b2[:, None]
            ) # shape (K, 3)
        else:
            uv = self._uv     # shape (V, 2), V is the number of vertices
            hit_point_uv = (  # shape (K, 2)
                uv[hit_faces[:, 0]] * (1 - b1 - b2)[:, None] +
                uv[hit_faces[:, 1]] * b1[:, None] +
                uv[hit_faces[:, 2]] * b2[:, None]
            )
            img = self._uv_img
            u_idx = jnp.clip((hit_point_uv[:, 0] * img.shape[0]).astype(jnp.int32), 0, img.shape[0] - 1)
            v_idx = jnp.clip((hit_point_uv[:, 1] * img.shape[1]).astype(jnp.int32), 0, img.shape[1] - 1)
            hit_point_color = img[u_idx, v_idx]  # shape (K, 3)

        point_normals = self.point_normals   # shape (V, 3)
        hit_point_normals = (
            point_normals[hit_faces[:, 0]] * (1 - b1 - b2)[:, None] +
            point_normals[hit_faces[:, 1]] * b1[:, None] +
            point_normals[hit_faces[:, 2]] * b2[:, None]
        ) # shape (K, 3)

        # Write information back into the record
        ray.t = ray.t.at[ray_idx].set(t)

        record.normal_vector = record.normal_vector.at[ray_idx].set(hit_point_normals)
        record.intensity_modu = record.intensity_modu.at[ray_idx].set(hit_point_color)
        record.hit_light_src = record.hit_light_src.at[ray_idx].set(self.light_source)

        record.enable_diffuse = record.enable_diffuse.at[ray_idx].set(self.enable_diffuse)
        record.diffuse_strength = record.diffuse_strength.at[ray_idx].set(self.diffuse_strength)
        
        record.enable_specular = record.enable_specular.at[ray_idx].set(self.enable_specular)
        record.specular_strength = record.specular_strength.at[ray_idx].set(self.specular_strength)
        record.specular_fuzz = record.specular_fuzz.at[ray_idx].set(self.specular_fuzz)

    
    @staticmethod
    def sphere(
        radius: float = 1, 
        center: Float[Array, "3"] = jnp.array([0, 0, 0]), 
        num_points: int = 100,
        uv_material_path: str = None,
    ) -> "Mesh":
        center = jnp.asarray(center, dtype=jnp.float32)

        theta = jnp.linspace(0, jnp.pi, num_points)
        theta = theta[1:-1]  # Exclude poles
        phi = jnp.linspace(0, 2 * jnp.pi, num_points, endpoint=False) # Exclude the last point to avoid duplication
        phi, theta = jnp.meshgrid(phi, theta)

        x = radius * jnp.sin(theta) * jnp.cos(phi) + center[0]
        y = radius * jnp.sin(theta) * jnp.sin(phi) + center[1]
        z = radius * jnp.cos(theta) + center[2]

        vertices = jnp.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
        north_idx = vertices.shape[0]
        south_idx = vertices.shape[0] + 1
        vertices = jnp.concatenate(
            (
                vertices, 
                jnp.array([
                    [center[0], center[1], radius + center[2]], 
                    [center[0], center[1], -radius + center[2]]
                ], dtype=jnp.float32)), 
            axis=0
        )  # shape (N, 3)

        point_normals = vertices - center # shape (N, 3)
        point_normals = point_normals / jnp.linalg.norm(point_normals, axis=-1, keepdims=True)

        faces = []
        # Connect poles to the polar circle
        for phi_idx in range(num_points):
            # North pole
            a = north_idx
            b = phi_idx
            c = (phi_idx + 1) % num_points
            faces.append((a, b, c))

            # South pole
            a = south_idx
            b = (num_points - 3) * num_points + phi_idx
            c = (num_points - 3) * num_points + (phi_idx + 1) % num_points
            faces.append((a, c, b))

        # Other faces
        for theta_idx in range(num_points - 3):
            for phi_idx in range(num_points):
                a = theta_idx * num_points + phi_idx
                b = theta_idx * num_points + (phi_idx + 1) % num_points
                c = (theta_idx + 1) * num_points + (phi_idx + 1) % num_points
                d = (theta_idx + 1) * num_points + phi_idx
                faces.append((a, c, b))
                faces.append((a, d, c))

        faces = jnp.array(faces, dtype=jnp.int32)
        
        mesh = Mesh(vertices, faces, point_normals=point_normals)

        if uv_material_path is not None:
            img = Image.open(uv_material_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
            img_jax = jnp.asarray(img_np)  # Convert to JAX array
            
            # Treat (u, v) as (theta, phi) and recompute (theta, phi) from
            # point_normals to avoid possible problems
            phi = jnp.arctan2(point_normals[:, 1], point_normals[:, 0])  # shape (N,)
            theta = jnp.arccos(point_normals[:, 2])

            u = (theta % jnp.pi) / jnp.pi
            v = ((phi + jnp.pi) % (2*jnp.pi)) / (2 * jnp.pi) # shape (N,), range [0, 1)

            mesh._uv_img = img_jax
            mesh._uv = jnp.stack((u, v), axis=-1)  # shape (N, 2)
        return mesh


    @staticmethod
    def bunny(
        center: Vec3 = jnp.array([0, 0, 0], dtype=jnp.float32),
        color: Vec3 = jnp.array([0.73781, 0.53139, 0.07541], dtype=jnp.float32),
        scale: float = 2,
    ):
        center = jnp.asarray(center, dtype=jnp.float32)

        path = importlib.resources.files("myraytrace") / "bunny.obj"
        obj = pv.read(path)
        obj = obj.decimate_pro(reduction=0.6)

        vertices = jnp.array(obj.points, dtype=jnp.float32)  # shape (N, 3)
        vertices = vertices * scale + center

        point_normals = jnp.array(obj.point_normals, dtype=jnp.float32)  # shape (N, 3)
        faces = jnp.array(obj.faces.reshape(-1, 4)[:, 1:], dtype=jnp.int32)  # shape (M, 3)

        mesh = Mesh(
            vertices=vertices,
            faces=faces,
            point_normals=point_normals,
            point_color=color,
            enable_diffuse=True,
            diffuse_strength=0.5,
            enable_specular=False,
            specular_strength=0.8,
            specular_fuzz=0.01,
        )

        return mesh