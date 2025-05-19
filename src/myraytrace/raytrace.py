from collections import namedtuple
from typing import Iterable
import jax.numpy as jnp
from jaxtyping import Float, Array, Int, Bool
import jax
from PIL import Image
import numpy as np
from .utils import unit_vector
from datetime import datetime

from loguru import logger

from . import utils
from .types import Vec3, Vec3arr

class Ray:
    def __init__(self, origin: Vec3arr, direction: Vec3arr, index: Int[Array, "n 2"]):  # noqa: F722
        """
        Ray class to represent N rays in 3D space.

        Parameters: 
            - origin: The starting point of the rays, shape (N, 3).
            - direction: The direction of the rays, shape (N, 3). It should be
              normalized.
            - index: The index of pixels in the image from which the rays are
              casted, shape (N, 2). The index is in the format (row, column).
        """
        self.origin = origin
        self.direction = direction
        self.index = index

        # Intensity of the RGB color of the ray.
        #
        # The final color of a ligth path:
        #     light_color * 1-st reflection * 2-nd reflection * ... * N-th reflection
        # where:
        #   - light_color is the color of the light source, a shape (3,)
        #   - reflection is the portion of light that is reflected
        #
        # For reverse path tracing, it is viewed as:
        #     ((((1-st reflection) * 2-nd reflection) * ... ) * N-th reflection) * light_color
        #
        # So we initialize the intensity to 1.0, multiply it by the reflection
        # coefficients as we go along the path, and finally multiply it by
        # the light color at the end. Its value only matters when we
        # accumulate the color at the end of the path.
        self.intensity = jnp.full((self.direction.shape[0], 3), 1.0, dtype=jnp.float32) # shape (N, 3)

        # Distance from the ray origin to the intersection point.
        #
        # Conventions:
        #   - t is never smaller than 0. (It is a ray, not a line.)
        #   - t > 0 & t < inf: The ray intersects the object.
        #   - t = inf: The ray does not intersect the object.
        self.t = jnp.full(self.direction.shape[0], jnp.inf, dtype=jnp.float32)  # shape (N,)


    def __len__(self):
        return self.origin.shape[0]
    

    def subset(self, mask: Bool[Array, "n"]):  # noqa: F821
        ray = Ray(
            origin=self.origin[mask],
            direction=self.direction[mask],
            index=self.index[mask],
        )
        ray.intensity = self.intensity[mask]
        ray.t = self.t[mask]
        return ray
    

    def slice(self, start: int, end: int):
        ray = Ray(
            origin=self.origin[start:end],
            direction=self.direction[start:end],
            index=self.index[start:end],
        )
        ray.intensity = self.intensity[start:end]
        ray.t = self.t[start:end]
        return ray
    
    
    @staticmethod
    def concatenate(rays: Iterable["Ray"]) -> "Ray":
        """
        Concatenate a list of rays into a single ray.
        """
        rays = list(rays)
        origin = jnp.concatenate([ray.origin for ray in rays], axis=0)
        direction = jnp.concatenate([ray.direction for ray in rays], axis=0)
        index = jnp.concatenate([ray.index for ray in rays], axis=0)
        intensity = jnp.concatenate([ray.intensity for ray in rays], axis=0)
        t = jnp.concatenate([ray.t for ray in rays], axis=0)

        new_ray = Ray(origin=origin, direction=direction, index=index)
        new_ray.intensity = intensity
        new_ray.t = t
        return new_ray
    

    def accumulate_back_to_image(self, image: Float[Array, "h w 3"]) -> Float[Array, "h w 3"]:  # noqa: F722
        """
        Accumulate the color of the ray to the image.
        """
        # Convert the pixel index to row and column
        rows = self.index[:, 0]
        cols = self.index[:, 1]

        # Accumulate the color to the image
        image = image.at[rows, cols].add(self.intensity)
        return image
    
    def end_point(self, offset: float = 0) -> Vec3arr:
        """
        Calculate the end point of the ray.
        """
        return self.origin + self.direction * (self.t + offset)[:, None]


class Camera:
    def __init__(self, pos: Vec3, lookat: Vec3, cam_x: Vec3, hfov: float, focus_dist: float, num_hpixels: int = 720):
        pos = jnp.asarray(pos, dtype=jnp.float32) # shape (3,)
        lookat = jnp.asarray(lookat, dtype=jnp.float32) # shape (3,)
        cam_x = jnp.asarray(cam_x, dtype=jnp.float32) # shape (3,)
        
        self.pos = pos
        self.focus_dist = focus_dist
        
        theta = jnp.deg2rad(hfov)
        self.viewport_width = 2 * focus_dist * jnp.tan(theta / 2)
        self.viewport_height = self.viewport_width / (16 / 9)  # 16:9 aspect ratio

        self.cam_z = unit_vector(lookat - pos) # z = depth axis, the larger the further
        self.cam_x = unit_vector(cam_x)        # x = right axis
        self.cam_y = jnp.cross(self.cam_z, self.cam_x) # y = down axis
        self.cam_x = jnp.cross(self.cam_y, self.cam_z) # re-orthogonalize x, so it is perpendicular to z

        self.viewport_origin = (
            pos + self.cam_z * focus_dist - 
            self.cam_x * (self.viewport_width / 2) - 
            self.cam_y * (self.viewport_height / 2)
        )

        self.num_hpixels = num_hpixels
        self.num_vpixels = int(self.num_hpixels * (self.viewport_height / self.viewport_width))


    def world2cam_mat(self) -> Float[Array, "4 4"]:  # noqa: F722
        """
        Calculate the world to camera transformation matrix.
        """
        R = jnp.stack([self.cam_x, self.cam_y, self.cam_z], axis=1)  # shape (3, 3)
        T = -R.T @ self.pos  # shape (3,)
        mat = jnp.eye(4, dtype=jnp.float32)
        mat = mat.at[:3, :3].set(R.T)
        mat = mat.at[:3, 3].set(T)
        return mat
        

class HitRecord:
    def __init__(self, ray: Ray):
        self.ray = ray  # N rays
        self.normal_vector = jnp.zeros((ray.direction.shape[0], 3), dtype=jnp.float32) # shape (N, 3)
        self.intensity_modu = jnp.zeros((ray.direction.shape[0], 3), dtype=jnp.float32) # shape (N, 3)

        self.enable_diffuse = jnp.ones((ray.direction.shape[0],), dtype=jnp.bool) # shape (N,)
        self.diffuse_strength = jnp.ones((ray.direction.shape[0],), dtype=jnp.float32) # shape (N,)

        self.enable_specular = jnp.zeros((ray.direction.shape[0],), dtype=jnp.bool) # shape (N,)
        self.specular_strength = jnp.ones((ray.direction.shape[0],), dtype=jnp.float32) # shape (N,)
        self.specular_fuzz = jnp.zeros((ray.direction.shape[0],), dtype=jnp.float32) # shape (N,)

        self.hit_light_src = jnp.zeros((ray.direction.shape[0],), dtype=jnp.bool) # shape (N,)

    def subset(self, mask: Bool[Array, "N"]):  # noqa: F821
        record = HitRecord(self.ray.subset(mask))
        record.normal_vector = self.normal_vector[mask]
        record.intensity_modu = self.intensity_modu[mask]

        record.enable_diffuse = self.enable_diffuse[mask]
        record.diffuse_strength = self.diffuse_strength[mask]

        record.enable_specular = self.enable_specular[mask]
        record.specular_strength = self.specular_strength[mask]
        record.specular_fuzz = self.specular_fuzz[mask]

        record.hit_light_src = self.hit_light_src[mask]
        return record
    
    def slice(self, start: int, end: int):
        record = HitRecord(self.ray.slice(start, end))
        record.normal_vector = self.normal_vector[start:end]
        record.intensity_modu = self.intensity_modu[start:end]

        record.enable_diffuse = self.enable_diffuse[start:end]
        record.diffuse_strength = self.diffuse_strength[start:end]

        record.enable_specular = self.enable_specular[start:end]
        record.specular_strength = self.specular_strength[start:end]
        record.specular_fuzz = self.specular_fuzz[start:end]

        record.hit_light_src = self.hit_light_src[start:end]

        return record
    
    @staticmethod
    def concatenate(records: Iterable["HitRecord"]) -> "HitRecord":
        """
        Concatenate a list of hit records into a single hit record.
        """
        records = list(records)
        ray = Ray.concatenate([record.ray for record in records])
        normal_vector = jnp.concatenate([record.normal_vector for record in records], axis=0)
        intensity_modu = jnp.concatenate([record.intensity_modu for record in records], axis=0)
        enable_diffuse = jnp.concatenate([record.enable_diffuse for record in records], axis=0)
        diffuse_strength = jnp.concatenate([record.diffuse_strength for record in records], axis=0)
        hit_light_src = jnp.concatenate([record.hit_light_src for record in records], axis=0)

        new_record = HitRecord(ray)
        new_record.normal_vector = normal_vector
        new_record.intensity_modu = intensity_modu
        new_record.enable_diffuse = enable_diffuse
        new_record.diffuse_strength = diffuse_strength
        new_record.hit_light_src = hit_light_src
        return new_record
    
    def __len__(self):
        return self.normal_vector.shape[0]


class Hitable:
    def hit(self, record: HitRecord):
        raise NotImplementedError("hit() method not implemented in base class")


class RaytraceRenderer:
    def __init__(
            self, 
            cam: Camera, 
            ray_per_pixel: int = 1, 
            maximum_ray_bounces: int = 1,
            diffuse_sample_num: int = 1,
            specular_sample_num: int = 1,
            ray_batch_size: int = 10000
        ):
        self.cam = cam
        self.ray_per_pixel = ray_per_pixel
        self.maximum_ray_bounces = maximum_ray_bounces

        self.diffuse_sample_num = diffuse_sample_num
        self.specular_sample_num = specular_sample_num

        self.hitable_objects: list[Hitable] = []
        self.ray_batch_size = ray_batch_size

    def _get_cam_ray(self, randkey = jax.random.key(0)) -> Iterable[Ray]:
        ray_per_pixel = self.ray_per_pixel
        cam = self.cam

        # split the (h, v) pixel grid into smaller square batches
        square_length = int(jnp.sqrt(self.ray_batch_size // ray_per_pixel))
        for h_idx_start in range(0, cam.num_hpixels, square_length):
            for v_idx_start in range(0, cam.num_vpixels, square_length):
                h_idx_end = min(h_idx_start + square_length, cam.num_hpixels)
                v_idx_end = min(v_idx_start + square_length, cam.num_vpixels)
                num_batch_hpixels = h_idx_end - h_idx_start
                num_batch_vpixels = v_idx_end - v_idx_start

                # Index grid for the current batch
                h_indices = jnp.arange(h_idx_start, h_idx_end) # shape (num_h,)
                v_indices = jnp.arange(v_idx_start, v_idx_end) # shape (num_v,)

                v_coords, h_coords = jnp.meshgrid(v_indices, h_indices, indexing='ij') # shapes (num_v, num_h)
                v_flat = v_coords.ravel() # shape (num_v * num_h,)
                h_flat = h_coords.ravel() # shape (num_v * num_h,)

                N = v_flat.shape[0] * ray_per_pixel
                v_repeated = jnp.repeat(v_flat, ray_per_pixel) # shape (N,), N = num_v * num_h * ray_per_pixel
                h_repeated = jnp.repeat(h_flat, ray_per_pixel) # shape (N,)

                pixel_indices = jnp.stack([v_repeated, h_repeated], axis=-1) # shape (N, 2)

                # Random offsets within each pixel
                random_offsets = jax.random.uniform(randkey, (N, 2), minval=0.0, maxval=1.0) # shape (N, 2)
                
                # Make sure the first ray in each pixel starts from the top-left corner
                random_offsets = random_offsets.reshape(
                    num_batch_vpixels, num_batch_hpixels, ray_per_pixel, 2
                )
                random_offsets = random_offsets.at[:, :, 0, :].set(0)
                random_offsets = random_offsets.reshape(N, 2)   # Reshape it back to (N, 2)

                rand_x_flat = random_offsets[:, 0] # shape (N,)
                rand_y_flat = random_offsets[:, 1] # shape (N,)

                # Calculate the 3D point on the viewport plane for each ray
                total_h_units = h_repeated + rand_x_flat # shape (N,)
                total_v_units = v_repeated + rand_y_flat # shape (N,)

                pixel_delta_x = cam.cam_x * (cam.viewport_width / cam.num_hpixels) # shape (3,)
                pixel_delta_y = cam.cam_y * (cam.viewport_height / cam.num_vpixels) # shape (3,)
                point_on_viewport = (
                    cam.viewport_origin[None, :]  # shape (1, 3) broadcast to (N, 3)
                    + total_h_units[:, None] * pixel_delta_x # shape (N, 1) * (3,) -> (N, 3)
                    + total_v_units[:, None] * pixel_delta_y   # shape (N, 1) * (3,) -> (N, 3)
                ) # => shape (N, 3)

                # Ray directions
                directions = point_on_viewport - cam.pos[None, :] # shape (N, 3)
                directions = unit_vector(directions) # shape (N, 3)

                ray = Ray(
                    origin=point_on_viewport, 
                    direction=directions, 
                    index=pixel_indices
                )
                ray.intensity /= ray_per_pixel # averaging factor because we sample ray_per_pixel rays
                yield ray
    

    def _get_diffuse_ray_without_batch(self, record: HitRecord, randkey = jax.random.key(0)) -> Ray:
        S = self.diffuse_sample_num

        # offset=-0.001 is the trick to avoid placing the ray origin inside the
        # object due to the numerical error.
        # No effect, disable it.
        source = record.ray.end_point() # shape (N, 3)
        normal = record.normal_vector   # shape (N, 3)

        diff_directions = utils.random_in_hemishpere(
            normal=normal,
            sample_count=S,
            randkey=randkey
        ) # shape (S, N, 3)
        origin = jnp.repeat(source[None, ...], S, axis=0) # shape (S, N, 3)
        index = jnp.repeat(record.ray.index[None, ...], S, axis=0) # shape (S, N, 2)

        ray = Ray(
            origin=origin.reshape(-1, 3), # shape (S * N, 3)
            direction=diff_directions.reshape(-1, 3), # shape (S * N, 3)
            index=index.reshape(-1, 2) # shape (S * N, 2)
        )
        intensity = jnp.repeat(record.ray.intensity[None, ...], S, axis=0) # shape (S, N, 3)
        ray.intensity = intensity.reshape(-1, 3) # shape (S * N, 3)

        return ray


    def _get_diffuse_ray(self, record: HitRecord, randkey = jax.random.key(0)) -> Iterable[Ray]:
        """
        Generate a diffuse ray from the hit record.
        """
        S = self.diffuse_sample_num

        # Calculate the intensity of the new rays
        intensity = (
            record.ray.intensity 
            * record.intensity_modu             # modulate the object color into the ray
            * record.diffuse_strength[:, None]  # antenuaion factor for diffuse reflection, broadcasted to (N, 3)
            / S                                 # averaging factor because we sample S rays
        )  # shape (N, 3)
        record.ray.intensity = intensity # write it back so we reuse the record.subset

        # Ingore the rays whose intensity is nearly 0
        mask = jnp.sum(intensity, axis=-1) > 0.001
        record = record.subset(mask)
        if len(record) == 0: 
            return # no rays to scatter
        
        # Split the rays into batches
        ray_per_batch = self.ray_batch_size // S
        for st_idx in range(0, len(intensity), ray_per_batch):
            ed_idx = min(st_idx + ray_per_batch, len(intensity))
            batch_record = record.slice(st_idx, ed_idx)
            yield self._get_diffuse_ray_without_batch(batch_record, randkey=randkey)


    def _get_specular_ray_without_batch(self, record: HitRecord, randkey = jax.random.key(0)) -> Ray:
        S = self.diffuse_sample_num

        origin = record.ray.end_point() # shape (N, 3)
        origin = jnp.repeat(origin[None, ...], S, axis=0) # shape (S, N, 3)

        normal = record.normal_vector   # shape (N, 3)
        incident_dot_normal = jnp.sum(record.ray.direction * normal, axis=-1) # shape (N,)
        reflected_directions = record.ray.direction - 2 * incident_dot_normal[:, None] * normal # shape (N, 3)
        new_directions = utils.random_in_disk(
            radius=record.specular_fuzz, # shape (N,)
            normal=reflected_directions, # shape (N, 3)
            sample_count=S,
            randkey=randkey
        )

        index = jnp.repeat(record.ray.index[None, ...], S, axis=0) # shape (S, N, 2)

        ray = Ray(
            origin=origin.reshape(-1, 3), # shape (S * N, 3)
            direction=new_directions.reshape(-1, 3), # shape (S * N, 3)
            index=index.reshape(-1, 2) # shape (S * N, 2)
        )

        intensity = jnp.repeat(record.ray.intensity[None, ...], S, axis=0) # shape (S, N, 3)
        ray.intensity = intensity.reshape(-1, 3) # shape (S * N, 3)

        return ray


    def _get_specular_ray(self, record: HitRecord, randkey = jax.random.key(0)) -> Iterable[Ray]:
        """
        Generate a specular ray from the hit record.
        """
        S = self.specular_sample_num

        # Calculate the intensity of the new rays
        intensity = (
            record.ray.intensity 
            * record.intensity_modu             # modulate the object color into the ray
            * record.specular_strength[:, None] # antenuaion factor for specular reflection, broadcasted to (N, 3)
            / S                                 # averaging factor because we sample S rays
        ) # shape (N, 3)
        record.ray.intensity = intensity        # write it back so we reuse the record.subset

        # Ingore the rays whose intensity is nearly 0
        mask = jnp.sum(intensity, axis=-1) > 0.001
        record = record.subset(mask)
        if len(record) == 0:
            return
        
        # Split the rays into batches
        ray_per_batch = self.ray_batch_size // S
        for st_idx in range(0, len(intensity), ray_per_batch):
            ed_idx = min(st_idx + ray_per_batch, len(intensity))
            batch_record = record.slice(st_idx, ed_idx)
            yield self._get_specular_ray_without_batch(batch_record, randkey=randkey)
    

    def _collect_rays(self, input_batches: Iterable[Ray]) -> Iterable[Ray]:
        """
        Merge small batches of rays into a fixed size (self.ray_batch_size)
        batch. Assuming that the input batches are always smaller than
        self.ray_batch_size.
        """
        buffer = []
        for batch in input_batches:
            buffer.append(batch)
            buffer_sz = sum(len(b) for b in buffer)
            if buffer_sz > self.ray_batch_size:  # require strictly larger
                merged_batch = Ray.concatenate(buffer)
                # assert len(merged_batch) == buffer_sz
                rtn = merged_batch.slice(0, self.ray_batch_size)
                yield rtn
                buffer = [merged_batch.slice(self.ray_batch_size, buffer_sz)]
        
        if len(buffer) > 0:
            merged_batch = Ray.concatenate(buffer)
            # assert len(merged_batch) == sum(len(b) for b in buffer)
            yield merged_batch


    def render(self) -> Image.Image:
        cam = self.cam
        maximum_ray_bounces = self.maximum_ray_bounces
        randkey_gen = utils.auto_split_randkey(jax.random.key(0))

        image = jnp.zeros((cam.num_vpixels, cam.num_hpixels, 3), dtype=jnp.float32) # shape (num_v, num_h, 3)
        
        start_time = datetime.now()

        next_round = [self._get_cam_ray()]
        round_count = 1
        while round_count <= maximum_ray_bounces and len(next_round) > 0:
            cur_round = next_round
            next_round = []

            last_ray_fetch_time = datetime.now()
            for ray in self._collect_rays(utils.iter_over_list_of_iterable(cur_round)):
                logger.info(f"Round {round_count}, ray count {len(ray)}")
                ray_fetch_eapsed_time = datetime.now() - last_ray_fetch_time
                last_ray_fetch_time = datetime.now()
                logger.info(f"Ray fetch time: {ray_fetch_eapsed_time}")
                
                # 1. Hit with all objects and record necessary information into 
                #    the record.
                record = HitRecord(ray)
                for obj in self.hitable_objects:
                    hit_start_at = datetime.now()
                    obj.hit(record)
                    hit_elapsed_time = datetime.now() - hit_start_at
                    from .objects import Mesh
                    if isinstance(obj, Mesh):
                        faces = obj.faces.shape[0]
                        logger.info(f"{len(ray)} rays x {faces} faces, time {hit_elapsed_time}")


                # 2. For rays that hit a light source, accumulate the color to
                #    the image. 
                hit_light = record.subset(record.hit_light_src)
                hit_light.ray.intensity = hit_light.ray.intensity * hit_light.intensity_modu
                image = hit_light.ray.accumulate_back_to_image(image)

                # 3. For rays that hit an object, scatter them and add them to
                #    the next round.
                if round_count == maximum_ray_bounces:
                    # If this is the last round, we don't need to scatter the rays
                    # that hit an object.
                    continue

                diffuse_obj = record.subset(
                    (record.ray.t < jnp.inf) & 
                    (~record.hit_light_src) & 
                    record.enable_diffuse
                )
                next_round.append(self._get_diffuse_ray(diffuse_obj, randkey=next(randkey_gen)))

                specular_obj = record.subset(
                    (record.ray.t < jnp.inf) & 
                    (~record.hit_light_src) & 
                    record.enable_specular
                )
                next_round.append(self._get_specular_ray(specular_obj, randkey=next(randkey_gen)))
            
            round_count += 1

        elapsed_time = datetime.now() - start_time
        logger.info(f"Rendering time: {elapsed_time}")

        image = (image * 255).clip(0, 255).astype(np.uint8) # shape (num_v, num_h, 3)
        image = np.asarray(image) # shape (num_v, num_h, 3)
        pil_image = Image.fromarray(image, mode='RGB')
        return pil_image