from myraytrace.raytrace import Camera, RaytraceRenderer
from myraytrace.objects import Mesh, Sky

cam = Camera(
    pos=[1, 0, 0],
    lookat=[0, 0, 0],
    cam_x=[0, 1, 0],
    hfov=60,
    focus_dist=0.1,
    num_hpixels=200
)

renderer = RaytraceRenderer(
    cam=cam,
    ray_per_pixel=20,
    maximum_ray_bounces=2,
    diffuse_sample_num=2,
    ray_batch_size=30000
)

sph1 = Mesh.sphere(radius=0.2, num_points=20)
sky = Sky()

renderer.hitable_objects.append(sph1)
renderer.hitable_objects.append(sky)

img = renderer.render()
img.save("output/single_sphere.png")
img.show()