from myraytrace.raytrace import Camera, RaytraceRenderer
from myraytrace.objects import Mesh, Sky

cam = Camera(
    pos=[2, 0, 0.2],
    lookat=[0, 0, 0],
    cam_x=[0, 1, 0],
    hfov=60,
    focus_dist=0.1,
    num_hpixels=720
)

renderer = RaytraceRenderer(
    cam=cam,
    ray_per_pixel=40,
    maximum_ray_bounces=4,
    diffuse_sample_num=1,
    specular_sample_num=1,
    ray_batch_size=100000,
)

sph1 = Mesh.sphere(radius=0.3, center=[0, 0, 0.3], num_points=20)
sph2 = Mesh.sphere(radius=5, center=[0, 0, -5.01], num_points=20)
sph2.point_color = [0.9, 0.9, 0.9] # light gray ground
sph2.enable_diffuse = True
sph2.enable_specular = True
sph2.specular_strength = 0.9
sph2.specular_fuzz = 0.03
sky = Sky()

renderer.hitable_objects.append(sph1)
renderer.hitable_objects.append(sph2)
renderer.hitable_objects.append(sky)

img = renderer.render()
img.save("output/02_two_sphere.jpg")
img.show()