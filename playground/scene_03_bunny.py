from myraytrace.raytrace import Camera, RaytraceRenderer
from myraytrace.objects import Mesh, Sky
import importlib.resources

cam = Camera(
    pos=[1.7, 0, 0.5],
    lookat=[0, 0, 0],
    cam_x=[0, 1, 0],
    hfov=60,
    focus_dist=0.1,
    num_hpixels=360
)

renderer = RaytraceRenderer(
    cam=cam,
    ray_per_pixel=1,
    maximum_ray_bounces=2,
    diffuse_sample_num=1,
    specular_sample_num=1,
    ray_batch_size=80000,
)

sph1 = Mesh.sphere(radius=0.3, center=[0.1, 0.4, 0.3], num_points=20)
sph1.point_color = [0.8, 0.2, 0.2] # red
sph1.diffuse_strength = 0.3
sph1.enable_specular = True
sph1.specular_strength = 0.8
sph1.specular_fuzz = 0.0001

sph2 = Mesh.sphere(radius=5, center=[0, 0, -5.01], num_points=20)
sph2.point_color = [0.7, 0.7, 0.7] # light gray ground
sph2.enable_diffuse = True
sph2.enable_specular = True
sph2.specular_strength = 0.8
sph2.specular_fuzz = 0.02

sph3 = Mesh.sphere(
    radius=0.3, center=[0.1, -0.4, 0.2], num_points=20,
    uv_material_path=importlib.resources.files("myraytrace") / "Sepia-Rainbow-map.jpg"
)
sph3.diffuse_strength = 0.8

bunny = Mesh.bunny(center=[0.6, -0.3, -0], scale=2)
bunny.diffuse_strength = 0.9
bunny.enable_specular = False

ball_light = Mesh.sphere(radius=0.1, center=[0.6, 0.13, 0.01], num_points=10)
ball_light.point_color = [0.96398, 0.84948, 0.25889] # yellow
ball_light.light_source = True
ball_light.enable_diffuse = False
ball_light.enable_specular = False

light2 = Mesh.sphere(radius=0.1, center=[0.5, -0.1, 0.5], num_points=10)
light2.point_color = [0.9, 0.9, 0.9] # white
light2.light_source = True

sky = Sky()

renderer.hitable_objects.append(sph1)
renderer.hitable_objects.append(sph2)
renderer.hitable_objects.append(sph3)
renderer.hitable_objects.append(bunny)
renderer.hitable_objects.append(ball_light)
renderer.hitable_objects.append(light2)
renderer.hitable_objects.append(sky)

img = renderer.render()
img.save("output/03_bunny.jpg")
# img.show()