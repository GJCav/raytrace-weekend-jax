from myraytrace.objects import Mesh as MyMesh
import pyvista as pv
import numpy as np
import jax.numpy as jnp
from datetime import datetime, timedelta
import time

plotter = pv.Plotter()

# Show the sphere
myshpere = MyMesh.sphere(radius=0.5, center=[0, 2, 0], num_points=10)
# myshpere = MyMesh(
#     vertices=jnp.array([
#         [0, 0, 0],
#         [0, 1, 0],
#         [0, 1, 1],
#         [0, 2, 0],
#         [0, 4, 0],
#         [0, 3, 2]
#     ]),
#     faces=jnp.array([
#         [0, 1, 2],
#         [3, 4, 5]
#     ])
# )
faces = jnp.hstack(  # noqa: F722
    (jnp.full((myshpere.faces.shape[0], 1), 3, dtype=jnp.int32), myshpere.faces)
)
faces = faces.flatten()
sphere_mesh = pv.PolyData(
    np.asarray(myshpere.vertices),
    faces
)
plotter.add_mesh(sphere_mesh, color="lightblue", show_edges=True, edge_color="black")

# Show the normals
point_normals = myshpere.point_normals
plotter.add_arrows(
    np.asarray(myshpere.vertices),
    np.asarray(point_normals),
    color="red",
)


plotter.show_axes()  # Show the axes
plotter.show()

