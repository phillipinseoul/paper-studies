'''
    Tutorial from https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb
    We learn how to deform an initial generic shape (e.g. sphere) to fit a target shape.

    Details:
        - Load a mesh from an .obj file
        - Use the PyTorch3D `Meshes` data structure
        - Use four different PyTorch3D `mesh loss functions`
        - Set up an `optimization loop`
        - Try to minimize the `chamfer_distance` between `deformed mesh` and `target mesh`
        - Increase `smoothness` by using `shape regularizers` (i.e. mesh_edge_length, mesh_normal_consistency, mesh_laplacian_smoothing)
'''

### 0. Install and import modules ###
import os
from pickletools import optimize
import sys
import torch
need_pytorch3d = False

try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True

# Install PyTorch3D if needed
if need_pytorch3d:
    if torch.__version__.startswith("1.10.") and sys.platform.startswith("linux"):
        pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
        version_str = "".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".", ""),
            f"_pyt{pyt_version_str}"
        ])
        os.system("pip install fvcore iopath")
        os.system("pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html")
    else:
        os.system("curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz")
        os.system("tar xzf 1.10.0.tar.gz")
        os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
        os.system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")


# Now install and import necessary modules
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
import numpy as np
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

# Set the device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    print("WARNING: CPU only, this will be slow!")

### 1. Load .obj file & Create a Meshes object ###
# !wget https://dl.fbaipublicfiles.com/pytorch3d/data/dolphin/dolphin.obj

# Load the dolphin mesh
trg_obj = os.path.join('dolphin.obj')

# Read the target 3D model using `load_obj`
verts, faces, aux = load_obj(trg_obj)

# verts: FloatTensor of shape (V, 3), where V is `the number of vertices in the mesh`
# faces: Object that contains the following LongTensors: `verts_idx, normals_idx and textures_idx`
# For this tutorial, `normals` and `textures` are ignored.
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

# We `scale noralize` and `center` the target mesh to fit in a sphere of radius 1 centered at (0, 0, 0)
# (scale, center) will be used to bring the predicted mesh to its original center and scale.
center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale

# Construct a Meshes structure for the target mesh
trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

# Initialise the `source shape` to be a sphere of radius 1
src_mesh = ico_sphere(4, device)

# Visualize the source/target meshes
def plot_pointcloud(mesh, title=""):
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, auto_add_to_figure=True)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

# Show the target/source meshes
plot_pointcloud(trg_mesh, "Target mesh")
plot_pointcloud(src_mesh, "Source mesh")

### 3. Optimization Loop ###

# Deform the source mesh by `offsetting` its vertices.
# `Shape of the deform parameters` is the same as the `Total number of vertices in source mesh`
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

# Defince the Optimizer
optimizer = torch.optime.SGD([deform_verts], lr=1.0, momentum=0.9)

Niter = 2000        # number of optimization steps
w_chamfer = 1.0     # weight for chamfer loss
w_edge = 1.0        # weight for mesh edge loss
w_normal = 0.01     # weight for mesh normal consistency
w_laplacian = 0.1   # weight for mesh laplacian smoothing
plot_period = 250   # plot period for the losses

loop = tqdm(range(Niter))

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []


for i in loop:
    # Initialize the optimizer

    # Deform the mesh

    # We sample 5k points from the surface of each mesh

    # We compare the two sets of point clouds by computing (i) the chamfer loss

    # and (ii) the edge length of the predicted mesh

    # mesh normal consistency

    # mesh laplacian smoothing

    # weighted sum of the losses

    # print the losses

    # save the losses for plotting

    # plot the mesh






### 4. Visualized the loss ###


### 5. Save the predicted (deformed) mesh



