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


### 1. Load .obj file & Create a Meshes object ###


### 3. Optimization Loop ###



### 4. Visualized the loss ###


### 5. Save the predicted (deformed) mesh



