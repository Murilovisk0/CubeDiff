import trimesh


def check_mesh_watertight(obj_path: str) -> bool:
    """
    Load mesh from .obj file and check if it's watertight,
    also report boundary edges if it's not.
    """
    mesh = trimesh.load(obj_path, force='mesh')
    print(f"Watertight: {mesh.is_watertight}")
    if not mesh.is_watertight:
        from trimesh.grouping import group_rows
        # find those edges appearing only once
        # edges_sorted: (E, 2) array of edge vertex indices, sorted within each edge
        unique_indices = group_rows(mesh.edges_sorted, require_count=1)
        print(unique_indices)
        print(f"Boundary edges count: {len(unique_indices)}")

    return mesh.is_watertight
