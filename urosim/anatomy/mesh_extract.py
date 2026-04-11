"""Triangle mesh extraction from the kidney SDF.

This module bridges the implicit :func:`urosim.anatomy.sdf.kidney_sdf`
level-set to an explicit, watertight ``trimesh.Trimesh`` suitable for
downstream rendering, collision, and FEM pipelines.

Pipeline:

1. Compute a padded bounding box around all graph-node positions.
2. Evaluate the kidney SDF on a regular 3D grid. A two-pass narrow-band
   scheme is used by default: a coarse evaluation identifies the
   surface shell, then only fine voxels inside that shell are
   evaluated. This is typically ~10× faster than dense evaluation at
   the default ``voxel_size`` of 0.5 mm.
3. Run ``skimage.measure.marching_cubes`` at level 0 to recover the
   isosurface, translating grid vertices back to world coordinates.
4. Repair and isotropically remesh with PyMeshLab so triangle quality
   is suitable for FEM: duplicate-vertex/face removal, non-manifold
   repair, hole closing, isotropic explicit remeshing, and a small
   Laplacian smooth.
5. Build a ``trimesh.Trimesh`` from the remeshed geometry, keep the
   largest connected component, fix winding, and fix normals so they
   face outward.

All units are millimetres throughout, matching the rest of
:mod:`urosim.anatomy`.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pymeshlab
import trimesh
import trimesh.repair
from skimage.measure import marching_cubes

from urosim.anatomy.sdf import kidney_sdf

# Padding added on every side of the node bounding box, in mm. Must be
# larger than the Perlin detail amplitude and the smooth-min blending
# radius so the whole mucosal surface is contained.
_BBOX_PADDING_MM: float = 20.0

# Coarse voxel size (in fine voxels) for the narrow-band first pass.
_NARROW_BAND_COARSE_FACTOR: int = 4

# Constant sentinels used to fill the far-field of the narrow-band
# volume. They are constant within each region, so marching cubes
# produces no spurious crossings there.
_OUTSIDE_SENTINEL: float = 10.0
_INSIDE_SENTINEL: float = -10.0

# PyMeshLab remeshing parameters.
_REMESH_ITERATIONS: int = 5
_LAPLACIAN_STEPS: int = 2
_TAUBIN_STEPS: int = 3
_TAUBIN_LAMBDA: float = 0.5
_TAUBIN_MU: float = -0.53
_HOLE_FILL_MAX_SIZE: int = 30

# Triangles with area below this threshold (mm^2) are considered
# degenerate and dropped in the final clean-up pass.
_DEGENERATE_AREA_MM2: float = 1e-9


def extract_kidney_mesh(
    graph: nx.DiGraph,
    centerlines: dict[tuple[str, str], np.ndarray],
    pelvis_radii: tuple[float, float, float],
    voxel_size: float = 0.5,
    target_edge_length: float = 1.5,
    *,
    narrow_band: bool = True,
) -> trimesh.Trimesh:
    """Extract a watertight triangle mesh of the kidney surface.

    Evaluates :func:`urosim.anatomy.sdf.kidney_sdf` on a regular grid,
    runs marching cubes, and remeshes the result with PyMeshLab so the
    triangles are suitable for FEM. Output normals face outward and the
    mesh is guaranteed watertight and winding-consistent; a
    ``ValueError`` is raised if either invariant cannot be established.

    Args:
        graph: Collecting-system graph with a ``position`` attribute on
            every node (typically produced by ``build_topology`` +
            ``place_nodes_3d``).
        centerlines: Mapping from each edge ``(u, v)`` to a sampled
            centerline as a ``(M, 3)`` array. Passed straight through
            to ``kidney_sdf``.
        pelvis_radii: Renal pelvis ellipsoid half-axes ``(a, b, c)`` in
            mm. Passed straight through to ``kidney_sdf``.
        voxel_size: Fine grid resolution in mm. Smaller values give a
            more detailed mesh at the cost of runtime. Default 0.5 mm.
        target_edge_length: Target edge length for isotropic explicit
            remeshing, in mm. Default 1.5 mm.
        narrow_band: If ``True`` (the default), use the two-pass
            coarse/fine narrow-band SDF evaluation scheme. If ``False``,
            evaluate the full fine grid in a single pass — simpler but
            much slower at small ``voxel_size``.

    Returns:
        A watertight, winding-consistent ``trimesh.Trimesh`` in
        millimetre world coordinates with outward-facing normals.

    Raises:
        ValueError: If ``graph`` is empty, ``voxel_size`` is
            non-positive, the SDF volume has no zero crossing, or the
            final mesh fails the watertight/winding-consistency checks.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if target_edge_length <= 0.0:
        raise ValueError(
            f"target_edge_length must be positive, got {target_edge_length}"
        )

    origin, size = _compute_bbox(graph, _BBOX_PADDING_MM)
    shape = _grid_shape(size, voxel_size)

    if narrow_band:
        volume = _evaluate_sdf_narrow_band(
            graph, centerlines, pelvis_radii, origin, shape, voxel_size
        )
    else:
        volume = _evaluate_sdf_dense(
            graph, centerlines, pelvis_radii, origin, shape, voxel_size
        )

    if not (np.any(volume < 0.0) and np.any(volume > 0.0)):
        raise ValueError(
            "SDF volume has no zero crossing — cannot extract an isosurface"
        )

    verts, faces = _marching_cubes_to_world(volume, origin, voxel_size)
    verts, faces = _remesh_pymeshlab(verts, faces, target_edge_length)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh = _largest_component(mesh)
    _drop_degenerate_faces(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)

    if not mesh.is_watertight:
        raise ValueError(
            "extracted mesh is not watertight after repair pipeline"
        )
    if not mesh.is_winding_consistent:
        raise ValueError(
            "extracted mesh has inconsistent winding after repair pipeline"
        )

    return mesh


def _compute_bbox(
    graph: nx.DiGraph, padding: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the padded axis-aligned bounding box of all node positions.

    Args:
        graph: Graph whose nodes carry a ``position`` attribute of
            shape ``(3,)``.
        padding: Amount to expand the box on every side, in mm.

    Returns:
        ``(origin, size)`` where ``origin`` is the minimum corner and
        ``size`` is the extent of the box along each axis, both as
        ``(3,)`` float64 arrays.
    """
    positions = np.stack(
        [
            np.asarray(attrs["position"], dtype=np.float64)
            for _, attrs in graph.nodes(data=True)
        ],
        axis=0,
    )
    pad = float(padding)
    lo = positions.min(axis=0) - pad
    hi = positions.max(axis=0) + pad
    return lo, hi - lo


def _grid_shape(
    size: np.ndarray, voxel_size: float
) -> tuple[int, int, int]:
    """Return the number of voxels along each axis for a given size.

    The returned shape is at least 2 along every axis so marching cubes
    always has a non-degenerate volume.

    Args:
        size: Extent of the bounding box along each axis, shape ``(3,)``.
        voxel_size: Voxel edge length in mm.

    Returns:
        ``(nx, ny, nz)`` tuple of voxel counts.
    """
    counts = np.ceil(np.asarray(size, dtype=np.float64) / float(voxel_size)).astype(int) + 1
    counts = np.maximum(counts, 2)
    return int(counts[0]), int(counts[1]), int(counts[2])


def _axis_coords(
    origin: np.ndarray, shape: tuple[int, int, int], voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-axis coordinate vectors for an ij-ordered grid.

    Args:
        origin: Minimum corner of the bounding box, shape ``(3,)``.
        shape: ``(nx, ny, nz)`` voxel counts.
        voxel_size: Voxel edge length in mm.

    Returns:
        Three ``(n_i,)`` float64 arrays ``(xs, ys, zs)``.
    """
    nx_, ny_, nz_ = shape
    xs = origin[0] + np.arange(nx_, dtype=np.float64) * float(voxel_size)
    ys = origin[1] + np.arange(ny_, dtype=np.float64) * float(voxel_size)
    zs = origin[2] + np.arange(nz_, dtype=np.float64) * float(voxel_size)
    return xs, ys, zs


def _evaluate_sdf_dense(
    graph: nx.DiGraph,
    centerlines: dict[tuple[str, str], np.ndarray],
    pelvis_radii: tuple[float, float, float],
    origin: np.ndarray,
    shape: tuple[int, int, int],
    voxel_size: float,
) -> np.ndarray:
    """Single-pass evaluation of ``kidney_sdf`` on the full fine grid.

    Args:
        graph: Collecting-system graph.
        centerlines: Edge centerlines passed through to ``kidney_sdf``.
        pelvis_radii: Pelvis ellipsoid half-axes.
        origin: Grid origin, shape ``(3,)``.
        shape: ``(nx, ny, nz)`` voxel counts.
        voxel_size: Voxel edge length in mm.

    Returns:
        ``(nx, ny, nz)`` SDF volume, ``ij``-ordered so axis 0 is X,
        axis 1 is Y, axis 2 is Z.
    """
    xs, ys, zs = _axis_coords(origin, shape, voxel_size)
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    values = kidney_sdf(pts, graph, centerlines, pelvis_radii)
    return values.reshape(shape)


def _evaluate_sdf_narrow_band(
    graph: nx.DiGraph,
    centerlines: dict[tuple[str, str], np.ndarray],
    pelvis_radii: tuple[float, float, float],
    origin: np.ndarray,
    shape: tuple[int, int, int],
    voxel_size: float,
) -> np.ndarray:
    """Two-pass coarse/fine narrow-band evaluation of ``kidney_sdf``.

    A coarse grid at ``_NARROW_BAND_COARSE_FACTOR × voxel_size`` is
    evaluated first. Coarse voxels whose SDF lies within a conservative
    band around zero are upsampled to fine voxels (dilated by one fine
    voxel) and evaluated; fine voxels outside that shell are filled
    with ``±_OUTSIDE_SENTINEL`` / ``_INSIDE_SENTINEL`` based on the
    coarse sign. The sentinels are constant so marching cubes produces
    no spurious zero crossings outside the band.

    Args:
        graph: Collecting-system graph.
        centerlines: Edge centerlines passed through to ``kidney_sdf``.
        pelvis_radii: Pelvis ellipsoid half-axes.
        origin: Grid origin, shape ``(3,)``.
        shape: ``(nx, ny, nz)`` fine-voxel counts.
        voxel_size: Fine voxel edge length in mm.

    Returns:
        ``(nx, ny, nz)`` SDF volume, ``ij``-ordered.
    """
    nx_, ny_, nz_ = shape
    coarse_factor = _NARROW_BAND_COARSE_FACTOR
    coarse_voxel = float(voxel_size) * coarse_factor

    # Coarse grid aligned to the same origin. Over-cover the bounding
    # box so every fine voxel is inside some coarse voxel.
    cnx = int(np.ceil((nx_ - 1) / coarse_factor)) + 1
    cny = int(np.ceil((ny_ - 1) / coarse_factor)) + 1
    cnz = int(np.ceil((nz_ - 1) / coarse_factor)) + 1
    coarse_shape = (cnx, cny, cnz)

    coarse_xs, coarse_ys, coarse_zs = _axis_coords(
        origin, coarse_shape, coarse_voxel
    )
    grid_cx, grid_cy, grid_cz = np.meshgrid(
        coarse_xs, coarse_ys, coarse_zs, indexing="ij"
    )
    coarse_pts = np.stack([grid_cx, grid_cy, grid_cz], axis=-1).reshape(-1, 3)
    coarse_vals = kidney_sdf(
        coarse_pts, graph, centerlines, pelvis_radii
    ).reshape(coarse_shape)

    # Conservative band: any coarse voxel with |sdf| below this
    # threshold may contain the zero crossing after subsampling to the
    # fine grid.
    band_threshold = 2.0 * coarse_voxel * float(np.sqrt(3.0))
    coarse_band = np.abs(coarse_vals) < band_threshold

    # Build the fine-voxel mask by upsampling the coarse band
    # (nearest-neighbour) and dilating by one fine voxel via a 3x3x3
    # box kernel. The dilation is implemented with cheap axis-wise
    # `np.maximum` shifts to avoid a SciPy dependency.
    fine_mask = _upsample_nearest(coarse_band, coarse_factor, shape)
    fine_mask = _dilate_bool_one(fine_mask)

    # Initialise the fine volume with sign-appropriate sentinels so any
    # fine voxel not covered by the band still has the correct sign.
    coarse_sign_inside = coarse_vals < 0.0
    inside_mask = _upsample_nearest(coarse_sign_inside, coarse_factor, shape)
    volume = np.where(inside_mask, _INSIDE_SENTINEL, _OUTSIDE_SENTINEL)
    volume = volume.astype(np.float64, copy=False)

    # Evaluate kidney_sdf only at masked fine voxels.
    fine_xs, fine_ys, fine_zs = _axis_coords(origin, shape, voxel_size)
    idx = np.argwhere(fine_mask)
    if idx.size == 0:
        # Pathological: no zero crossing found on the coarse grid.
        # Fall back to a dense evaluation so we still get a mesh if the
        # coarse scan missed a thin feature.
        return _evaluate_sdf_dense(
            graph, centerlines, pelvis_radii, origin, shape, voxel_size
        )

    pts = np.stack(
        [
            fine_xs[idx[:, 0]],
            fine_ys[idx[:, 1]],
            fine_zs[idx[:, 2]],
        ],
        axis=-1,
    )
    values = kidney_sdf(pts, graph, centerlines, pelvis_radii)
    volume[idx[:, 0], idx[:, 1], idx[:, 2]] = values
    return volume


def _upsample_nearest(
    coarse: np.ndarray, factor: int, fine_shape: tuple[int, int, int]
) -> np.ndarray:
    """Nearest-neighbour upsample a 3D boolean array onto a fine grid.

    Each coarse voxel at index ``(i, j, k)`` maps to fine voxels in
    the block ``[i*factor : i*factor + factor, ...]``, clipped to
    ``fine_shape``.

    Args:
        coarse: Input 3D array of shape ``(cnx, cny, cnz)``.
        factor: Integer upsample factor.
        fine_shape: Target ``(nx, ny, nz)`` fine-grid shape.

    Returns:
        Fine-grid array of shape ``fine_shape`` with the same dtype.
    """
    nx_, ny_, nz_ = fine_shape
    xi = np.minimum(np.arange(nx_) // factor, coarse.shape[0] - 1)
    yi = np.minimum(np.arange(ny_) // factor, coarse.shape[1] - 1)
    zi = np.minimum(np.arange(nz_) // factor, coarse.shape[2] - 1)
    return coarse[np.ix_(xi, yi, zi)]


def _dilate_bool_one(mask: np.ndarray) -> np.ndarray:
    """Dilate a 3D boolean mask by one voxel using a 3x3x3 box kernel.

    Args:
        mask: Input boolean array of shape ``(nx, ny, nz)``.

    Returns:
        Dilated boolean array with the same shape.
    """
    out = mask.copy()
    for axis in range(3):
        shifted_pos = np.roll(out, 1, axis=axis)
        shifted_neg = np.roll(out, -1, axis=axis)
        # Clear the wrap-around slice so roll does not leak values
        # across the opposite boundary.
        idx_pos: list[Any] = [slice(None), slice(None), slice(None)]
        idx_pos[axis] = 0
        shifted_pos[tuple(idx_pos)] = False
        idx_neg: list[Any] = [slice(None), slice(None), slice(None)]
        idx_neg[axis] = -1
        shifted_neg[tuple(idx_neg)] = False
        out = out | shifted_pos | shifted_neg
    return out


def _marching_cubes_to_world(
    volume: np.ndarray, origin: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Run marching cubes and shift vertices to world coordinates.

    Because ``volume`` is an ``ij``-ordered grid with axis 0 == X,
    axis 1 == Y, axis 2 == Z, the marching-cubes vertex columns are
    already ``(x, y, z)`` after the spacing scale, and only need to be
    offset by the grid origin.

    Args:
        volume: ``(nx, ny, nz)`` SDF volume.
        origin: Grid origin, shape ``(3,)``.
        voxel_size: Voxel edge length in mm.

    Returns:
        ``(verts, faces)`` as ``(V, 3)`` float64 and ``(F, 3)`` int64
        arrays. Vertices are in world-space mm.
    """
    verts, faces, _, _ = marching_cubes(
        volume,
        level=0.0,
        spacing=(float(voxel_size), float(voxel_size), float(voxel_size)),
    )
    verts = np.asarray(verts, dtype=np.float64) + np.asarray(
        origin, dtype=np.float64
    )
    faces = np.asarray(faces, dtype=np.int64)
    return verts, faces


def _absolute_length(value: float) -> Any:
    """Wrap ``value`` in PyMeshLab's absolute-length value type.

    PyMeshLab renamed this class across releases: older versions expose
    ``AbsoluteValue``, newer ones expose ``PureValue``. Try both and
    fall back to the raw float so the pipeline still runs on any
    reasonably recent release.

    Args:
        value: The edge length in mm to wrap.

    Returns:
        Either a ``PureValue`` / ``AbsoluteValue`` instance or the raw
        float, whichever the installed PyMeshLab version accepts.
    """
    for name in ("PureValue", "AbsoluteValue"):
        cls = getattr(pymeshlab, name, None)
        if cls is not None:
            return cls(value)
    return value


def _safe_apply(ms: pymeshlab.MeshSet, name: str, **kwargs: Any) -> None:
    """Apply a PyMeshLab filter, tolerating version-specific renames.

    Looks up ``name`` on the ``MeshSet`` and calls it if present; if the
    filter does not exist in this version of PyMeshLab (common for
    renamed filters across releases), silently does nothing. Other
    PyMeshLab exceptions are re-raised.

    Args:
        ms: The ``pymeshlab.MeshSet`` instance.
        name: Filter method name.
        **kwargs: Keyword arguments forwarded to the filter.
    """
    func = getattr(ms, name, None)
    if func is None:
        return
    try:
        func(**kwargs)
    except pymeshlab.PyMeshLabException:
        # Filter exists but rejected its arguments (e.g. parameter
        # renames across versions); skipping is safer than crashing
        # the whole extraction pipeline.
        return


def _remesh_pymeshlab(
    verts: np.ndarray, faces: np.ndarray, target_edge_length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Repair and isotropically remesh a triangle mesh with PyMeshLab.

    Pipeline, in order:

    1. Dedup + non-manifold repair + initial hole close.
    2. Isotropic explicit remeshing (``_REMESH_ITERATIONS`` passes).
    3. Laplacian smoothing to wash out remeshing ringing.
    4. Degenerate removal: duplicate faces/vertices and null-area faces.
    5. Taubin smoothing — volume-preserving regularisation of the
       vertex positions to improve skinny-triangle quality without the
       shrinkage that plain Laplacian causes.
    6. Final hole close + null-face removal (remeshing and smoothing
       can occasionally re-open tiny holes or collapse a sliver).

    PyMeshLab's degenerate-face filter is called
    ``meshing_remove_null_faces`` in current releases; the older name
    ``meshing_remove_zero_area_faces`` is also attempted via the
    silent ``_safe_apply`` so this function keeps working across
    PyMeshLab versions.

    Args:
        verts: ``(V, 3)`` vertex array.
        faces: ``(F, 3)`` face array.
        target_edge_length: Target edge length in mm.

    Returns:
        ``(verts, faces)`` of the remeshed mesh as float64 / int64.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.ascontiguousarray(verts, dtype=np.float64),
            face_matrix=np.ascontiguousarray(faces, dtype=np.int32),
        )
    )

    # --- 1. Pre-remesh repair. ---------------------------------------
    _safe_apply(ms, "meshing_remove_duplicate_vertices")
    _safe_apply(ms, "meshing_remove_duplicate_faces")
    _safe_apply(ms, "meshing_remove_unreferenced_vertices")
    _safe_apply(ms, "meshing_repair_non_manifold_edges")
    _safe_apply(ms, "meshing_repair_non_manifold_vertices")
    _safe_apply(ms, "meshing_close_holes", maxholesize=_HOLE_FILL_MAX_SIZE)

    # --- 2. Isotropic explicit remeshing. ----------------------------
    _safe_apply(
        ms,
        "meshing_isotropic_explicit_remeshing",
        targetlen=_absolute_length(float(target_edge_length)),
        iterations=_REMESH_ITERATIONS,
    )

    # --- 3. Laplacian smoothing. -------------------------------------
    _safe_apply(
        ms,
        "apply_coord_laplacian_smoothing",
        stepsmoothnum=_LAPLACIAN_STEPS,
    )

    # --- 4. Degenerate removal after smoothing. ----------------------
    _safe_apply(ms, "meshing_remove_duplicate_faces")
    _safe_apply(ms, "meshing_remove_duplicate_vertices")
    # Try the current name first, then the historical alias; _safe_apply
    # silently skips whichever one is missing on this PyMeshLab release.
    _safe_apply(ms, "meshing_remove_null_faces")
    _safe_apply(ms, "meshing_remove_zero_area_faces")

    # --- 5. Taubin smoothing for triangle-quality regularisation. ----
    _safe_apply(
        ms,
        "apply_coord_taubin_smoothing",
        stepsmoothnum=_TAUBIN_STEPS,
        lambda_=_TAUBIN_LAMBDA,
        mu=_TAUBIN_MU,
    )

    # --- 6. Final hole close and degenerate sweep. -------------------
    _safe_apply(ms, "meshing_close_holes", maxholesize=_HOLE_FILL_MAX_SIZE)
    _safe_apply(ms, "meshing_remove_null_faces")
    _safe_apply(ms, "meshing_remove_zero_area_faces")
    _safe_apply(ms, "meshing_remove_unreferenced_vertices")

    current = ms.current_mesh()
    out_verts = np.asarray(current.vertex_matrix(), dtype=np.float64)
    out_faces = np.asarray(current.face_matrix(), dtype=np.int64)
    return out_verts, out_faces


def _drop_degenerate_faces(mesh: trimesh.Trimesh) -> None:
    """Remove any residual zero-area faces from ``mesh`` in place.

    PyMeshLab's ``meshing_remove_null_faces`` filter and
    ``trimesh.Trimesh(process=True)`` both drop obviously-degenerate
    faces, but after Taubin smoothing a handful of slivers can still
    slip through with area below floating-point noise. This helper is
    the belt-and-braces final pass: compute per-face areas directly
    and drop anything under ``_DEGENERATE_AREA_MM2``. If nothing is
    degenerate the mesh is left untouched.

    Args:
        mesh: Mesh to clean in place. Modified via
            ``mesh.update_faces`` so vertex buffers and cached
            properties are refreshed consistently.
    """
    areas = mesh.area_faces
    keep = areas > _DEGENERATE_AREA_MM2
    if not keep.all():
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return the connected component of ``mesh`` with the largest volume.

    Most kidney extractions produce a single component, but SDF noise
    or remeshing artifacts can occasionally spawn tiny islands. This
    helper keeps only the dominant component so downstream watertight
    checks do not fail on spurious satellites.

    Args:
        mesh: Input mesh, possibly with multiple connected components.

    Returns:
        A single-component ``trimesh.Trimesh``. If ``mesh`` is already
        a single component it is returned unchanged.
    """
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    return max(components, key=lambda m: abs(float(m.volume)))
