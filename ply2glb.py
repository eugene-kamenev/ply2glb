#!/usr/bin/env python3
"""
High‑quality 3D Gaussian Splat to Mesh conversion.
@author: Eugene Kamenev
"""

import argparse
import os
import time
import numpy as np
import trimesh as Trimesh
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import numba

SH_C0 = 0.28209479177387814

# ----------------------------------------------------------------------
@numba.njit
def compute_prec_mats(rotations, scales):
    n = rotations.shape[0]
    prec_mats = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        q = rotations[i]
        norm = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
        if norm < 1e-8:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            q = q / norm
        w, x, y, z = q[0], q[1], q[2], q[3]

        r00 = 1 - 2*y*y - 2*z*z
        r01 = 2*x*y - 2*z*w
        r02 = 2*x*z + 2*y*w
        r10 = 2*x*y + 2*z*w
        r11 = 1 - 2*x*x - 2*z*z
        r12 = 2*y*z - 2*x*w
        r20 = 2*x*z - 2*y*w
        r21 = 2*y*z + 2*x*w
        r22 = 1 - 2*x*x - 2*y*y

        s = scales[i]
        s0 = 1.0 / (s[0]*s[0] + 1e-10)
        s1 = 1.0 / (s[1]*s[1] + 1e-10)
        s2 = 1.0 / (s[2]*s[2] + 1e-10)

        m00, m01, m02 = r00*s0, r01*s1, r02*s2
        m10, m11, m12 = r10*s0, r11*s1, r12*s2
        m20, m21, m22 = r20*s0, r21*s1, r22*s2

        prec_mats[i, 0, 0] = m00*r00 + m01*r01 + m02*r02
        prec_mats[i, 0, 1] = m00*r10 + m01*r11 + m02*r12
        prec_mats[i, 0, 2] = m00*r20 + m01*r21 + m02*r22
        prec_mats[i, 1, 0] = prec_mats[i, 0, 1]
        prec_mats[i, 1, 1] = m10*r10 + m11*r11 + m12*r12
        prec_mats[i, 1, 2] = m10*r20 + m11*r21 + m12*r22
        prec_mats[i, 2, 0] = prec_mats[i, 0, 2]
        prec_mats[i, 2, 1] = prec_mats[i, 1, 2]
        prec_mats[i, 2, 2] = m20*r20 + m21*r21 + m22*r22
    return prec_mats

# ----------------------------------------------------------------------
@numba.njit
def accumulate_serial(points, indices, prec_mats, weights, radii_vox, resolution, voxel_size, grid):
    for i in range(points.shape[0]):
        idx = indices[i]
        r = radii_vox[i]
        if r <= 0:
            continue
        prec = prec_mats[i]
        w = weights[i]

        x_min = max(0, idx[0] - r)
        x_max = min(resolution - 1, idx[0] + r)
        y_min = max(0, idx[1] - r)
        y_max = min(resolution - 1, idx[1] + r)
        z_min = max(0, idx[2] - r)
        z_max = min(resolution - 1, idx[2] + r)

        for x in range(x_min, x_max + 1):
            dx = (x - idx[0]) * voxel_size[0]
            for y in range(y_min, y_max + 1):
                dy = (y - idx[1]) * voxel_size[1]
                for z in range(z_min, z_max + 1):
                    dz = (z - idx[2]) * voxel_size[2]
                    # d2 = v^T Prec v
                    d2 = (dx * prec[0, 0] + dy * prec[1, 0] + dz * prec[2, 0]) * dx + \
                         (dx * prec[0, 1] + dy * prec[1, 1] + dz * prec[2, 1]) * dy + \
                         (dx * prec[0, 2] + dy * prec[1, 2] + dz * prec[2, 2]) * dz
                    if d2 < 16.0:   # 4‑sigma cutoff
                        grid[x, y, z] += w * np.exp(-0.5 * d2)

# ----------------------------------------------------------------------
def _fast_density_field_numba(points, scales, rotations, opacity,
                              resolution=256, padding=0.1):
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = pmax - pmin
    pad = extent * padding
    pmin -= pad
    pmax += pad
    extent = pmax - pmin

    voxel_size = (extent / resolution).astype(np.float32)
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    indices = ((points - pmin) / voxel_size).astype(np.int32)
    indices = np.clip(indices, 0, resolution - 1)

    print("  Computing precision matrices...")
    prec_mats = compute_prec_mats(rotations.astype(np.float32), scales.astype(np.float32))

    # Radius in voxels: 4 * max(scale) / min(voxel_size) (captures elongated splats)
    radii_vox = np.ceil(4.0 * np.max(scales, axis=1) / np.min(voxel_size)).astype(np.int32)
    radii_vox = np.clip(radii_vox, 1, 32)

    weights = opacity.astype(np.float32)

    print(f"  Splatting {len(points)} points into {resolution}^3 grid "
          f"(max radius {radii_vox.max()} voxels)...")
    t0 = time.time()
    accumulate_serial(points, indices, prec_mats, weights, radii_vox, resolution, voxel_size, grid)
    print(f"  Splatting took {time.time()-t0:.1f} seconds")

    return grid, pmin, extent

# ----------------------------------------------------------------------
def _transfer_colors_to_vertices(vertices, source_points, source_colors, k=8):
    if k == 1:
        tree = cKDTree(source_points)
        _, idxs = tree.query(vertices, k=1)
        return (np.clip(source_colors[idxs], 0, 1) * 255).astype(np.uint8)

    tree = cKDTree(source_points)
    dists, idxs = tree.query(vertices, k=k)
    weights = 1.0 / (dists + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)
    vert_colors = np.einsum('nk,nkc->nc', weights, source_colors[idxs])
    vert_colors = np.clip(vert_colors, 0.0, 1.0)
    return (vert_colors * 255).astype(np.uint8)

# ----------------------------------------------------------------------
def _read_ply_gaussian(filepath):
    """Robust PLY reader for Gaussian Splats."""
    with open(filepath, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header': break

    num_verts = 0
    props = []
    for line in header:
        if line.startswith('element vertex'):
            num_verts = int(line.split()[-1])
        elif line.startswith('property'):
            props.append(line.split()[-1])

    dtype = []
    ply_to_np = {
        'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
        'uchar': 'u1', 'uint8': 'u1', 'int': 'i4', 'int32': 'i4'
    }
    for line in header:
        if line.startswith('property'):
            parts = line.split()
            dtype.append((parts[-1], ply_to_np.get(parts[1], 'f4')))

    header_size = 0
    with open(filepath, 'rb') as f:
        for line in header:
            header_size += len(line) + 1
        f.seek(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype(dtype), count=num_verts)

    points = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)

    def get_prop(names, default_val=None):
        for name in names:
            if name in data.dtype.names:
                return data[name]
        return default_val

    s0 = get_prop(['scale_0', 'scales_0'])
    s1 = get_prop(['scale_1', 'scales_1'])
    s2 = get_prop(['scale_2', 'scales_2'])
    if s0 is not None:
        scales = np.exp(np.stack([s0, s1, s2], axis=1)).astype(np.float64)
    else:
        scales = np.ones((num_verts, 3)) * 0.01

    r0 = get_prop(['rot_0', 'rots_0'])
    r1 = get_prop(['rot_1', 'rots_1'])
    r2 = get_prop(['rot_2', 'rots_2'])
    r3 = get_prop(['rot_3', 'rots_3'])
    if r0 is not None:
        rots = np.stack([r0, r1, r2, r3], axis=1).astype(np.float64)
    else:
        rots = np.zeros((num_verts, 4))
        rots[:, 0] = 1.0

    op = get_prop(['opacity'])
    if op is not None:
        op = op.astype(np.float64)
        if op.min() < 0 or op.max() > 1:
            opacity = 1.0 / (1.0 + np.exp(-op))
        else:
            opacity = op
    else:
        opacity = np.ones(num_verts)

    f0 = get_prop(['f_dc_0', 'features_dc_0'])
    f1 = get_prop(['f_dc_1', 'features_dc_1'])
    f2 = get_prop(['f_dc_2', 'features_dc_2'])
    if f0 is not None:
        r = 0.5 + SH_C0 * f0
        g = 0.5 + SH_C0 * f1
        b = 0.5 + SH_C0 * f2
        colors = np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0).astype(np.float64)
    else:
        red = get_prop(['red', 'diffuse_red'])
        green = get_prop(['green', 'diffuse_green'])
        blue = get_prop(['blue', 'diffuse_blue'])
        if red is not None:
            colors = np.stack([red, green, blue], axis=1).astype(np.float64)
            if colors.max() > 1.0: colors /= 255.0
        else:
            colors = np.ones((num_verts, 3)) * 0.7

    return points, colors, scales, rots, opacity

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D Gaussian Splat PLY files to textured GLB meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s scene.ply                          # output scene.glb\n"
            "  %(prog)s scene.ply out.glb --resolution 1024\n"
            "  %(prog)s scene.ply --smooth laplacian --smooth_iterations 3\n"
            "  %(prog)s scene.ply --threshold 0.5 --padding 0.2\n"
        )
    )
    parser.add_argument("input_ply",
                        help="Path to input Gaussian Splat PLY file (e.g. from 3DGS, SuGaR, or similar)")
    parser.add_argument("output_glb", nargs="?",
                        help="Output GLB file path (default: <input>.glb)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Voxel grid resolution per axis (default: %(default)s; 512 or 1024 recommended)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Density isosurface level (default: %(default)s; lower = thicker, higher = thinner)")
    parser.add_argument("--padding", type=float, default=0.1,
                        help="Fractional padding around point cloud bounding box (default: %(default)s)")
    parser.add_argument("--smooth", type=str, default="none",
                        choices=["none", "laplacian", "humphrey"],
                        help="Mesh smoothing method: laplacian = uniform, humphrey = edge-preserving (default: %(default)s)")
    parser.add_argument("--smooth_iterations", type=int, default=1,
                        help="Number of smoothing iterations (default: %(default)s)")
    parser.add_argument("--humphrey_beta", type=float, default=0.1,
                        help="Edge-preservation strength for Humphrey smoothing (default: %(default)s; 0 = no effect, higher = more edge retention)")
    parser.add_argument("--k_neighbors", type=int, default=3,
                        help="Nearest neighbours for colour transfer (default: %(default)s; 1 = nearest only, 8 = smoother blending)")
    args = parser.parse_args()

    if not args.output_glb:
        args.output_glb = os.path.splitext(args.input_ply)[0] + ".glb"

    print(f"Loading {args.input_ply}...")
    points, colors, scales, rots, opacity = _read_ply_gaussian(args.input_ply)

    print(f"Generating density field (res={args.resolution})...")
    grid, pmin, extent = _fast_density_field_numba(
        points, scales, rots, opacity,
        resolution=args.resolution,
        padding=args.padding
    )

    # --- threshold selection ---
    thresh = args.threshold
    print(f"Using manual threshold: {thresh:.4f}")

    # --- marching cubes ---
    print("Running Marching Cubes...")
    verts, faces, _, _ = marching_cubes(grid, level=thresh)

    voxel_size = extent / args.resolution
    verts = verts * voxel_size + pmin

    print(f"Transferring colours to {len(verts)} vertices (k={args.k_neighbors})...")
    v_colors = _transfer_colors_to_vertices(verts, points, colors, k=args.k_neighbors)

    v_colors_rgba = np.column_stack([v_colors, np.full(len(v_colors), 255, dtype=np.uint8)])

    mesh = Trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_colors=v_colors_rgba, process=True)
    mesh.fix_normals()
    # --- optional smoothing ---
    if args.smooth != "none":
        print(f"Applying {args.smooth} smoothing ({args.smooth_iterations} iter)...")
        if args.smooth == "laplacian":
            mesh = Trimesh.smoothing.filter_laplacian(mesh, iterations=args.smooth_iterations)
        elif args.smooth == "humphrey":
            mesh = Trimesh.smoothing.filter_humphrey(
                mesh, iterations=args.smooth_iterations, beta=args.humphrey_beta
            )

    print(f"Exporting to {args.output_glb}...")
    mesh.export(args.output_glb)
    print("Done!")

if __name__ == "__main__":
    main()
