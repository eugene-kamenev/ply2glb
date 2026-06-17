# ply2glb

Convert 3D Gaussian Splat PLY files to textured GLB meshes.

Reads Gaussian Splat data (positions, scales, rotations, opacity, SH colour features), reconstructs a density field via splatting, and extracts a mesh with Marching Cubes. Colours are transferred from the original splats to the mesh vertices using neighbour interpolation.

## Requirements

- Python 3.9+
- [numba](https://numba.pydata.org/) (for JIT-compiled splatting)
- [numpy](https://numpy.org/)
- [trimesh](https://trimesh.org/)
- [scikit-image](https://scikit-image.org/) (Marching Cubes)
- [scipy](https://scipy.org/) (KD-tree colour transfer)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python ply2glb.py input.ply [output.glb] [options]
```

If `output.glb` is omitted, the output file is `<input>.glb`.

### Options

| Argument | Default | Description |
|---|---|---|
| `--resolution` | 512 | Voxel grid resolution per axis (512 or 1024 recommended) |
| `--threshold` | 0.3 | Density isosurface level; lower = thicker mesh, higher = thinner |
| `--padding` | 0.1 | Fractional padding around the point cloud bounding box |
| `--smooth` | none | Mesh smoothing method: `laplacian` (uniform) or `humphrey` (edge-preserving) |
| `--smooth_iterations` | 1 | Number of smoothing passes |
| `--humphrey_beta` | 0.1 | Edge-preservation strength for Humphrey smoothing |
| `--k_neighbors` | 3 | Nearest neighbours for colour transfer (1 = nearest only, 8 = smoother) |

### Examples

```bash
# Basic conversion
python ply2glb.py scene.ply

# Higher quality with smoothing
python ply2glb.py scene.ply --resolution 1024 --smooth laplacian --smooth_iterations 3

# Adjust density field
python ply2glb.py scene.ply --threshold 0.5 --padding 0.2

# Edge-preserving smoothing
python ply2glb.py scene.ply --smooth humphrey --humphrey_beta 0.5
```

## How it works

1. **PLY parsing** -- reads positions, scales (log-space), quaternion rotations, opacity (with sigmoid unfolding), and SH DC colour coefficients from Gaussian Splat PLY files (compatible with 3DGS, SuGaR, and similar exporters).
2. **Density field** -- each Gaussian splat is rasterised into a voxel grid using its precision matrix (inverse covariance) with a 4-sigma cutoff. The accumulation is JIT-compiled with Numba for performance.
3. **Marching Cubes** -- extracts an isosurface from the density field using scikit-image's `marching_cubes`.
4. **Colour transfer** -- vertex colours are interpolated from the nearest K source splats using inverse-distance weighting via a KD-tree.
5. **Export** -- the mesh is written as a binary GLB file with vertex colours and corrected normals.

## Input format

The script expects a PLY file with Gaussian Splat properties:

- `x`, `y`, `z` (float) -- position
- `scale_0`, `scale_1`, `scale_2` (float, log-space) -- anisotropic scale
- `rot_0`..`rot_3` (float) -- rotation as (w, x, y, z) quaternion
- `opacity` (float) -- opacity (raw logit or sigmoid)
- `f_dc_0`..`f_dc_2` (float) -- SH DC colour coefficients

Fallbacks exist for flat colour properties (`red`/`green`/`blue`, `diffuse_red`/`diffuse_green`/`diffuse_blue`) and alternative naming conventions (`scales_0`/`rots_0`/`features_dc_0`).

## License

MIT
