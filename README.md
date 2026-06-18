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
| `--threshold` | 0.1 | Density isosurface level; lower = thicker mesh, higher = thinner |
| `--padding` | 0.1 | Fractional padding around the point cloud bounding box |
| `--smooth` | none | Mesh smoothing method: `laplacian` (uniform) or `humphrey` (edge-preserving) |
| `--smooth_iterations` | 1 | Number of smoothing passes |
| `--humphrey_beta` | 0.1 | Edge-preservation strength for Humphrey smoothing |
| `--k_neighbors` | 3 | Nearest neighbours for colour transfer (1 = nearest only, 8 = smoother) |
| `--texture_size` | 2048 | Texture resolution for GLB baking (requires cumesh) |
| `--decimation_target` | 100000 | Target face count for mesh simplification (requires cumesh) |
| `--remesh` | false | Perform remeshing using Dual Contouring (requires cumesh) |
| `--no_to_glb` | false | Disable CUDA-based textured export; use trimesh vertex-colour export |
| `--rotation` | `gs_to_glb` | Coordinate transform preset: `gs_to_glb` (3DGS→GLB) or `none` |
| `--verbose` | false | Enable verbose logging |

## How it works

1. **PLY parsing** -- reads positions, scales (log-space), quaternion rotations, opacity (with sigmoid unfolding), and SH DC colour coefficients from Gaussian Splat PLY files (compatible with 3DGS, SuGaR, and similar exporters).
2. **Density field** -- each Gaussian splat is rasterised into a voxel grid using its precision matrix (inverse covariance) with a 4-sigma cutoff. The accumulation is JIT-compiled with Numba for performance.
3. **Marching Cubes** -- extracts an isosurface from the density field using scikit-image's `marching_cubes`.
4. **Colour transfer** -- vertex colours are interpolated from the nearest K source splats using inverse-distance weighting via a KD-tree.
5. **Export** -- the mesh is written as a binary GLB file with vertex colours and corrected normals.

## Post-processing (CUDA textured export)

By default, the script uses a CUDA-based pipeline (via [cumesh](https://github.com/nv-tlabs/cumesh), nvdiffrast, and flex_gemm) adapted from [TRELLIS.2's o-voxel](https://github.com/microsoft/TRELLIS.2/blob/main/o-voxel/o_voxel/postprocess.py). This produces a GLB with baked PBR textures (base colour, metallic, roughness, alpha) and UV coordinates, rather than per-vertex colours.

The pipeline:
1. Bakes the splat colours into a texture atlas (`--texture_size`, default 2048)
2. Simplifies the mesh to a target face count (`--decimation_target`, default 100k)
3. Optionally runs Dual Contouring remeshing (`--remesh`) for better topology
4. Assigns a PBR material (metallic=0, roughness=1)

Use `--no_to_glb` to skip the CUDA path and fall back to trimesh vertex-colour export (no CUDA dependencies needed).

The included Docker image (`docker compose up -d`) provides the required CUDA dependencies.

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
