# Project Context — mlx_sam3

## Overview

This project integrates **D4M/AA (Dynamic Distributed Dimensional Data Model / Associative Arrays)**
with **GraphBLAS** to manipulate Blender mesh topology using sparse linear algebra.

The goal is to represent Blender mesh elements (vertices, edges, faces) as associative arrays,
perform graph-algebraic operations on them via GraphBLAS, and write results back into Blender.

---

## D4M / Associative Arrays

D4M was developed by Jeremy Kepner et al. at MIT Lincoln Laboratory.
The central object is the **Associative Array (AA)** — a sparse matrix whose rows and columns
are labelled with arbitrary strings rather than integer indices.

### Key properties
- Row and column keys are strings (e.g. `"v:001"`, `"e:042"`, `"f:117"`)
- Values can be numeric or string
- Supports standard linear-algebra operations (multiply, add, transpose) via semirings
- Generalises both relational databases and sparse matrices in one structure

### Canonical operations
| Operation | Meaning |
|-----------|---------|
| `A * B`   | Matrix multiply (semiring selectable) |
| `A + B`   | Element-wise add |
| `A & B`   | Element-wise intersection (logical AND / min) |
| `A \| B`  | Element-wise union (logical OR / max) |
| `A[r, c]` | Row/column slicing by key |
| `A.T`     | Transpose |

### Semirings used in this project
- **plus_times** — standard linear algebra (default)
- **plus_min** — shortest paths (geodesic distance on mesh)
- **min_plus** — tropical semiring (distance propagation)
- **any_pair** — reachability / flood fill (does a path exist?)
- **plus_first**, **plus_second** — one-sided propagation

### D4M Python library
> **TODO**: Record the specific library name, version, and import convention here once chosen.
> Candidates: `py4dm`, custom implementation, Julia via subprocess.
> Note import alias used throughout project, e.g. `import d4m as d4m`.

---

## GraphBLAS

We use **python-graphblas** (wrapping SuiteSparse:GraphBLAS) as the computation engine.

```python
import graphblas as gb
```

### Key conventions
- Matrices are `gb.Matrix`, vectors are `gb.Vector`
- Use `<<` assignment to trigger lazy expressions: `C << A.mxm(B, semiring)`
- Masks go on the left: `C(mask) << expr`
- Accumulate with: `C(accum=gb.binary.plus) << expr`
- Descriptors control transpose, complement: use `A.T` for transpose input

### Common patterns in this project
```python
# Adjacency matrix from mesh edges
A = gb.Matrix.from_coo(rows, cols, vals, nrows=n, ncols=n)

# One BFS step (spread labels one hop)
next_frontier << A.mxv(frontier, gb.semiring.any_pair)

# Laplacian smoothing (one iteration)
smoothed << L.mxv(positions, gb.semiring.plus_times)

# Geodesic distance (SSSP via tropical semiring)
dist << A.mxv(dist, gb.semiring.min_plus)
```

---

## Blender Mesh Representation

### Extracting topology via bpy
```python
import bpy

obj  = bpy.context.active_object
mesh = obj.data
mesh.calc_loop_triangles()

n_verts = len(mesh.vertices)
n_edges = len(mesh.edges)
n_faces = len(mesh.polygons)
```

### Associative array key conventions
| Element | Row/col key format | Example |
|---------|--------------------|---------|
| Vertex  | `v:{index:05d}`    | `v:00042` |
| Edge    | `e:{index:05d}`    | `e:00117` |
| Face    | `f:{index:05d}`    | `f:00008` |

### Core matrices
| Matrix  | Rows   | Cols   | Value            | Purpose                  |
|---------|--------|--------|------------------|--------------------------|
| `V_adj` | vertex | vertex | edge weight or 1 | vertex adjacency         |
| `VE`    | vertex | edge   | ±1 / 1           | vertex-edge incidence    |
| `VF`    | vertex | face   | 1                | vertex-face incidence    |
| `EF`    | edge   | face   | 1                | edge-face incidence      |
| `L`     | vertex | vertex | Laplacian weights| mesh Laplacian           |

### Writing results back to Blender
```python
# Move vertices after smoothing
for i, v in enumerate(mesh.vertices):
    v.co.x = new_positions[i][0]
    v.co.y = new_positions[i][1]
    v.co.z = new_positions[i][2]

mesh.update()
```

---

## Planned Operations

- [ ] Laplacian smoothing (iterative vertex position averaging)
- [ ] Flood fill (propagate vertex colour / weight across connected region)
- [ ] Connected component / island detection
- [ ] Geodesic distance from selected vertices
- [ ] k-hop neighbourhood selection
- [ ] Vertex influence / weight propagation (bones, masks)

---

## Project Status

- D4M/AA library: **not yet chosen**
- GraphBLAS: `python-graphblas` via pip
- Blender MCP: connected (Claude can inspect live Blender scenes)
- LoRA dataset pipeline: complete (Little Nemo panels, see `little-nemo-ge25/`)

---

## Notes for Claude

- Always prefer associative array (string-keyed) representations over raw integer index matrices
  when the operation involves selection, slicing, or joining across mesh element types
- Integer-indexed `gb.Matrix` is fine for inner-loop numeric computation (smoothing iterations etc.)
- When writing Blender code, use `bpy.ops` for standard operations and `bpy.data` for precise access
- The Blender MCP tools are available — inspect the live scene before assuming mesh structure
- Ask for clarification before destructively modifying mesh data
