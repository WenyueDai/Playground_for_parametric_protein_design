# Parameters Overview

## Helix Generation Parameters

```python
helix_len = 28       # Number of residues in the helix
phi_deg = -57.8      # Typical alpha-helix phi angle
psi_deg = -47.0      # Typical alpha-helix psi angle
omega_deg = 180.0    # Typical alpha-helix omega angle
```

---

## Input PDB Options

```python
input_pdb_path  = ""   # Path to input PDB file; empty = disabled
input_chain     = ""   # Example: "A"; empty = use whole pose or infer
input_pdb_range = ""   # Example: "5-42" in PDB numbering; empty = full chain
```

---

## Input Axis & Alignment Parameters

```python
input_axis_mode    = "auto_pca"   # "auto_pca" | "auto_ends"
                                  # "auto_ends" = use N→C direction (faster, may fail for short/curved helices)

input_target_axis  = "1,0,0"      # Aligns to +Z (0,0,1) if None/""/"none"
                                  # Provide vector "x,y,z" to align to a custom direction

roll_deg           = 0.0          # Roll around the aligned target axis (deg)
```

---

## C<sub>n</sub> Symmetry Parameters

```python
sym_n            = 0        # Number of symmetry units; 0 = no symmetry
sym_axis         = "0,0,1"  # Symmetry axis direction (default Z)
sym_center       = "0,0,0"  # Symmetry center point
sym_radius       = 15.0     # Radius from symmetry axis to helix center (Å)
sym_start_angle  = 0.0      # Starting azimuth angle of first helix (deg)
global_tilt_deg  = 0.0      # Tilt each helix away from sym_axis (deg)
```

---

## Dihedral Extension (C<sub>n</sub> → D<sub>n</sub>)

```python
dihedral_enable        = False    # Enable partner ring via 180° rotation
dihedral_axis          = "1,0,0"  # C2 axis perpendicular to sym_axis
dihedral_post_shift_z  = 0.0      # Translate along sym_axis (Å)
dihedral_post_delta_r  = 0.0      # Radial push outward (Å)
dihedral_twist_deg     = 0.0      # Extra twist about sym_axis (deg)
```

---

## Ring Stacking / Expansion

```python
ring_stack_copies   = 0       # Number of additional rings
ring_delta_radius   = 0.0     # Outward shift per ring (Å)
ring_delta_z        = 0.0     # Axial shift per ring (Å)
ring_delta_twist_deg = 0.0    # Extra twist per ring (deg)
```

---

## Output Options

```python
resname = "ALA"     # Residue name to display in PDB
out_pdb = "outs/helix_output_CA_only.pdb" # output PDB file path, hash tag will be added based on params

```

---

## Input Structure Generation

There are **two ways** to generate the initial structure:

1. **Parametric design**
   Define `phi`, `psi`, `chi` angles to create an *idealized helix* using PyRosetta.

2. **Read from PDB**
   Load structure by specifying chain and residue range.
   *(Currently, discontinuous regions are not supported.)*

After loading:

* The **center of mass** is moved to the origin.
* The **axis** is calculated by:

  * Principal Component Analysis → if `input_axis_mode = "auto_pca"`, or
  * N→C terminal direction → if `input_axis_mode = "auto_ends"`.
* The structure is **aligned to the Z-axis** by default (or to your defined axis via `input_target_axis`).
* The `roll_deg` parameter controls rotation around its own axis.

---

## Practical Exploration Tips

* Disable PDB input and use helix generation only.
* Experiment with `input_target_axis` values:

| Axis Vector | Axis | Color    |
| ----------- | ---- | -------- |
| `0,0,1`     | Z    | Blue  |
| `0,1,0`     | Y    | Green |
| `1,0,0`     | X    | Red   |

![Axis Alignment Example](images/f1.png)

* Adjust `roll_deg` (e.g., 0°, 30°, 60°, 90°, 120°) to visualize twisting around the selected axis:

![Roll Comparison Example](images/f2.png)

---


