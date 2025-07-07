<img src="assets/logo.png" width="300">

This repository contains a collection of benchmark problems demonstrating the application of mathematical optimization to inorganic materials design using the [MatOpt framework](https://github.com/xyin-anl/matopt). Each example represents a different materials design challenge, from nanoclusters to bulk oxides, surfaces, and nanowires.

## Overview

MatOpt (Materials Optimization) is a Python package for modeling and solving materials design problems using mathematical optimization. This benchmark suite showcases various applications where materials properties are optimized subject to physical and chemical constraints.

## Repository Structure

```
MatOpt-bench/
├── examples/              # Numbered example scripts (01-11)
├── configs/               # JSON configuration files
├── data/                  # Organized data files
├── lp_files/              # Generated LP benchmark files (MILP & MIQCQP)
├── utils/                 # Utility scripts
├── models.md              # Mathematical formulations
└── README.md              # This file
```

## Examples Overview

### Categorization Table

| Category | Example | Material System | Design Space | Key Features |
|----------|---------|-----------------|--------------|--------------|
| **Nanoclusters** | `01_nanocluster_mono` | Configurable (default: Pt) | FCC lattice sites | Square-root cohesive energy, piecewise linear |
| | `02_nanocluster_bimetal` | Configurable (default: Cu-Ag) | Composition & structure | Two-step optimization, Gupta potentials |
| **Surfaces** | `03_surface_design` | Configurable metal | Surface atom positions | GCN descriptor, activity-stability trade-off |
| | `04_surface_bifunctional` | Pt-Ni catalyst | 7 predefined motifs | Configuration-based bifunctional design |
| **Bulk Materials** | `05_bulk_oxide_vacancy` | Ba(Fe,In)O₃ | Oxygen vacancy configs | Local/global doping constraints |
| | `06_crystal_ionic` | 5 crystal types | Ion positions | Ewald summation, space group orbits |
| **Nanowires** | `07_nanowire_design` | InAs wurtzite | Core-shell structure | Cylindrical geometry, 20% core constraint |
| **Alloys & Solutions** | `08_alloy_cluster_expansion` | Ru/Zr/CuNiPdAg | Composition space | ML cluster expansion, multi-property |
| | `09_solid_solution_qubo` | Graphene/AlGaN/Ta-W | Binary substitutions | Quantum annealing ready, chemical potential |
| **ML-Enhanced** | `10_crystal_ml_fm` | Crystal structures | Bit-string encoding | CRYSIM factorization machine |
| | `11_hea_ml_fm` | Configurable HEA | BCC supercell | Entropy at finite T, composition limits |

## Detailed Example Descriptions

### 1. Monometallic Nanocluster Design (`01_nanocluster_mono.py`)
- **Problem**: Find the optimal structure of monometallic nanoclusters (configurable element and size)
- **Objective**: Maximize cohesive energy using square-root coordination model
- **Features**: FCC lattice, piecewise linear approximation for square-root function
- **Reference**: [Mol. Syst. Des. Eng., 2020](https://pubs.rsc.org/en/content/articlelanding/2020/me/c9me00108e)

### 2. Bimetallic Nanocluster Design (`02_nanocluster_bimetal.py`)
- **Problem**: Design bimetallic nanoclusters (configurable elements and composition)
- **Approach**: Two-step optimization - first shape, then composition
- **Model**: Gupta-like potentials with coordination-dependent bond energies (Gkl coefficients)
- **Reference**: [Mol. Syst. Des. Eng., 2021](https://pubs.rsc.org/en/content/articlelanding/2021/me/d1me00027f)

### 3. Surface Design (`03_surface_design.py`)
- **Problem**: Design nanopatterned metallic surfaces for optimal catalytic performance
- **Objective**: Balance activity (ideal sites) and stability (surface energy)
- **Features**: 3D slab model, generalized coordination number (GCN) descriptor
- **Reference**: [AIChE J. 2017](https://doi.org/10.1002/aic.15359)

### 4. Bifunctional Surface Design (`04_surface_bifunctional.py`)
- **Problem**: Design bifunctional Pt-Ni catalyst surface
- **Approach**: Configuration-based design with 7 predefined atomic patterns
- **Objective**: Maximize combined activity from Type A and Type B sites
- **Reference**: [Ind. Eng. Chem. Res. 2019](https://pubs.acs.org/doi/full/10.1021/acs.iecr.8b04801)

### 5. Metal Oxide Bulk Design (`05_bulk_oxide_vacancy.py`)
- **Problem**: Optimize oxygen vacancy distribution in Ba(Fe,In)O₃ perovskite
- **Features**: Local and global In doping constraints, pre-computed vacancy configurations
- **Application**: Solid oxide fuel cells and catalysis
- **Reference**: [Comput. Chem. Eng. 2019](https://www.sciencedirect.com/science/article/abs/pii/S0098135418310998)

### 6. Ionic Crystal Structure Prediction (`06_crystal_ionic.py`)
- **Problem**: Predict stable ionic crystal structures with space group symmetry constraints
- **Systems**: Perovskites (SrTiO3), spinels (MgAl2O4), garnets (Ca3Al2Si3O12), pyrochlores (Y2Ti2O7), bixbyites (Y2O3)
- **Model**: Long-range Coulomb (Ewald summation) + short-range Buckingham potentials
- **Features**: Space group orbit constraints, charge neutrality
- **Reference**: [Nature 2023](https://www.nature.com/articles/s41586-023-06071-y)

### 7. Nanowire Design (`07_nanowire_design.py`)
- **Problem**: Design InAs wurtzite nanowire with core-shell structure
- **Geometry**: Cylindrical with periodic boundary conditions
- **Constraints**: 20% core ratio, directional growth
- **Reference**: [Curr. Opin. Chem. Eng. 2022](https://www.sciencedirect.com/science/article/pii/S2211339821000587)

### 8. Cluster Expansion (`08_alloy_cluster_expansion.py`)
- **Problem**: Multi-property optimization using machine learning cluster expansion
- **Systems**: Configurable - Ru oxides, Zr oxides, or CuNiPdAg alloys
- **Features**: Integration with ICET, trains cluster expansion with LassoCV, multiple property targets
- **Reference**: [Matter 2023](https://www.cell.com/matter/fulltext/S2590-2385(22)00662-2)

### 9. Solid Solution Design (`09_solid_solution_qubo.py`)
- **Problem**: Design substitutional solid solutions using QUBO formulation
- **Systems**: N-doped graphene, AlGaN, Ta-W alloys
- **Features**: Quantum annealing-ready QUBO matrix, chemical potential term
- **Modes**: Constrained (fixed substitutions) or unconstrained optimization
- **Reference**: [Sci. Adv. 2024](https://www.science.org/doi/10.1126/sciadv.adt7156)

### 10. Factorization Machine - Crystal Structure (`10_crystal_ml_fm.py`)
- **Problem**: Crystal structure prediction using machine learning
- **Model**: CRYSIM-trained factorization machine with automatic bit-string structure detection
- **Variables**: Space groups, Wyckoff positions, lattice parameters (one-hot encoded)
- **Reference**: [arXiv:2024](https://arxiv.org/abs/2504.06878)

### 11. Factorization Machine - High-Entropy Alloys (`11_hea_ml_fm.py`)
- **Problem**: Design high-entropy alloys with entropy contribution
- **Systems**: Default Nb-Mo-Ta-W, configurable elements
- **Features**: Includes configurational entropy at finite temperature, BCC supercell
- **Model**: QALO-kit factorization machine format, composition constraints (max 60% per element)
- **Reference**: [npj Comput. Mater. 2025](https://www.nature.com/articles/s41524-024-01505-1)

## Configuration System

Each example can be configured using JSON files in the `configs/` directory. Configuration options include:

- **Material parameters**: Elements, lattice constants, interaction energies
- **Geometry parameters**: System size, boundaries, orientations
- **Optimization settings**: Solver choice, time limits, tolerances
- **Model-specific options**: Energy models, constraints, special features

### Using Custom Configurations

```bash
# Run with default configuration
python examples/01_nanocluster_mono.py

# Run with custom configuration
python examples/01_nanocluster_mono.py --config configs/01_nanocluster_mono_large.json

# Export LP file
python examples/01_nanocluster_mono.py --export-lp
```

## Benchmark LP Files

The `lp_files/` directory contains LP format files that can be used to benchmark different solvers:

### Compatible Solvers

- **Conventional solvers**: CPLEX, Gurobi, or NEOS server
- **Quantum solvers**: D-Wave or other quantum solvers

## Requirements

- Python 3.9+
- [MatOpt](https://github.com/idaes/MatOpt)
- Additional dependencies per example (see individual files)


```bash
# Clone repository
git clone https://github.com/xyin-anl/MatOpt-bench.git
cd MatOpt-bench

# Install dependencies
pip install pyomo numpy scipy pandas matopt
```


## Quick Start

1. **Run a simple example**:
   ```bash
   python examples/01_nanocluster_mono.py
   ```

2. **Export an LP file**:
   ```bash
   python examples/03_surface_design.py --export-lp
   ```

3. **Use a custom configuration**:
   ```bash
   python examples/01_nanocluster_mono.py --config configs/01_nanocluster_mono_large.json
   ```

4. **Solve an LP file with your solver**:


## Citation

If you use this benchmark suite, please cite:

```bibtex
@software{matopt_bench,
  title = {MatOpt-bench: A Benchmark Suite for Materials Design Optimization},
  author = {Xiangyu Yin},
  year = {2025},
  url = {https://github.com/xyin-anl/MatOpt-bench}
}
```

## Contributing

Contributions are welcome! Please submit pull requests with:
- New benchmark problems
- Alternative formulations
- Performance comparisons
- Bug fixes and improvements

## License

This benchmark suite is released under the MIT License. Individual examples may reference papers with their own licensing terms.

## Acknowledgments

This work builds upon the MatOpt framework developed by the IDAES project. We thank all contributors to the original research papers that inspired these benchmarks.