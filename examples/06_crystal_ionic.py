# Optimality guarantees for crystal structure prediction
# https://www.nature.com/articles/s41586-023-06071-y
# https://github.com/lrcfmd/ipcsp
from __future__ import annotations
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, math, pathlib
from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

from matopt.materials.atom import Atom
from matopt.materials.canvas import Canvas
from matopt.aml.expr import (
    SumSites,
    SumSitesAndTypes,
    SumBondTypes,
    LinearExpr,
    SumBonds,
)
from matopt.aml.rule import EqualTo, FixedTo
from matopt.aml.model import MatOptModel
from utils.config_handler import ConfigHandler


# metadata
@dataclass
class ProblemSpec:
    phase: str
    grid: int
    cell: float
    group: str
    ions: dict[str, int]  # per orbit stoichiometry


@dataclass
class Phase:
    name: str
    path: pathlib.Path
    charges: dict[str, float]
    radii: dict[str, float]
    buck_pars: dict[tuple[str, str], tuple[float, float, float, float, float]]
    closest: dict[tuple[str, str], float]


_here = pathlib.Path(__file__).resolve().parent


# helper functions
def cubic(grid: int, step_divisor: float = 1.0) -> np.ndarray:
    step = step_divisor / grid
    pts = np.indices((grid,) * 3).reshape(3, -1).T * step
    return pts.astype(float)


def load_orbits(
    grid: int, sg: str, grid_dir: pathlib.Path, orbit_ext: str = ".json"
) -> dict[str, list[int]]:
    fn = grid_dir / f"CO{grid}G{sg}{orbit_ext}"
    if not fn.exists():
        raise FileNotFoundError(
            f"Orbit file {fn} not found, copy or generate it first."
        )
    with open(fn) as fp:
        return json.load(fp)


def _qewald(pts, cell_len, alpha, real_d, recip_d) -> np.ndarray:
    """Calculate Ewald sum without numba (scipy erfc not supported in numba)."""
    n = len(pts)
    H = np.zeros((n, n))
    recip_vol = (2 * math.pi / cell_len) ** 3
    prefac = 4 * math.pi / recip_vol

    # real space
    for g_idx in range(-real_d, real_d + 1):
        for g_idy in range(-real_d, real_d + 1):
            for g_idz in range(-real_d, real_d + 1):
                for ii in range(n):
                    for jj in range(ii, n):
                        r_vec = pts[ii] - pts[jj].copy()
                        r_vec[0] += g_idx
                        r_vec[1] += g_idy
                        r_vec[2] += g_idz
                        r = cell_len * np.linalg.norm(r_vec)
                        if (g_idx, g_idy, g_idz) != (0, 0, 0) or ii != jj:
                            val = erfc(alpha * r) / r
                            H[ii, jj] += val
                            H[jj, ii] += val
                        if (g_idx, g_idy, g_idz) == (0, 0, 0) and ii == jj:
                            # special ii=jj correction
                            H[ii, jj] -= 2 * alpha / math.sqrt(math.pi)

    # reciprocal space
    recip_scale = 2 * math.pi / cell_len
    for k_idx in range(-recip_d, recip_d + 1):
        for k_idy in range(-recip_d, recip_d + 1):
            for k_idz in range(-recip_d, recip_d + 1):
                if (k_idx, k_idy, k_idz) == (0, 0, 0):
                    continue
                k_vec = recip_scale * np.array([k_idx, k_idy, k_idz])
                k_sq = np.dot(k_vec, k_vec)
                exp_fac = prefac * np.exp(-k_sq / (4 * alpha**2)) / k_sq
                for ii in range(n):
                    r_i = pts[ii] * cell_len
                    exp_i = np.exp(1j * np.dot(k_vec, r_i))
                    for jj in range(n):
                        r_j = pts[jj] * cell_len
                        exp_j = np.exp(-1j * np.dot(k_vec, r_j))
                        val = exp_fac * (exp_i * exp_j).real
                        H[ii, jj] += val

    return H


def Ewald_from_C2(
    C2: np.ndarray,
    cell_len: float,
    grid: int,
    grid_dir: pathlib.Path,
    grid_ext: str = ".txt",
    hartree_to_ev: float = 27.2114,
    nm_to_angstrom: float = 10.0,
) -> np.ndarray:
    """Expand C2 Ewald matrix to full grid."""
    # Read G2 mapping
    with open(grid_dir / f"C{grid}{grid_ext}") as fp:
        G2 = np.array([int(line.strip()) for line in fp], dtype=int) - 1

    n_sites = grid**3
    H = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            g2_i, g2_j = G2[i], G2[j]
            H[i, j] = C2[g2_i, g2_j]

    conversion = hartree_to_ev * nm_to_angstrom / cell_len
    return H * conversion


def get_ew(
    e1: str,
    e2: str,
    cell: float,
    grid: int,
    ewald_dir: pathlib.Path,
    grid_dir: pathlib.Path,
    ewald_ext: str = ".npy",
    grid_ext: str = ".txt",
    step_divisor: float = 1.0,
    alpha_factor: float = 5.0,
    real_d: int = 3,
    recip_d: int = 3,
    hartree_to_ev: float = 27.2114,
    nm_to_angstrom: float = 10.0,
    pars=None,
):
    """Get Ewald matrix for given element pair."""
    # Prefer C2 files if available
    if pars is None:
        # Try loading C2 files
        c2_files = {
            ("O", "O"): ewald_dir / f"C2_OO_{cell:.1f}{ewald_ext}",
            ("O", "Sr"): ewald_dir / f"C2_OSr_{cell:.1f}{ewald_ext}",
            ("Sr", "O"): ewald_dir / f"C2_OSr_{cell:.1f}{ewald_ext}",
            ("O", "Ti"): ewald_dir / f"C2_OTi_{cell:.1f}{ewald_ext}",
            ("Ti", "O"): ewald_dir / f"C2_OTi_{cell:.1f}{ewald_ext}",
        }

        c2_file = c2_files.get((e1, e2))
        if c2_file and c2_file.exists():
            C2 = np.load(c2_file)
            return Ewald_from_C2(
                C2, cell, grid, grid_dir, grid_ext, hartree_to_ev, nm_to_angstrom
            )

        # Default parameters if no C2 file
        alpha = alpha_factor / cell
    else:
        alpha, real_d, recip_d = pars

    pts = cubic(grid, step_divisor)
    return _qewald(pts, cell, alpha, real_d, recip_d)


def get_buck(grid, cell, e1, e2, phase, step_divisor, max_penalty):
    """Calculate Buckingham potential."""
    A, rho, C, d, delta = phase.buck_pars.get((e1, e2), (0, 1, 0, 0, 0))
    if A == 0:
        return np.zeros((grid**3, grid**3))

    # Calculate without numba due to compatibility issues
    def _get_buck_inner(grid, cell, A, rho, C, cutoff, step_divisor, max_penalty):
        step = step_divisor / grid
        pts = np.indices((grid, grid, grid)).reshape(3, -1).T * step
        pts = pts.astype(np.float64)

        n = len(pts)
        H = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Minimum image convention
                r_vec = pts[i] - pts[j]
                r_vec = r_vec - np.round(r_vec)
                r = cell * np.linalg.norm(r_vec)

                if r < cutoff:
                    val = max_penalty
                else:
                    val = A * np.exp(-r / rho) - C / r**6

                H[i, j] = val
                H[j, i] = val

        return H

    cutoff = phase.closest.get((e1, e2), 0.0)
    return _get_buck_inner(grid, cell, A, rho, C, cutoff, step_divisor, max_penalty)


# phase definitions
def load_phase(name: str, data_dir: pathlib.Path, lib_files: dict) -> Phase:
    """Load phase parameters from data files."""
    path = data_dir / name

    # Initialize
    charges = {}
    radii = {}
    buck_pars = {}

    # Load radii from radii.lib (only has element and radius)
    radii_file = path / lib_files["radii"]
    if radii_file.exists():
        with open(radii_file) as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 2:
                    elem = parts[0]
                    radius = float(parts[1])
                    radii[elem] = radius

    # Load Buckingham parameters and charges from buck.lib
    buck_file = path / lib_files["buckingham"]
    if buck_file.exists():
        with open(buck_file) as fp:
            in_species_section = False
            in_buck_section = False

            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Check for section headers
                if line == "species":
                    in_species_section = True
                    in_buck_section = False
                    continue
                elif line == "buck":
                    in_species_section = False
                    in_buck_section = True
                    continue

                parts = line.split()

                # Parse species section for charges
                if in_species_section and len(parts) >= 3:
                    elem = parts[0]
                    if parts[1] == "core" and len(parts) >= 3:
                        charge = float(parts[2])
                        charges[elem] = charge

                # Parse buck section for parameters
                elif in_buck_section and len(parts) >= 8:
                    # Format: elem1 core elem2 core A rho C d delta
                    if parts[1] == "core" and parts[3] == "core":
                        e1, e2 = parts[0], parts[2]
                        A = float(parts[4])
                        rho = float(parts[5])
                        C = float(parts[6])
                        d = float(parts[7]) if len(parts) > 7 else 0.0
                        delta = float(parts[8]) if len(parts) > 8 else 0.0
                        buck_pars[(e1, e2)] = (A, rho, C, d, delta)
                        buck_pars[(e2, e1)] = (A, rho, C, d, delta)

    # Load closest approach distances
    closest = {}
    dist_file = path / lib_files["distances"]
    if dist_file.exists():
        with open(dist_file) as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 3:
                    e1, e2 = parts[0], parts[1]
                    dist = float(parts[2])
                    closest[(e1, e2)] = dist
                    closest[(e2, e1)] = dist

    return Phase(
        name=name,
        path=path,
        charges=charges,
        radii=radii,
        buck_pars=buck_pars,
        closest=closest,
    )


# main optimization
if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "06_crystal_ionic")

    # Define required fields
    required_fields = {
        "material": ["phase", "ions"],
        "geometry": ["grid", "cell", "group", "grid_step_divisor"],
        "optimization": ["solver", "time_limit", "minimize_sense"],
        "energy_parameters": [
            "max_penalty",
            "ewald",
            "conversion_factors",
            "energy_coefficients",
        ],
        "constraint_parameters": ["yi_fixed_value"],
        "data_paths": [
            "base_directory",
            "grid_subdirectory",
            "ewald_subdirectory",
            "file_extensions",
            "library_files",
        ],
        "output": ["lp_export_dir", "lp_file_prefix"],
    }

    # Validate configuration
    ConfigHandler.validate_config(config, required_fields)

    # Extract parameters from config
    phase_name = config["material"]["phase"]
    ions = config["material"]["ions"]
    grid = config["geometry"]["grid"]
    cell = config["geometry"]["cell"]
    group = config["geometry"]["group"]
    step_divisor = config["geometry"]["grid_step_divisor"]

    solver = config["optimization"]["solver"]
    time_limit = config["optimization"]["time_limit"]
    minimize_sense = config["optimization"]["minimize_sense"]

    max_penalty = config["energy_parameters"]["max_penalty"]
    ewald_params = config["energy_parameters"]["ewald"]
    conv_factors = config["energy_parameters"]["conversion_factors"]
    energy_coefs = config["energy_parameters"]["energy_coefficients"]

    yi_fixed = config["constraint_parameters"]["yi_fixed_value"]

    # Set up data paths
    data_base = _here.parent / config["data_paths"]["base_directory"]
    DATA_DIR = data_base
    GRID_DIR = DATA_DIR / config["data_paths"]["grid_subdirectory"]
    EWALD_DIR = DATA_DIR / config["data_paths"]["ewald_subdirectory"]
    file_exts = config["data_paths"]["file_extensions"]
    lib_files = config["data_paths"]["library_files"]

    lp_export_dir = config["output"]["lp_export_dir"]
    lp_file_prefix = config["output"]["lp_file_prefix"]

    # Create ProblemSpec from config
    spec = ProblemSpec(phase=phase_name, grid=grid, cell=cell, group=group, ions=ions)

    print(f"Running crystal ionic optimization...")
    print(
        f"Phase: {spec.phase}, Grid: {spec.grid}, Cell: {spec.cell} Å, Group: {spec.group}"
    )
    print(f"Ions: {spec.ions}")

    # Load phase data
    phase = load_phase(spec.phase, DATA_DIR, lib_files)

    # lattice & orbits
    pts = cubic(spec.grid, step_divisor)
    n_sites = len(pts)
    print(f"Grid points: {n_sites}")

    # Load orbit information
    orbits = load_orbits(spec.grid, spec.group, GRID_DIR, file_exts["orbit"])
    print(f"Loaded {len(orbits)} orbits")

    # canvas
    # For ionic crystals, we use a complete graph
    neighbors = [[j for j in range(n_sites) if j != i] for i in range(n_sites)]

    # Pad to make rectangular
    max_deg = max(len(row) for row in neighbors)
    neighbors = [row + [None] * (max_deg - len(row)) for row in neighbors]

    Canv = Canvas(pts.tolist(), neighbors)

    # atoms & basic constraints
    elements = list(spec.ions.keys())
    AllAtoms = [Atom(e) for e in elements]

    m = MatOptModel(Canv, AllAtoms)

    # Each orbit gets the prescribed number of each ion type
    for orbit_name, site_indices in orbits.items():
        if not site_indices:  # skip empty orbits
            continue

        # For each element, fix the count in this orbit
        for elem, total_count in spec.ions.items():
            if total_count > 0:
                # Distribute ions across orbits proportionally
                orbit_count = len(site_indices) * total_count // n_sites
                if orbit_count > 0:
                    m.addSitesTypesDescriptor(
                        f"Orbit_{orbit_name}_{elem}",
                        sites=site_indices,
                        site_types=[Atom(elem)],
                        rules=EqualTo(SumSites(m.Yik), orbit_count),
                    )

    # One atom per site
    m.Yi.rules.append(FixedTo(yi_fixed))

    # energy model
    H_lin = {}  # {(site, Atom): coef}
    H_quad = {}  # {(i, j, Atom1, Atom2): coef}

    # Ewald
    for e1 in elements:
        for e2 in elements:
            z1 = phase.charges.get(e1, 0.0)
            z2 = phase.charges.get(e2, 0.0)
            if abs(z1) < 1e-6 or abs(z2) < 1e-6:
                continue

            mat = get_ew(
                e1,
                e2,
                spec.cell,
                spec.grid,
                EWALD_DIR,
                GRID_DIR,
                file_exts["ewald_matrix"],
                file_exts["grid_mapping"],
                step_divisor,
                ewald_params["alpha_factor"],
                ewald_params["real_d"],
                ewald_params["recip_d"],
                conv_factors["hartree_to_ev"],
                conv_factors["nm_to_angstrom"],
            )
            a1, a2 = Atom(e1), Atom(e2)

            for i in range(n_sites):
                for j in range(i, n_sites):
                    d = mat[i, j]
                    if abs(d) < 1e-12:
                        continue

                    coef = d * z1 * z2

                    if e1 == e2 and i == j:  # self
                        H_lin[(i, a1)] = H_lin.get((i, a1), 0.0) + coef
                    elif e1 == e2:  # like‑like
                        H_quad[(i, j, a1, a1)] = coef
                    else:  # unlike (need both orderings)
                        coef = d * z1 * z2
                        H_quad[(i, j, a1, a2)] = coef
                        H_quad[(i, j, a2, a1)] = coef

    # Buckingham
    for (e1, e2), pars in phase.buck_pars.items():
        mat = get_buck(spec.grid, spec.cell, e1, e2, phase, step_divisor, max_penalty)
        a1, a2 = Atom(e1), Atom(e2)
        for i in range(n_sites):
            for j in range(i, n_sites):
                coef = mat[i, j]
                if abs(coef) < 1e-12:
                    continue
                if e1 == e2 and i == j:  # self
                    H_lin[(i, a1)] += coef
                elif e1 == e2:  # like‑like
                    H_quad[(i, j, a1, a1)] = H_quad.get((i, j, a1, a1), 0.0) + coef
                elif i != j:  # unlike
                    H_quad[(i, j, a1, a2)] = H_quad.get((i, j, a1, a2), 0.0) + coef
                    H_quad[(i, j, a2, a1)] = H_quad.get((i, j, a2, a1), 0.0) + coef

    # plug energy into MatOpt
    m.addGlobalDescriptor("Hlin", rules=EqualTo(SumSitesAndTypes(m.Yik, coefs=H_lin)))
    m.addBondsDescriptor(
        "Hpair",
        rules=EqualTo(SumBondTypes(m.Xijkl, coefs=H_quad)),
        symmetric_bonds=True,
    )
    m.addGlobalDescriptor("Hquad", rules=EqualTo(SumBonds(m.Hpair)))

    obj = LinearExpr([m.Hlin, m.Hquad], energy_coefs)  # total energy

    # Export LP file if requested
    if "--export-lp" in sys.argv:
        # Determine formulation type from command line argument
        formulation = "milp"  # default
        if "--miqcqp" in sys.argv:
            formulation = "miqcqp"
        elif "--milp" in sys.argv:
            formulation = "milp"
        
        # Create subdirectory based on formulation type
        # Keep using 'miqcp' for subdirectory name for consistency
        subdir_name = 'miqcp' if formulation == 'miqcqp' else formulation
        lp_export_subdir = os.path.join(lp_export_dir, subdir_name)
        os.makedirs(lp_export_subdir, exist_ok=True)
        
        preset_name = f"{spec.phase}_{spec.grid}_{spec.group}_{spec.cell}"
        # Use config filename for LP file
        config_basename = ConfigHandler.get_config_basename(config_file)
        if config_basename:
            lp_file = f"{lp_export_subdir}/{config_basename}.lp"
        else:
            # Fallback to original naming if no config file provided
            lp_file = f"{lp_export_subdir}/{lp_file_prefix}.lp"
        # Create Pyomo model and export
        pyomo_model = m._make_pyomo_model(obj, sense=minimize_sense, formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    # optimize
    D = m.minimize(obj, tilim=time_limit, solver=solver)

    if D:
        print("\nOptimization successful!")
        # Count composition
        from collections import Counter

        composition = Counter(atom.symbol for atom in D.atoms)
        print(f"Final composition: {dict(composition)}")

        # Calculate final energy
        energy = obj.getValue(m)
        print(f"Final energy: {energy:.6f} eV")
