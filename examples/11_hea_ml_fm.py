# Quantum annealing-assisted lattice optimization
# https://www.nature.com/articles/s41524-024-01505-1
# https://github.com/ZhihaoXu0313/qalo-kit
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import itertools
import numpy as np
from matopt.materials.atom import Atom
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSites, SumSitesAndTypes, SumBondsAndTypes, LinearExpr
from matopt.aml.rule import EqualTo, FixedTo
from matopt.aml.model import MatOptModel
from utils.config_handler import ConfigHandler


def load_fm_model(model_txt, N, Nf):
    idx, idf = 1, 1
    offset, L, V = [], [], [[] for _ in range(N)]
    with open(model_txt) as f:
        for line in f:
            nums = list(map(float, line.strip().split()[1:]))
            if idx == 1:
                offset.append(nums)
            elif 1 < idx <= 1 + N:
                L.append(nums)
            else:
                V[idf - 1].append(nums)
                if len(V[idf - 1]) == Nf:
                    idf += 1
            idx += 1
    offset = np.array(offset, dtype=float)
    L = np.array(L, dtype=float)
    V = np.array(V, dtype=float)

    Q = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                Q[i, j] = L[i]
            else:
                Q[i, j] = np.dot(V[i, j % Nf], V[j, i % Nf])
                Q[j, i] = Q[i, j]  # make symmetric
    return Q, offset


def lattice_points(supercell, basis, a0):
    Nx, Ny, Nz = supercell
    pts = []
    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                base = np.array([ix, iy, iz], dtype=float)
                for frac in basis:
                    pts.append((base + frac) * a0)
    return pts


def split_Q_and_entropy(
    Q,
    atoms,
    nsites,
    entropy_lin,
    entropy_pair,
    fm_threshold=1e-12,
    entropy_threshold=1e-10,
):
    linear, quad = {}, {}
    nbrs = [set() for _ in range(nsites)]

    # FM part
    for a in range(Q.shape[0]):
        ia, ka = divmod(a, nsites)
        for b in range(a, Q.shape[1]):
            val = Q[a, b]
            if abs(val) < fm_threshold:
                continue
            ib, kb = divmod(b, nsites)
            if ka == kb and a != b:
                # same site, different species → already forbidden by MatOpt
                continue
            if a == b:  # diagonal → linear
                linear[(ka, atoms[ia])] = linear.get((ka, atoms[ia]), 0.0) + val
            else:  # off‑diagonal → quadratic
                if ka < kb:
                    key = (ka, kb, atoms[ia], atoms[ib])
                    quad[key] = quad.get(key, 0.0) + val
                    # Also add the symmetric entry
                    key_sym = (kb, ka, atoms[ib], atoms[ia])
                    quad[key_sym] = quad.get(key_sym, 0.0) + val
                elif kb < ka:
                    key = (kb, ka, atoms[ib], atoms[ia])
                    quad[key] = quad.get(key, 0.0) + val
                    # Also add the symmetric entry
                    key_sym = (ka, kb, atoms[ia], atoms[ib])
                    quad[key_sym] = quad.get(key_sym, 0.0) + val
                else:
                    raise ValueError("ka == kb")
                nbrs[ka].add(kb)
                nbrs[kb].add(ka)

    # entropy linear term  Σ_i entropy_lin * s_i
    for site in range(nsites):
        for sp in atoms:
            linear[(site, sp)] = linear.get((site, sp), 0.0) + entropy_lin

    # entropy quadratic term  Σ_i entropy_pair * x_i_j x_i_k (j<k)
    if abs(entropy_pair) > entropy_threshold:
        all_sites = range(nsites)
        for sp in atoms:
            for j, k in itertools.combinations(all_sites, 2):
                # Add both directions
                key1 = (j, k, sp, sp)
                quad[key1] = quad.get(key1, 0.0) + entropy_pair
                key2 = (k, j, sp, sp)
                quad[key2] = quad.get(key2, 0.0) + entropy_pair
                nbrs[j].add(k)  # ensure connectivity lists cover all pairs
                nbrs[k].add(j)

    # pad neighbour lists for Canvas
    max_deg = max(len(s) for s in nbrs) if nbrs else 0
    nb_lists = [sorted(s) + [None] * (max_deg - len(s)) for s in nbrs]

    return linear, quad, nb_lists


if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "11_hea_ml_fm")

    # Define required fields
    required_fields = {
        "material": ["elements", "max_composition_ratio", "unit_sites", "alat"],
        "geometry": ["spc_size"],
        "thermodynamics": ["temperature", "physical_constants", "entropy_factors"],
        "model": ["model_txt", "data_directory"],
        "optimization": [
            "time_limit",
            "solver",
            "minimize_sense",
            "composition_bounds",
        ],
        "numerical": ["fm_threshold", "entropy_threshold"],
        "constraint_parameters": ["yi_fixed_value"],
        "energy_parameters": ["linear_coefficients"],
        "output": ["lp_export", "lp_export_dir", "lp_file_prefix", "lp_filename"],
    }

    # Validate configuration
    ConfigHandler.validate_config(config, required_fields)

    # ────────────── USER SETTINGS ───────────────────────────────────────────────
    # Get required parameters from config
    ELEMENTS = config["material"]["elements"]  # order must match FM
    MAX_COMPOSITION_RATIO = config["material"]["max_composition_ratio"]  #  Σ = 1.0
    UNIT_SITES = config["material"]["unit_sites"]  # BCC basis
    ALAT = config["material"]["alat"]  # Å (for POSCAR export)

    SPC_SIZE = config["geometry"]["spc_size"]  # super‑cell (Nx,Ny,Nz)

    TEMPERATURE = config["thermodynamics"]["temperature"]  # K
    phys_constants = config["thermodynamics"]["physical_constants"]
    entropy_factors = config["thermodynamics"]["entropy_factors"]

    MODEL_TXT = config["model"]["model_txt"]  # trained FM file
    data_directory = config["model"]["data_directory"]

    time_limit = config["optimization"]["time_limit"]
    solver = config["optimization"]["solver"]
    minimize_sense = config["optimization"]["minimize_sense"]
    comp_bounds_config = config["optimization"]["composition_bounds"]

    fm_threshold = config["numerical"]["fm_threshold"]
    entropy_threshold = config["numerical"]["entropy_threshold"]

    yi_fixed_val = config["constraint_parameters"]["yi_fixed_value"]
    linear_coefs = config["energy_parameters"]["linear_coefficients"]

    lp_export = config["output"]["lp_export"]
    lp_export_dir = config["output"]["lp_export_dir"]
    lp_file_prefix = config["output"]["lp_file_prefix"]
    lp_filename = config["output"]["lp_filename"]

    # ────────────── derived constants ───────────────────────────────────────────
    NSITES = len(UNIT_SITES) * np.prod(SPC_SIZE, dtype=int)
    NSPECIES = len(ELEMENTS)
    MAX_COUNTS = int(NSITES * MAX_COMPOSITION_RATIO)

    # factor  R⋅T  (eV per site)
    R_per_site = phys_constants["boltzmann_ev"] / phys_constants["avogadro"]  # eV K⁻¹
    RT_total = R_per_site * TEMPERATURE * NSITES  # eV · NSITES
    EQ_ATOMIC_COMP = entropy_factors["linear_multiplier"] / NSPECIES

    # entropy linear & quadratic coefficients
    C1 = (
        entropy_factors["linear_multiplier"]
        + math.log(EQ_ATOMIC_COMP)
        - NSPECIES * EQ_ATOMIC_COMP
    ) / NSITES
    C2 = (NSPECIES * entropy_factors["species_factor"]) / (NSITES**2)  # for s_i² term
    ENTROPY_LIN = RT_total * C1  # (eV) × s_i
    ENTROPY_PAIR = (
        RT_total * entropy_factors["quadratic_multiplier"] * C2
    )  # (eV) × x_ij (i<j)
    ENTROPY_CONST = (
        RT_total
        * NSPECIES
        * (
            EQ_ATOMIC_COMP * math.log(EQ_ATOMIC_COMP)
            - (entropy_factors["linear_multiplier"] + math.log(EQ_ATOMIC_COMP))
            * EQ_ATOMIC_COMP
            + (NSPECIES * entropy_factors["species_factor"]) * (EQ_ATOMIC_COMP**2)
        )
    )

    # Handle model file path - check if it's already absolute or relative to project root
    if os.path.isabs(MODEL_TXT):
        model_file = MODEL_TXT
    elif MODEL_TXT.startswith("data/"):
        # Path relative to project root
        model_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), MODEL_TXT)
    else:
        # Legacy format - just filename
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), data_directory
        )
        model_file = os.path.join(data_dir, MODEL_TXT)

    if not os.path.exists(model_file):
        sys.exit(f"[ERROR]  FM parameter file '{model_file}' not found.")

    N_total = NSPECIES * NSITES
    Qmat, base_offset = load_fm_model(model_file, N=N_total, Nf=NSITES)

    atoms = [Atom(e) for e in ELEMENTS]
    points = lattice_points(SPC_SIZE, UNIT_SITES, ALAT)

    linear, quad, nb_lists = split_Q_and_entropy(
        Qmat, atoms, NSITES, ENTROPY_LIN, ENTROPY_PAIR, fm_threshold, entropy_threshold
    )

    canv = Canvas(points, nb_lists)
    m = MatOptModel(canv, atoms)

    # FM + entropy linear descriptor (GLOBAL)
    m.addGlobalDescriptor(
        "Hlin",
        rules=EqualTo(SumSitesAndTypes(m.Yik, coefs=linear)),
    )

    # FM + entropy quadratic descriptor (GLOBAL)
    m.addGlobalDescriptor(
        "Hquad",
        rules=EqualTo(SumBondsAndTypes(m.Xijkl, coefs=quad)),
    )

    # fixed global composition  Σ_i Y_{ik}  =  specified count
    comp_bounds = {
        Atom(sym): (
            comp_bounds_config["lower"],
            int(MAX_COUNTS * comp_bounds_config["upper_multiplier"]),
        )
        for sym in ELEMENTS
    }
    m.addGlobalTypesDescriptor(
        "Composition",
        bounds=comp_bounds,
        rules=EqualTo(SumSites(m.Yik)),
    )
    m.Yi.rules.append(FixedTo(yi_fixed_val))

    constant_shift = float(base_offset[0]) + ENTROPY_CONST
    free_energy = LinearExpr(
        descs=[m.Hlin, m.Hquad], coefs=linear_coefs, offset=constant_shift
    )

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
        
        # Use config filename for LP file
        config_basename = ConfigHandler.get_config_basename(config_file)
        if config_basename:
            lp_file = f"{lp_export_subdir}/{config_basename}.lp"
        else:
            # Fallback to original naming if no config file provided
            lp_file = f"{lp_export_subdir}/{lp_file_prefix}.lp"
        # Create Pyomo model and export
        pyomo_model = m._make_pyomo_model(free_energy, sense=minimize_sense, formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    Design = m.minimize(free_energy, tilim=time_limit, solver=solver)

    if Design:
        print(f"Optimal free energy: {Design.free_energy}")
        print(f"Solver status: {Design.SolverStatus}")
    else:
        print("No solution found within time limit.")
