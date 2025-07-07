# CRYSIM: PREDICTION OF SYMMETRIC STRUCTURES OF LARGE CRYSTALS WITH GPU-BASED ISING MACHINES
# https://arxiv.org/abs/2504.06878
# https://github.com/tsudalab/CRYSIM
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_handler import ConfigHandler

import itertools
import numpy as np
import torch

from matopt.materials.atom import Atom
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSitesAndTypes, SumBondsAndTypes, LinearExpr
from matopt.aml.rule import EqualTo
from matopt.aml.model import MatOptModel


def load_fm(
    pt_file: str,
    device: str = "cpu",
    crysim_constants: dict = None,
    search_ranges: dict = None,
):
    """
    Read a CRYSIM TorchFM checkpoint and return

        w       - 1-D ndarray  (length N)          linear weights
        V       - 2-D ndarray  (N × k)             latent factors
        groups  - list[list[int]]                  one-hot blocks

    The routine deduces the split number *p*, number of atoms *n*
    and spg_precision automatically.  It relies on the fact that
    CRYSIM hard-codes  wp_precision = 300  and  crystal-system block = 7.
    """
    sd = torch.load(pt_file, map_location=device)
    try:
        w = sd["lin.weight"].squeeze().cpu().numpy()
        V = sd["V"].cpu().numpy()
    except KeyError as exc:
        raise RuntimeError(f"{pt_file} is not a TorchFM checkpoint - missing key {exc}")

    N = w.size  # total length of bit-string

    # Use config values or defaults
    if crysim_constants is None:
        crysim_constants = {
            "wp_precision": 300,
            "crystal_system_bits": 7,
            "max_spg": 120,
        }
    if search_ranges is None:
        search_ranges = {"split_numbers": [3, 61], "n_atoms": [1, 101]}

    WP = crysim_constants["wp_precision"]
    CS = crysim_constants["crystal_system_bits"]
    base = WP + CS  # bits whose size is fixed
    MAX_SPG = crysim_constants["max_spg"]

    # We need integers  p ≥ 3, n ≥ 1, 1 ≤ spg ≤ MAX_SPG  such that
    # N = 6p           (lat & ang)
    #   + CS
    #   + spg
    #   + WP
    #   + 3pn          (coords)
    #  ⇒  N - base = p(6 + 3n) + spg
    candidates = []
    p_min, p_max = search_ranges["split_numbers"]
    n_min, n_max = search_ranges["n_atoms"]
    for p in range(p_min, p_max):  # split numbers seen in CRYSIM
        for n in range(n_min, n_max):  # atoms (adjust if needed)
            rhs = N - base - p * (6 + 3 * n)
            if 1 <= rhs <= MAX_SPG:
                candidates.append((p, n, rhs))
    if not candidates:
        raise ValueError(
            "Could not factorise bit string - "
            "pt file inconsistent with CRYSIM layout"
        )
    # choose the candidate with the *largest* spg (most conservative)
    p, n_atoms, spg_precision = max(candidates, key=lambda t: t[2])

    # build one-hot blocks
    groups = []
    start = 0

    def push(sz):
        nonlocal start, groups
        groups.append(list(range(start, start + sz)))
        start += sz

    # 6 blocks: a,b,c,alpha,beta,gamma
    for _ in range(6):
        push(p)
    push(CS)  # crystal system
    push(spg_precision)  # space group
    push(WP)  # Wyckoff template
    # fractional coordinates: 3*p bits per atom
    for _ in range(n_atoms):
        for _ in range(3):  # x,y,z
            push(p)

    # final sanity-check
    if start != N:
        raise RuntimeError(
            f"Internal error: reconstructed {start} bits, " f"checkpoint contains {N}"
        )

    return w, V, groups


if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "10_crystal_ml_fm")

    # Define required fields
    required_fields = {
        "data": ["pt_filename", "data_directory"],
        "model": ["device", "crysim_constants", "search_ranges", "atom_on_type"],
        "optimization": ["time_limit", "solver", "minimize_sense"],
        "numerical": ["coefficient_threshold"],
        "geometry": ["coordinate_template"],
        "constraint_parameters": ["one_hot_bounds", "one_hot_coefficient"],
        "energy_parameters": ["linear_coefficients"],
        "output": ["lp_export", "lp_export_dir", "lp_file_prefix", "lp_filename"],
    }

    # Validate configuration
    ConfigHandler.validate_config(config, required_fields)

    # Get required parameters from config
    pt_filename = config["data"]["pt_filename"]
    data_directory = config["data"]["data_directory"]

    device = config["model"]["device"]
    crysim_constants = config["model"]["crysim_constants"]
    search_ranges = config["model"]["search_ranges"]
    atom_on_type = config["model"]["atom_on_type"]

    time_limit = config["optimization"]["time_limit"]
    solver = config["optimization"]["solver"]
    minimize_sense = config["optimization"]["minimize_sense"]

    coef_threshold = config["numerical"]["coefficient_threshold"]
    coord_template = config["geometry"]["coordinate_template"]
    one_hot_bounds = config["constraint_parameters"]["one_hot_bounds"]
    one_hot_coef = config["constraint_parameters"]["one_hot_coefficient"]
    linear_coefs = config["energy_parameters"]["linear_coefficients"]

    lp_export = config["output"]["lp_export"]
    lp_export_dir = config["output"]["lp_export_dir"]
    lp_file_prefix = config["output"]["lp_file_prefix"]
    lp_filename = config["output"]["lp_filename"]

    # Use data directory for model file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_directory)
    pt_file = os.path.join(data_dir, pt_filename)

    w, V, groups = load_fm(
        pt_file=pt_file,
        device=device,
        crysim_constants=crysim_constants,
        search_ranges=search_ranges,
    )

    N = w.size
    bits = list(range(N))
    k_lat = V.shape[1]

    # generate quadratic coefficients  Q_{i<j} = Σ_k V_ik V_jk
    Q_pairs = {}
    for i, j in itertools.combinations(bits, 2):
        coef = float(np.dot(V[i], V[j]))
        if abs(coef) > coef_threshold:
            Q_pairs[(i, j)] = coef
            Q_pairs[(j, i)] = coef  # MatOpt needs both directions

    # linear coefficients
    W_lin = {i: float(c) for i, c in enumerate(w) if abs(c) > coef_threshold}

    # neighbourhood list - MatOpt must know which sites interact
    max_deg = 0
    nbrs = [[] for _ in bits]
    for i, j in Q_pairs:
        nbrs[i].append(j)
    max_deg = max(len(s) for s in nbrs) if nbrs else 0
    nbrs = [sorted(li) + [None] * (max_deg - len(li)) for li in nbrs]

    # trivial 1-D coordinates (no physics implied)
    pts = np.array([[i] + coord_template for i in bits], dtype=float)

    # assemble the MatOpt model
    atom_on = Atom(atom_on_type)  # artificial atom that means x_i = 1
    canv = Canvas(pts, nbrs)
    m = MatOptModel(canv, [atom_on])

    # linear term
    lin_dict = {(site, atom_on): coef for site, coef in W_lin.items()}
    m.addGlobalDescriptor(
        "Hlin", rules=EqualTo(SumSitesAndTypes(m.Yik, coefs=lin_dict))
    )

    # quadratic term
    quad_dict = {(i, j, atom_on, atom_on): coef for (i, j), coef in Q_pairs.items()}
    m.addGlobalDescriptor(
        "Hquad", rules=EqualTo(SumBondsAndTypes(m.Xijkl, coefs=quad_dict))
    )

    # one-hot "exactly-one" constraints per CRYSIM group
    for gidx, grp in enumerate(groups):
        # Sum_{i in grp} x_i  == 1
        m.addGlobalDescriptor(
            f"group{gidx}",
            bounds=tuple(one_hot_bounds),
            rules=EqualTo(
                SumSitesAndTypes(m.Yik, coefs=one_hot_coef, sites_to_sum=grp)
            ),
        )

    E_tot = LinearExpr(descs=[m.Hlin, m.Hquad], coefs=linear_coefs)

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
        pyomo_model = m._make_pyomo_model(E_tot, sense=minimize_sense, formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    Design = m.minimize(E_tot, tilim=time_limit, solver=solver)

    if Design:
        print(f"Optimal energy: {Design.Etot}")
        print(f"Solver status: {Design.SolverStatus}")
    else:
        print("No solution found within time limit.")
