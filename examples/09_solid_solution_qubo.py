# Exploring the thermodynamics of disordered materials with quantum computing
# https://www.science.org/doi/10.1126/sciadv.adt7156
# https://github.com/cmc-ucl/QA_solid_solutions
from __future__ import annotations
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, List, Tuple
import numpy as np
from pymatgen.core.structure import Structure

from CRYSTALpytools.crystal_io import Crystal_gui
from CRYSTALpytools.convert import cry_gui2pmg

from matopt.materials.atom import Atom
from matopt.materials.canvas import Canvas
from matopt.aml.expr import (
    LinearExpr,
    SumSites,
    SumSitesAndTypes,
    SumBondTypes,
    SumBonds,
)
from matopt.aml.model import MatOptModel
from matopt.aml.rule import EqualTo, FixedTo
from utils.config_handler import ConfigHandler


def build_structure(conf: dict) -> Structure:
    """Build structure from configuration"""
    gui = Crystal_gui().read_gui(conf["data"]["structure_file"])
    st = cry_gui2pmg(gui)

    # Handle special cases
    special_cases = conf["material"].get("special_cases", {})
    for filename_pattern, spec in special_cases.items():
        if filename_pattern in conf["data"]["structure_file"]:
            # Replace element as specified
            st.replace_species(
                {
                    spec["replace_element"]: conf["material"]["substitution"][
                        "base_element"
                    ]
                }
            )

    return st


def detect_sites(structure: Structure, conf: dict) -> Tuple[List[int], List[int]]:
    """Detect substitutable and frozen sites"""
    n_sites = len(structure)
    sub_cfg = conf["material"]["substitution"]

    if sub_cfg["allowed_sites"] == "all":
        sub_sites = list(range(n_sites))
        frozen_sites = []
    elif sub_cfg["allowed_sites"] == "cations":
        fixed_el = sub_cfg.get("fixed_element")
        sub_sites = [
            i for i, site in enumerate(structure) if site.specie.symbol != fixed_el
        ]
        frozen_sites = [i for i in range(n_sites) if i not in sub_sites]
    else:  # explicit list
        sub_sites = list(sub_cfg["allowed_sites"])
        frozen_sites = [i for i in range(n_sites) if i not in sub_sites]

    return sub_sites, frozen_sites


def create_canvas(structure: Structure) -> Canvas:
    """Create MatOpt canvas"""
    positions = structure.cart_coords.tolist()
    n_sites = len(structure)

    # Build neighbor list - use complete graph to ensure all QUBO pairs are covered
    neighbors = []
    for i in range(n_sites):
        # Complete graph excluding self
        nbr_indices = [j for j in range(n_sites) if j != i]
        neighbors.append(nbr_indices)

    # Pad neighbor list to make it rectangular (Canvas requirement)
    max_deg = max(len(row) for row in neighbors)
    neighbors = [row + [None] * (max_deg - len(row)) for row in neighbors]

    return Canvas(positions, neighbors)


def get_atom_types(conf: dict) -> List[Atom]:
    """Get atom types for the system"""
    sub_config = conf["material"]["substitution"]
    atoms = [Atom(sub_config["base_element"]), Atom(sub_config["sub_element"])]

    return atoms


def add_basic_constraints(m: MatOptModel, conf: dict, frozen_sites: List[int]) -> None:
    """Pin frozen sites first, then one-hot rule for all sites."""
    fixed_vals = conf["constraint_parameters"]["fixed_values"]
    if frozen_sites:
        fixed_el = conf["material"]["substitution"]["fixed_element"]
        # Pin the fixed element to 1 on frozen sites
        m.addSitesTypesDescriptor(
            "Pinned",
            rules=FixedTo(fixed_vals["occupied"]),
            sites=frozen_sites,
            site_types=[Atom(fixed_el)],
        )
        # Pin all other elements to 0 on frozen sites (prevent double occupancy)
        for a in m.atoms:
            if a.symbol != fixed_el:
                m.addSitesTypesDescriptor(
                    f"NoSubOn{a.symbol}",
                    rules=FixedTo(fixed_vals["unoccupied"]),
                    sites=frozen_sites,
                    site_types=[a],
                )
    # one atom per site
    m.Yi.rules.append(FixedTo(fixed_vals["occupied"]))


def qubo_objective(
    m: MatOptModel,
    Q: np.ndarray,
    conf: dict,
    sub_sites: List[int],
    n_sites: int,
    mu: float = 0.0,
) -> LinearExpr:
    """Build QUBO objective with chemical potential."""
    # Get atom types
    sub_config = conf["material"]["substitution"]
    base_atom = Atom(sub_config["base_element"])
    sub_atom = Atom(sub_config["sub_element"])

    # Get numerical parameters
    num_config = conf["numerical"]
    energy_config = conf["energy_parameters"]

    # QUBO linear terms - initialize all to zero first
    lin_coefs = {}
    for site in range(n_sites):
        for atom in m.atoms:
            lin_coefs[(site, atom)] = 0.0

    # Set QUBO values for substitution sites
    for idx, i in enumerate(sub_sites):
        E_i = Q[idx, idx]
        lin_coefs[(i, sub_atom)] = float(E_i)
        lin_coefs[(i, base_atom)] = 0.0

    # Build the linear expression
    m.addGlobalDescriptor(
        "H_lin", rules=EqualTo(SumSitesAndTypes(m.Yik, coefs=lin_coefs))
    )

    # QUBO quadratic terms
    quad_coefs = {}

    # First, initialize all coefficients to zero for all site pairs and atom types
    for i in range(n_sites):
        for j in range(n_sites):
            if j <= i:  # Skip diagonal and lower triangle
                continue
            for k in m.atoms:
                for l in m.atoms:
                    quad_coefs[(i, j, k, l)] = 0.0

    # Then set the actual QUBO coefficients for substitutable site pairs
    for idx_i, i in enumerate(sub_sites):
        for idx_j, j in enumerate(sub_sites):
            if idx_j <= idx_i:  # Skip diagonal and lower triangle
                continue
            # Use indices into sub_sites to access QUBO
            J_ij = Q[idx_i, idx_j]
            if abs(J_ij) > num_config["quadratic_threshold"]:
                quad_coefs[(i, j, sub_atom, sub_atom)] = float(
                    num_config["quadratic_multiplier"] * J_ij
                )

    # Build the quadratic expression
    m.addBondsDescriptor(
        "H_pair",
        rules=EqualTo(SumBondTypes(m.Xijkl, coefs=quad_coefs)),
        symmetric_bonds=True,
    )
    m.addGlobalDescriptor("H_quad", rules=EqualTo(SumBonds(m.H_pair)))

    # Add chemical potential if non-zero
    if abs(mu) > 1e-12:
        # Create chemical potential descriptor
        m.addGlobalDescriptor(
            "ChemPot",
            rules=EqualTo(
                SumSites(m.Yik, sites_to_sum=sub_sites, site_types=[sub_atom])
            ),
        )
        # Return with mu as coefficient
        return LinearExpr([m.H_lin, m.H_quad, m.ChemPot], [1.0, 1.0, mu])

    # Total QUBO energy
    return LinearExpr([m.H_lin, m.H_quad], energy_config["linear_coefficients"])


def constrained_optimization(
    m: MatOptModel,
    obj: LinearExpr,
    conf: dict,
    sub_sites: List[int],
    config_file: str = None,
) -> Any:
    """Perform constrained optimization with fixed number of substitutions."""
    opt_config = conf["optimization"]
    output_config = conf["output"]
    n_sub = opt_config["target_substitutions"]

    # Add constraint for exact number of substitutions
    sub_atom = Atom(conf["material"]["substitution"]["sub_element"])
    m.addGlobalTypesDescriptor(
        "NSubst",
        bounds=(n_sub, n_sub),
        rules=EqualTo(SumSites(desc=m.Yik, site_types=[sub_atom])),
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
        lp_export_subdir = os.path.join(output_config["lp_export_dir"], subdir_name)
        os.makedirs(lp_export_subdir, exist_ok=True)

        # Use config filename for LP file
        config_basename = ConfigHandler.get_config_basename(config_file)
        if config_basename:
            lp_file = f"{lp_export_subdir}/{config_basename}.lp"
        else:
            # Fallback to original naming if no config file provided
            system_name = os.path.splitext(
                os.path.basename(conf["data"]["structure_file"])
            )[0]
            lp_file = f"{lp_export_subdir}/{output_config['lp_file_prefix']}_{system_name}_constrained.lp"

        # Create Pyomo model and export
        pyomo_model = m._make_pyomo_model(obj, sense=opt_config["minimize_sense"], formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    # Optimize
    D = m.minimize(
        obj,
        tilim=opt_config["time_limit"],
        solver=opt_config["solver"],
        disp=opt_config["verbose"],
    )

    return D


def unconstrained_search(
    m: MatOptModel,
    obj: LinearExpr,
    conf: dict,
    sub_sites: List[int],
    mu: float,
    config_file: str = None,
) -> Tuple[Any, int]:
    """Perform unconstrained optimization and count substitutions."""
    opt_config = conf["optimization"]
    output_config = conf["output"]

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
        lp_export_subdir = os.path.join(output_config["lp_export_dir"], subdir_name)
        os.makedirs(lp_export_subdir, exist_ok=True)

        # Use config filename for LP file
        config_basename = ConfigHandler.get_config_basename(config_file)
        if config_basename:
            lp_file = f"{lp_export_subdir}/{config_basename}.lp"
        else:
            # Fallback to original naming if no config file provided
            system_name = os.path.splitext(
                os.path.basename(conf["data"]["structure_file"])
            )[0]
            lp_file = f"{lp_export_subdir}/{output_config['lp_file_prefix']}_{system_name}_unconstrained.lp"

        # Create Pyomo model and export
        pyomo_model = m._make_pyomo_model(obj, sense=opt_config["minimize_sense"], formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    # Optimize
    D = m.minimize(
        obj,
        tilim=opt_config["time_limit"],
        solver=opt_config["solver"],
        disp=opt_config["verbose"],
    )

    if D:
        # Count substitutions
        sub_atom_symbol = conf["material"]["substitution"]["sub_element"]
        n_sub = sum(1 for atom in D.atoms if atom.symbol == sub_atom_symbol)
        return D, n_sub
    else:
        return None, 0


def print_structure_info(
    structure: Structure, sub_sites: List[int], frozen_sites: List[int]
) -> None:
    """Print structure information"""
    n_sites = len(structure)
    print(f"Total sites: {n_sites}")
    print(f"Substitutable sites: {len(sub_sites)}")
    print(f"Frozen sites: {len(frozen_sites)}")

    # Composition info
    composition = structure.composition
    print(f"Initial composition: {composition}")


def print_results(D: Any, conf: dict, n_sub: int = None) -> None:
    """Print optimization results"""
    if D:
        print("\nOptimization successful!")

        # Count each element type
        from collections import Counter

        composition = Counter(atom.symbol for atom in D.atoms)
        print(f"Final composition: {dict(composition)}")

        if n_sub is not None:
            print(f"Number of substitutions: {n_sub}")
    else:
        print("\nOptimization failed - no solution found within time limit")


if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "09_solid_solution_qubo")

    # Define required fields
    required_fields = {
        "data": ["structure_file", "qubo_path"],
        "material": ["substitution", "special_cases"],
        "optimization": ["mode", "time_limit", "solver", "verbose", "minimize_sense"],
        "numerical": ["quadratic_threshold", "quadratic_multiplier"],
        "energy_parameters": ["linear_coefficients"],
        "constraint_parameters": ["fixed_values"],
        "output": ["lp_export_dir", "lp_file_prefix"],
    }

    # Validate configuration
    ConfigHandler.validate_config(config, required_fields)

    print(f"Running solid solution optimization...")

    # Build structure
    structure = build_structure(config)

    # Detect sites
    sub_sites, frozen_sites = detect_sites(structure, config)
    n_sites = len(structure)

    # Print structure info
    print_structure_info(structure, sub_sites, frozen_sites)

    # Load QUBO matrix
    Q = np.load(config["data"]["qubo_path"])
    print(f"\nLoaded QUBO matrix: {Q.shape}")

    # Create canvas and model
    canvas = create_canvas(structure)
    atoms = get_atom_types(config)
    m = MatOptModel(canvas, atoms)

    # Add basic constraints
    add_basic_constraints(m, config, frozen_sites)

    # Get optimization parameters
    opt_config = config["optimization"]
    mu = opt_config.get("chemical_potential", 0.0)

    # Build QUBO objective
    obj = qubo_objective(m, Q, config, sub_sites, n_sites, mu)

    # Perform optimization based on mode
    if opt_config["mode"] == "constrained":
        print(
            f"\nRunning constrained optimization with {opt_config['target_substitutions']} substitutions..."
        )
        D = constrained_optimization(m, obj, config, sub_sites, config_file)
        print_results(D, config)
    else:  # unconstrained
        print(f"\nRunning unconstrained optimization with μ = {mu}...")
        D, n_sub = unconstrained_search(m, obj, config, sub_sites, mu, config_file)
        print_results(D, config, n_sub)
