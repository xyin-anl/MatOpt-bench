# Identification of optimally stable nanocluster geometries via mathematical optimization and density-functional theory
# https://pubs.rsc.org/en/content/articlelanding/2020/me/c9me00108e

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from math import sqrt
from matopt.materials.atom import Atom
from matopt.materials.lattices.fcc_lattice import FCCLattice
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSites
from matopt.aml.rule import PiecewiseLinear, EqualTo
from matopt.aml.model import MatOptModel
from utils.config_handler import ConfigHandler

if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "01_nanocluster_mono")

    # Validate required fields
    required_fields = {
        "material": ["element", "inter_atomic_distance"],
        "geometry": ["cluster_size", "n_shells", "initial_location"],
        "optimization": ["solver", "time_limit", "maximize_sense"],
        "energy_parameters": [
            "coordination_bounds",
            "piecewise_points",
            "cnri_lower_bound",
            "ecoh_normalization_factor",
        ],
        "output": ["lp_export_dir", "lp_file_prefix"],
    }
    ConfigHandler.validate_config(config, required_fields)

    # Extract parameters from config
    element = config["material"]["element"]
    IAD = config["material"]["inter_atomic_distance"]
    N = config["geometry"]["cluster_size"]
    n_shells = config["geometry"]["n_shells"]
    initial_location = np.array(config["geometry"]["initial_location"], dtype=float)
    solver = config["optimization"]["solver"]
    tilim = config["optimization"]["time_limit"]
    mip_gap = config["optimization"].get("mip_gap")
    maximize_sense = config["optimization"]["maximize_sense"]
    cn_bounds = config["energy_parameters"]["coordination_bounds"]
    pw_points = config["energy_parameters"]["piecewise_points"]
    cnri_lower_bound = config["energy_parameters"]["cnri_lower_bound"]
    ecoh_norm_factor = config["energy_parameters"]["ecoh_normalization_factor"]
    lp_export_dir = config["output"]["lp_export_dir"]
    lp_file_prefix = config["output"]["lp_file_prefix"]

    # Build model
    Lat = FCCLattice(IAD=IAD)
    Canv = Canvas()
    Canv.addLocation(initial_location)
    Canv.addShells(n_shells, Lat.getNeighbors)
    Canv.setNeighborsFromFunc(Lat.getNeighbors)
    Atoms = [Atom(element)]

    m = MatOptModel(Canv, Atoms)
    m.addSitesDescriptor(
        "CNRi",
        bounds=(cnri_lower_bound, sqrt(cn_bounds[1])),
        integer=False,
        rules=PiecewiseLinear(
            values=[sqrt(CN) for CN in range(pw_points + 1)],
            breakpoints=[CN for CN in range(pw_points + 1)],
            input_desc=m.Ci,
            con_type="UB",
        ),
    )
    m.addGlobalDescriptor(
        "Ecoh",
        rules=EqualTo(
            SumSites(desc=m.CNRi, coefs=(ecoh_norm_factor / sqrt(cn_bounds[1]) / N))
        ),
    )
    m.addGlobalDescriptor("Size", bounds=(N, N), rules=EqualTo(SumSites(desc=m.Yi)))

    # Solve
    solver_options = {"tilim": tilim}
    if mip_gap is not None:
        solver_options["mipgap"] = mip_gap

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
            lp_file = f"{lp_export_subdir}/{lp_file_prefix}_N{N}.lp"

        # Create Pyomo model and export
        pyomo_model = m._make_pyomo_model(m.Ecoh, sense=maximize_sense, formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    # Solve the optimization problem
    D = m.maximize(m.Ecoh, solver=solver, **solver_options)

    # Print results
    if D:
        print(f"Optimal cohesive energy: {D.Ecoh}")
        print(f"Solver status: {D.SolverStatus}")
