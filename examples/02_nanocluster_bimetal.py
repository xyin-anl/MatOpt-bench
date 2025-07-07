# Designing stable bimetallic nanoclusters via an iterative two-step optimization approach
# https://pubs.rsc.org/en/content/articlelanding/2021/me/d1me00027f
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import sqrt
import numpy as np
from matopt.materials.atom import Atom
from matopt.materials.lattices.fcc_lattice import FCCLattice
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSites, SumBonds, SumBondTypes
from matopt.aml.rule import FixedTo, EqualTo, PiecewiseLinear
from matopt.aml.model import MatOptModel
from utils.config_handler import ConfigHandler

if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "02_nanocluster_bimetal")

    # Define required fields
    required_fields = {
        "material": ["iad", "atom1", "atom2", "comp_bounds_cu", "comp_bounds_ag"],
        "geometry": ["n_atoms", "initial_location", "n_shells_monometallic"],
        "optimization": [
            "solver",
            "time_limit_monometallic",
            "time_limit_bimetallic",
            "memory_limit",
            "maximize_sense",
        ],
        "energy_parameters": [
            "coordination_number_min",
            "coordination_number_max",
            "coordination_range",
            "cnri_lower_bound",
            "ecoh_normalization_factor",
            "yi_fixed_value",
            "gkl_coefs",
        ],
        "output": [
            "lp_export",
            "lp_export_dir",
            "lp_file_prefix",
            "lp_filename_monometallic",
            "lp_filename_bimetallic",
        ],
        "misc": ["core_ratio"],
    }

    # Validate configuration
    ConfigHandler.validate_config(config, required_fields)

    # Get required parameters from config
    IAD = config["material"]["iad"]
    atom1 = config["material"]["atom1"]
    atom2 = config["material"]["atom2"]
    comp_bounds_cu = config["material"]["comp_bounds_cu"]
    comp_bounds_ag = config["material"]["comp_bounds_ag"]

    n_atoms = config["geometry"]["n_atoms"]
    initial_location = np.array(config["geometry"]["initial_location"], dtype=float)
    n_shells_mono = config["geometry"]["n_shells_monometallic"]

    solver = config["optimization"]["solver"]
    time_limit1 = config["optimization"]["time_limit_monometallic"]
    time_limit2 = config["optimization"]["time_limit_bimetallic"]
    memory_limit = config["optimization"]["memory_limit"]
    maximize_sense = config["optimization"]["maximize_sense"]

    cn_min = config["energy_parameters"]["coordination_number_min"]
    cn_max = config["energy_parameters"]["coordination_number_max"]
    cn_range = config["energy_parameters"]["coordination_range"]
    cnri_lower_bound = config["energy_parameters"]["cnri_lower_bound"]
    ecoh_norm_factor = config["energy_parameters"]["ecoh_normalization_factor"]
    yi_fixed_value = config["energy_parameters"]["yi_fixed_value"]
    gkl_config = config["energy_parameters"]["gkl_coefs"]

    lp_export = config["output"]["lp_export"]
    lp_export_dir = config["output"]["lp_export_dir"]
    lp_file_prefix = config["output"]["lp_file_prefix"]
    lp_filename1 = config["output"]["lp_filename_monometallic"]
    lp_filename2 = config["output"]["lp_filename_bimetallic"]

    core_ratio = config["misc"]["core_ratio"]

    # MONOMETALIC OPT
    Lat = FCCLattice(IAD=IAD)
    Canv = Canvas()
    Canv.addLocation(initial_location)
    Canv.addShells(n_shells_mono, Lat.getNeighbors)
    Atoms = [Atom(atom1)]
    N = n_atoms
    m = MatOptModel(Canv, Atoms)
    Vals = [sqrt(CN) for CN in range(cn_min, cn_range)]
    BPs = [CN for CN in range(cn_min, cn_range)]
    m.addSitesDescriptor(
        "CNRi",
        bounds=(cnri_lower_bound, sqrt(cn_max)),
        integer=False,
        rules=PiecewiseLinear(values=Vals, breakpoints=BPs, input_desc=m.Ci),
    )
    m.addGlobalDescriptor(
        "Ecoh",
        rules=EqualTo(
            SumSites(desc=m.CNRi, coefs=(ecoh_norm_factor / (N * sqrt(cn_max))))
        ),
    )
    m.addGlobalDescriptor("Size", bounds=(N, N), rules=EqualTo(SumSites(desc=m.Yi)))

    D = m.maximize(m.Ecoh, tilim=time_limit1, solver=solver)

    # PROCESSING SOLUTION
    Canv = Canvas()
    for i in range(len(D)):
        if D.Contents[i] is not None:
            Canv.addLocation(D.Canvas.Points[i])
    Canv.setNeighborsFromFunc(Lat.getNeighbors)

    Atoms = [Atom(atom1), Atom(atom2)]
    CompBounds = {Atom(atom1): comp_bounds_cu, Atom(atom2): comp_bounds_ag}

    m = MatOptModel(Canv, Atoms)

    m.Yi.rules.append(FixedTo(yi_fixed_value))
    # Get Gkl coefficients from config
    GklCoefs = {
        (Atom(atom1), Atom(atom1)): gkl_config[f"{atom1}_{atom1}"],
        (Atom(atom1), Atom(atom2)): gkl_config[f"{atom1}_{atom2}"],
        (Atom(atom2), Atom(atom2)): gkl_config[f"{atom2}_{atom2}"],
        (Atom(atom2), Atom(atom1)): gkl_config[f"{atom2}_{atom1}"],
    }
    BEijCoefs = {}
    for i in range(len(Canv)):
        CNi = sum(1 for _ in Canv.NeighborhoodIndexes[i] if _ is not None)
        for j in Canv.NeighborhoodIndexes[i]:
            if j is not None:
                CNj = sum(1 for _ in Canv.NeighborhoodIndexes[j] if _ is not None)
                for k in Atoms:
                    for l in Atoms:
                        BEijCoefs[i, j, k, l] = GklCoefs[k, l] * 1 / sqrt(
                            CNi
                        ) + GklCoefs[l, k] * 1 / sqrt(CNj)
    m.addBondsDescriptor(
        "BEij",
        rules=EqualTo(SumBondTypes(m.Xijkl, coefs=BEijCoefs)),
        symmetric_bonds=True,
    )

    m.addGlobalDescriptor(
        "Ecoh",
        rules=EqualTo(
            SumBonds(desc=m.BEij, coefs=ecoh_norm_factor / (N * sqrt(cn_max)))
        ),
    )
    m.addGlobalTypesDescriptor(
        "Composition", bounds=CompBounds, rules=EqualTo(SumSites(desc=m.Yik))
    )

    # Export LP file if requested (after second step)
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
        pyomo_model = m._make_pyomo_model(m.Ecoh, sense=maximize_sense, formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    D = m.maximize(m.Ecoh, tilim=time_limit2, trelim=memory_limit, solver=solver)

    if D:
        print(f"Optimal cohesive energy: {D.Ecoh}")
        print(f"Solver status: {D.SolverStatus}")
