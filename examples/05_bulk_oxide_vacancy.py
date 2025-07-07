# A framework for optimizing oxygen vacancy formation in doped perovskites
# https://www.sciencedirect.com/science/article/abs/pii/S0098135418310998
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tempfile import TemporaryDirectory
from zipfile import ZipFile
import numpy as np
from matopt.materials.atom import Atom
from matopt.materials.lattices.perovskite_lattice import PerovskiteLattice
from matopt.materials.geometry import RectPrism
from matopt.materials.tiling import CubicTiling
from matopt.materials.canvas import Canvas
from matopt.materials.design import loadFromPDBs
from matopt.aml.expr import SumSites, SumNeighborSites, SumSitesAndConfs
from matopt.aml.rule import FixedTo, EqualTo
from matopt.aml.model import MatOptModel
from utils.config_handler import ConfigHandler

if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "05_bulk_oxide_vacancy")

    # Define required fields
    required_fields = {
        "material": ["lattice_parameters", "atom_types", "atoms"],
        "geometry": ["n_unit_cells_on_edge", "shift_vector"],
        "optimization": [
            "solver",
            "time_limit",
            "maximize_sense",
            "pct_local_lb",
            "pct_local_ub",
            "pct_global_lb",
            "pct_global_ub",
        ],
        "configurations": [
            "desired_conf_indices",
            "data_directory",
            "zip_filename",
            "conf_subdirectory",
        ],
        "constraint_parameters": ["fixed_values"],
        "output": ["lp_export", "lp_export_dir", "lp_file_prefix", "lp_filename"],
    }

    # Validate configuration
    ConfigHandler.validate_config(config, required_fields)

    # Get required parameters from config
    lattice_params = config["material"]["lattice_parameters"]
    A = lattice_params["a"]
    B = lattice_params["b"]
    C = lattice_params["c"]
    atom_types = config["material"]["atom_types"]
    atom_config = config["material"]["atoms"]

    nUnitCellsOnEdge = config["geometry"]["n_unit_cells_on_edge"]
    shift_vector = np.array(config["geometry"]["shift_vector"])

    solver = config["optimization"]["solver"]
    time_limit = config["optimization"]["time_limit"]
    maximize_sense = config["optimization"]["maximize_sense"]
    pct_local_lb = config["optimization"]["pct_local_lb"]
    pct_local_ub = config["optimization"]["pct_local_ub"]
    pct_global_lb = config["optimization"]["pct_global_lb"]
    pct_global_ub = config["optimization"]["pct_global_ub"]

    iDesiredConfs = config["configurations"]["desired_conf_indices"]
    data_directory = config["configurations"]["data_directory"]
    zip_filename = config["configurations"]["zip_filename"]
    conf_subdirectory = config["configurations"]["conf_subdirectory"]

    fixed_values = config["constraint_parameters"]["fixed_values"]
    occupied_val = fixed_values["occupied"]
    unoccupied_val = fixed_values["unoccupied"]

    lp_export = config["output"]["lp_export"]
    lp_export_dir = config["output"]["lp_export_dir"]
    lp_file_prefix = config["output"]["lp_file_prefix"]
    lp_filename = config["output"]["lp_filename"]

    Lat = PerovskiteLattice(A, B, C)
    S = RectPrism(nUnitCellsOnEdge * A, nUnitCellsOnEdge * B, nUnitCellsOnEdge * C)
    S.shift(shift_vector)
    T = CubicTiling(S)

    Canv = Canvas.fromLatticeAndTilingScan(Lat, T)
    Atoms = [Atom(atom_type) for atom_type in atom_types]

    # Use data directory for configuration files
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_directory)
    confs_file = os.path.join(data_dir, zip_filename)

    with TemporaryDirectory() as ConfDir:
        ZipFile(confs_file).extractall(ConfDir)
        ConfDesigns = loadFromPDBs(
            [str(i) + ".pdb" for i in iDesiredConfs],
            folder=ConfDir + "/" + conf_subdirectory + "/",
        )
    Confs = [Conf.Contents for Conf in ConfDesigns]

    Sites = [i for i in range(len(Canv))]
    ASites = [i for i in Sites if Lat.isASite(Canv.Points[i])]
    BSites = [i for i in Sites if Lat.isBSite(Canv.Points[i])]
    OSites = [i for i in Sites if Lat.isOSite(Canv.Points[i])]
    pctLocalLB, pctLocalUB = pct_local_lb, pct_local_ub
    pctGlobalLB, pctGlobalUB = pct_global_lb, pct_global_ub
    LocalBounds = {
        (i, Atom("In")): (
            round(pctLocalLB * len(Canv.NeighborhoodIndexes[i])),
            round(pctLocalUB * len(Canv.NeighborhoodIndexes[i])),
        )
        for i in OSites
    }
    GlobalLB = round(pctGlobalLB * len(BSites))
    GlobalUB = round(pctGlobalUB * len(BSites))

    m = MatOptModel(Canv, Atoms, Confs)

    m.Yik.rules.append(
        FixedTo(occupied_val, sites=ASites, site_types=[Atom(atom_config["a_site"])])
    )
    m.Yik.rules.append(
        FixedTo(occupied_val, sites=OSites, site_types=[Atom(atom_config["o_site"])])
    )
    m.Yik.rules.append(
        FixedTo(
            unoccupied_val,
            sites=BSites,
            site_types=[Atom(atom_config["a_site"]), Atom(atom_config["o_site"])],
        )
    )
    m.Yi.rules.append(FixedTo(occupied_val, sites=BSites))

    m.addGlobalDescriptor(
        "Activity",
        rules=EqualTo(
            SumSitesAndConfs(
                m.Zic, coefs=occupied_val / len(OSites), sites_to_sum=OSites
            )
        ),
    )
    m.addGlobalTypesDescriptor(
        "GlobalBudget",
        bounds=(GlobalLB, GlobalUB),
        rules=EqualTo(SumSites(m.Yik, site_types=[Atom("In")], sites_to_sum=BSites)),
    )
    m.addSitesTypesDescriptor(
        "LocalBudget",
        bounds=LocalBounds,
        rules=EqualTo(SumNeighborSites(m.Yik, sites=OSites, site_types=[Atom("In")])),
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
        pyomo_model = m._make_pyomo_model(m.Activity, sense=maximize_sense, formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    D = m.maximize(m.Activity, tilim=time_limit, solver=solver)

    if D:
        print(f"Optimal activity: {D.Activity}")
        print(f"Solver status: {D.SolverStatus}")
