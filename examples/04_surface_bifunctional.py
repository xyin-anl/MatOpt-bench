# Multiscale Modeling Combined with Active Learning for Microstructure Optimization of Bifunctional Catalysts
# https://pubs.acs.org/doi/full/10.1021/acs.iecr.8b04801
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import sqrt
import numpy as np
from matopt.materials.atom import Atom
from matopt.materials.lattices.fcc_lattice import FCCLattice
from matopt.materials.geometry import Parallelepiped
from matopt.materials.tiling import PlanarTiling
from matopt.materials.canvas import Canvas
from matopt.materials.design import Design
from matopt.aml.expr import SumSites, SumConfs, LinearExpr, SumBonds
from matopt.aml.rule import (
    FixedTo,
    Implies,
    ImpliesSiteCombination,
    EqualTo,
    GreaterThan,
    PiecewiseLinear,
)
from matopt.aml.model import MatOptModel
from utils.config_handler import ConfigHandler

if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "04_surface_bifunctional")

    # Validate required fields
    required_fields = {
        "materials": ["iad_multiplier", "lattice_type", "atoms", "canvas_atom"],
        "geometry": [
            "n_unit_cells_on_edge",
            "n_layers",
            "alpha",
            "beta",
            "gamma",
            "shift_factor",
            "fixed_pt_layers",
            "canvas_bottom_layers",
        ],
        "motifs": [
            "canvas_neighbors",
            "configurations",
            "type_a_confs",
            "type_b_confs",
        ],
        "optimization": ["cat_weight", "time_limit", "solver", "cat_norm_factor"],
        "energy_parameters": [
            "undefected_surf_e",
            "max_surf_e",
            "e_i_values",
            "e_i_breakpoints",
            "energy_offset",
        ],
        "output": ["lp_export", "lp_filename", "lp_export_dir"],
    }
    ConfigHandler.validate_config(config, required_fields)

    # Extract material parameters
    mat_config = config["materials"]
    IAD = sqrt(2) * mat_config["iad_multiplier"]
    Atoms = [Atom(atom_type) for atom_type in mat_config["atoms"]]
    canvas_atom = Atom(mat_config["canvas_atom"])

    # Extract geometry parameters
    geom_config = config["geometry"]
    nUnitCellsOnEdge = geom_config["n_unit_cells_on_edge"]
    nLayers = geom_config["n_layers"]
    alpha = geom_config["alpha"]
    beta = geom_config["beta"]
    gamma = geom_config["gamma"]
    shift_factor = geom_config["shift_factor"]
    fixed_pt_layers = geom_config["fixed_pt_layers"]
    canvas_bottom_layers = geom_config["canvas_bottom_layers"]

    # Extract motif parameters
    motif_config = config["motifs"]
    canvas_neighbors = motif_config["canvas_neighbors"]
    motif_configurations = motif_config["configurations"]
    TypeAConfs = motif_config["type_a_confs"]
    TypeBConfs = motif_config["type_b_confs"]

    # Extract optimization parameters
    opt_config = config["optimization"]
    CatWeight = opt_config["cat_weight"]
    time_limit = opt_config["time_limit"]
    solver = opt_config["solver"]
    cat_norm_factor = opt_config["cat_norm_factor"]

    # Extract energy parameters
    energy_config = config["energy_parameters"]
    UndefectedSurfE = energy_config["undefected_surf_e"]
    maxSurfE = energy_config["max_surf_e"]
    EiVals = energy_config["e_i_values"]
    EiBPs = energy_config["e_i_breakpoints"]
    energy_offset = energy_config["energy_offset"]

    # Build lattice
    Lat = FCCLattice.alignedWith111(IAD)
    a = nUnitCellsOnEdge * IAD
    b = a
    c = nLayers * Lat.FCC111LayerSpacing

    # Create parallelepiped
    S = Parallelepiped.fromEdgesAndAngles(a, b, c, alpha, beta, gamma)
    S.shift(np.array([shift_factor * a, shift_factor * b, shift_factor * c]))
    T = PlanarTiling(S)

    # Create canvas
    Canv = Canvas.fromLatticeAndTilingScan(Lat, T)

    # Create motif canvas
    MotifCanvas = Canvas()
    MotifCanvas.addLocation(
        np.array([0, 0, 0], dtype=float), NNeighbors=canvas_neighbors
    )
    MotifCanvas.addShell(Lat.getNeighbors)

    # Build configurations from config
    Confs = [
        [None] * len(MotifCanvas.NeighborhoodIndexes[0])
        for _ in range(len(motif_configurations))
    ]

    for iConf, conf_data in enumerate(motif_configurations):
        for i in conf_data["ni_indices"]:
            Confs[iConf][i] = Atom("Ni")
        for i in conf_data["pt_indices"]:
            Confs[iConf][i] = Atom("Pt")

    # Define location sets
    LocsToFixPt = [
        i
        for i in range(len(Canv))
        if Canv.Points[i][2] < Lat.FCC111LayerSpacing * fixed_pt_layers
    ]
    LocsToExcludePt = [i for i in range(len(Canv)) if i not in LocsToFixPt]
    CanvTwoBotLayers = [
        i
        for i in range(len(Canv))
        if Canv.Points[i][2] < Lat.FCC111LayerSpacing * canvas_bottom_layers
    ]
    CanvMinusTwoBotLayers = [i for i in range(len(Canv)) if i not in CanvTwoBotLayers]

    # Handle case where all sites are fixed
    if LocsToExcludePt:
        OneLocToFix = [min(LocsToExcludePt)]
    else:
        # If all sites are fixed, just use empty list
        OneLocToFix = []

    # Calculate normalization factors
    TileSizeSquared = nUnitCellsOnEdge**2
    CatNorm = TileSizeSquared * cat_norm_factor

    # Create model
    m = MatOptModel(Canv, Atoms, Confs)

    # Add rules
    m.Yik.rules.append(FixedTo(1, sites=LocsToFixPt, site_types=[Atom("Pt")]))
    m.Yik.rules.append(FixedTo(0, sites=LocsToExcludePt, site_types=[Atom("Pt")]))

    m.Zic.rules.append(FixedTo(1, sites=OneLocToFix, confs=TypeAConfs))
    m.Zic.rules.append(Implies(concs=(m.Yik, EqualTo(1, site_types=[Atom("Ni")]))))

    # Add descriptors
    SumAConfsExpr = SumConfs(m.Zic, confs_to_sum=TypeAConfs)
    SumBConfsExpr = SumConfs(m.Zic, confs_to_sum=TypeBConfs)
    m.addBondsDescriptor(
        "SiteCombinations",
        binary=True,
        rules=ImpliesSiteCombination(
            Canv, (SumAConfsExpr, GreaterThan(1)), (SumBConfsExpr, GreaterThan(1))
        ),
    )
    m.addGlobalDescriptor(
        "Activity", rules=EqualTo(SumBonds(m.SiteCombinations, coefs=1 / CatNorm))
    )

    # Energy descriptors
    m.addSitesDescriptor(
        "Ei",
        rules=PiecewiseLinear(values=EiVals, breakpoints=EiBPs, input_desc=m.Ci),
        sites=CanvMinusTwoBotLayers,
    )
    m.addGlobalDescriptor(
        "Esurf",
        rules=EqualTo(SumSites(m.Ei, coefs=1 / TileSizeSquared, offset=energy_offset)),
    )
    m.addGlobalDescriptor(
        "Stability", rules=EqualTo(LinearExpr(m.Esurf, 1 / UndefectedSurfE))
    )

    m.addGlobalDescriptor(
        "ActAndStab",
        rules=EqualTo(
            LinearExpr(
                descs=[m.Stability, m.Activity], coefs=[-(1 - CatWeight), CatWeight]
            )
        ),
    )

    # Export LP file if requested
    if "--export-lp" in sys.argv or config["output"]["lp_export"]:
        # Determine formulation type from command line argument
        formulation = "milp"  # default
        if "--miqcqp" in sys.argv:
            formulation = "miqcqp"
        elif "--milp" in sys.argv:
            formulation = "milp"
        
        # Create subdirectory based on formulation type
        # Keep using 'miqcp' for subdirectory name for consistency
        subdir_name = 'miqcp' if formulation == 'miqcqp' else formulation
        lp_export_subdir = os.path.join(config["output"]["lp_export_dir"], subdir_name)
        os.makedirs(lp_export_subdir, exist_ok=True)

        # Use config filename for LP file
        config_basename = ConfigHandler.get_config_basename(config_file)
        if config_basename:
            lp_file = f"{lp_export_subdir}/{config_basename}.lp"
        else:
            # Fallback to config filename
            lp_file = (
                f"{lp_export_subdir}/{config['output']['lp_filename']}"
            )

        pyomo_model = m._make_pyomo_model(m.Activity, sense=-1, formulation=formulation)  # -1 for maximize
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        if "--export-lp" in sys.argv:
            sys.exit(0)

    # Solve
    D = m.maximize(m.ActAndStab, tilim=time_limit, solver=solver)

    # Print results
    if D:
        print(f"Optimal activity and stability: {D.ActAndStab}")
        print(f"Activity: {D.Activity}")
        print(f"Stability: {D.Stability}")
        print(f"Surface energy: {D.Esurf}")
