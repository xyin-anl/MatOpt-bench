# A mathematical optimization framework for the design of nanopatterned surfaces.
# https://doi.org/10.1002/aic.15359

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matopt.materials.atom import Atom
from matopt.materials.lattices.fcc_lattice import FCCLattice
from matopt.materials.geometry import Parallelepiped
from matopt.materials.tiling import PlanarTiling
from matopt.materials.canvas import Canvas
from matopt.materials.design import Design
from matopt.aml.expr import SumSites, SumNeighborSites
from matopt.aml.rule import (
    FixedTo,
    EqualTo,
    PiecewiseLinear,
    ImpliesNeighbors,
    GreaterThan,
    LessThan,
    Implies,
    LinearExpr,
)
from matopt.aml.model import MatOptModel
from utils.config_handler import ConfigHandler

if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "03_surface_design")

    # Validate required fields
    required_fields = {
        "material": ["element", "inter_atomic_distance"],
        "geometry": [
            "atoms_on_edge",
            "n_layers",
            "fixed_bottom_layers",
            "parallelepiped",
            "layer_spacing_multipliers",
        ],
        "optimization": [
            "target_gcn",
            "catalyst_weight",
            "solver",
            "time_limit",
            "maximize_sense",
        ],
        "energy_parameters": [
            "surface_CN_bounds",
            "undefected_surface_energy",
            "max_surface_energy",
            "E_i_values",
            "E_i_breakpoints",
            "gcni_bounds",
            "gcni_normalization",
            "esurf_offset",
            "stability_normalization_sign",
            "stability_offset",
        ],
        "numerical": ["double_tolerance"],
        "constraint_parameters": ["yi_fixed_value"],
        "output": ["lp_export_dir", "lp_file_prefix"],
    }
    ConfigHandler.validate_config(config, required_fields)

    # Extract parameters from config
    element = config["material"]["element"]
    IAD = config["material"]["inter_atomic_distance"]
    nAtomsOnEdge = config["geometry"]["atoms_on_edge"]
    nLayers = config["geometry"]["n_layers"]
    fixed_layers = config["geometry"]["fixed_bottom_layers"]
    alpha = config["geometry"]["parallelepiped"]["alpha"]
    beta = config["geometry"]["parallelepiped"]["beta"]
    gamma = config["geometry"]["parallelepiped"]["gamma"]
    shift_factor = config["geometry"]["parallelepiped"]["shift_factor"]
    bottom_layer_mult = config["geometry"]["layer_spacing_multipliers"]["bottom_layers"]
    top_layer_mult = config["geometry"]["layer_spacing_multipliers"]["top_layer"]

    TargetGCN = config["optimization"]["target_gcn"]
    CatWeight = config["optimization"]["catalyst_weight"]
    solver = config["optimization"]["solver"]
    tilim = config["optimization"]["time_limit"]
    mip_gap = config["optimization"].get("mip_gap")
    maximize_sense = config["optimization"]["maximize_sense"]

    CNsurfMin = config["energy_parameters"]["surface_CN_bounds"][0]
    CNsurfMax = config["energy_parameters"]["surface_CN_bounds"][1]
    UndefectedSurfE = config["energy_parameters"]["undefected_surface_energy"]
    maxSurfE = config["energy_parameters"]["max_surface_energy"]
    EiVals = config["energy_parameters"]["E_i_values"]
    EiBPs = config["energy_parameters"]["E_i_breakpoints"]
    gcni_bounds = config["energy_parameters"]["gcni_bounds"]
    gcni_norm = config["energy_parameters"]["gcni_normalization"]
    esurf_offset = config["energy_parameters"]["esurf_offset"]
    stab_norm_sign = config["energy_parameters"]["stability_normalization_sign"]
    stab_offset = config["energy_parameters"]["stability_offset"]

    DBL_TOL = config["numerical"]["double_tolerance"]
    yi_fixed_val = config["constraint_parameters"]["yi_fixed_value"]
    lp_export_dir = config["output"]["lp_export_dir"]
    lp_file_prefix = config["output"]["lp_file_prefix"]

    # Build lattice and canvas
    Lat = FCCLattice.alignedWith111(IAD)

    a = nAtomsOnEdge * IAD
    b = a
    c = nLayers * Lat.FCC111LayerSpacing
    S = Parallelepiped.fromEdgesAndAngles(a, b, c, alpha, beta, gamma)
    S.shift(np.array([shift_factor * a, shift_factor * b, shift_factor * c]))
    T = PlanarTiling(S)

    Canv = Canvas.fromLatticeAndTilingScan(Lat, T)

    D = Design(Canv, Atom(element))

    # Build optimization model
    Atoms = [Atom(element)]
    TileSizeSquared = nAtomsOnEdge**2

    m = MatOptModel(Canv, Atoms)

    # Fixed bottom layers
    CanvTwoBotLayers = [
        i
        for i in range(len(Canv))
        if Canv.Points[i][2] < bottom_layer_mult * Lat.FCC111LayerSpacing
    ]
    CanvMinusTwoBotLayers = [i for i in range(len(Canv)) if i not in CanvTwoBotLayers]
    OneSiteInTopLayer = [
        min(
            [
                i
                for i in range(len(Canv))
                if Canv.Points[i][2]
                > (nLayers - top_layer_mult) * Lat.FCC111LayerSpacing
            ]
        )
    ]

    m.Yi.rules.append(FixedTo(yi_fixed_val, sites=OneSiteInTopLayer))
    m.Yi.rules.append(FixedTo(yi_fixed_val, sites=CanvTwoBotLayers))

    # Support constraint
    NeighborsBelow = [
        [
            j
            for j in Canv.NeighborhoodIndexes[i]
            if (j is not None and Canv.Points[j][2] < Canv.Points[i][2] - DBL_TOL)
        ]
        for i in range(len(Canv))
    ]
    m.Yi.rules.append(
        ImpliesNeighbors(
            concs=(m.Yi, GreaterThan(1)),
            sites=CanvMinusTwoBotLayers,
            neighborhoods=NeighborsBelow,
        )
    )

    # Coordination descriptors
    m.addSitesDescriptor(
        "GCNi",
        bounds=tuple(gcni_bounds),
        integer=False,
        rules=EqualTo(SumNeighborSites(desc=m.Ci, coefs=1 / gcni_norm)),
        sites=CanvMinusTwoBotLayers,
    )
    m.addSitesDescriptor(
        "IdealSitei",
        binary=True,
        rules=[
            Implies(concs=(m.Ci, GreaterThan(CNsurfMin))),
            Implies(concs=(m.Ci, LessThan(CNsurfMax))),
            Implies(concs=(m.GCNi, EqualTo(TargetGCN))),
        ],
        sites=CanvMinusTwoBotLayers,
    )

    # Activity descriptor
    m.addGlobalDescriptor(
        "Activity", rules=EqualTo(SumSites(m.IdealSitei, coefs=1 / TileSizeSquared))
    )

    # Surface energy model
    m.addSitesDescriptor(
        "Ei",
        rules=PiecewiseLinear(values=EiVals, breakpoints=EiBPs, input_desc=m.Ci),
        sites=CanvMinusTwoBotLayers,
    )
    m.addGlobalDescriptor(
        "Esurf",
        rules=EqualTo(SumSites(m.Ei, coefs=1 / TileSizeSquared, offset=esurf_offset)),
    )
    m.addGlobalDescriptor(
        "Stability",
        rules=EqualTo(
            LinearExpr(
                descs=m.Esurf,
                coefs=stab_norm_sign / UndefectedSurfE,
                offset=stab_offset,
            )
        ),
    )

    # Combined objective
    m.addGlobalDescriptor(
        "ActAndStab",
        rules=EqualTo(
            LinearExpr(
                descs=[m.Activity, m.Stability], coefs=[CatWeight, (1 - CatWeight)]
            )
        ),
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
        pyomo_model = m._make_pyomo_model(m.ActAndStab, sense=maximize_sense, formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    # Solve
    solver_options = {"tilim": tilim}
    if mip_gap is not None:
        solver_options["mipgap"] = mip_gap

    D = m.maximize(m.ActAndStab, solver=solver, **solver_options)

    # Print results
    if D:
        print(f"Optimal objective (Activity + Stability): {D.ActAndStab}")
        print(f"Activity: {D.Activity}")
        print(f"Stability: {D.Stability}")
        print(f"Surface energy: {D.Esurf}")
        print(f"Solver status: {D.SolverStatus}")
