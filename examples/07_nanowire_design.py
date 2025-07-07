# Search methods for inorganic materials crystal structure prediction
# https://www.sciencedirect.com/science/article/pii/S2211339821000587
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matopt.materials.atom import Atom
from matopt.materials.lattices.wurtzite_lattice import WurtziteLattice
from matopt.materials.geometry import Cylinder
from matopt.materials.tiling import LinearTiling
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSites
from matopt.aml.rule import (
    FixedTo,
    EqualTo,
    PiecewiseLinear,
    ImpliesNeighbors,
    GreaterThan,
)
from matopt.aml.model import MatOptModel
from utils.config_handler import ConfigHandler

if __name__ == "__main__":
    # Load configuration
    config_file = ConfigHandler.parse_command_line_config()
    config = ConfigHandler.load_config(config_file, "07_nanowire_design")

    # Define required fields
    required_fields = {
        "material": ["iad", "atom_type"],
        "geometry": [
            "orientation",
            "n_atom_radius",
            "n_atom_unit_length",
            "size_unit_length",
            "core_ratio",
            "origin",
            "axis_direction",
            "shape_shift_factor",
        ],
        "optimization": ["solver", "time_limit", "maximize_sense"],
        "energy_parameters": [
            "p",
            "q",
            "alpha",
            "cn_bounds",
            "ecoh_normalization_factor",
        ],
        "constraint_parameters": ["yi_fixed_value"],
        "numerical": ["double_tolerance"],
        "output": ["lp_export", "lp_export_dir", "lp_file_prefix", "lp_filename"],
    }

    # Validate configuration
    ConfigHandler.validate_config(config, required_fields)

    # Get required parameters from config
    IAD = config["material"]["iad"]
    atom_type = config["material"]["atom_type"]

    orientation = config["geometry"]["orientation"]
    nAtomRadius = config["geometry"]["n_atom_radius"]
    nAtomUnitLength = config["geometry"]["n_atom_unit_length"]
    sizeUnitLength = config["geometry"]["size_unit_length"]
    coreRatio = config["geometry"]["core_ratio"]
    origin = np.array(config["geometry"]["origin"], dtype=float)
    axisDirection = np.array(config["geometry"]["axis_direction"], dtype=float)
    shape_shift_factor = config["geometry"]["shape_shift_factor"]

    solver = config["optimization"]["solver"]
    time_limit = config["optimization"]["time_limit"]
    maximize_sense = config["optimization"]["maximize_sense"]

    p = config["energy_parameters"]["p"]
    q = config["energy_parameters"]["q"]
    alpha = config["energy_parameters"]["alpha"]
    CNBounds = config["energy_parameters"]["cn_bounds"]
    ecoh_norm_factor = config["energy_parameters"]["ecoh_normalization_factor"]

    yi_fixed_val = config["constraint_parameters"]["yi_fixed_value"]
    DBL_TOL = config["numerical"]["double_tolerance"]

    lp_export = config["output"]["lp_export"]
    lp_export_dir = config["output"]["lp_export_dir"]
    lp_file_prefix = config["output"]["lp_file_prefix"]
    lp_filename = config["output"]["lp_filename"]

    BPs = list(range(CNBounds[0], CNBounds[1] + 1))
    Vals = [(p * pow(cn, 1 - alpha) - q * cn) for cn in BPs]

    lattice = WurtziteLattice.alignedWith(IAD, orientation)
    radius = lattice.getShellSpacing(orientation) * (nAtomRadius - 1)
    height = (
        lattice.getLayerSpacing(orientation)
        * lattice.getUniqueLayerCount(orientation)
        * nAtomUnitLength
    )
    shape = Cylinder(origin, radius, height, axisDirection)
    shape.shift(
        shape_shift_factor * shape.Vh
    )  # shift downwards so that the seed is in the shape
    canvas = Canvas.fromLatticeAndShape(lattice, shape)
    tiling = LinearTiling.fromCylindricalShape(shape)
    canvas.makePeriodic(tiling, lattice.getNeighbors)

    CoreLayers = [
        i
        for i, p in enumerate(canvas.Points)
        if p[0] ** 2 + p[1] ** 2 < (coreRatio * radius) ** 2
    ]
    CanvasMinusCoreLayers = [
        i for i, p in enumerate(canvas.Points) if i not in CoreLayers
    ]
    NeighborsInside = [
        [
            j
            for j in canvas.NeighborhoodIndexes[i]
            if (
                j is not None
                and canvas.Points[j][0] ** 2 + canvas.Points[j][1] ** 2
                < p[0] ** 2 + p[1] ** 2 - DBL_TOL
            )
        ]
        for i, p in enumerate(canvas.Points)
    ]

    m = MatOptModel(canvas, [Atom(atom_type)])

    m.Yi.rules.append(FixedTo(yi_fixed_val, sites=CoreLayers))
    m.Yi.rules.append(
        ImpliesNeighbors(
            concs=(m.Yi, GreaterThan(1)),
            sites=CanvasMinusCoreLayers,
            neighborhoods=NeighborsInside,
        )
    )
    m.addSitesDescriptor(
        "Vi",
        bounds=(min(Vals), max(Vals)),
        rules=PiecewiseLinear(
            values=Vals, breakpoints=BPs, input_desc=m.Ci, con_type="UB"
        ),
    )
    m.addGlobalDescriptor(
        "Ecoh",
        rules=EqualTo(SumSites(desc=m.Vi, coefs=(ecoh_norm_factor / sizeUnitLength))),
    )
    m.addGlobalDescriptor(
        "Size",
        bounds=(sizeUnitLength, sizeUnitLength),
        rules=EqualTo(SumSites(desc=m.Yi)),
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
        
        lp_file = (
            f"{lp_export_subdir}/{lp_file_prefix}_r{nAtomRadius}_l{sizeUnitLength}.lp"
        )
        # Create Pyomo model and export
        pyomo_model = m._make_pyomo_model(m.Ecoh, sense=maximize_sense, formulation=formulation)
        pyomo_model.write(lp_file)
        print(f"LP file exported to: {lp_file} (formulation: {formulation})")
        sys.exit(0)

    optimalDesign = m.maximize(m.Ecoh, tilim=time_limit, solver=solver)

    if optimalDesign:
        print(f"Optimal cohesive energy: {optimalDesign.Ecoh}")
        print(f"Solver status: {optimalDesign.SolverStatus}")
