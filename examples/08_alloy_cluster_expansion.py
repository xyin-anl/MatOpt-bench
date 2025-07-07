# Accelerated chemical space search using a quantum-inspired cluster expansion approach
# https://www.cell.com/matter/fulltext/S2590-2385(22)00662-2
# https://github.com/hitarth64/quantum-inspired-cluster-expansion
from __future__ import annotations
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_handler import ConfigHandler

import itertools, math
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from ase.io import read
from ase.data import chemical_symbols

from icet import ClusterSpace, StructureContainer, ClusterExpansion
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from matopt.materials.atom import Atom
from matopt.materials.canvas import Canvas
from matopt.aml.expr import SumSites, SumClusters
from matopt.aml.rule import EqualTo, FixedTo
from matopt.aml.model import MatOptModel

# Load configuration
config = ConfigHandler.load_config(
    ConfigHandler.parse_command_line_config(), "08_alloy_cluster_expansion"
)

# Define required fields
required_fields = {
    "data": [
        "structure_file",
        "property_file",
        "prop_columns",
        "index_column",
        "structure_slice",
    ],
    "material": ["allowed_cations", "has_oxygen", "oxygen_symbol"],
    "composition": ["bounds"],
    "cluster_expansion": [
        "train_multiple_ce",
        "cutoffs",
        "weights",
        "evaluate_model",
        "test_size",
        "cv_verbosity",
    ],
    "optimization": ["time_limit", "solver", "minimize_sense"],
    "model_training": ["lasso_cv", "random_state", "n_jobs"],
    "numerical": ["tolerance", "zero_mixing_energy"],
    "constraint_parameters": ["fixed_values"],
    "output": ["lp_export_dir", "lp_file_prefix"],
}

# Validate configuration
ConfigHandler.validate_config(config, required_fields)

# Load local data  (json + csv)
print(f"Loading system data...")

# Extract parameters from config
data_config = config["data"]
mat_config = config["material"]
comp_config = config["composition"]
ce_config = config["cluster_expansion"]
opt_config = config["optimization"]
train_config = config["model_training"]
num_config = config["numerical"]
constraint_config = config["constraint_parameters"]
output_config = config["output"]

# Handle file paths - check if they're already absolute
if not os.path.isabs(data_config["structure_file"]):
    # Relative path - prepend the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    struct_file = os.path.join(project_root, data_config["structure_file"])
    prop_file = os.path.join(project_root, data_config["property_file"])
else:
    struct_file = data_config["structure_file"]
    prop_file = data_config["property_file"]

atoms_list = read(
    struct_file, data_config["structure_slice"]
)  # list[ase.Atoms] - all structures
props = pd.read_csv(prop_file, index_col=data_config["index_column"])

# Get indices from CSV
indices = props.index.values
print(
    f"Found {len(indices)} structures with properties (out of {len(atoms_list)} total structures)"
)

# Build properties dictionary based on configuration
properties = {}
for prop_name, col_spec in data_config["prop_columns"].items():
    if col_spec is None:
        # Skip null entries
        continue
    elif isinstance(col_spec, list):
        # Sum multiple columns
        properties[prop_name] = props[col_spec].sum(axis=1)
    else:
        # Single column
        properties[prop_name] = props[col_spec]

# Handle different mixing energy calculations
if mat_config.get("reference_energies") and len(mat_config["reference_energies"]) > 0:
    # CuNiPdAg system: calculate mixing energy from reference energies
    print("Calculating mixing energy from reference energies...")
    df_data = []
    for idx, struct_idx in enumerate(indices):
        atoms = atoms_list[struct_idx]  # Use CSV index to access correct structure
        energy = properties["total_energy"].iloc[idx]
        composition = Counter(atoms.get_chemical_symbols())
        n_atoms = len(atoms)

        # Calculate mixing energy per atom
        mixing_energy = (
            energy
            - sum(
                composition[element] / n_atoms * ref_energy
                for element, ref_energy in mat_config["reference_energies"].items()
                if element in composition
            )
        ) / n_atoms

        df_data.append(
            {
                "atoms": atoms,
                "MixingEnergy": mixing_energy,
                "TotalEnergy": energy,
            }
        )

    df = pd.DataFrame(df_data)
else:
    # Ru/Zr systems: read mixing energy directly and normalize
    atoms_per_cell = mat_config["atoms_per_cell"]
    df_data = []

    for idx, struct_idx in enumerate(indices):
        atoms = atoms_list[struct_idx]  # Use CSV index to access correct structure

        entry = {
            "atoms": atoms,
            "MixingEnergy": properties["mixing_energy"].iloc[idx] / atoms_per_cell,
        }

        # Add other properties if training multiple CEs
        if ce_config["train_multiple_ce"]:
            entry["pband"] = properties["pband"].iloc[idx]
            entry["dband"] = properties["dband"].iloc[idx]

        df_data.append(entry)

    df = pd.DataFrame(df_data)

# ensure deterministic site ordering
for i in df.index:
    pos = df.at[i, "atoms"].positions
    order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
    df.at[i, "atoms"] = df.at[i, "atoms"][order]

# discard pure phases if present
df = df[df["MixingEnergy"] != num_config["zero_mixing_energy"]].reset_index(drop=True)
print(f"Loaded {len(df)} structures with non-zero mixing energy")

# Set template structure
ref_idx = mat_config.get("reference_structure_index")
if ref_idx is not None:
    template_atoms = atoms_list[ref_idx]
else:
    template_atoms = df.at[0, "atoms"]

n_sites = len(template_atoms)

# Handle different substitution schemes
if mat_config["has_oxygen"]:
    # Oxide systems: only cation sites are substitutable
    cation_ids = [
        i
        for i, s in enumerate(template_atoms.get_chemical_symbols())
        if s != mat_config["oxygen_symbol"]
    ]
    oxygen_ids = [i for i in range(n_sites) if i not in cation_ids]
    n_cation = len(cation_ids)
    substitutions = [
        (
            mat_config["allowed_cations"]
            if i in cation_ids
            else [mat_config["oxygen_symbol"]]
        )
        for i in range(n_sites)
    ]
else:
    # Metallic systems: all sites are substitutable
    cation_ids = list(range(n_sites))
    oxygen_ids = []
    n_cation = n_sites
    substitutions = [mat_config["allowed_cations"] for _ in range(n_sites)]


# Train cluster expansion(s)
def train_ce(
    target,
    cutoffs,
    evaluate=False,
    lasso_cv=5,
    random_state=42,
    n_jobs=8,
    test_size=0.2,
    cv_verbose=1,
):
    print(f"Training cluster expansion for {target}...")
    cs = ClusterSpace(template_atoms, cutoffs=cutoffs, chemical_symbols=substitutions)
    sc = StructureContainer(cs)
    for idx in tqdm(df.index, desc=f"Adding structures for {target}"):
        sc.add_structure(df.at[idx, "atoms"], properties={target: df.at[idx, target]})

    print("Getting fit data...")
    X, y = sc.get_fit_data(key=target)
    print("Training model...")

    if evaluate:
        # Train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        mdl = LassoCV(
            cv=lasso_cv, random_state=random_state, verbose=cv_verbose, n_jobs=n_jobs
        ).fit(X_train, y_train)

        train_r2 = mdl.score(X_train, y_train)
        test_r2 = mdl.score(X_test, y_test)
        test_mae = mean_absolute_error(mdl.predict(X_test), y_test)

        print(
            f"[{target:12}]  Train R² {train_r2:6.3f}  Test R² {test_r2:6.3f}  Test MAE {test_mae:7.4f}"
        )
    else:
        mdl = LassoCV(
            cv=lasso_cv, random_state=random_state, verbose=cv_verbose, n_jobs=n_jobs
        ).fit(X, y)
        print(
            f"[{target:12}]  MAE {mean_absolute_error(mdl.predict(X), y):7.4f}  "
            f"R² {r2_score(y, mdl.predict(X)):6.3f}"
        )

    ce = ClusterExpansion(cs, mdl.coef_)
    ce.parameters[0] += mdl.intercept_
    return cs, ce


# Train cluster expansion(s) based on configuration
cluster_expansions = {}
if ce_config["train_multiple_ce"]:
    # Train three cluster expansions
    cs_mix, ce_mix = train_ce(
        "MixingEnergy",
        ce_config["cutoffs"]["MixingEnergy"],
        lasso_cv=train_config["lasso_cv"],
        random_state=train_config["random_state"],
        n_jobs=train_config["n_jobs"],
        test_size=ce_config["test_size"],
        cv_verbose=ce_config["cv_verbosity"],
    )
    cs_p, ce_p = train_ce(
        "pband",
        ce_config["cutoffs"]["pband"],
        lasso_cv=train_config["lasso_cv"],
        random_state=train_config["random_state"],
        n_jobs=train_config["n_jobs"],
        test_size=ce_config["test_size"],
        cv_verbose=ce_config["cv_verbosity"],
    )
    cs_d, ce_d = train_ce(
        "dband",
        ce_config["cutoffs"]["dband"],
        lasso_cv=train_config["lasso_cv"],
        random_state=train_config["random_state"],
        n_jobs=train_config["n_jobs"],
        test_size=ce_config["test_size"],
        cv_verbose=ce_config["cv_verbosity"],
    )
    cluster_expansions = {
        "MixingEnergy": (cs_mix, ce_mix),
        "pband": (cs_p, ce_p),
        "dband": (cs_d, ce_d),
    }
else:
    # Train only mixing energy
    cs_mix, ce_mix = train_ce(
        "MixingEnergy",
        ce_config["cutoffs"]["MixingEnergy"],
        evaluate=ce_config["evaluate_model"],
        lasso_cv=train_config["lasso_cv"],
        random_state=train_config["random_state"],
        n_jobs=train_config["n_jobs"],
        test_size=ce_config["test_size"],
        cv_verbose=ce_config["cv_verbosity"],
    )
    cluster_expansions = {"MixingEnergy": (cs_mix, ce_mix)}


# Expand CEs to explicit MatOpt clusters
def _basis_value(cs, site_basis_index, Z):
    """value of φ_{α}(species Z)"""
    return cs.evaluate_cluster_function(
        len(cs.species_maps[0]), site_basis_index, cs.species_maps[0][Z]
    )


def expand_ce(cs, ce, atoms, weight=1.0, tol=None):
    if tol is None:
        tol = num_config["tolerance"]
    """Expand cluster expansion to explicit MatOpt cluster specifications."""
    specs, coeffs = [], []
    species_numbers = list(cs.species_maps[0].keys())
    nsp = len(species_numbers)

    # constant term
    c0 = ce.parameters[0] * weight
    if abs(c0) > tol:
        specs.append([])
        coeffs.append(c0)

    lolg = LocalOrbitListGenerator(
        cs.orbit_list, Structure.from_atoms(atoms), ce.fractional_position_tolerance
    )
    orbit_list = lolg.generate_full_orbit_list().get_orbit_list()

    p = 1  # parameter index
    for orbit in orbit_list:
        for cve in orbit.cluster_vector_elements:
            outer = ce.parameters[p] / cve["multiplicity"] * weight
            p += 1
            if abs(outer) < tol:
                continue

            basis_indices = cve["multicomponent_vector"]  # α‑indices

            for cluster in orbit.clusters:
                sites = [s.index for s in cluster.lattice_sites]

                # enumerate all species assignments
                for combo in itertools.product(range(nsp), repeat=len(sites)):
                    phi = 1.0
                    for site_idx, sp_idx, alpha in zip(sites, combo, basis_indices):
                        Z = species_numbers[sp_idx]
                        phi *= _basis_value(cs, alpha, Z)
                    coeff = outer * phi
                    if abs(coeff) > tol:
                        specs.append(list(zip(sites, combo)))
                        coeffs.append(coeff)
    return specs, coeffs


print("Expanding cluster expansions...")

# Expand CEs based on configuration
all_specs, all_coeffs = [], []
weights = ce_config["weights"]

if ce_config["train_multiple_ce"]:
    # Multiple CEs: combine with different weights
    W1, W2 = weights["W1"], weights["W2"]

    cs_mix, ce_mix = cluster_expansions["MixingEnergy"]
    spec_mix, coeff_mix = expand_ce(
        cs_mix, ce_mix, template_atoms, weight=W1 - W2, tol=num_config["tolerance"]
    )

    cs_p, ce_p = cluster_expansions["pband"]
    spec_p, coeff_p = expand_ce(
        cs_p,
        ce_p,
        template_atoms,
        weight=weights["pband"] * W2,
        tol=num_config["tolerance"],
    )

    cs_d, ce_d = cluster_expansions["dband"]
    spec_d, coeff_d = expand_ce(
        cs_d,
        ce_d,
        template_atoms,
        weight=weights["dband"] * W2,
        tol=num_config["tolerance"],
    )

    all_specs = spec_mix + spec_p + spec_d
    all_coeffs = coeff_mix + coeff_p + coeff_d
else:
    # Single CE
    cs_mix, ce_mix = cluster_expansions["MixingEnergy"]
    all_specs, all_coeffs = expand_ce(
        cs_mix,
        ce_mix,
        template_atoms,
        weight=weights["W1"],
        tol=num_config["tolerance"],
    )

# merge duplicate clusters
combined = defaultdict(float)
for s, c in zip(all_specs, all_coeffs):
    combined[tuple(sorted(s))] += c

cluster_specs = [list(k) for k in combined]
cluster_coeffs = [combined[k] for k in combined]

# pull out constant shift
constant_shift = 0.0
if [] in cluster_specs:
    idx = cluster_specs.index([])
    constant_shift = cluster_coeffs[idx]
    cluster_specs.pop(idx)
    cluster_coeffs.pop(idx)

print(f"Generated {len(cluster_specs)} unique clusters")


# MatOpt MILP
print("Setting up MatOpt model...")
Canv = Canvas(template_atoms.get_positions().tolist())

# atoms in EXACT order of ICET species-map
species_numbers = list(cs_mix.species_maps[0].keys())
species_symbols = [chemical_symbols[z] for z in species_numbers]
AllAtoms = [Atom(sym) for sym in species_symbols]

# convert (site, sp_idx) → (site, Atom)
spec_expanded = [[(i, AllAtoms[k]) for i, k in cl] for cl in cluster_specs]

m = MatOptModel(Canv, AllAtoms, clusters=spec_expanded)

# fix O sites if present
fixed_vals = constraint_config["fixed_values"]
if mat_config["has_oxygen"] and mat_config["oxygen_symbol"] in species_symbols:
    O_atom = AllAtoms[species_symbols.index(mat_config["oxygen_symbol"])]
    for i in oxygen_ids:
        m.Yik.rules.append(
            FixedTo(fixed_vals["occupied"], sites=[i], site_types=[O_atom])
        )
        for a in AllAtoms:
            if a is not O_atom:
                m.Yik.rules.append(
                    FixedTo(fixed_vals["unoccupied"], sites=[i], site_types=[a])
                )

# composition bounds
bounds = {}
composition_bounds = comp_config["bounds"]
for element, (min_frac, max_frac) in composition_bounds.items():
    if element in species_symbols:
        if isinstance(min_frac, (int, float)) and min_frac < 1:
            # fraction of substitutable sites
            min_atoms = math.ceil(min_frac * n_cation) if min_frac > 0 else 0
        else:
            # absolute number of atoms
            min_atoms = int(min_frac)

        if max_frac is None:
            max_atoms = n_cation
        elif isinstance(max_frac, (int, float)) and max_frac <= 1:
            # fraction of substitutable sites
            max_atoms = math.floor(max_frac * n_cation)
        else:
            # absolute number of atoms
            max_atoms = int(max_frac)

        bounds[Atom(element)] = (min_atoms, max_atoms)
        print(f"  {element}: {min_atoms}-{max_atoms} atoms")

m.addGlobalTypesDescriptor("Comp", bounds=bounds, rules=EqualTo(SumSites(desc=m.Yik)))

# every site must be filled
m.Yi.rules.append(FixedTo(fixed_vals["occupied"]))

# objective
obj_expr = SumClusters(desc=m.Zn, coefs=cluster_coeffs)

print("Solving optimization...")

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
    
    system_name = os.path.splitext(os.path.basename(data_config["structure_file"]))[0]
    lp_file = f"{lp_export_subdir}/{output_config['lp_file_prefix']}_{system_name}.lp"
    # Create Pyomo model and export
    pyomo_model = m._make_pyomo_model(obj_expr, sense=opt_config["minimize_sense"], formulation=formulation)
    pyomo_model.write(lp_file)
    print(f"LP file exported to: {lp_file} (formulation: {formulation})")
    sys.exit(0)


D = m.minimize(obj_expr, tilim=opt_config["time_limit"], solver=opt_config["solver"])

print(f"\nOptimization complete!")
if D is not None:
    print("Optimal design found.")
    # Print composition summary
    if D and hasattr(D, "atoms"):
        composition = Counter([atom.symbol for atom in D.atoms])
        print("Optimal composition:", dict(composition))
