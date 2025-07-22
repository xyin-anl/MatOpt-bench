"""
Auto-generated experiment script based on problem categorization.
"""

import os
import time
import json
import dimod
from dimod import ConstrainedQuadraticModel, SampleSet
from dwave.system import LeapHybridCQMSampler


def call_solver(
    cqm: ConstrainedQuadraticModel, time_limit: float, label: str
) -> SampleSet:
    """Helper function to call the CQM Solver.

    Args:
        cqm: A ``CQM`` object that defines the 3D bin packing problem.
        time_limit: Time limit parameter to pass on to the CQM sampler.
        label: Label for the solver run.

    Returns:
        A ``dimod.SampleSet`` that represents the best feasible solution found.

    """
    sampler = LeapHybridCQMSampler()
    res = sampler.sample_cqm(cqm, time_limit=time_limit, label=label)

    res.resolve()
    feasible_sampleset = res.filter(lambda d: d.is_feasible)

    try:
        best_feasible = feasible_sampleset.first.sample

        return best_feasible

    except ValueError:
        raise RuntimeError(
            "Sampleset is empty, try increasing time limit or "
            + "adjusting problem config."
        )


def process_lp_file(file_path: str, time_limit: float = 10, iteration: int = 1) -> dict:
    """Process a single LP file and return results with timing.

    Args:
        file_path: Path to the LP file
        time_limit: Time limit for the solver
        iteration: Current iteration number for labeling

    Returns:
        Dictionary containing results and timing information
    """
    filename = os.path.basename(file_path)
    base_label = os.path.splitext(filename)[0]  # Remove .lp extension
    label = f"{base_label}_iter{iteration}_{time_limit}s"  # Include iteration and time limit

    print(
        f"Processing: {filename} with time limit: {time_limit}s (iteration {iteration})"
    )

    # Load the CQM from LP file
    try:
        with open(file_path, "rb") as f:
            cqm = dimod.lp.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load LP file {file_path}: {str(e)}")

    # Time the execution
    start_time = time.time()
    try:
        result = call_solver(cqm, time_limit, label)
        execution_time = time.time() - start_time

        return {
            "filename": filename,
            "base_label": base_label,
            "solver_label": label,
            "iteration": iteration,
            "execution_time": execution_time,
            "time_limit_used": time_limit,
            "status": "success",
            "solution": result,
        }
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "filename": filename,
            "base_label": base_label,
            "solver_label": label,
            "iteration": iteration,
            "execution_time": execution_time,
            "time_limit_used": time_limit,
            "status": "error",
            "error": str(e),
            "solution": None,
        }


def save_individual_result(
    result: dict, output_dir: str = "results", iteration: int = 1, time_limit: float = 5
):
    """Save individual result to a file.

    Args:
        result: Single result dictionary
        output_dir: Output directory for results
        iteration: Current iteration number
        time_limit: Time limit used for this iteration
    """
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory {output_dir}: {str(e)}")

    # Create filename for individual result
    base_name = os.path.splitext(result["filename"])[0]
    output_file = os.path.join(
        output_dir, f"{base_name}_iter{iteration}_{time_limit}s.json"
    )

    # Convert numpy arrays if present
    serializable_result = result.copy()
    if result["solution"] is not None:
        serializable_solution = {}
        for key, value in result["solution"].items():
            if hasattr(value, "tolist"):
                serializable_solution[key] = value.tolist()
            else:
                serializable_solution[key] = value
        serializable_result["solution"] = serializable_solution

    with open(output_file, "w") as f:
        json.dump(serializable_result, f, indent=2)

    return output_file


def run_experiments():
    """Run experiments based on categorization."""

    # 5s problems - run at 5s and 10s
    five_second_problems = [
        "lp_files/miqcp/01_nanocluster_mono_Au_large.lp",
        "lp_files/miqcp/02_nanocluster_bimetal_AuAg_medium.lp",
        "lp_files/miqcp/01_nanocluster_mono_Au_small.lp",
        "lp_files/miqcp/01_nanocluster_mono_Pt_medium.lp",
        "lp_files/miqcp/07_nanowire_design_r6_l216.lp",
        "lp_files/miqcp/03_surface_design_Pt_FCC_medium.lp",
        "lp_files/miqcp/02_nanocluster_bimetal_CuAg_small.lp",
        "lp_files/miqcp/01_nanocluster_mono_Au_medium.lp",
        "lp_files/miqcp/01_nanocluster_mono_Pd_small.lp",
        "lp_files/miqcp/01_nanocluster_mono_Ag_large.lp",
        "lp_files/miqcp/01_nanocluster_mono_Ag_medium.lp",
        "lp_files/miqcp/01_nanocluster_mono_Pd_large.lp",
        "lp_files/miqcp/01_nanocluster_mono_Ag_small.lp",
        "lp_files/miqcp/07_nanowire_design_r4_l120.lp",
        "lp_files/miqcp/09_solid_solution_qubo_graphene_qubo.lp",
        "lp_files/miqcp/09_solid_solution_qubo_AlGaN_constrained.lp",
        "lp_files/miqcp/09_solid_solution_qubo_TaW_constrained.lp",
        "lp_files/miqcp/02_nanocluster_bimetal_CuAu_small.lp",
        "lp_files/miqcp/07_nanowire_design_r8_l360.lp",
        "lp_files/miqcp/01_nanocluster_mono_Cu_medium.lp",
        "lp_files/miqcp/02_nanocluster_bimetal_CuAu_medium.lp",
        "lp_files/miqcp/01_nanocluster_mono_Cu_large.lp",
        "lp_files/miqcp/09_solid_solution_qubo_graphene_constrained.lp",
        "lp_files/miqcp/04_surface_bifunctional_PtNi_small.lp",
        "lp_files/miqcp/05_bulk_oxide_vacancy_BaFeInO_medium.lp",
        "lp_files/miqcp/04_surface_bifunctional_PtNi_large.lp",
        "lp_files/miqcp/01_nanocluster_mono_Cu_small.lp",
        "lp_files/miqcp/03_surface_design_Pt_FCC_large.lp",
        "lp_files/miqcp/01_nanocluster_mono_Pt_large.lp",
        "lp_files/miqcp/03_surface_design_Pt_FCC_small.lp",
        "lp_files/miqcp/09_solid_solution_qubo_AlGaN_qubo.lp",
        "lp_files/miqcp/01_nanocluster_mono_Pt_small.lp",
        "lp_files/miqcp/05_bulk_oxide_vacancy_BaFeInO_small.lp",
        "lp_files/miqcp/01_nanocluster_mono_Pd_medium.lp",
        "lp_files/miqcp/05_bulk_oxide_vacancy_BaFeInO_large.lp",
        "lp_files/miqcp/09_solid_solution_qubo_TaW_qubo.lp",
        "lp_files/miqcp/02_nanocluster_bimetal_CuAg_medium.lp",
        "lp_files/miqcp/04_surface_bifunctional_PtNi_medium.lp",
        "lp_files/miqcp/02_nanocluster_bimetal_AuAg_small.lp",
    ]

    # 5-10s problems - run at 10s and 20s
    five_to_ten_second_problems = [
        "lp_files/miqcp/06_crystal_ionic_SrTiO3_3.lp",
        "lp_files/miqcp/06_crystal_ionic_SrTiO3_2.lp",
        "lp_files/miqcp/06_crystal_ionic_Y2O3_1.lp",
        "lp_files/miqcp/06_crystal_ionic_MgAl2O4_3.lp",
        "lp_files/miqcp/06_crystal_ionic_MgAl2O4_2.lp",
        "lp_files/miqcp/06_crystal_ionic_Y2Ti2O7_1.lp",
        "lp_files/miqcp/06_crystal_ionic_Ca3Al2Si3O12_2.lp",
        "lp_files/miqcp/06_crystal_ionic_MgAl2O4_1.lp",
        "lp_files/miqcp/06_crystal_ionic_Y2Ti2O7_2.lp",
        "lp_files/miqcp/06_crystal_ionic_Ca3Al2Si3O12_1.lp",
        "lp_files/miqcp/06_crystal_ionic_Y2O3_3.lp",
        "lp_files/miqcp/06_crystal_ionic_MgAl2O4_4.lp",
        "lp_files/miqcp/06_crystal_ionic_Y2O3_2.lp",
        "lp_files/miqcp/06_crystal_ionic_SrTiO3_5.lp",
        "lp_files/miqcp/06_crystal_ionic_SrTiO3_1.lp",
        "lp_files/miqcp/06_crystal_ionic_SrTiO3_4.lp",
    ]

    # 120-240s problems - run at 240s
    one20_to_240_second_problems = [
        "lp_files/miqcp/08_alloy_cluster_expansion_CuNiPdAg.lp",
    ]

    # 240-500s problems - run at 500s
    two40_to_500_second_problems = [
        "lp_files/miqcp/08_alloy_cluster_expansion_Zr.lp",
    ]

    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    print("Running 5s problems at 5s and 10s...")
    for i, lp_file in enumerate(five_second_problems, 1):
        print(
            f"  [{i}/{len(five_second_problems)}] Processing {os.path.basename(lp_file)} at 5s..."
        )
        result_5s = process_lp_file(lp_file, time_limit=5, iteration=1)
        save_individual_result(result_5s, output_dir, iteration=1, time_limit=5)

        print(
            f"  [{i}/{len(five_second_problems)}] Processing {os.path.basename(lp_file)} at 10s..."
        )
        result_10s = process_lp_file(lp_file, time_limit=10, iteration=2)
        save_individual_result(result_10s, output_dir, iteration=2, time_limit=10)

    print("\nRunning 5-10s problems at 10s and 20s...")
    for i, lp_file in enumerate(five_to_ten_second_problems, 1):
        print(
            f"  [{i}/{len(five_to_ten_second_problems)}] Processing {os.path.basename(lp_file)} at 10s..."
        )
        result_10s = process_lp_file(lp_file, time_limit=10, iteration=1)
        save_individual_result(result_10s, output_dir, iteration=1, time_limit=10)

        print(
            f"  [{i}/{len(five_to_ten_second_problems)}] Processing {os.path.basename(lp_file)} at 20s..."
        )
        result_20s = process_lp_file(lp_file, time_limit=20, iteration=2)
        save_individual_result(result_20s, output_dir, iteration=2, time_limit=20)

    print("\nRunning 120-240s problems at 240s...")
    for i, lp_file in enumerate(one20_to_240_second_problems, 1):
        print(
            f"  [{i}/{len(one20_to_240_second_problems)}] Processing {os.path.basename(lp_file)} at 240s..."
        )
        result_240s = process_lp_file(lp_file, time_limit=240, iteration=1)
        save_individual_result(result_240s, output_dir, iteration=1, time_limit=240)

    print("\nRunning 240-500s problems at 500s...")
    for i, lp_file in enumerate(two40_to_500_second_problems, 1):
        print(
            f"  [{i}/{len(two40_to_500_second_problems)}] Processing {os.path.basename(lp_file)} at 500s..."
        )
        result_500s = process_lp_file(lp_file, time_limit=500, iteration=1)
        save_individual_result(result_500s, output_dir, iteration=1, time_limit=500)

    print("\nRunning additional experiments...")

    print("Running 120-240s problem at 500s...")
    for i, lp_file in enumerate(one20_to_240_second_problems, 1):
        print(
            f"  [{i}/{len(one20_to_240_second_problems)}] Processing {os.path.basename(lp_file)} at 500s..."
        )
        result_500s = process_lp_file(lp_file, time_limit=500, iteration=2)
        save_individual_result(result_500s, output_dir, iteration=2, time_limit=500)

    # Additional experiment: 240-500s problem at 900s
    print("Running 240-500s problem at 900s...")
    for i, lp_file in enumerate(two40_to_500_second_problems, 1):
        print(
            f"  [{i}/{len(two40_to_500_second_problems)}] Processing {os.path.basename(lp_file)} at 900s..."
        )
        result_900s = process_lp_file(lp_file, time_limit=900, iteration=2)
        save_individual_result(result_900s, output_dir, iteration=2, time_limit=900)

    print("\nAll experiments completed!")


if __name__ == "__main__":
    run_experiments()
