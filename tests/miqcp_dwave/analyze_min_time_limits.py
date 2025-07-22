"""
Script to analyze minimum time limits for CQM problems from LP files.
This helps determine appropriate time limits for experimentation.
"""

import os
import json
import dimod
from dimod import ConstrainedQuadraticModel
from dwave.system.samplers import LeapHybridCQMSampler


def calculate_min_time_limit(cqm: ConstrainedQuadraticModel) -> float:
    """Calculate the minimum time limit for a CQM problem.

    Args:
        cqm: A ConstrainedQuadraticModel object

    Returns:
        The minimum time limit in seconds
    """
    sampler = LeapHybridCQMSampler()
    return sampler.min_time_limit(cqm)


def analyze_lp_file_min_time_limit(file_path: str) -> dict:
    """Analyze a single LP file to get its minimum time limit.

    Args:
        file_path: Path to the LP file

    Returns:
        Dictionary containing file info and minimum time limit
    """
    filename = os.path.basename(file_path)

    try:
        # Load the CQM from LP file
        with open(file_path, "rb") as f:
            cqm = dimod.lp.load(f)

        # Calculate minimum time limit
        min_time_limit = calculate_min_time_limit(cqm)

        # Get some basic info about the CQM
        num_variables = len(cqm.variables)
        num_constraints = len(cqm.constraints)

        return {
            "filename": filename,
            "file_path": file_path,
            "min_time_limit": min_time_limit,
            "num_variables": num_variables,
            "num_constraints": num_constraints,
            "status": "success",
        }

    except Exception as e:
        return {
            "filename": filename,
            "file_path": file_path,
            "min_time_limit": None,
            "num_variables": None,
            "num_constraints": None,
            "status": "error",
            "error": str(e),
        }


def save_min_time_limit_analysis(
    results: list, output_file: str = "min_time_limit_analysis.json"
):
    """Save minimum time limit analysis results to a JSON file.

    Args:
        results: List of analysis result dictionaries
        output_file: Output file path
    """
    # Create summary statistics
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "error"]

    if successful_results:
        min_time_limits = [r["min_time_limit"] for r in successful_results]
        summary = {
            "total_files": len(results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "min_time_limit_stats": {
                "min": min(min_time_limits),
                "max": max(min_time_limits),
                "mean": sum(min_time_limits) / len(min_time_limits),
                "median": sorted(min_time_limits)[len(min_time_limits) // 2],
            },
            "results": results,
        }
    else:
        summary = {
            "total_files": len(results),
            "successful": 0,
            "failed": len(failed_results),
            "min_time_limit_stats": None,
            "results": results,
        }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Analysis results saved to {output_file}")
    return summary


def analyze_directory_min_time_limits(folder: str) -> list:
    """Analyze all LP files in a directory to get their minimum time limits.

    Args:
        folder: Directory containing LP files

    Returns:
        List of analysis results for each LP file
    """
    results = []

    if not os.path.exists(folder):
        print(f"Directory {folder} does not exist")
        return results

    # Get all LP files in the folder
    lp_files = []
    for file in os.listdir(folder):
        if file.endswith(".lp"):
            lp_files.append(os.path.join(folder, file))

    if not lp_files:
        print(f"No LP files found in {folder}")
        return results

    print(f"Analyzing {len(lp_files)} LP files for minimum time limits...")

    for i, lp_file in enumerate(lp_files, 1):
        print(f"  [{i}/{len(lp_files)}] Analyzing {os.path.basename(lp_file)}...")
        result = analyze_lp_file_min_time_limit(lp_file)
        results.append(result)

        if result["status"] == "success":
            print(f"    Min time limit: {result['min_time_limit']:.2f}s")
            print(
                f"    Variables: {result['num_variables']}, Constraints: {result['num_constraints']}"
            )
        else:
            print(f"    Error: {result['error']}")

    return results


def main():
    """Main function to analyze minimum time limits for all LP file directories."""

    # Define the directories to analyze
    directories = ["lp_files/milp", "lp_files/miqcp", "lp_files/milp_preprocessed"]

    all_results = {}

    for directory in directories:
        print(f"\n{'='*60}")
        print(f"Analyzing directory: {directory}")
        print(f"{'='*60}")

        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist, skipping...")
            continue

        # Analyze the directory
        results = analyze_directory_min_time_limits(directory)

        if results:
            # Save individual directory results
            output_file = f"min_time_limit_analysis_{os.path.basename(directory)}.json"
            summary = save_min_time_limit_analysis(results, output_file)
            all_results[directory] = summary

            # Print summary for this directory
            if summary["min_time_limit_stats"]:
                stats = summary["min_time_limit_stats"]
                print(f"\nSummary for {directory}:")
                print(f"  Files analyzed: {summary['total_files']}")
                print(f"  Successful: {summary['successful']}")
                print(f"  Failed: {summary['failed']}")
                print(
                    f"  Min time limit range: {stats['min']:.2f}s - {stats['max']:.2f}s"
                )
                print(f"  Average min time limit: {stats['mean']:.2f}s")
                print(f"  Median min time limit: {stats['median']:.2f}s")
        else:
            print(f"No results for {directory}")

    # Create combined analysis
    if all_results:
        print(f"\n{'='*60}")
        print("COMBINED ANALYSIS")
        print(f"{'='*60}")

        # Collect all successful results
        all_successful_results = []
        for directory, summary in all_results.items():
            if summary["min_time_limit_stats"]:
                all_successful_results.extend(summary["results"])

        if all_successful_results:
            # Calculate combined statistics
            min_time_limits = [
                r["min_time_limit"]
                for r in all_successful_results
                if r["status"] == "success"
            ]

            combined_stats = {
                "total_files": len(all_successful_results),
                "successful": len(min_time_limits),
                "failed": len(all_successful_results) - len(min_time_limits),
                "min_time_limit_stats": {
                    "min": min(min_time_limits),
                    "max": max(min_time_limits),
                    "mean": sum(min_time_limits) / len(min_time_limits),
                    "median": sorted(min_time_limits)[len(min_time_limits) // 2],
                },
                "directory_summaries": all_results,
            }

            # Save combined results
            with open("min_time_limit_analysis_combined.json", "w") as f:
                json.dump(combined_stats, f, indent=2)

            print(f"Combined analysis saved to: min_time_limit_analysis_combined.json")
            print(f"\nOverall Statistics:")
            print(f"  Total files analyzed: {combined_stats['total_files']}")
            print(f"  Successful: {combined_stats['successful']}")
            print(f"  Failed: {combined_stats['failed']}")

            stats = combined_stats["min_time_limit_stats"]
            print(
                f"  Global min time limit range: {stats['min']:.2f}s - {stats['max']:.2f}s"
            )
            print(f"  Global average min time limit: {stats['mean']:.2f}s")
            print(f"  Global median min time limit: {stats['median']:.2f}s")

            # Show distribution by directory
            print(f"\nDistribution by directory:")
            for directory, summary in all_results.items():
                if summary["min_time_limit_stats"]:
                    dir_stats = summary["min_time_limit_stats"]
                    print(
                        f"  {os.path.basename(directory)}: {dir_stats['min']:.2f}s - {dir_stats['max']:.2f}s (avg: {dir_stats['mean']:.2f}s)"
                    )

        # Print recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS FOR EXPERIMENTATION")
        print(f"{'='*60}")

        if all_successful_results:
            min_time_limits = [
                r["min_time_limit"]
                for r in all_successful_results
                if r["status"] == "success"
            ]

            # Group by problem type (based on filename patterns)
            problem_types = {}
            for result in all_successful_results:
                if result["status"] == "success":
                    filename = result["filename"]
                    # Extract problem type from filename (e.g., "01_nanocluster_mono" from "01_nanocluster_mono_Ag_large.lp")
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        problem_type = "_".join(parts[:3])  # First 3 parts
                    else:
                        problem_type = "unknown"

                    if problem_type not in problem_types:
                        problem_types[problem_type] = []
                    problem_types[problem_type].append(result["min_time_limit"])

            print("Recommended time limits by problem type:")
            for problem_type, limits in problem_types.items():
                min_limit = min(limits)
                max_limit = max(limits)
                avg_limit = sum(limits) / len(limits)
                print(
                    f"  {problem_type}: {min_limit:.2f}s - {max_limit:.2f}s (avg: {avg_limit:.2f}s)"
                )

            # General recommendations
            global_min = min(min_time_limits)
            global_max = max(min_time_limits)
            global_avg = sum(min_time_limits) / len(min_time_limits)

            print(f"\nGeneral recommendations:")
            print(f"  Minimum time limit for any problem: {global_min:.2f}s")
            print(f"  Maximum time limit for any problem: {global_max:.2f}s")
            print(f"  Average time limit across all problems: {global_avg:.2f}s")
            print(f"  Suggested starting time limit: {max(global_min * 2, 5):.2f}s")
            print(f"  Suggested maximum time limit: {global_max * 2:.2f}s")


if __name__ == "__main__":
    main()
