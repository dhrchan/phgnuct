import argparse
import os
import csv
import sys
import importlib

sys.path.insert(0, ".")

# Unified Planning imports
from unified_planning.shortcuts import *
from unified_planning.model.phgn import *
from unified_planning.model.phgn.goal_network import PartialOrderGoalNetwork

# PHGN Planner config
from phgn_planner.config import UCTConfig

# Define the UCTConfig parameters that will be logged, in a specific order
# This helps ensure consistent CSV column order.
UCT_CONFIG_PARAMS_TO_LOG = [
    "n_rollouts",
    "horizon",
    "budget",
    "exploration_const",
    "normalize_exploration_const",
    "n_init",
    "risk_factor",
    "goal_utility",
    "seed",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run a PHGN probabilistic planning algorithm and log results."
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="The name of the planning domain (e.g., 'transport').",
    )
    parser.add_argument(
        "--problem_instance",
        type=int,
        required=True,
        help="The problem instance number for the domain.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["factored", "unfactored"],
        help="The UCT variant to use ('factored' or 'unfactored').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the CSV file where results will be appended.",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=1000,
        help="Number of rollouts to perform (default: 1000).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Maximum depth of each rollout (default: 50).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="The maximum cost budget for a single run (default: 100).",
    )

    args = parser.parse_args()

    # --- 1. Dynamically load the domain ---
    try:
        # Assuming domains are in phgn_planner.experiments.domains.{domain_name}
        domain_module = importlib.import_module(
            f"phgn_planner.experiments.domains.{args.domain}"
        )
        # Assuming the function to load the problem is named the same as the domain
        load_problem_function = getattr(domain_module, args.domain)
        problem = load_problem_function(problem_instance=args.problem_instance)
        print(
            f"Loaded domain '{args.domain}' problem instance {args.problem_instance}",
            flush=True,
        )
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading domain '{args.domain}': {e}", flush=True)
        print(
            "Please ensure the domain module and its loading function exist.",
            flush=True,
        )
        return

    # --- 2. Dynamically select and import the PHGNPlanner class ---
    PHGNPlanner = None
    if args.variant == "unfactored":
        try:
            from phgn_planner.unfactored_uct import PHGNPlanner
        except ImportError:
            print("Error: Could not import 'unfactored_uct.PHGNPlanner'.", flush=True)
            return
    elif args.variant == "factored":
        try:
            from phgn_planner.factored_uct import PHGNPlanner
        except ImportError:
            print("Error: Could not import 'factored_uct.PHGNPlanner'.", flush=True)
            return

    if PHGNPlanner is None:
        print(
            f"Error: Planner for variant '{args.variant}' could not be loaded.",
            flush=True,
        )
        return

    # --- 3. Configure UCT parameters ---
    # Note: h_util and h_ptg are excluded as requested for logging
    cfg = UCTConfig(
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        budget=args.budget,
        exploration_const=2**0.5,
        normalize_exploration_const=True,
        n_init=5,
        risk_factor=-0.1,
        goal_utility=1,
        h_util=lambda _: 1,
        h_ptg=lambda _: 1,
        seed=None,
        show_progress=True,
    )

    print(f"Running planner with variant: {args.variant}", flush=True)
    # --- 4. Run the planner ---
    try:
        planner = PHGNPlanner(cfg)
        result, cost, num_nodes = planner.run(problem)
        print(
            f"Planner finished. Result: '{result}', Cost: {cost}, Num nodes: {num_nodes}",
            flush=True,
        )
    except Exception as e:
        print(f"An error occurred during planner execution: {e}", flush=True)
        result = "ERROR"
        cost = -1  # Indicate an error cost
        num_nodes = -1

    # --- 5. Prepare data for CSV logging ---
    # Determine if header needs to be written
    file_exists = os.path.exists(args.output_file)
    write_header = not file_exists or os.path.getsize(args.output_file) == 0

    # Define the fields for the CSV, ensuring order
    fieldnames = [
        "domain",
        "problem_instance",
        "variant",
        "result",
        "cost",
        "num_nodes",
    ] + UCT_CONFIG_PARAMS_TO_LOG

    # Create the data row dictionary
    row_data = {
        "domain": args.domain,
        "problem_instance": args.problem_instance,
        "variant": args.variant,
        "result": result,
        "cost": cost,
        "num_nodes": num_nodes,
    }

    # Add UCTConfig parameters to the row_data
    for param_name in UCT_CONFIG_PARAMS_TO_LOG:
        row_data[param_name] = getattr(cfg, param_name)

    # --- 6. Write results to CSV ---
    try:
        with open(args.output_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()
            writer.writerow(row_data)
        print(f"Results appended to '{args.output_file}' successfully.", flush=True)
    except Exception as e:
        print(f"Error writing to output file '{args.output_file}': {e}", flush=True)


if __name__ == "__main__":
    main()
