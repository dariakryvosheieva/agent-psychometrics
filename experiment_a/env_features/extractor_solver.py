"""Deterministic solver for extracting environment features.

This solver runs bash commands in the SWE-bench Docker sandbox and stores
the results in state.metadata. It does NOT call any LLM.
"""

from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.util import sandbox

from experiment_a.env_features.feature_definitions import FEATURE_DEFINITIONS


@solver
def env_feature_extractor():
    """Solver that extracts environment features without using an LLM.

    Runs predefined bash commands in the sandbox and stores results in metadata.
    All features are deterministic and should produce identical results on repeated runs.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Store errors separately for debugging
        errors: list[str] = []

        for feature_def in FEATURE_DEFINITIONS:
            try:
                # Execute the command in the sandbox
                result = await sandbox().exec(["bash", "-c", feature_def.command])

                if result.success:
                    # Parse the output to numeric value
                    value = feature_def.parse(result.stdout)
                    state.metadata[feature_def.name] = value
                else:
                    # Command failed - store -1 and log error
                    state.metadata[feature_def.name] = -1
                    errors.append(f"{feature_def.name}: {result.stderr.strip()}")

            except Exception as e:
                # Unexpected error - store -1 and log
                state.metadata[feature_def.name] = -1
                errors.append(f"{feature_def.name}: {type(e).__name__}: {e}")

        # Store errors for debugging
        if errors:
            state.metadata["_extraction_errors"] = errors

        # Mark as completed - we don't need to call the LLM
        state.completed = True

        # Set a dummy completion so logs look clean
        feature_count = len(FEATURE_DEFINITIONS)
        error_count = len(errors)
        state.output.completion = f"Extracted {feature_count} features ({error_count} errors)"

        return state

    return solve
