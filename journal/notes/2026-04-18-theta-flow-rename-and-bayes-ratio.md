# Rename: `theta_flow` = Bayes-ratio ODE, `theta_path_integral` = score + θ-axis integral

## Summary

The CLI flag `--theta-field-method` and `HMatrixEstimator.field_method` were renamed for clarity (hard break; no legacy aliases).

| Old CLI / internal meaning | New |
|----------------------------|-----|
| Default / documented `theta_flow` = velocity→score + trapezoid along sorted θ | **`theta_path_integral`** (`HMatrixEstimator`: `field_method="theta_path_integral"`) |
| Rejected legacy `flow_likelihood` = ODE `compute_likelihood` on θ (log p(θ|x) − log p(θ)) | **`theta_flow`** (`field_method="theta_flow"`) |

Internal strings: `flow` → `theta_path_integral`; `flow_likelihood` → `theta_flow`. `flow_x_likelihood` and `x_flow` are unchanged.

## Code

- `fisher/h_matrix.py` — `run()` branches and solver wiring.
- `fisher/shared_fisher_est.py` — `normalize_theta_field_method`, shared training block for both theta methods, `HMatrixEstimator(..., field_method=...)`.
- `fisher/cli_shared_fisher.py`, `bin/study_h_decoding_convergence.py`, `bin/visualize_h_matrix_binned.py` — help text and figure/summary labels.

## Artifacts

Saved NPZ keys may still say `theta_field_method`; **semantic** of the string `theta_flow` in old archives does not match the new meaning (hard break per project choice—re-run studies for consistent interpretation).
