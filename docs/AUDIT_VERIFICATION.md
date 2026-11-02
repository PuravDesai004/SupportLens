# Code audit verification — 3 November 2026

The audit fixes validate model metadata and every required artifact checksum before deserialization, reject malformed probability outputs and class labels, and distinguish artifact failures from invalid questions in the app. The model cache now follows changes to the active release pointer. Review assessment and promotion check their score settings against the frozen decisions. Evidence export validates its inputs and totals before writing, protects existing reports, and reproduction verification hashes the actual comparison CSV. Legacy evaluation also verifies the training data and supports a saved PyTorch winner.

Verification completed on the local Windows machine with Python 3.12:

| Check | Result |
| --- | --- |
| Full pytest suite | **48 passed** in 13.24 seconds |
| Dependency consistency | `pip check` passed in `.venv` and `.venv-app` |
| Python syntax | `compileall` passed for the app, source, scripts and tests |
| Git patch whitespace | `git diff --check` passed |
| Release switching | Streamlit AppTest switched from v2 to v3 in one session, preserved the displayed score and applied the new review flag |
| Broken artifacts | Invalid release JSON produced a model-file error without a traceback or prediction |
| Minimal app package | Passed using current source and v3 artifacts, with no dataset, training outputs, PyTorch or DuckDB |
| Real browser | Local app displayed **Card arrival**, **94.8%**, and the second-model review warning for a manually authored example |

A separate ignored output directory replayed both review development/assessment stages, promotion and export from the saved v2 predictions. Thirteen pipeline and rejection checks passed. The cutoff, assessment comparison, CSV bytes and v3 metadata matched the shipped release. Changed temperature metadata, altered CSV bytes, evidence of test rescoring, and attempts to overwrite historical reports were rejected.

Using the current code, calibration was refitted from the saved development predictions and metrics were recomputed for all three candidates on development and assessment. Temperatures and numerical metrics matched the frozen records within `1e-12`; review policies and counts matched exactly. The existing independent-directory training reproduction also passed its comparison check.

The trained models and historical evidence files are unchanged. Training was not repeated during this audit, and the official test was not rescored. These are correctness and reproducibility checks on existing evidence, not new evidence of real-world accuracy. The app remains local at `http://127.0.0.1:8501/`.
