# CPU and document evidence package

This package runs the analyses that require neither a GPU nor an external API:

| Component | Script | Primary output |
|---|---|---|
| P0 release verification | ../shared/p0_snapshot.py | results/p0_snapshot.json |
| Audit I: response-process documentary trace | render_e1_response_process.py | results/e1_response_process.md and JSON |
| Audit I: recording architecture, taxonomy structure, absolute-state decomposition | run_native_audit.py | results/native_audit.json and CSV tables |
| Audit I: unsafe-conditioned tetrachoric EFA and parallel analysis | run_e2_efa.R | results/e2_efa.json, loadings, and correlation matrices |
| Audit II: field-to-objective routing | render_e4_routing.py | results/e4_field_routing.md and JSON |

Run from the experiments repository root:

    python3 shared/p0_snapshot.py
    python3 cpu/render_e1_response_process.py
    python3 cpu/run_native_audit.py
    Rscript cpu/run_e2_efa.R
    python3 cpu/render_e4_routing.py

The Python script uses only the standard library. The R script requires
psych, jsonlite, Matrix, and GPArotation. If the latter is absent, run

    Rscript cpu/bootstrap_r_dependencies.R

from the experiments root; it is installed in the ignored cpu/.Rlib directory.
The script writes only
aggregate or metadata output to results/; its response-category matrices are
placed in the ignored intermediate/ directory.

run_e2_efa.R uses 1,000 tetrachoric parallel-analysis simulations by default.
Its primary matrix includes only responses released as unsafe. This separates
the harm-taxonomy description from the all-zero category pattern that encodes
the binary safe state. The factor count is descriptive, and the report records
whether correlation-matrix smoothing was required. The one-response-per-pair
run is a dependence sensitivity analysis, not a second dataset. The script
uses a temporary output directory and an E2 run lock, publishing results only
after both analyses finish.
