# E6 primary and repeat CCAI audit

This directory contains the reviewed, text-free outputs for the completed
200-pair Audit IV primary run and independent 20-pair repeat batch. Each pair
was judged under 21 CCAI principles in both response orders.

- `e6_primary_summary.json` is the authoritative primary aggregate.
- `e6_primary_by_stratum.csv` gives CCAI state shares and 10,000-replicate
  stratified pair-bootstrap intervals by native L1--L4 stratum.
- `e6_primary_direction_by_stratum.csv` gives retained principle directions
  relative to PKU's `safer` selection.
- `e6_primary_orientation.csv` gives all 4,200 reconciled primary
  pair--principle outcomes.
- The corresponding `e6_repeat_*` files give the independent repeat results
  and primary--repeat agreement diagnostics.

The primary run completed all 8,400 order judgements with zero failures. The
repeat completed all 840 judgements with zero failures. Exact order-label
agreement was 0.97738 and exact reconciled CCAI-state agreement was 0.90000
across the 20 repeated pairs.

These outputs relate a fixed CCAI reference system to PKU native strata. They
are neither a safety truth table nor an exact reproduction of Buyl et al. The
files contain source row identifiers, native strata, principle indices, and
reconciled directions, but no prompt or response text, credentials, API
request identifiers, or vendor payloads.

To reproduce from an authorised private run directory:

```bash
python gpu_api/scripts/e6_aggregate.py \
  --private-run gpu_api/private_runs/e6/e6-primary-deepseek-json-200 \
  --aggregate-dir /tmp/e6-primary \
  --phase primary \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260811 \
  --sample-manifest gpu_api/config/e6_sample_manifest.csv
python gpu_api/scripts/e6_aggregate.py \
  --private-run gpu_api/private_runs/e6/e6-primary-deepseek-json-200 \
  --aggregate-dir /tmp/e6-primary \
  --phase repeat \
  --sample-manifest gpu_api/config/e6_sample_manifest.csv
```
