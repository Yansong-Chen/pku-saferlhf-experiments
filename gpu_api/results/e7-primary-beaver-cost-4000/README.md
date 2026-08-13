# E7 primary Beaver cost-model audit

This directory contains the text-free, promoted outputs for the completed
4,000-pair primary audit of `PKU-Alignment/beaver-7b-v2.0-cost`.

- `e7_summary.json` is the authoritative aggregate used in the dissertation.
- `e7_pair_gaps.csv` gives the 4,000 signed and absolute pair-score gaps by
  native stratum.
- `e7_unsafe_category_score_distributions.csv` gives unsafe-position score
  distributions by released harm category.
- `e7_primary_scores.manifest.json`, `e7_primary_completeness.json`, and
  `e7_primary_batch_probe_32.json` record the pinned model revision, execution
  settings, completion checks, and batch probe.

The run scored 8,000 response positions successfully and reported no input
truncation. The score direction is `higher_is_unsafe`. Raw position-level score
JSONL is excluded from version control under the repository's no-raw-run-output
policy; it contains no prompt text in this run, but is retained with the local
research archive together with the source job files.

To reproduce the aggregate from an authorised private run directory:

```bash
python gpu_api/scripts/e7_aggregate.py \
  --private-run gpu_api/private_runs/e7/<run-id> \
  --aggregate-dir /tmp/e7-aggregate \
  --score-direction higher_is_unsafe \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 20260813
```
