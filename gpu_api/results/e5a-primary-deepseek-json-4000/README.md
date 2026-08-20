# E5 DeepSeek primary external-boundary audit

This directory contains the reviewed, text-free aggregate for the completed
4,000-pair Audit III DeepSeek V4 Flash run. Both response positions were judged
under a prompt--response rendering and a response-only sensitivity rendering.
All 16,000 position--rendering records completed successfully.

- `e5_summary.json` is the authoritative aggregate used in the dissertation.
- `e5_three_way_by_stratum.csv` records PKU/external state combinations by
  native L1--L4 stratum.
- `e5_safer_external_pair_relation.csv` records whether PKU's `safer`
  selection follows the externally safe response when the external states
  differ.
- `e5_by_category_set.csv` and `e5_by_severity.csv` retain descriptive
  category/severity relations.
- `e5_failure_summary.json` records zero failed primary records.

The prompt--response rendering has a design-weighted overall agreement of
0.94499. This is agreement with a frozen external LLM rubric, not validation of
either system as safety truth. The response-only result is a separate rendering
sensitivity analysis and must not be pooled with the primary rendering.

The published files contain no prompts, response text, credentials, API request
identifiers, or vendor payloads. Raw API records remain in the private research
archive and are excluded from version control.

To reproduce the aggregate from an authorised private run directory:

```bash
python gpu_api/scripts/e5_aggregate.py \
  --private-run gpu_api/private_runs/e5/e5a-primary-deepseek-json-4000 \
  --aggregate-dir /tmp/e5-deepseek-primary
```
