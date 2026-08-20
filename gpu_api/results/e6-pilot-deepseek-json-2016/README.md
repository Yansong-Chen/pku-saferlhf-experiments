# E6 operational pilot diagnostics

This directory contains the reviewed, text-free aggregate for the separate
48-pair Audit IV operational pilot. The run submitted each pair under 21 CCAI
principles in both response orders: all 2,016 planned order judgements
completed successfully, with no failed or unclassifiable pairs.

- `e6_pilot_summary.json` is the authoritative pilot summary.
- `e6_pilot_orientation.csv` records the reconciled outcome for each of 1,008
  pair--principle comparisons.
- `e6_pilot_by_stratum.csv` gives the resulting pilot state counts.

This pilot used a pre-primary manifest and shares no observations with the
200-pair primary estimate. It is an execution diagnostic, not an additional
estimate of CCAI-state prevalence.

The files contain source row identifiers, native strata, principle indices,
and reconciled directions, but no prompt or response text, credentials, API
request identifiers, or vendor payloads. Raw order-level API records remain
private.

To reproduce from an authorised private run directory:

```bash
python gpu_api/scripts/e6_aggregate.py \
  --private-run gpu_api/private_runs/e6/e6-pilot-deepseek-json-2016 \
  --aggregate-dir /tmp/e6-pilot \
  --phase pilot
```
