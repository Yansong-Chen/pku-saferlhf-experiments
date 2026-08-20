# E5 ShieldGemma primary external-boundary audit

This directory contains the reviewed, text-free aggregate for the completed
4,000-pair Audit III ShieldGemma 9B run at revision
`b8b636016df4540721a098c7aab91c97ec6ee508`. Both response positions were
scored under prompt--response and response-only renderings, for 16,000
successful position--rendering records and no failures.

- `e5_summary.json` is the authoritative aggregate used in the dissertation.
- `e5_shieldgemma_detail_summary.json` describes the four-policy diagnostic
  scope.
- `e5_shieldgemma_policy_agreement.csv`,
  `e5_shieldgemma_policy_by_pku_state.csv`,
  `e5_shieldgemma_category_policy_relation.csv`, and
  `e5_shieldgemma_severity_relation.csv` retain policy-level diagnostics.
- `e5_shieldgemma_rendering_transitions.csv` records changes between the two
  renderings.
- The remaining CSV files retain the same stratum, pair-selection, category,
  and severity summaries as the DeepSeek aggregate.

The prompt--response rendering has a design-weighted overall agreement of
0.68920. The four-policy union is an external operationalisation, not safety
truth and not a validation of PKU's 19-category taxonomy. Category rows overlap
because PKU categories are multi-label.

The published files contain no prompts, response text, credentials, or model
inputs. Raw score records remain in the private research archive.

To reproduce the aggregate and diagnostics from an authorised private run:

```bash
python gpu_api/scripts/e5_aggregate.py \
  --private-run gpu_api/private_runs/e5/e5-sg-primary-4000 \
  --aggregate-dir /tmp/e5-shieldgemma-primary
python gpu_api/scripts/e5_shieldgemma_detail.py \
  --private-run gpu_api/private_runs/e5/e5-sg-primary-4000 \
  --aggregate-dir /tmp/e5-shieldgemma-primary
```
