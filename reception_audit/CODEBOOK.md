# Coding sheet

One row represents one document. Leave a code blank only when the document is
not full-text accessible; use `NR` when the full text is available but provides
no reportable information.

| Variable | Allowed code(s) | Decision rule |
|---|---|---|
| `release_identity` | `target`; `related_legacy`; `related_30k`; `related_beavertails`; `ambiguous`; `not_dataset_use` | `target` requires a citation to Ji et al. (2024/2025) or an unambiguous identification of the full multi-level release. |
| `consumer` | `policy`; `reward_model`; `cost_model`; `moderation`; `evaluation`; `model_provenance`; `other` | Create a separate record for each consuming component. |
| `route_to_consumer` | `prompt_only`; `response_pair`; `named_field`; `intermediate_score`; `unknown` | Code the immediate input route, not the fields used by an upstream component. |
| `reported_field_*` | `1`; `0`; `NR` | Five variables for helpfulness preference, safety preference, binary state, category, and severity. These record the document's dataset description. |
| `direct_field_*` | `1`; `0`; `NR` | The same five variables, applied only to fields the named consumer receives directly. Do not infer these from `reported_field_*`. |
| `intermediate_artifact` | free text; `NA` | Name a frozen reward/cost/classifier, when a component receives its score rather than native data fields. |
| `description_granularity` | `generic_safety`; `dual_preference`; `multilevel_claim`; `NR` | Select every explicit description. A `multilevel_claim` is not evidence of using categories or severity. |
| `use_evidence` | `direct_documentation`; `asserted_undocumented`; `citation_only`; `not_applicable` | Direct documentation names a repository/split/column/configuration or a concrete procedure. |
| `evidence_location` | free text | Give page/table/appendix/repository location. |
| `evidence_note` | free text | Paraphrase the supporting passage. Do not infer beyond it. |

### Priority rules

1. Resolve the release identity before coding fields.
2. Treat `PKU-SafeRLHF-30K` as a separate derivative unless the document also
   establishes use of the full 2024/2025 release.
3. A statement about what a dataset contains belongs in `reported_field_*`; it
   does not identify the direct inputs to a policy, reward model, or cost model.
4. A policy that receives a frozen reward or cost score is coded
   `intermediate_score`, even where the paper describes the upstream dataset in
   detail. Training provenance for that frozen artefact is a separate component
   and needs separate documentation.
5. A named model trained on PKU-SafeRLHF establishes `model_provenance`; it does
   not establish the fields used to train that model unless the source provides
   them.
6. `NR` means full text available but no relevant report; it is distinct from
   `0`, which means the document clearly indicates the field is absent from its
   documented procedure.
