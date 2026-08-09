# E4: field-to-objective and branch-routing trace

Protocol SHA-256: 9351a8380890e28f26a76797c2dfb7e455622276b886be21eb996ecc020291e1

| Released field | Released role | Moderation branch | Reward loss | Cost loss | Evidence |
|---|---|---|---|---|---|
| better_response_id | pairwise helpfulness selection | no direct term documented | pairwise Bradley--Terry term | no direct term | Dai et al. (2024), equation 5 |
| safer_response_id | pairwise safety selection | no direct term documented | no direct term | pairwise Bradley--Terry term | Dai et al. (2024), equation 6 |
| is_response_0_safe / is_response_1_safe | response-level binary safety state | binary safety target is documented | no direct term | classification/origin term | Dai et al. (2024), equation 6 and virtual-response derivation |
| response_0_harm_category / response_1_harm_category | 19-label response-level harm taxonomy | available to the severity-sensitive moderation branch | no direct term | no direct term | Ji et al. (2025), section 4.1; Dai et al. (2024), equations 5--6 |
| response_0_severity_level / response_1_severity_level | three-level response-level severity annotation | all severity meta-labels are reported as training inputs | no direct term | no direct term | Ji et al. (2025), section 4.1; Dai et al. (2024), equations 5--6 |

## Source hierarchy

1. Released training implementation and configuration — not inspected in this CPU package.
2. Dai et al. (2024), equations 5--6 — used for the documented Safe RLHF reward and cost objectives.
3. Ji et al. (2025), sections 4.1--4.2 — used for PKU-SafeRLHF application-branch context.

## Registered discrepancy

- **D1** — Ji et al. section 4.2 describes an amendment of the original pairwise cost loss but prints classification terms only; Dai et al. equation 6 prints both pairwise and classification terms. Treatment: Report this as a documentary discrepancy. The safer-response pairwise route is documented from Dai et al.; no statement in this result asserts that a PKU-specific executable configuration has been inspected.

## Interpretive boundary

A missing direct loss term establishes that the published objective has no declared direct supervision channel for that field. It does not establish that the trained model cannot recover correlated textual information.
