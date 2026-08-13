# Pilot report — 13 August 2026

## Pilot aim

The pilot tests whether a citation/reuse audit can reliably identify the
specific multi-level PKU-SafeRLHF release and recover field-level information
from documents. It is a protocol test, not an estimate of community practice.
No pilot percentage should be reported in the dissertation.

## Frame check

OpenAlex returned 16 citing records for the ACL work
`10.18653/v1/2025.acl-long.1544` and two for the arXiv work
`10.48550/arXiv.2406.15513` on the retrieval date. These are pre-deduplication
counts. An exact full-text search for `PKU-SafeRLHF` returned 426 works.
The contrast confirms that a name-search frame would substantially mix the
target release with SafeRLHF, BeaverTails, and `PKU-SafeRLHF-30K` lineage
records. Formal citations alone may also omit papers that name a repository or
an older version rather than the final ACL paper.

## Full-text screen

Eight accessible documents were selected to stress the proposed identity and
field codes. The selection was purposive: it included papers that named the
full release, the 30K derivative, the older SafeRLHF data, and a model-card
provenance case. The coding below is one researcher's pilot coding and has not
been independently double-coded.

| Document | Pilot release decision | Declared relationship | What the document reports | Protocol lesson |
|---|---|---|---|---|
| *Gradient-Adaptive Policy Optimization* (2025) | `target` | Policy fine-tuning and evaluation | Cites Ji et al. (2024) and describes safety meta-labels plus helpfulness/harmlessness preferences. Its Appendix B identifies frozen `beaver-7b-v1.0-reward` and `beaver-7b-v1.0-cost` scorers. The policy's documented immediate route is PKU prompts $\rightarrow$ generated responses $\rightarrow$ reward/cost scores; the paper does not document a new reward/cost-model training step or its direct label columns. | A dataset description cannot be treated as a statement about the direct inputs to the policy. Consumer-level route coding is required. |
| *SafeDPO* (2025) | `related_30k` | Training and evaluation | Uses `PKU-SafeRLHF-30K`; reports more-helpful, safer, and binary safety annotations. | A 30K derivative can share some fields with the target while remaining a different release. |
| *Unmasking and Improving Data Credibility* (2023) | `related_legacy` | Dataset-quality evaluation | Identifies a 300K SafeRLHF dataset attributed to Dai et al. (2023), with response labels. | String matching would wrongly bring predecessor data into the target corpus. |
| *RLHFPoison* (2024) | `related_legacy` | Training | Uses a 330K `PKU-SafeRLHF-Dataset` attributed to Dai et al. (2023) and reports harmlessness preferences. | Version identity must be checked from references, counts, and repository names. |
| *Interpretable Preferences* (2024) | `related_30k` | Model-training provenance | Lists `PKU-SafeRLHF-30K (Ji et al., 2023)` among preference datasets. | A listed component of another model establishes provenance, not field use. |
| *RewardBench* (2025) | `related_legacy` | Model-training provenance | A model-data table names SafeRLHF/PKU-SafeRLHF variants, without identifying which fields enter the constituent model. | `model_provenance` and `direct_documentation` must remain separate codes. |
| *Rejected Dialects* (2025) | `related_30k` | Model-training provenance | Identifies an `RLHFlow/PKU-SafeRLHF-30K-standard` model in an evaluated model list. | A downstream model name alone is insufficient evidence of its source fields. |
| *SOUL* (2024) | `ambiguous` | Training and evaluation | States that it draws negative samples from a PKU-SafeRLHF training set and uses a PKU test set, yet its visible reference entry is BeaverTails rather than the target paper. | Preserve an `ambiguous` identity state and resolve it from supplementary code or author documentation; do not force inclusion. |

## Pilot conclusion

The coding scheme is viable, and the central question is answerable. The pilot
also establishes three requirements for a defensible main study:

1. The study must be called a **reception and declared-reuse audit**, rather
   than a simple citation count. It needs exact-paper cited-by records and
   full-text repository-name discovery.
2. The 100-paper design from Chehbouni et al. should not be copied. The exact
   target-paper citation frame is currently small enough for a census. A larger
   name-search frame requires manual release-identity screening before any
   count is calculated.
3. A document's description of the dataset and the direct inputs to its trained
   component must be coded separately. GAPO makes the distinction concrete:
   its policy is trained against frozen scorer outputs, so its dataset paragraph
   cannot establish which native PKU labels reached the policy update.

The next step is to freeze the full candidate ledger, screen every record, and
have an independent human coder double-code the pilot documents before the
codebook is frozen.

## Sources used for the pilot

- Ji et al., *PKU-SafeRLHF* (ACL 2025):
  https://aclanthology.org/2025.acl-long.1544/
- OpenAlex cited-by queries for the ACL and arXiv work identifiers, retrieved
  on 13 August 2026.
- Full texts cited in the table, retrieved from their official ACL, ACM, or
  arXiv pages during the pilot.
