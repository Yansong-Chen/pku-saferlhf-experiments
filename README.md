# PKU-SafeRLHF experiment workbench

This directory is an independent Git repository for the executable evidence
package of the dissertation. It deliberately does not contain a copy of the
raw PKU-SafeRLHF releases. Scripts read the pinned files from
../data/raw/, and every run records the versioned release manifest, code digest, seed,
denominator, and output location.

## Two execution classes

| Class | Audits | Resources | Current role |
|---|---|---|---|
| cpu | P0, Audit I (former E1--E3), and the field-routing component of Audit II (former E4) | Local CPU plus the checked-in papers and raw JSONL | Fully runnable now |
| gpu_api | Audit III (former E5), Audit IV (former E6), and the cost-model component of Audit II (former E7) | GPU and/or external API credentials | Frozen plan, runnable executors, and promoted text-free aggregate outputs for completed primary runs |
| conditional_human | Reception and unhelpful-selection extensions (former E8--E9) | Human coding and, if selected, literature collection | Claim-dependent; not automated or silently replaced by an LLM |

The dissertation's headline claims have direct evidence anchors:

| Claim | Direct evidence | Boundary |
|---|---|---|
| A safer selection can mean a less-unsafe choice | P0 plus Audit I absolute-state decomposition | Uses PKU's own absolute field |
| Categories and severity remain available to moderation, while alignment objectives receive a binary boundary and relative order | Audit I taxonomy description plus Audit II field routing | Field routing does not prove a neural model cannot recover correlated text features |
| The released safety boundary has a specified relation to outside operationalisations | Audit III | Neither external operationalisation is safety ground truth |
| PKU's native distinctions have a describable relation to Buyl et al.'s CCAI system | Audit IV | It is a cross-system sensitivity analysis, not a truth test or exact reproduction |
| The released cost model expresses some subset of the available distinctions | Audit II artefact probe | Evaluation is release-matched, not held-out generalisation |

## Reproducibility rules

1. The versioned P0 runner (shared/p0_snapshot.py) writes
   cpu/results/p0_snapshot.json before CPU analyses; its SHA-256 is embedded in
   every aggregate result.
2. Raw prompts and responses are never written to tracked output files.
3. A configuration, script change, or input-manifest change requires a new
   committed run.
4. GPU/API results are committed only as aggregate, text-free outputs after
   manual inspection.

See cpu/README.md for immediately runnable analyses and gpu_api/RUNBOOK.md for
the credentialed execution package.

## Promoted E7 primary output

`gpu_api/results/e7-primary-beaver-cost-4000/` contains the reviewed,
text-free aggregate from the 4,000-pair Beaver cost-model audit. The primary
run completed all 8,000 response positions at model revision
`26bf7161f09fee958ae64c8b4bb70fb420f7ba39`, with score direction
`higher_is_unsafe`. Its authoritative summary reports a design-weighted
unsafe-versus-safe AUROC of 0.9582 (2,000-bootstrap interval 0.9517--0.9645),
a severity coefficient of 0.4550 after the specified controls, and a 0.5900
same-category both-unsafe lower-severity ordering rate. See the directory
README for the output inventory and the private-run aggregation command.
