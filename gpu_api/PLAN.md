# GPU/API execution plan

This package is ready for execution, but no model weights are loaded and no
external request is sent by default. Its three audits depend on the committed
CPU results and on the frozen P0 release manifest. The exact commands,
environment setup, pilot gates, aggregation steps, and privacy controls are in
gpu_api/RUNBOOK.md.

| Audit | Direct question | Fixed population | Required resource | Primary output |
|---|---|---:|---|---|
| E5 | Does PKU's absolute boundary correspond with an external safety adjudication? | 4,000 stratified pairs (8,000 positions) | DeepSeek JSON-mode API | weighted site-specific agreement and three-way tables |
| E6 | How do PKU native distinctions co-occur with Buyl's CCAI states? | nested 200-pair stratified sample | DeepSeek JSON-mode API | weighted native-stratum by CCAI-state table |
| E7 | Which release distinctions does the published cost model express? | 4,000 stratified pairs (8,000 positions) | local GPU | weighted boundary, severity, and gap diagnostics |

## Fixed sequencing

1. Run the preflight command for the audit. It validates P0 and the native
   audit and reports planned volume; it never sends a request.
2. Run the stated pilot. Freeze model revisions, prompts, score direction,
   precision, batching, truncation, and retry behaviour immediately after the
   pilot. A pilot diagnoses execution only and cannot tune a decision rule.
3. Commit the frozen run configuration before a primary-sample job begins.
4. Write raw prompts, responses, model outputs, and API payloads only under an
   ignored private run directory. Commit reviewed text-free aggregate tables,
   output manifests, code revisions, and deviations from the protocol.

## E5: external safety-boundary relation

Preflight command:

    python3 gpu_api/scripts/preflight.py e5 --check-runtime

The pilot comprises 500 seeded pairs, hence 1,000 response positions per
rendering. The primary rendering includes the user request and assistant
response; response-only scoring is a separate sensitivity run. The primary
analysis uses the common 4,000-pair sample, which contains 1,000 independently
sampled pairs from each L1--L4 stratum. DeepSeek therefore makes 8,000 binary
adjudications in each rendering, or 16,000 across the primary and sensitivity
renderings. The report includes raw sample counts beside inverse-probability
weighted population estimates, site-specific two-by-two tables against PKU
is_safe, proportional agreement, Cohen's kappa, directional conditional
agreement with PKU named as reference, and three-way tables stratified by
L1--L4, category set, and severity. The fixed DeepSeek rubric supplies only a
binary state. It is an external operationalisation of a safety boundary, rather
than a test of the full 19-category PKU taxonomy.

DeepSeek uses JSON mode, temperature zero, disabled thinking mode, and the
frozen safe/unsafe rubric in
\texttt{e5\_external\_boundary.json}. The configured credential and base URL
remain outside the repository. Its labels are external LLM judgements, not
safety truth labels or independent human annotations. ShieldGemma and OpenAI
moderation remain optional secondary sites rather than execution prerequisites.

## E6: CCAI reference-system comparison

The text-free shared 4,000-pair manifest and nested 200-pair E6 manifest are
fixed by:

    python3 gpu_api/scripts/e6_make_sample.py
    python3 gpu_api/scripts/preflight.py e6 --check-runtime

The shared manifest is first created with:

    python3 gpu_api/scripts/make_primary_sample.py

The manifests record source coordinates, hashes, strata, inclusion
probabilities, and weights. The API runner must re-read each selected raw row
from the pinned release rather than copy text into version control. The primary
job contains 200 times 21 principles times two response orders, or 8,400
judgements. The 10 percent repeat batch adds 840 judgements, for 9,240
planned judgements after the pilot. Temperature is zero; the execution-time model
snapshot, API date, retry policy, and malformed-output policy are committed
before submission.

For each pair-principle, reverse the response order. A substantive direction
that reverses is position-unstable and leaves the primary three-state
classification while remaining in the sensitivity output. One substantive
direction paired with an abstention is orientation-inconclusive and also
remains outside the primary table. Aggregate
estimators use the stored stratum weights. Buyl et al.'s published proportions
are contextual references only: their evaluation population, release mixture,
and oracle run differ from this fixed dual-training-split sample. The plan
therefore makes no model-only comparison or direct reproduction claim.

## E7: released cost-model artefact probe

Preflight command:

    python3 gpu_api/scripts/preflight.py e7 --check-runtime

Use PKU-Alignment/beaver-7b-v2.0-cost with its model-card safe_rlhf scorer.
Its documented conversation rendering begins BEGINNING OF CONVERSATION followed
by USER and ASSISTANT turns; the pilot verifies this rendering, the model
revision, tokenisation, truncation, score sign, and safe batch size. A 24 GB GPU is recommended for
the first BF16 run; lower-memory or quantised execution is permissible only if
the resulting precision, batch size, and equivalence check are frozen in the
run manifest. Score both responses in the shared 4,000-pair sample after the
pilot. The boundary AUROC and severity regression use inverse-probability
weights; pair-level bootstrap resampling remains stratified by L1--L4.

The aggregate report contains E7a's is_safe distributions, AUROC, and the
author-documented zero-sign boundary; E7b's category- and length-adjusted
severity trace plus the locked both-unsafe severity comparator; and E7c's
signed and absolute score gaps by L1--L4. This is a cross-version,
within-lineage transfer audit of an earlier released artefact on the later Dual
record release. It is neither an independent validation nor a generalisation
evaluation.

## Release gate for all three jobs

A primary-sample run may begin only when all of these are recorded:

- P0 SHA-256 and CPU native-audit SHA-256;
- exact model identifier and revision, or returned API model identifier;
- input rendering and normalisation;
- pilot manifest and frozen batch/retry/truncation policy;
- private-output location, reviewed aggregate-output location, and a rule that
  failed requests remain a distinct result state.
