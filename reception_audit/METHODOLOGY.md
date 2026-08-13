# PKU-SafeRLHF reception and reuse audit

## Purpose

This documentary study asks how later scholarly works publicly describe and
document their reuse of the **2024/2025 multi-level PKU-SafeRLHF release**
audited in this dissertation. It follows the reception component of Chehbouni
et al.'s audit of HH, while keeping a narrower object: the relationship between
published descriptions of a dataset and the released fields that a later work
reports using.

The study does not verify a safety label, infer an implementation that a paper
does not document, or measure the effect of the dataset on a trained model. It
is evidence about published reception and declared reuse.

## Research question

> In later scholarly works, how is PKU-SafeRLHF described, and, for each
> documented consumer, which released fields enter it directly rather than only
> through an intermediate model or score?

## Corpus construction

The retrieval date, provider response, query URL, and the two target works are
frozen before screening:

1. ACL version: `10.18653/v1/2025.acl-long.1544`.
2. arXiv version: `10.48550/arXiv.2406.15513`.

OpenAlex provides the primary cited-by frame. Semantic Scholar may be queried
only as a coverage check. Exact full-text searches for `PKU-SafeRLHF` and the
Hugging Face repository name are a *discovery supplement*, since a name match
can refer to a predecessor or a derivative dataset without citing either target
work. Candidate lists are deduplicated by DOI, arXiv identifier, and manually
checked title/version pairs.

Every candidate is screened against the full text and its bibliography. It is
included in the target corpus if it either cites one of the two target works or
unambiguously identifies the full multi-level repository/release. It is placed
in a separate **related-release ledger** if it uses or cites the original
SafeRLHF dataset, BeaverTails, or `PKU-SafeRLHF-30K` without establishing that
it uses the 2024/2025 multi-level release. These records are useful for
understanding ambiguous names, but are never pooled with the target corpus.

If the frozen target corpus has 100 or fewer eligible documents, it is a
census. If it exceeds 100, the main sample is a fixed-date, reproducibly
selected stratified sample of 100, stratified by publication year and declared
relationship to the dataset. The screening ledger reports inaccessible full
texts and all exclusions.

## Unit, coding, and evidence standard

The unit is one scholarly work. A work can receive several non-exclusive
field and use codes. Each substantive code must include a page, table, appendix
or repository location plus a short evidence note. Missing documentation is
coded as **not reported**, never as a claim that the field was not used.

The codebook has five parts:

1. **Release identity**: target multi-level release; related derivative or
   predecessor; ambiguous; not a dataset use.
2. **Consumer and data route**: policy, reward model, cost model,
   moderation/classifier, or evaluation; then record whether that component
   receives a prompt, response pair, named dataset field, or a frozen model's
   scalar output. A policy trained by PPO from a frozen cost model therefore
   has a different route from the prior training of that cost model.
3. **Released fields named in the document's dataset description**:
   helpfulness preference; safety preference (`safer`); response-level binary
   safety state (`is_safe`); harm categories; severity; no field named.
4. **Fields directly received by the named consumer**: the same five fields,
   plus `prompt_only`, `intermediate_score`, and `not_documented`. This is the
   primary analytic code; it is never inferred from a general dataset
   description.
5. **Granularity of the description**: generic safety data; separate
   helpfulness and safety preferences; multi-level/category/severity claim;
   no substantive dataset description.
6. **Evidence for use**: an exact repository, split, column, configuration, or
   training/evaluation procedure is shown; reuse is asserted but the mechanism
   is not documented; citation only; not applicable.

For a multi-coder study, two human coders independently code a common pilot of
10 documents, revise only definitions (not observed outcomes), then
independently double-code at least 30 documents or the whole corpus when the
corpus is smaller. Report agreement before adjudication for release identity,
relationship, and each field indicator. An LLM may locate likely passages but
cannot substitute for a second independent human coder.

## Analysis and reporting

Report a flow table: retrieved, deduplicated, full text obtained, target
eligible, related-release, and excluded. The primary result is a
component-level table giving the direct input route and direct PKU fields for
every named consumer. A separate document-level table reports how authors
describe the release. This separation prevents a sentence such as "the dataset
has safety meta-labels" from being read as evidence that a policy received
those meta-labels. Short, attributed examples illustrate patterns; they do not
stand in for counts.

The interpretation remains documentary. For example, a paper can call the
dataset "multi-level" while its policy receives only prompts and scalar
feedback from a frozen cost model. The audit records the difference between the
dataset description and the policy's direct input route; it does not allege
misuse or infer the contents of unreported code.

## Place in the dissertation

This is a consequences-and-use extension. It can support a claim about later
published description and declared reuse. It should not be used to strengthen
the core claims about the native label, the published Safe RLHF objective, or
the cost-model artefact. If included, it belongs after the documented
field-to-objective routing analysis, with the coding detail and full ledger in
an appendix.
