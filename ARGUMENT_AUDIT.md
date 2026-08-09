# Experiment-to-chapter argument audit

This audit checks the executable package against the dissertation question in
Background section 2.5 and the evidence structure in Methodology chapter 3.
It is a design record, not a result claim.

## Claim-to-evidence map

| Chapter claim | Necessary evidence | Experiment decision | Boundary carried into reporting |
|---|---|---|---|
| A safer winner can be unsafe because both candidates are unsafe | Native absolute-state decomposition | Retain E3 as the central Chehbouni transfer | The absolute state is PKU's own field; E3 does not validate that field |
| The native fields reflect one documented measurement record and response process | Schema support, identity exceptions, annotation-card documentation | Retain E1 | Co-recorded associations are recording-architecture evidence, not independent convergence |
| Moderation and alignment retain different distinctions | Field-to-branch and loss trace | Retain E4 | A missing direct loss term does not show a neural model cannot recover correlated text features |
| The released safety boundary has a stated relation to outside safety readings | An operationalisation independent of the annotation card | Retain E5; ShieldGemma is primary, OpenAI moderation is robustness | Neither external site supplies ground truth; lineage differences cannot identify GPT-4 anchoring |
| The released cost artefact visibly retains or collapses native distinctions | Release-matched cost-model scores | Retain E7 | This is an artefact audit, not a causal policy or held-out-generalisation study |
| PKU native distinctions can be placed beside an explicit principle system | New stratified mapping to Buyl's CCAI states | Retain E6 as a horizontal comparison | CCAI is neither a safety truth label nor an arbitrariness estimate |

## Decisions made after reading the chapters

### E2 is secondary and conditionally useful

E2 cannot bear the dissertation's headline construct-validity claim. All safe
responses normally have an all-zero category vector, and a full-population EFA
would mainly rediscover the binary safety boundary. The executable protocol
therefore conditions EFA on responses released as unsafe. The category
frequency, conditional co-occurrence, severity profile, and unsafe-conditioned
EFA are retained as supporting evidence for the moderation-facing taxonomy.
The completed EFA required tetrachoric-matrix smoothing and retained seven
factors, including near-single-item factors. It consequently remains an
appendix diagnostic, rather than evidence for a seven-dimensional latent
taxonomy. Conditional co-occurrence, solo rates, and severity profiles replace
it as the primary E2 presentation.

### E6 is retained with a narrower comparative claim

Buyl et al.'s published prevalence estimates are based on a distinct PKU
evaluation population and release mixture. Comparing a new sample from the
dual training split to those values cannot isolate an oracle-model change. E6
therefore reports the weighted native-stratum by CCAI-state table as its
contribution. The published values are contextual reference only.

### E5 requires policy-level reporting before binary collapse

ShieldGemma has four policy areas, whereas PKU releases 19 harm categories.
Its any-policy binary collapse can support a relation to that declared external
operationalisation. It cannot verify PKU's full category taxonomy. E5 retains
the policy-specific outputs and treats their aggregate as a secondary summary;
the OpenAI moderation site is a distinct robustness operationalisation rather
than a coverage repair.

### E8 and E9 remain conditional human extensions

Reception and unhelpful-selection audits answer valid questions, yet neither is
necessary for the chapter's core construct-to-pipeline argument. They are
excluded from the executable CPU/GPU package unless the dissertation retains a
claim about later community description or an empirical claim about the
protocol's helpfulness exclusion. An API response cannot replace the stated
human coding in either extension.

### Proposals intentionally excluded

A post-hoc GPT-family agreement test cannot estimate GPT-4 anchoring; recasting
PKU categories as CCAI principles changes the reference construct; and a
Bradley--Terry reliability ceiling cannot be inferred from BeaverTails
agreement. These analyses remain excluded because their apparent numerical
precision would exceed their inferential warrant.

## Reconciled implementation details

1. The four E6/E7 native strata are now exhaustive: L4 is an is_safe
   difference; otherwise L3 is a severity difference; otherwise L2 is a
   complete-category-set difference; remaining pairs are L1. This includes
   the eight field-hierarchy exceptions without dropping them.
2. Unsafe-only subsets are named explicitly where the substantive question
   concerns severity or taxonomy internal structure.
3. The cost-model protocol uses the published Beaver scorer and its documented
   conversation rendering rather than borrowing the external-classifier
   rendering.
