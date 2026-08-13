# E5--E7 execution runbook

The three executors read the exact dual-dimension release under ../data/raw/.
They never write prompts or responses to version-controlled files. Each run
uses an ignored directory below gpu_api/private_runs/; only reviewed aggregate
tables should later be force-added to gpu_api/results/.

## One-time setup

Run the work from the experiments repository:

    cd "/Users/yansongchen/Downloads/untitled folder 21/experiments"
    python3 gpu_api/scripts/preflight.py e5 --check-runtime
    python3 gpu_api/scripts/preflight.py e6 --check-runtime
    python3 gpu_api/scripts/preflight.py e7 --check-runtime
    python3 gpu_api/scripts/make_primary_sample.py
    python3 gpu_api/scripts/e6_make_sample.py

For an API machine, create an isolated environment and install:

    python3 -m venv gpu_api/.venv
    source gpu_api/.venv/bin/activate
    pip install -r gpu_api/requirements-api.txt
    export OPENAI_API_KEY='...'

For a CUDA machine, use Python 3.10 or 3.11, install the CUDA-compatible
PyTorch build recommended for that driver, then:

    pip install -r gpu_api/requirements-gpu.txt
    pip install git+https://github.com/PKU-Alignment/safe-rlhf.git
    export HF_TOKEN='...'

Before E5, accept the ShieldGemma terms on Hugging Face with the account that
owns HF_TOKEN. E7 uses the author-released safe_rlhf scorer; keep its
environment separate if its dependency set conflicts with the current
Transformers stack.

## E5: external safety-boundary relation

Question: how does PKU's released binary state relate to a pre-specified
external safety adjudication? The primary site is DeepSeek V4 Flash through its
JSON-mode chat API. It evaluates a frozen safe/unsafe rubric that was not
produced on the PKU annotation card. Its result is an external LLM judgement,
not safety truth or an independently collected human label.

The primary input contains prompt and response. The response-only run is a
sensitivity analysis and uses the model card's prompt-only guidelines, so it
must be reported as a different rendering rather than pooled with the primary
result.

1. Validate volume without loading a model or calling an API:

       python3 gpu_api/scripts/e5_run.py --site all --phase pilot --run-id e5-pilot --dry-run

2. Run the fixed 500-pair pilot. It covers 1,000 response positions per
   rendering. The safe/unsafe rubric, JSON contract, temperature, disabled
   thinking mode, and 32-token completion limit are already fixed; the pilot
   never changes them.

       python3 gpu_api/scripts/e5_run.py --site deepseek --phase pilot --run-id e5-pilot

3. Inspect gpu_api/private_runs/e5/e5-pilot/run_manifest.json. Record the
   returned DeepSeek model identifier, failed-record count, worker count,
   completion diagnostics, and retry settings. Freeze those fields before the
   primary-sample run. Any parsing or request failure remains a failure record;
   it is never recoded as safe.

4. Run the common 4,000-pair primary sample (8,000 response positions) in its
   context-preserving and response-only renderings. The runner retains inverse
   pair-inclusion-probability weights, and the aggregator returns both raw
   sample facts and weighted population estimates.

       python3 gpu_api/scripts/e5_run.py --site deepseek --phase primary --run-id e5-primary

5. Aggregate after coverage review:

       python3 gpu_api/scripts/e5_aggregate.py \
           --private-run gpu_api/private_runs/e5/e5-primary \
           --aggregate-dir gpu_api/results/e5-primary

The report gives a binary relation to the frozen DeepSeek safety rubric. It
does not validate PKU's 19-category taxonomy. ShieldGemma and OpenAI moderation
can be run later as separately labelled secondary operationalisations.

## E6: CCAI reference-system comparison

Question: how do the release's L1--L4 native distinctions co-occur with
Buyl et al.'s 21-principle CCAI states? The committed manifest contains a
seeded, nested 200-pair stratified probability sample and no text. Its fixed
allocation is 50 pairs in each L1--L4 stratum; every selected E6 pair is also
in the shared E5/E7 sample.

The primary job has 200 pairs times 21 principles times two response orders:
8,400 judgements. The independent 10% repeat has 840 further judgements.
Run the pilot before authorising this volume: it has 50 pairs and 2,100
judgements, so it gives a real estimate of malformed outputs, token use,
latency, and cost without altering the prompt or classification rule.

1. Confirm the exact sample and pilot plan:

       python3 gpu_api/scripts/e6_run.py --phase pilot --run-id e6-pilot --dry-run

2. Run and aggregate the pilot:

       python3 gpu_api/scripts/e6_run.py --phase pilot --run-id e6-pilot
       python3 gpu_api/scripts/e6_aggregate.py \
           --private-run gpu_api/private_runs/e6/e6-pilot \
           --aggregate-dir gpu_api/results/e6-pilot --phase pilot

3. Freeze the returned model identifier, API date, worker count, retry rule,
malformed-output rate, token-use total, and any run deviation. Do not edit
the 21 principles, prompt, output parser, or sample after this point.

E6 uses DeepSeek JSON mode rather than a bare-text label. The request sets
`response_format={"type":"json_object"}` and the system instruction requires
exactly one object with the key `label`, whose value is `A`, `B`, or `No
preference`. It also disables DeepSeek thinking mode, so the completion budget
is available to the JSON response; the completion limit is 32 tokens to allow
the object to complete. The parser accepts only that one-field object. This is a
DeepSeek-specific output protocol, so the audit retains Buyl et al.'s 21
principles and pairwise user prompt without claiming an exact reproduction of
their GPT-4o elicitation. Any other response remains an
`UnparseableModelOutput` after the configured retries. It is recorded as a
failed request, excluded from orientation reconciliation, and never recoded as
`No preference`. A saved failure is terminal for that run ID; begin a new,
separately documented run if it is to be attempted again.

4. Execute the primary and repeat jobs, resuming safely after interruption:

       python3 gpu_api/scripts/e6_run.py --phase primary --run-id e6-primary
       python3 gpu_api/scripts/e6_run.py --phase repeat --run-id e6-primary
       python3 gpu_api/scripts/e6_aggregate.py \
           --private-run gpu_api/private_runs/e6/e6-primary \
           --aggregate-dir gpu_api/results/e6-primary --phase primary \
           --bootstrap-replicates 10000 \
           --bootstrap-seed 20260811 \
           --sample-manifest gpu_api/config/e6_sample_manifest.csv
       python3 gpu_api/scripts/e6_aggregate.py \
           --private-run gpu_api/private_runs/e6/e6-primary \
           --aggregate-dir gpu_api/results/e6-primary --phase repeat \
           --sample-manifest gpu_api/config/e6_sample_manifest.csv

The executor maps both orderings back to response identifiers. A reversed
substantive direction is position-unstable. A substantive result in one order
and abstention in the other is orientation-inconclusive. Both are reported
separately and excluded from the primary state table. Remaining principle
votes produce indifference, consensus, or conflict. The weighted L1--L4 table
is the contribution; Buyl et al.'s published overall percentages are
contextual reference only.
The aggregate also records whether the released `safer_response_id`
selected the released-safe response in the analysed L4 pairs. This is a
release-record direction check, distinct from a validation of the safety label.

## E7: released Beaver cost-model probe

Question: which release distinctions are observable in a released,
release-matched cost artefact? This is not a held-out generalisation
evaluation. The runner uses the model-card conversation string:

    BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT: {response}

1. Confirm the 500-pair, 1,000-position pilot without loading weights:

       python3 gpu_api/scripts/e7_run.py --phase pilot --run-id e7-pilot --dry-run

2. Score the pilot at the candidate revision:

       python3 gpu_api/scripts/e7_run.py --phase pilot --run-id e7-pilot
       python3 gpu_api/scripts/e7_pilot_report.py \
           --private-run gpu_api/private_runs/e7/e7-pilot \
           --output gpu_api/results/e7-pilot-orientation.json

3. Record the resolved model commit and score direction. The pilot report
   gives AUROC under both possible orientations, yet deliberately makes no
   automatic choice. Choose higher_is_unsafe or lower_is_unsafe using the
   author convention and pilot distribution; write the rationale in the primary
   run manifest.

4. Run the shared 4,000-pair sample (8,000 response positions), then aggregate:

       python3 gpu_api/scripts/e7_run.py --phase primary --run-id e7-primary \
           --model-revision RESOLVED_HF_COMMIT \
           --score-direction higher_is_unsafe
       python3 gpu_api/scripts/e7_aggregate.py \
           --private-run gpu_api/private_runs/e7/e7-primary \
           --aggregate-dir gpu_api/results/e7-primary \
           --score-direction higher_is_unsafe \
           --bootstrap-replicates 2000

E7a reports unsafe-versus-safe AUROC and a clearly labelled zero-threshold
diagnostic. E7b fits the category- and length-adjusted severity association
among released-unsafe responses and reports the same-category,
unequal-severity ordering rate. E7c reports signed gaps by L1--L4, where a
positive value means greater risk for the response the dataset calls less
safe. Pair-level bootstrap resampling keeps both response positions together.

## Final release gate

Before copying any aggregate table into the dissertation, verify all of the
following:

- the run manifest names the exact P0 and native-audit hashes;
- model/API identifier, model revision, prompt rendering, tokenisation,
  truncation, batch and retry settings are present;
- expected coverage, failures, and resumptions are reported;
- no file headed for Git contains prompt text, response text, or vendor raw
  payloads;
- the table caption names the operationalisation and its inferential boundary.
