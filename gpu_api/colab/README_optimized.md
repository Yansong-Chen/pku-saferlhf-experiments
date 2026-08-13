# E5 ShieldGemma: validated optimised Colab run

This bundle separates the completed pilot from the optimised primary scorer.
The candidate scorer uses left padding and Gemma's supported one-last-logit
argument.  It only becomes eligible for the primary job after it reproduces
the completed right-padding pilot's policy labels at the frozen 0.5 threshold.

## Required order in Colab

1. Upload `e5_shieldgemma_optimised_colab_bundle.tgz` and run the environment,
   GPU, and Hugging Face access cells.
2. Run the direction checks at the commit stated in the supplied pilot
   manifest.
3. Score the candidate on the 1,000-position pilot, then run the equivalence
   check.  The check requires all 2,000 position/rendering terminal results,
   unchanged truncation states, and zero policy-label or any-policy changes.
4. Probe batches 32, 16, 8, and 4 against deterministically selected long
   primary inputs.  Use the largest passing batch size reported by the
   notebook.
5. Only after both gates pass, start the 4,000-position primary job.

The scorer is append-only and resumes a position/rendering only after its
terminal record is successful.  Re-running a score cell after a transient
failure adds a new attempt; ingestion uses the final attempt.  Download the
primary score file and manifest before ending the Colab runtime.

## Files to retain

Keep the downloaded `e5-primary-leftpad.scores.jsonl`,
`e5-primary-leftpad.scores.manifest.json`,
`e5-leftpad-equivalence.json`, and selected batch-probe report.  The first two
are the primary experimental artefacts.  The latter two document the
execution change and its validation.

## Local ingestion after download

From the repository's `experiments` directory, use a new run identifier:

```bash
python gpu_api/scripts/ingest_gpu_scores.py \
  --phase primary \
  --run-id e5-sg-primary-leftpad-4000 \
  --scores /path/to/e5-primary-leftpad.scores.jsonl \
  --score-manifest /path/to/e5-primary-leftpad.scores.manifest.json
```

Do not ingest a primary output if the equivalence report says `"passed": false`.
