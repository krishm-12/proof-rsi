# Experimental evidence

- `gpu-rsi/` contains the first completed CUDA run. Its original runner tried up to two candidates per level-zero curriculum problem.
- `gpu-rsi-resample/` contains the second completed CUDA run. It used the configured five attempts per curriculum problem, including three sampled retries after the initial attempt and repair. This is the full-run configuration documented in the main README.
- Each `metrics.json` preserves the recorded metrics and run configuration with paths made relative. The `evidence` field summarizes the full attempt log.
- Each `attempts.jsonl` is a representative subset with intact initial and repair sequences, sampled retries where present, and the verified smoke attempt. `verified_smoke.jsonl` records the accepted smoke proof.
- `model.json` records the upstream model revision and logical parameter counts calculated from the base and smoke adapter checkpoint tensor shapes. The denominator counts the tied embedding once.

The two runs verified the same reflexive smoke theorem. The held-out fixtures and the examples dataset are hand-authored. They do not count as model-generated successes. No round produced an accepted training adapter.
