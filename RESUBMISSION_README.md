# Resubmission experiments — how to run

Your original code is untouched in its behaviour: the S=3 path reproduces the
published numbers (the first-40 headline metric is identical; later episodes
differ only by PyTorch CPU floating-point nondeterminism, which would also
happen re-running the original). We have only ADDED what the reviewers asked
for.

## What was added
- `agents/mappo.py`      — MAPPO (CTDE, centralised critic). New baseline (R2.3, R1.5).
- `agents/ewc_ippo.py`   — EWC-IPPO (continual-learning baseline). Closes the
                            "you frame this as continual learning but never
                            compare to one" gap.
- env + drift generalised to S>3 (S=3 unchanged and verified identical).
- `train.py` registers the new methods, accepts `--n_slices`, and logs overhead
  (params/agent, optimisers/agent, inter-agent comm messages/step).
- `run_resubmission_experiments.py` — idempotent runner for the new runs only.
- `analyze_resubmission.py`         — all tables + significance (Holm-corrected).

## Install
```bash
pip install torch numpy matplotlib pandas
```

## Sanity check (~1 min) — confirm new methods run
```bash
python3 train.py --method mappo    --episodes 10 --horizon 100 --seed 0 --severity 1.5 --log_every 5
python3 train.py --method ewc_ippo --episodes 10 --horizon 100 --seed 0 --severity 1.5 --log_every 5
```

## Run the new experiments (idempotent; skips existing; resumable)
Run in chunks or overnight. Total ~3-5 h on a laptop CPU.
```bash
python3 run_resubmission_experiments.py --phase A   # new baselines, seeds 0-9  (~1 h)
python3 run_resubmission_experiments.py --phase B   # scaling N{6,10} x S{3,5}   (~2-3 h)
python3 run_resubmission_experiments.py --phase C   # severity sweep to n=10     (~1 h)
# or all at once:
python3 run_resubmission_experiments.py --phase all
```

## Produce the tables
```bash
python3 analyze_resubmission.py
```
Then paste the printed tables back to me and we draft the response letter,
reviewer by reviewer.

## What each phase answers
| Phase | Adds | Reviewer concern |
|---|---|---|
| A | MAPPO + EWC-IPPO at kappa=1.5, n=10 | R2.3, R1.5, AE (baselines) |
| B | N in {6,10}, S in {3,5} + overhead/comm log | R1.4/R2.4 (scale), R1.3/R1.7 + AE (overhead) |
| C | severity sweep at n=10 (was n=3) | robustness; fixes the thin-data soft spot |

## Integrity note
The new baselines share the actor-critic, PPO update, and hyperparameters with
your existing methods, so the comparison isolates the training paradigm, not
tuning. Report whatever the runs produce. Your genuine, checkable advantages —
sample efficiency (first-40), policy stability (switching cost), and
decentralised zero-communication scaling — are what to build the response
around. If MAPPO wins on asymptotic reward, that is expected (it is
centralised); the honest, strong framing is that Nested-MARL approaches
centralised reward WITHOUT the communication cost (see the comm/step column).
```
```
