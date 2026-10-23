# 08 · Distributed compute (work units)

Status: agreed direction 2026-10-24. The CLI and JSON schemas are implemented in `kepler-train`. This section is final once the shard commands land.

## 1. Policy

- **Local machine:** light runs only, at most about **1 hour** at high CPU per session, at most 6 threads by default. Everything checkpoints, so a run can stop and resume.
- **Heavy runs** (full generations at depth ≥ 4, Law Ledger, Stockfish ladder, long SPRTs) run on remote compute:
  - first on GitHub Actions in a separate compute repository;
  - later through the maintainer's mesh orchestrator.
- Every remote result is reduced into the same files a local run would produce, and committed to `engine/results/` or `engine/data/generations/` here. A number's provenance doesn't depend on where it was computed.

## 2. Why work units

Everything expensive in v1 is a sum of independent games:

| job | unit | combine |
|---|---|---|
| TDLeaf round | play N self-play games with constants X; return the gradient sum | sum gradients, one Adam step |
| SPRT / Law Ledger | play N opening pairs A vs B; return a pentanomial | sum pentanomials, one test |
| Stockfish ladder | play N games vs one rung; return W/D/L | sum per rung |
| suites | run M positions; return rows | concatenate |

So a unit is: **one binary, one or two JSON inputs, a seed, and a count, writing one JSON output.** No network, no shared state, no secrets. Any worker that can run a Linux x86-64 binary can do any unit.

## 3. Unit contract

**Inputs:**
- the `kepler-train` binary, built from a specific commit, recorded in the output;
- generation JSON(s);
- the opening suite (`engine/data/openings/balanced-200.epd`);
- CLI flags.

Stockfish units also need a Stockfish binary (downloaded by the worker, path in `KEPLER_STOCKFISH`).

**Determinism:**
- Units are depth- or node-limited, so a unit's output is a pure function of (binary, inputs, seed), whatever the hardware.
- Time-limited units (`--movetime`, `--tc`) are not deterministic. They're allowed for ladders only, and must be labelled as time-limited.

**Seeds:** a round seed `R` and the shard index `k` give the shard seed through the documented mix function in `kepler-train` (`util.rs`). The reducer checks that the shard seeds match the expected ones, so a duplicated or missing shard is detected.

**Integrity:** every shard records `spec`, `gen_hash` (or `a_hash`/`b_hash`) and `config`. The reducer refuses mixed shards. A shard that fails validation is re-queued, never patched.

**Output schemas** (version 1):
- `tdleaf-shard`: spec, gen_hash, seed, games, plies, grad_sum[12], abs_d_sum, results, elapsed_ms, config
- `sprt-shard`: spec, a_hash, b_hash, seed, pentanomial[5], wdl, elapsed_ms, config

Exact field definitions live next to the code (`kepler-train` source, `tdleaf.rs` and `sprt.rs` docs).

## 4. Commands

```bash
# one TDLeaf shard
kepler-train tdleaf-shard --gen gen.json --openings balanced-200.epd --games 100 --seed <S_k> --depth 4 --threads 4 --out shard-k.json
# combine a round and step
kepler-train tdleaf-reduce --state checkpoint-NNN.json --gen gen.json --shard-dir shards/ --lr 0.01 --out-dir run/
# SPRT in batches
kepler-train sprt-shard --a new.json --b old.json --openings balanced-200.epd --pairs 50 --seed <S_k> --depth 4 --threads 4 --out s-k.json
kepler-train sprt-reduce --shards s-*.json --elo0 0 --elo1 10 --alpha 0.05 --beta 0.05 --out sprt.json
# cosmological natural selection (SPSA): theta+ vs theta- mini-match, then one step
kepler-train evolve-shard --gen centre.json --state none --openings balanced-200.epd --iter 0 --seed <S> --shard-index <k> --pairs 50 --depth 4 --threads 4 --out e-k.json
kepler-train evolve-reduce --state evolve-checkpoint-NNN.json --gen centre.json --shard-dir shards/ --a0 0.1 --A 10 --c0 0.05 --out-dir run/
```

TDLeaf flags added by the universe methods (`10_UNIVERSE_METHODS.md`), on `tdleaf`, `tdleaf-shard` and `tdleaf-reduce` unless noted:
- `--temperature <T0>`: starting temperature T in `tanh(eval / T)` (default: `training.temperature` of `--gen`, else 4.0; a checkpoint overrides it). It replaces the old `--scale`. Shards record it and `tdleaf-reduce` refuses shards played at another T. Shards output `grad_logT_sum`.
- `--learn-temperature` (default on) / `--no-learn-temperature` (`tdleaf`, `tdleaf-reduce`): learn T with Adam on `log T`, bounds [0.5, 50], or keep it fixed. T is written to the checkpoint and `candidate.json` (`training.temperature`); feed `candidate.json` back as `--gen` for the next round.
- `--capped-target drop|score|draw` (default `drop`; `tdleaf`, `tdleaf-shard`): what a game stopped by `--max-plies` teaches. Recorded in the shard config; the shard JSON counts `dropped_plies`. Shards in one reduce must agree.
- `--conserve-mass` (`tdleaf`, `tdleaf-reduce`, default off): rescale the four masses after each step so `8 + 2·m_n + 2·m_b + 2·m_r + m_q` keeps its value from the start of the run (stored in the checkpoint).
- The TD leaf is now the quiet leaf (end of the quiescence line from the PV leaf); there is no flag.

Evolve units:
- `evolve-shard`: `--gen`, `--state <checkpoint|none>`, `--openings`, `--iter K`, `--seed S`, `--shard-index k`, `--pairs N`, `--c0` (default: the state's, else 0.05), `--depth D` or `--movetime ms`, `--threads`, `--max-plies`, `--explore-plies`, `--hash`, the adjudication flags of `sprt-shard`, `--out`. The shard draws its ±1 vector Δ (13 parameters: the 12 constants and `f_acc`) from `mix(mix(S, K), k)`, so the seed and index alone reproduce it. Output (version 1): `kind`, `spec`, `centre_hash`, `iter`, `seed`, `shard_index`, `c_k`, `delta[13]`, `score_plus`, `pentanomial[5]`, `wdl`, `elapsed_ms`, `config`.
- `evolve-reduce`: `--state`, `--gen` (the centre; its hash must equal the shards' `centre_hash`), `--shards ...` or `--shard-dir`, `--a0` (0.1), `--A` (10), `--c0` (0.05) (state values are used unless a flag is given), `--out-dir`. It refuses shards with a different `centre_hash`, `iter`, seed, `c_k` or Δ, duplicate shard indexes, and an `iter` that differs from the checkpoint's. It writes `evolve-checkpoint-KKK.json`, `evolve-iter-KKK.json` and `candidate.json` (training method `cosmological-natural-selection-spsa`). Iteration K needs the checkpoint of iteration K-1 as `--state` and its `candidate.json` as `--gen`.
- Local loop: `kepler-train evolve --gen gen.json --openings balanced-200.epd --out-dir run/ --iters 20 --shards 4 --pairs 50 --depth 4 [--a0 --A --c0 --seed --max-minutes --resume]` runs the same shard and reduce code in-process.

The local `tdleaf` command runs the same shard and reduce code in-process. A local round and a distributed round with the same seeds give identical constants (this is tested).

## 5. Backend 1: GitHub Actions in a compute repository

- **A separate public repository**, [`r-baruah/kepler-64-compute`](https://github.com/r-baruah/kepler-64-compute) (created 2026-10-24), holds a copy of `engine/` and the workflows. It's separate so compute runs don't clutter this repo's Actions history, and so it never needs this repo's git history. Public repositories get standard runners at no cost; private ones have a monthly minute cap.
- **Runner:** 4 vCPU and 16 GB RAM, 6 h per job, up to 20 concurrent jobs on the free plan.
- **One workflow run = one round:**
  1. `build`: compile `kepler-train` once, upload it as an artifact.
  2. `shard` (matrix of K ≤ 20): download the binary and inputs, run one unit with `--threads 4`, upload `shard-k.json`.
  3. `reduce`: download all shards, run the reducer, and commit the checkpoint, `round-NNN.json` and `candidate.json` to the compute repo.
  4. Trigger the next round (`workflow_dispatch` with the round number) until `--rounds` is reached or a stop file is committed.
- **Expected speed** at depth 4, 2,000 games per round, K = 20: about 2–3 min of play plus about 2 min of job overhead per round. That's roughly 1.5–2 h for a 20-round generation.
- **Usage rule:** runs are purposeful (a generation, a ledger, a ladder), started by the maintainer, never scheduled to run continuously. The workload is testing and tuning the software in that repository.
- **Results** come back here by downloading the reduced files and committing them, with a link to the workflow run in the result's JSON.

## 6. Backend 2: the maintainer's mesh orchestrator (planned)

The orchestrator schedules work units across free compute (CI runners, notebook sessions, volunteer machines). What Kepler needs from it is only what §3 defines: ship a binary and inputs, run the command, collect the JSON, re-queue failures, then run the reduce step. No Kepler-specific code should be needed in the orchestrator. If it is, the unit contract is incomplete and gets fixed here.

## 7. Other options considered

- **Kaggle CPU notebooks:** 4 cores, long sessions, so no faster than the laptop and hard to automate. A backup for one long single-machine run only.
- **Hugging Face Spaces (free tier):** 2 vCPUs, sleeps when idle; meant for demos. Not used for compute.
