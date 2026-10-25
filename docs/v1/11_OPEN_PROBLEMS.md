# 11 Open problems

Kepler-64 is a small universe with open questions. These are real ones, taken from our own findings (`09_FINDINGS.md`), each with a way to tell whether it's solved.

The one rule: a new law must be a quantity of the field in `02_LAWS_SPEC.md`. Pull, mass, exposure, the squares each side holds. No chess heuristics (no piece-square tables, king-safety tables, pawn-structure rules, books or tablebases). If it can't be written as physics on the board, it doesn't go in.

How a law is judged: switch it on, let self-play learn its weight, then play the result against the current champion. It stays only if it wins a sequential test (SPRT) over thousands of games, the same Law Ledger test every current law passed (`04_TRAINING_AND_GATES.md`).

## 1. The king problem
Two versions of a Roche limit (the king "torn apart" by enemy pull) both failed to make the engine stronger (F27, F31, F33, F35). The universe has no law about danger to the king that it keeps.

**Solved when:** a king-danger law, written only from field quantities, passes the Ledger against the champion.

## 2. Make accretion work
Accretion lets a capturing piece take on mass from the piece it captured. At Gen 1, switching it off was worth about +103 Elo (`engine/results/ledger-gen001.md`), so it was cut.

**Solved when:** a version of mass transfer on capture passes the Ledger, or someone shows why no version can help.

## 3. Any new law
Propose a law of your own. It might be borrowed from real physics or invented for this board.

**Solved when:** it passes the Ledger against the champion.

## 4. Explain a result (no code needed)
- **Why does a fixed temperature beat a learned one?** Learning the training temperature lost to keeping it fixed at 4 (F32).
- **Why do the self-found piece values land where they do?** From equal weights, two independent runs settled near knight 2.6, bishop 3.3, rook 4.4, queen 8.2 (pawn = 1). What in the field produces those ratios?

**Solved when:** an explanation makes a prediction that a run or measurement confirms.

## Taking one on
Open an issue on GitHub naming the problem number. The commands for training and testing are in `04_TRAINING_AND_GATES.md`. Heavy runs can go to remote compute as work units (`08_DISTRIBUTED_COMPUTE.md`).
