import os, sys, random
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np, chess
from kepler64.core.constants import load_constants
from kepler64.core.universe import GameState
from kepler64.core.evaluate import score_white_terms
from kepler64.core.transitions import child_mass_vector

C = load_constants()
random.seed(1)
names = None; rows = []; decisive = None
NPOS = 120
for g in range(NPOS):
    b = chess.Board(); st = GameState(b, c=C.c)
    for _ in range(random.randint(8, 50)):
        if b.is_game_over(): break
        mv = random.choice(list(b.legal_moves)); st.push(mv); b.push(mv)
    if b.is_game_over(): continue
    parent = st.masses
    per_move = []
    for mv in list(b.legal_moves):
        s2 = st.copy(); s2.push(mv)
        t = score_white_terms(s2.masses, C, parent=parent)
        per_move.append([float(x) for x in t])
    names = score_white_terms.__annotations__ and None
    arr = np.array(per_move)  # moves x terms
    sign = 1 if b.turn == chess.WHITE else -1
    rows.append((arr, sign))

from kepler64.core.evaluate import EvalTerms
names = EvalTerms._fields
k = len(names) - 1
mag = np.zeros(k); spread = np.zeros(k); flips = np.zeros(k)
for arr, sign in rows:
    mag += np.abs(arr[:, :k]).mean(0)
    spread += arr[:, :k].std(0)
    best = np.argmax(sign * arr[:, k])
    for j in range(k):
        alt = np.argmax(sign * (arr[:, k] - arr[:, j]))
        flips[j] += (alt != best)
n = len(rows)
print(f"positions={n}")
print(f"{'term':16s} {'mean|v|':>9s} {'spread across moves':>20s} {'best-move changes if removed':>30s}")
for j in range(k):
    print(f"{names[j]:16s} {mag[j]/n:9.3f} {spread[j]/n:20.3f} {100*flips[j]/n:29.1f}%")
