import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import chess
from kepler64.core.constants import load_constants
from kepler64.core.universe import GameState
from kepler64.core.evaluate import score_white_terms
C = load_constants()
b = chess.Board()
for s in "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6".split(): b.push_san(s)
st = GameState.from_moves(b.move_stack, c=C.c)
prev = None
for s in ["Qd3", "c4", "Qxc4", "dxc4", "Bxc4"]:
    parent = st.masses
    mv = st.board.parse_san(s); st.push(mv)
    t = score_white_terms(st.masses, C, parent=parent)
    print(f"after {s:5s} material={float(t.material):7.2f} force_enemy_king={float(t.force_enemy_king):7.2f} force_own_king={float(t.force_own_king):7.2f} total={float(t.total):7.2f}")
