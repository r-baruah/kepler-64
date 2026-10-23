import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import chess
from kepler64 import RocheEngine
from kepler64.uci import search_full
eng = RocheEngine()
lines = {
 "French advance (W to move)": "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6",
 "French advance (B to move)": "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6 Bd3",
 "KID closed (W to move)": "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7",
 "KID closed (B to move)": "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1",
}
for name, sans in lines.items():
    b = chess.Board()
    for s in sans.split(): b.push_san(s)
    out = []
    for _ in range(6):
        uci, sc, d = search_full(b, eng, 3, None, use_multiverse=False)
        mv = chess.Move.from_uci(uci); out.append(b.san(mv)); b.push(mv)
    print(f"{name}: Kepler plays both sides ->", " ".join(out))
