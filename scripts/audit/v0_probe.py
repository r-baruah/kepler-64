import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import chess
from kepler64 import RocheEngine
from kepler64.uci import search_full

eng = RocheEngine()
depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3

def mate_in_1(b):
    for m in b.legal_moves:
        b.push(m); ok = b.is_checkmate(); b.pop()
        if ok: return m
    return None

def attacker_move(b, plan):
    m = mate_in_1(b)
    if m: return m
    for san in plan:
        try:
            mv = b.parse_san(san)
            return mv
        except ValueError:
            continue
    return None

def kepler(b):
    t = time.time(); uci, score, d = search_full(b, eng, depth, None, use_multiverse=False)
    return chess.Move.from_uci(uci), time.time() - t, score

# Test 1: scholar's mate attack, Kepler as Black
plans = {
 "scholar": ["e4", "Bc4", "Qh5", "Qf3", "Qxf7#", "Bxf7+"],
 "wayward-queen": ["e4", "Qh5", "Bc4", "Qxf7#"],
 "fools-for-white?": ["f3", "g4"],
}
for name, plan in plans.items():
    b = chess.Board(); times = []
    for ply in range(24):
        if b.is_game_over(): break
        if b.turn == chess.WHITE:
            mv = attacker_move(b, [s for s in plan if s])
            if mv is None:
                mv, dt, _ = kepler(b)  # fall back: let kepler play white too
            b.push(mv)
        else:
            mv, dt, sc = kepler(b); times.append(dt); b.push(mv)
    g = chess.Board().variation_san(b.move_stack)
    print(f"[{name}] depth={depth} result={b.result(claim_draw=True)} mate={b.is_checkmate()} plies={len(b.move_stack)} avg_move_s={sum(times)/max(1,len(times)):.2f}")
    print("   ", g[:300])
