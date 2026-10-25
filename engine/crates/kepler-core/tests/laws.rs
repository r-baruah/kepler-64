//! Invariants required by `docs/v1/02_LAWS_SPEC.md` §11, plus L6 transitions.

use kepler_core::cozy_chess::{Board, Color, Move, Piece, Square};
use kepler_core::laws::field::cone;
use kepler_core::{Dual, Params, Position, Scalar, evaluate};

fn mv(b: &Board, uci: &str) -> Move {
    // cozy-chess castling is king-takes-rook; the tests only use normal moves
    // except where they build the rook square explicitly.
    let m: Move = uci.parse().expect("uci");
    assert!(b.is_legal(m), "illegal {uci} in {b}");
    m
}

fn play_san_line(p: &Params, line: &[&str]) -> Vec<Position> {
    let mut pos = Position::startpos();
    let mut out = vec![pos.clone()];
    for &s in line {
        let m = san(&pos.board, s);
        pos = pos.play(m, p);
        out.push(pos.clone());
    }
    out
}

/// Minimal SAN resolver for test lines: match by generating moves and
/// comparing their SAN-relevant parts (piece, destination, promotion, file hint).
fn san(b: &Board, s: &str) -> Move {
    let s = s.trim_end_matches(['+', '#']);
    let mut found = Vec::new();
    b.generate_moves(|ms| {
        for m in ms {
            let piece = b.piece_on(m.from).unwrap();
            let letter = match piece {
                Piece::Pawn => "",
                Piece::Knight => "N",
                Piece::Bishop => "B",
                Piece::Rook => "R",
                Piece::Queen => "Q",
                Piece::King => "K",
            };
            let dest = m.to.to_string();
            let body = s.replace('x', "");
            let ok = if letter.is_empty() {
                // pawn: "e4", "dxc4" -> "dc4"
                (body == dest && m.from.file() == m.to.file())
                    || (body.len() == 3
                        && body[1..] == dest
                        && body.as_bytes()[0] as char == m.from.file().into())
            } else {
                body.starts_with(letter) && body.ends_with(&dest) && {
                    let hint = &body[1..body.len() - 2];
                    hint.is_empty() || m.from.to_string().contains(hint)
                }
            };
            if ok && m.promotion.is_none() {
                found.push(m);
            }
        }
        false
    });
    assert_eq!(found.len(), 1, "SAN {s} matched {found:?} in {b}");
    found[0]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Positions from seeded random play, with accretion carried.
fn random_positions(n: usize, p: &Params) -> Vec<Position> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::new();
    while out.len() < n {
        let mut pos = Position::startpos();
        let plies = 6 + (rng.next() % 50) as usize;
        for _ in 0..plies {
            let mut moves = Vec::new();
            pos.board.generate_moves(|ms| {
                moves.extend(ms);
                false
            });
            if moves.is_empty() {
                break;
            }
            let m = moves[(rng.next() % moves.len() as u64) as usize];
            pos = pos.play(m, p);
        }
        if pos.board.status() == kepler_core::cozy_chess::GameStatus::Ongoing {
            out.push(pos);
        }
    }
    out
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn colour_symmetry() {
    for lambda_tide in [0.0, 0.5] {
        let p = Params {
            lambda_tide,
            ..Params::default()
        };
        for pos in random_positions(200, &p) {
            let s = evaluate::<f32>(&pos, &p).score_white;
            let m = evaluate::<f32>(&pos.mirrored(), &p).score_white;
            assert!(close(s, -m, 1e-4), "{s} vs mirrored {m} for {}", pos.board);
        }
    }
}

#[test]
fn file_symmetry() {
    let p = Params::default();
    for pos in random_positions(100, &p) {
        let fen = format!("{}", pos.board);
        let mut parts: Vec<String> = fen.split(' ').map(String::from).collect();
        parts[0] = parts[0]
            .split('/')
            .map(|row| {
                let mut cells = Vec::new();
                for ch in row.chars() {
                    if let Some(d) = ch.to_digit(10) {
                        cells.extend(std::iter::repeat_n('1', d as usize));
                    } else {
                        cells.push(ch);
                    }
                }
                cells.reverse();
                cells.into_iter().collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("/");
        parts[2] = "-".into();
        parts[3] = "-".into();
        let Ok(mut flipped) = Position::from_fen(&parts.join(" ")) else {
            continue;
        };
        for i in 0..64 {
            let sq = Square::index(i);
            let f = Square::new(sq.file().flip(), sq.rank());
            flipped.acc[f as usize] = pos.acc[i];
        }
        let s = evaluate::<f32>(&pos, &p).score_white;
        let t = evaluate::<f32>(&flipped, &p).score_white;
        assert!(close(s, t, 1e-4), "{s} vs file-flipped {t}");
    }
}

#[test]
fn material_is_not_laundered_by_accretion() {
    // v0 audit §3: Qd3 c4 Qxc4 dxc4 Bxc4 gives a queen for two pawns.
    let pre = [
        "e4", "e6", "d4", "d5", "e5", "c5", "c3", "Nc6", "Nf3", "Qb6", "Qd3", "c4",
    ];
    let tail = ["Qxc4", "dxc4", "Bxc4"];
    for f_acc in [0.0, 0.5, 0.8, 1.0] {
        let p = Params {
            f_acc,
            ..Params::default()
        };
        let line: Vec<&str> = pre.iter().chain(tail.iter()).copied().collect();
        let ps = play_san_line(&p, &line);
        let before = evaluate::<f32>(&ps[pre.len()], &p).material;
        let after = evaluate::<f32>(ps.last().unwrap(), &p).material;
        let expected = -p.m_q + 2.0;
        assert!(
            (after - before - expected).abs() < 1e-5,
            "f_acc {f_acc}: {}",
            after - before
        );
        assert!(after - before < 0.0);
    }
}

#[test]
fn accretion_transitions() {
    let p = Params::default();
    // Capture: the captor absorbs f_acc of the victim's carried mass.
    let pos = Position::from_fen("4k3/8/8/3p4/4N3/8/8/4K3 w - - 0 1").unwrap();
    let after = pos.play(mv(&pos.board, "e4d6"), &p); // knight to d6 is not a capture
    assert_eq!(after.acc.iter().sum::<f32>(), 0.0);
    let pos = Position::from_fen("4k3/8/8/3p4/4N3/8/8/4K3 w - - 0 1").unwrap();
    let cap = pos.play(mv(&pos.board, "e4c5"), &p);
    assert_eq!(cap.acc.iter().sum::<f32>(), 0.0, "c5 is empty");
    let pos = Position::from_fen("4k3/8/2p5/8/3N4/8/8/4K3 w - - 0 1").unwrap();
    let cap = pos.play(mv(&pos.board, "d4c6"), &p);
    assert!((cap.acc[Square::C6 as usize] - 0.8).abs() < 1e-6);
    // A recapture takes the carried mass along: victim M = 3 + 0.8.
    let pos2 = Position {
        board: Board::from_fen("4k3/1p6/2N5/8/8/8/8/4K3 b - - 0 1", false).unwrap(),
        acc: cap.acc,
    };
    let re = pos2.play(mv(&pos2.board, "b7c6"), &p);
    assert!((re.acc[Square::C6 as usize] - 0.8 * 3.8).abs() < 1e-5);
    // En passant: the victim beside the captor is removed and absorbed.
    let mut ep = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    ep.acc[Square::D5 as usize] = 2.0;
    let e = ep.play(mv(&ep.board, "e5d6"), &p);
    assert!((e.acc[Square::D6 as usize] - 0.8 * 3.0).abs() < 1e-6);
    assert_eq!(e.acc[Square::D5 as usize], 0.0);
    // Castling: king and rook keep their own mass (cozy encodes e1h1).
    let mut c = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
    c.acc[Square::H1 as usize] = 1.5;
    c.acc[Square::E1 as usize] = 0.25;
    let c2 = c.play(mv(&c.board, "e1h1"), &p);
    assert_eq!(c2.board.piece_on(Square::F1), Some(Piece::Rook));
    assert_eq!(c2.acc[Square::F1 as usize], 1.5);
    assert_eq!(c2.acc[Square::G1 as usize], 0.25);
    assert_eq!(
        c2.acc[Square::H1 as usize] + c2.acc[Square::E1 as usize],
        0.0
    );
}

#[test]
fn no_xray_and_blocking() {
    let b = Board::from_fen("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1", false).unwrap();
    let r = cone(Piece::Rook, Color::White, Square::A1, b.occupied(), 1);
    assert!(
        r.hops[0].has(Square::A2),
        "the blocker itself feels the pull"
    );
    assert!(!r.hops[0].has(Square::A3), "no x-ray through the own pawn");
    let open = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", false).unwrap();
    let r_open = cone(Piece::Rook, Color::White, Square::A1, open.occupied(), 1);
    assert!(r.hops[0].len() < r_open.hops[0].len());
}

#[test]
fn locality_light_speed() {
    let b = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", false).unwrap();
    let c1 = cone(Piece::Rook, Color::White, Square::A1, b.occupied(), 1);
    assert_eq!(c1.hops[1].len() + c1.hops[2].len(), 0);
    let c2 = cone(Piece::Knight, Color::White, Square::A1, b.occupied(), 2);
    assert!((c2.hops[0] & c2.hops[1]).is_empty());
    assert!(!c2.hops[1].has(Square::A1), "no pull on its own square");
    assert!(!c2.hops[1].is_empty());
    // A pawn's field is its capture squares only; pushes emit nothing.
    let pb = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", false).unwrap();
    let pc = cone(Piece::Pawn, Color::White, Square::E2, pb.occupied(), 1);
    assert!(pc.hops[0].has(Square::D3) && pc.hops[0].has(Square::F3));
    assert!(!pc.hops[0].has(Square::E3));
}

#[test]
fn breakdown_sums_exactly() {
    let p = Params::default();
    for pos in random_positions(50, &p) {
        let b = evaluate::<f32>(&pos, &p);
        assert_eq!(b.material + b.bind + b.space + b.roche, b.score_white);
        assert_eq!(b.tide, 0.0);
    }
    let p = Params {
        lambda_tide: 0.34,
        ..Params::default()
    };
    let mut nonzero = 0;
    for pos in random_positions(50, &p) {
        let b = evaluate::<f32>(&pos, &p);
        assert_eq!(
            b.material + b.bind + b.space + b.roche + b.tide,
            b.score_white
        );
        nonzero += (b.tide != 0.0) as usize;
    }
    assert!(nonzero > 40);
}

#[test]
fn tide_uses_the_l5_stress() {
    // U_K equals the reported L5 stress tau_K; T_tide = lambda_tide (U_KB - U_KW).
    let p = Params {
        lambda_tide: 0.34,
        ..Params::default()
    };
    for pos in random_positions(100, &p) {
        let b = evaluate::<f32>(&pos, &p);
        let r = kepler_core::eval::report(&pos, &p, None);
        assert_eq!(r.kings[0].stress, b.kings[0].stress);
        assert_eq!(
            b.tide,
            p.lambda_tide * (r.kings[1].stress - r.kings[0].stress)
        );
        assert_eq!(r.terms.tide, b.tide);
        // Enemy pull summed over the body, recomputed from the field.
        for (c, color) in [Color::White, Color::Black].into_iter().enumerate() {
            let sum: f32 = b.kings[c]
                .body
                .into_iter()
                .map(|s| b.field[1 - c][s as usize])
                .sum();
            assert!(close(sum, b.kings[c].stress, 1e-4), "{color:?}");
        }
        // The lean path agrees with the generic one.
        assert_eq!(
            kepler_core::laws::fast::score_white_f32(&pos, &p).to_bits(),
            b.score_white.to_bits()
        );
    }
}

#[test]
fn tide_zero_is_spec_1() {
    let p0 = Params::default();
    let pt = Params {
        lambda_tide: 1e-9,
        ..p0
    };
    for pos in random_positions(50, &p0) {
        let a = evaluate::<f32>(&pos, &p0);
        assert_eq!(a.tide, 0.0);
        assert_eq!(a.score_white, a.material + a.bind + a.space + a.roche);
        // Pull on the king exists, so a tiny weight must differ in the term only.
        let b = evaluate::<f32>(&pos, &pt);
        assert!((b.score_white - a.score_white).abs() < 1e-3);
    }
}

#[test]
fn gradient_matches_finite_difference() {
    // lambda_tide is nonzero so its central difference stays inside [0, 2].
    let base = Params {
        lambda_tide: 0.3,
        ..Params::default()
    };
    let positions = random_positions(200, &base);
    let v0 = base.grad_values();
    for (i, name) in Params::GRAD_NAMES.iter().enumerate() {
        let h = 1e-2f32 * v0[i].abs().max(0.1);
        let (mut up, mut dn) = (base, base);
        let mut vu = v0;
        let mut vd = v0;
        vu[i] += h;
        vd[i] -= h;
        up.set_grad_values(vu);
        dn.set_grad_values(vd);
        let (mut num, mut ana) = (0.0f64, 0.0f64);
        for pos in &positions {
            let d = evaluate::<Dual>(pos, &base).score_white;
            let fu = evaluate::<f32>(pos, &up).score_white as f64;
            let fd = evaluate::<f32>(pos, &dn).score_white as f64;
            num += (fu - fd) / (2.0 * h as f64);
            ana += d.d[i] as f64;
            assert!((d.val() - evaluate::<f32>(pos, &base).score_white).abs() < 1e-3);
        }
        let rel = (num - ana).abs() / (1.0 + num.abs().max(ana.abs()));
        assert!(rel < 1e-3, "{name}: analytic {ana} vs numeric {num}");
    }
}
