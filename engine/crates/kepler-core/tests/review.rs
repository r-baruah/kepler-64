use kepler_core::cozy_chess::{Board, Color, Move, Piece, Square};
use kepler_core::laws::field::cone;
use kepler_core::{Dual, Params, Position, evaluate};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
fn random_positions(n: usize, p: &Params, seed: u64, maxp: u64) -> Vec<Position> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let mut pos = Position::startpos();
        let plies = 6 + (rng.next() % maxp) as usize;
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
fn mv(b: &Board, uci: &str) -> Move {
    let m: Move = uci.parse().unwrap();
    assert!(b.is_legal(m), "illegal {uci}");
    m
}

#[test]
fn stats_random_positions() {
    let p = Params::default();
    let ps = random_positions(200, &p, 0x9E37_79B9_7F4A_7C15, 50);
    let mut nacc = 0;
    let mut promo = 0;
    let mut maxs: f32 = 0.0;
    for pos in &ps {
        if pos.acc.iter().any(|&a| a > 0.0) {
            nacc += 1;
        }
        let pr =
            pos.board.pieces(Piece::Queen).len() > 2 || pos.board.pieces(Piece::Knight).len() > 4;
        if pr {
            promo += 1;
        }
        maxs = maxs.max(evaluate::<f32>(pos, &p).score_white.abs());
    }
    eprintln!("with acc {nacc}/200, promo-ish {promo}, max |score| {maxs}");
}

#[test]
fn colour_symmetry_strict() {
    for c in [1u8, 2, 3] {
        let p = Params {
            c,
            ..Params::default()
        };
        let mut worst: f32 = 0.0;
        for pos in random_positions(200, &p, 0x1234_5678_9ABC_DEF1, 120) {
            let s = evaluate::<f32>(&pos, &p).score_white;
            let m = evaluate::<f32>(&pos.mirrored(), &p).score_white;
            worst = worst.max((s + m).abs());
        }
        eprintln!("c={c} worst |s+m| = {worst:e}");
        assert!(worst <= 1e-5, "c={c} worst {worst}");
    }
}

#[test]
fn per_position_gradient() {
    for c in [1u8, 2, 3] {
        let base = Params {
            c,
            ..Params::default()
        };
        let positions = random_positions(200, &base, 0xABCDEF, 80);
        let v0 = base.grad_values();
        for (i, name) in Params::GRAD_NAMES.iter().enumerate() {
            let h = 1e-2f32 * v0[i].abs().max(0.1);
            let mut vu = v0;
            let mut vd = v0;
            vu[i] += h;
            vd[i] -= h;
            let (mut up, mut dn) = (base, base);
            up.set_grad_values(vu);
            dn.set_grad_values(vd);
            let mut worst = 0.0f64;
            for pos in &positions {
                let d = evaluate::<Dual>(pos, &base).score_white.d[i] as f64;
                let fu = evaluate::<f32>(pos, &up).score_white as f64;
                let fd = evaluate::<f32>(pos, &dn).score_white as f64;
                let num = (fu - fd) / (2.0 * h as f64);
                let rel = (num - d).abs() / (num.abs().max(d.abs()).max(1e-3));
                if rel > worst {
                    worst = rel;
                }
            }
            eprintln!("c={c} {name}: worst per-position rel {worst:e}");
        }
    }
}

#[test]
fn pawn_cone() {
    let b = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", false).unwrap();
    let c = cone(Piece::Pawn, Color::White, Square::E2, b.occupied(), 3);
    eprintln!(
        "h1 {:?}\nh2 {:?}\nh3 {:?}",
        c.hops[0].into_iter().collect::<Vec<_>>(),
        c.hops[1].into_iter().collect::<Vec<_>>(),
        c.hops[2].into_iter().collect::<Vec<_>>()
    );
    // promotion
    let b = Board::from_fen("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", false).unwrap();
    let c = cone(Piece::Pawn, Color::White, Square::B7, b.occupied(), 3);
    eprintln!(
        "b7 h1 {:?} h2 {:?} h3 {:?}",
        c.hops[0].into_iter().collect::<Vec<_>>(),
        c.hops[1].into_iter().collect::<Vec<_>>(),
        c.hops[2].into_iter().collect::<Vec<_>>()
    );
    // blocked double push
    let b = Board::from_fen("4k3/8/8/8/4p3/8/4P3/4K3 w - - 0 1", false).unwrap();
    let c = cone(Piece::Pawn, Color::White, Square::E2, b.occupied(), 3);
    eprintln!(
        "e2 blocked e4: h2 {:?} h3 {:?}",
        c.hops[1].into_iter().collect::<Vec<_>>(),
        c.hops[2].into_iter().collect::<Vec<_>>()
    );
    // rook c3 through own square
    let b = Board::from_fen("4k3/8/8/8/3R4/8/8/4K3 w - - 0 1", false).unwrap();
    let c = cone(Piece::Rook, Color::White, Square::D4, b.occupied(), 2);
    assert!(!c.hops[0].has(Square::D4) && !c.hops[1].has(Square::D4));
}

#[test]
fn accretion_more() {
    let p = Params::default();
    // white queenside castle
    let mut c = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    c.acc[Square::A1 as usize] = 1.5;
    c.acc[Square::E1 as usize] = 0.25;
    let c2 = c.play(mv(&c.board, "e1a1"), &p);
    assert_eq!(c2.board.piece_on(Square::D1), Some(Piece::Rook));
    assert_eq!(c2.board.piece_on(Square::C1), Some(Piece::King));
    assert_eq!(c2.acc[Square::D1 as usize], 1.5);
    assert_eq!(c2.acc[Square::C1 as usize], 0.25);
    // black both sides
    let mut c = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1").unwrap();
    c.acc[Square::A8 as usize] = 1.0;
    c.acc[Square::H8 as usize] = 2.0;
    c.acc[Square::E8 as usize] = 3.0;
    let k = c.play(mv(&c.board, "e8h8"), &p);
    assert_eq!(k.acc[Square::F8 as usize], 2.0);
    assert_eq!(k.acc[Square::G8 as usize], 3.0);
    assert_eq!(k.acc[Square::A8 as usize], 1.0);
    assert_eq!(k.acc[Square::E8 as usize] + k.acc[Square::H8 as usize], 0.0);
    let q = c.play(mv(&c.board, "e8a8"), &p);
    assert_eq!(q.acc[Square::D8 as usize], 1.0);
    assert_eq!(q.acc[Square::C8 as usize], 3.0);
    // promotion quiet and capture
    let mut pr = Position::from_fen("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    pr.acc[Square::A7 as usize] = 0.5;
    pr.acc[Square::B8 as usize] = 1.0;
    let a = pr.play(mv(&pr.board, "a7a8q"), &p);
    assert_eq!(a.acc[Square::A8 as usize], 0.5);
    let b = pr.play(mv(&pr.board, "a7b8n"), &p);
    assert!((b.acc[Square::B8 as usize] - (0.5 + 0.8 * 6.0)).abs() < 1e-6);
    assert_eq!(b.acc[Square::A7 as usize], 0.0);
    // king capture
    let mut kc = Position::from_fen("4k3/8/8/8/8/8/3r4/4K3 w - - 0 1").unwrap();
    kc.acc[Square::D2 as usize] = 1.0;
    kc.acc[Square::E1 as usize] = 0.5;
    let r = kc.play(mv(&kc.board, "e1d2"), &p);
    assert!((r.acc[Square::D2 as usize] - (0.5 + 0.8 * 6.0)).abs() < 1e-6);
    // emission of king grows, material unchanged by A
    let e0 = evaluate::<f32>(&r, &p);
    let mut r0 = r.clone();
    r0.acc = [0.0; 64];
    let e1 = evaluate::<f32>(&r0, &p);
    assert_eq!(e0.material, e1.material);
    // black en passant
    let mut ep = Position::from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1").unwrap();
    ep.acc[Square::E4 as usize] = 2.0;
    ep.acc[Square::D4 as usize] = 0.1;
    let e = ep.play(mv(&ep.board, "d4e3"), &p);
    assert!((e.acc[Square::E3 as usize] - (0.1 + 0.8 * 3.0)).abs() < 1e-6);
    assert_eq!(e.acc[Square::E4 as usize], 0.0);
}

#[test]
fn kings_only_and_kappa_zero() {
    for kappa in [0.0f32, 1.0] {
        let p = Params {
            kappa,
            ..Params::default()
        };
        for fen in [
            "8/8/8/3k4/8/3K4/8/8 w - - 0 1",
            "k7/8/8/8/8/8/8/7K w - - 0 1",
            "8/8/8/8/8/2k5/8/K7 w - - 0 1",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let e = evaluate::<Dual>(&pos, &p);
            let f = evaluate::<f32>(&pos, &p);
            eprintln!(
                "kappa {kappa} {fen}: score {} eta {:?} grads {:?}",
                f.score_white,
                [f.kings[0].eta, f.kings[1].eta],
                e.score_white.d
            );
            assert!(f.score_white.is_finite() && e.score_white.d.iter().all(|x| x.is_finite()));
        }
    }
}

/// Spec 7: A = f_acc * M_victim with M = mu + A, so A depends on m_n. The
/// Dual path treats A as a constant. Compare against replaying with m_n + h.
#[test]
fn accretion_gradient_path() {
    let base = Params::default();
    let fen = "4k3/8/2n5/8/3N4/8/8/4K3 w - - 0 1";
    let run = |p: &Params| {
        let pos = Position::from_fen(fen).unwrap();
        pos.play(mv(&pos.board, "d4c6"), p)
    };
    let h = 1e-2;
    let up = Params {
        m_n: base.m_n + h,
        ..base
    };
    let dn = Params {
        m_n: base.m_n - h,
        ..base
    };
    let num = (evaluate::<f32>(&run(&up), &up).score_white
        - evaluate::<f32>(&run(&dn), &dn).score_white)
        / (2.0 * h);
    let ana = evaluate::<Dual>(&run(&base), &base).score_white.d[0];
    eprintln!("d score / d m_n with replay: numeric {num} analytic {ana}");
}

#[test]
fn per_position_gradient_worst_detail() {
    let base = Params::default();
    let positions = random_positions(200, &base, 0xABCDEF, 80);
    let v0 = base.grad_values();
    for (i, name) in Params::GRAD_NAMES.iter().enumerate() {
        let mut worst = (0.0f64, 0.0, 0.0, 0.0);
        for pos in &positions {
            let d = evaluate::<Dual>(pos, &base).score_white.d[i] as f64;
            let fdh = |hh: f32| {
                let mut vu = v0;
                let mut vd = v0;
                vu[i] += hh;
                vd[i] -= hh;
                let (mut up, mut dn) = (base, base);
                up.set_grad_values(vu);
                dn.set_grad_values(vd);
                (evaluate::<f32>(pos, &up).score_white as f64
                    - evaluate::<f32>(pos, &dn).score_white as f64)
                    / (2.0 * hh as f64)
            };
            let h = 1e-2f32 * v0[i].abs().max(0.1);
            let n1 = fdh(h);
            let n2 = fdh(4.0 * h);
            let rel = (n1 - d).abs() / (1.0 + n1.abs().max(d.abs()));
            if rel > worst.0 {
                worst = (rel, d, n1, n2);
            }
        }
        eprintln!(
            "{name}: worst rel(1+) {:e} ana {} fd(h) {} fd(4h) {}",
            worst.0, worst.1, worst.2, worst.3
        );
    }
}
