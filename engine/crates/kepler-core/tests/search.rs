use cozy_chess::{Board, GameStatus, Move};
use kepler_core::search::{Limits, Searcher, move_from_uci, move_to_uci};
use kepler_core::{Params, Position};

fn run(fen: &str, history: &[u64], depth: u32) -> kepler_core::search::SearchResult {
    let pos = Position::from_fen(fen).unwrap();
    let mut s = Searcher::new(16);
    let lim = Limits {
        depth: Some(depth),
        ..Default::default()
    };
    s.search(&pos, history, &Params::default(), &lim, &mut |_| {})
}

fn legal(b: &Board) -> Vec<Move> {
    let mut v = vec![];
    b.generate_moves(|pm| {
        v.extend(pm);
        false
    });
    v
}

/// Can the side to move force mate within `n` of its own moves? (brute force)
fn forced_mate(b: &Board, n: u32) -> bool {
    if n == 0 {
        return false;
    }
    legal(b).into_iter().any(|m| {
        let mut c = b.clone();
        c.play_unchecked(m);
        if c.status() == GameStatus::Won {
            return true;
        }
        if c.status() != GameStatus::Ongoing {
            return false;
        }
        legal(&c).into_iter().all(|r| {
            let mut d = c.clone();
            d.play_unchecked(r);
            forced_mate(&d, n - 1)
        })
    })
}

// (fen, mate in N)
const MATES: &[(&str, u32)] = &[
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 1),
    ("k7/8/1K6/8/8/8/8/7R w - - 0 1", 1),
    (
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        1,
    ),
    ("6k1/8/6K1/8/8/8/8/7Q w - - 0 1", 1),
    ("kbK5/pp6/1P6/8/8/8/8/R7 w - - 0 1", 2),
    (
        "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1",
        2,
    ),
    (
        "2bqkbn1/2pppp2/np2N3/r3P1p1/p2N2B1/5Q2/PPPPKPP1/RNB2r2 w - - 0 1",
        2,
    ),
];

/// After `c` (opponent to move), is every reply answered by a mate within `n`?
fn all_replies_mated(c: &Board, n: u32) -> bool {
    c.status() == GameStatus::Ongoing
        && legal(c).into_iter().all(|r| {
            let mut d = c.clone();
            d.play_unchecked(r);
            forced_mate(&d, n)
        })
}

#[test]
fn solves_mates() {
    for &(fen, n) in MATES {
        let b = Board::from_fen(fen, false).unwrap();
        assert!(
            forced_mate(&b, n),
            "test position is not mate in {n}: {fen}"
        );
        let r = run(fen, &[], 4);
        let m = r
            .mate_in
            .unwrap_or_else(|| panic!("no mate score for {fen}: {}", r.score_ku));
        assert!(
            m > 0 && m as u32 <= n,
            "{fen}: mate_in {m}, expected <= {n}"
        );
        let best = r.best.unwrap();
        let mut c = b.clone();
        c.play_unchecked(best);
        let ok = c.status() == GameStatus::Won || all_replies_mated(&c, n - 1);
        assert!(ok, "{fen}: {} does not force mate", move_to_uci(&b, best));
    }
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

#[test]
fn random_selfplay_legal() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let params = Params::default();
    let mut searcher = Searcher::new(8);
    for _game in 0..50 {
        searcher.clear();
        let mut pos = Position::startpos();
        let mut hist: Vec<u64> = vec![];
        for ply in 0..200 {
            let moves = legal(&pos.board);
            if moves.is_empty() || pos.board.status() != GameStatus::Ongoing {
                break;
            }
            // A few random opening plies for variety, then search at depth 2.
            let mv = if ply < 4 {
                moves[(rng.next() % moves.len() as u64) as usize]
            } else {
                let lim = Limits {
                    depth: Some(2),
                    ..Default::default()
                };
                let r = searcher.search(&pos, &hist, &params, &lim, &mut |_| {});
                r.best.expect("a move exists")
            };
            assert!(moves.contains(&mv), "illegal move {mv}");
            assert_eq!(
                move_from_uci(&pos.board, &move_to_uci(&pos.board, mv)),
                Some(mv)
            );
            hist.push(pos.board.hash());
            pos = pos.play(mv, &params);
        }
    }
}

#[test]
fn repetition_scores_zero() {
    // White's only legal move (Kg1) is hopeless (Black mates next), but if the
    // resulting position already occurred in the game, the search scores it 0.
    let fen = "k7/8/8/8/8/5q2/r7/7K w - - 10 40";
    let params = Params::default();
    let root = Position::from_fen(fen).unwrap();
    assert_eq!(legal(&root.board).len(), 1);
    let child = root.play(legal(&root.board)[0], &params);

    let free = run(fen, &[], 3);
    assert!(free.score_ku < -100.0 || free.mate_in.is_some_and(|m| m < 0));

    let hist = [child.board.hash(), 1, 2];
    let mut s = Searcher::new(8);
    let lim = Limits {
        depth: Some(3),
        ..Default::default()
    };
    let r = s.search(&root, &hist, &params, &lim, &mut |_| {});
    assert_eq!(r.score_ku, 0.0);
    assert_eq!(r.mate_in, None);
    assert!(r.best.is_some());
}
