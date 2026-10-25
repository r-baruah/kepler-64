use cozy_chess::Board;

fn perft(b: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut n = 0;
    if depth == 1 {
        b.generate_moves(|pm| {
            n += pm.len() as u64;
            false
        });
        return n;
    }
    b.generate_moves(|pm| {
        for mv in pm {
            let mut c = b.clone();
            c.play_unchecked(mv);
            n += perft(&c, depth - 1);
        }
        false
    });
    n
}

fn check(fen: &str, depth: u32, expected: u64) {
    let b = Board::from_fen(fen, false).unwrap();
    assert_eq!(perft(&b, depth), expected, "{fen} depth {depth}");
}

#[test]
fn perft_start() {
    assert_eq!(perft(&Board::default(), 5), 4_865_609);
}

#[test]
fn perft_kiwipete() {
    check(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        4,
        4_085_603,
    );
}

#[test]
fn perft_pos3() {
    check("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674_624);
}

#[test]
fn perft_pos4() {
    check(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        4,
        422_333,
    );
}

#[test]
fn perft_pos5() {
    check(
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        4,
        2_103_487,
    );
}

#[test]
fn perft_pos6() {
    check(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        4,
        3_894_594,
    );
}

#[test]
#[ignore]
fn perft_start_timing() {
    let t = std::time::Instant::now();
    let n = perft(&Board::default(), 5);
    eprintln!("perft(5) startpos = {n} in {:?}", t.elapsed());
}
