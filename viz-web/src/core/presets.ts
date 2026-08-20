/**
 * Curated Preset Games for Kepler-64 Observatory Replay
 */

export interface PresetPly {
  ply: number;
  moveSan: string;
  fen: string;
  comment?: string;
}

export interface PresetGame {
  id: string;
  title: string;
  subtitle: string;
  white: string;
  black: string;
  date: string;
  event: string;
  initialFen: string;
  pgn: string;
  highlightPly: number;
}

export const PRESET_GAMES: PresetGame[] = [
  {
    id: 'kepler64-selfplay-v2',
    title: 'Kepler-64 Autonomous Clash (V2)',
    subtitle: 'Post-fix JAX gravitational search, mass accretion & tidal rook checkmate',
    white: 'Kepler-64 (Gravitational Vanguard)',
    black: 'Kepler-64 (Roche Defender)',
    date: '2026.08.20',
    event: 'Kepler-64 Observatory Harness V2',
    initialFen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    highlightPly: 44,
    pgn: `[Event "Kepler-64 Observatory Harness V2"]
[Site "Kepler-64 Observatory"]
[Date "2026.08.20"]
[Round "2"]
[White "Kepler-64 (Gravitational Vanguard)"]
[Black "Kepler-64 (Roche Defender)"]
[Result "0-1"]

1. Nf3 Nf6 2. Ne5 Ne4 3. Nxd7 Nxd2 4. Qxd2 Qxd7 5. Qxd7+ Nxd7 6. Bh6 gxh6 7. Nc3 Ne5 8. Nd5 Nf3+ 9. gxf3 Bh3 10. Nxc7+ Kd8 11. Ne8 Bxf1 12. O-O-O+ Kxe8 13. Rhxf1 b5 14. Rd5 b4 15. Re5 a5 16. Rxe7+ Bxe7 17. e4 Bc5 18. Kd1 Bd4 19. Kc1 Bc3 20. Kb1 Ra6 21. bxc3 bxc3 22. Ka1 a4 23. e5 a3 24. f4 Ra4 25. f5 Kd8 26. e6 fxe6 27. fxe6 Kc8 28. e7 Kb8 29. f4 Ka8 30. Rf3 Rc8 31. Rd3 Rcc4 32. e8=Q+ Ka7 33. Qa8+ Kxa8 34. Rd8+ Ka7 35. Rc8 Rxc8 36. h4 Rxf4 37. h5 Ra4 38. Kb1 Rh4 39. Ka1 Rh1# 0-1`,
  },
  {
    id: 'kasparov-topalov-1999',
    title: "Kasparov's Immortal",
    subtitle: "Rook sacrifice creates runaway tidal strain on Topalov's King",
    white: 'Garry Kasparov',
    black: 'Veselin Topalov',
    date: '1999.01.20',
    event: 'Wijk aan Zee / Hoogovens',
    initialFen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    highlightPly: 48,
    pgn: `[Event "Hoogovens A"]
[Site "Wijk aan Zee NED"]
[Date "1999.01.20"]
[Round "4"]
[White "Kasparov, Garry"]
[Black "Topalov, Veselin"]
[Result "1-0"]

1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. Be3 Bg7 5. Qd2 c6 6. f3 b5 7. Nge2 Nbd7 8. Bh6 Bxh6 9. Qxh6 Bb7 10. a3 e5 11. O-O-O Qe7 12. Kb1 a6 13. Nc1 O-O-O 14. Nb3 exd4 15. Rxd4 c5 16. Rd1 Nb6 17. g3 Kb8 18. Na5 Ba8 19. Bh3 d5 20. Qf4+ Ka7 21. Rhe1 d4 22. Nd5 Nbxd5 23. exd5 Qd6 24. Rxd4 cxd4 25. Re7+ Kb6 26. Qxd4+ Kxa5 27. b4+ Ka4 28. Qc3 Qxd5 29. Ra7 Bb7 30. Rxb7 Qc4 31. Qxf6 Kxa3 32. Qxa6+ Kxb4 33. c3+ Kxc3 34. Qa1+ Kd2 35. Qb2+ Kd1 36. Bf1 Rd2 37. Rd7 Rxd7 38. Bxc4 bxc4 39. Qxh8 Rd3 40. Qa8 c3 41. Qa4+ Ke1 42. f4 f5 43. Kc1 Rd2 44. Qa7 1-0`,
  },
  {
    id: 'alphazero-stockfish-2017',
    title: 'AlphaZero Queen Infiltration',
    subtitle: 'Gravitational mass concentration crushes defensive perimeter',
    white: 'AlphaZero',
    black: 'Stockfish 8',
    date: '2017.12.04',
    event: 'DeepMind AlphaZero Match',
    initialFen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    highlightPly: 36,
    pgn: `[Event "AlphaZero vs Stockfish Match"]
[Site "London ENG"]
[Date "2017.12.04"]
[Round "9"]
[White "AlphaZero"]
[Black "Stockfish 8"]
[Result "1-0"]

1. d4 Nf6 2. c4 e6 3. Nf3 b6 4. g3 Ba6 5. Qc2 Bb7 6. Bg2 c5 7. d5 exd5 8. cxd5 Bxd5 9. Nc3 Bc6 10. e4 Be7 11. Bf4 O-O 12. O-O-O Qc8 13. h4 Re8 14. e5 Nh5 15. Ng5 g6 16. Bd5 Rf8 17. Nce4 Na6 18. Bc4 Nb4 19. Qe2 b5 20. Bxf7+ Rxf7 21. Nxf7 Kxf7 22. a3 Na6 23. Rhe1 Nc7 24. Nd6+ Bxd6 25. exd6 Ne6 26. Bh6 1-0`,
  },
  {
    id: 'kepler64-legacy-match',
    title: 'Kepler-64 Legacy (Pre-Fix Artifact)',
    subtitle: 'Early broken physics where White & Black evaluation bled into each other',
    white: 'Kepler-64 (Broken Prototype)',
    black: 'Stockfish-1300',
    date: '2026.08.05',
    event: 'Kepler-64 Early Prototype',
    initialFen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    highlightPly: 22,
    pgn: `[Event "Kepler-64 Early Prototype"]
[Site "Kepler-64 Observatory"]
[Date "2026.08.05"]
[Round "1"]
[White "Kepler-64"]
[Black "Stockfish-1300"]
[Result "0-1"]

1. Nc3 d5 2. Rb1 d4 3. a3 dxc3 4. bxc3 Nf6 5. a4 e5 6. a5 e4 7. a6 Nxa6 8. h3 e3 9. fxe3 Ne4 10. h4 Bd6 11. h5 Bg3# 0-1`,
  },
];
