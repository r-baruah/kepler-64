use kepler_core::cozy_chess::GameStatus;
use kepler_core::{Params, Position, evaluate};
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
fn main() {
    let a: Vec<f32> = std::env::args()
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect();
    let mut p = Params::default();
    if a.len() == 3 {
        p.lambda_bind = a[0];
        p.lambda_space = a[1];
        p.beta = a[2];
    }
    let mut rng = Rng(12345);
    let (mut n, mut sstd, mut smax, mut mx) = (0usize, 0f64, 0f64, 0f32);
    let (mut tb, mut ts, mut tr) = (0f64, 0f64, 0f64);
    let mut etas = Vec::new();
    let t0 = std::time::Instant::now();
    let mut evals = 0u64;
    while n < 300 {
        let mut pos = Position::startpos();
        let plies = 6 + (rng.next() % 60) as usize;
        for _ in 0..plies {
            let mut ms = Vec::new();
            pos.board.generate_moves(|m| {
                ms.extend(m);
                false
            });
            if ms.is_empty() {
                break;
            }
            let m = ms[(rng.next() % ms.len() as u64) as usize];
            pos = pos.play(m, &p);
        }
        if pos.board.status() != GameStatus::Ongoing {
            continue;
        }
        let mut ms = Vec::new();
        pos.board.generate_moves(|m| {
            ms.extend(m);
            false
        });
        let mut vals = Vec::new();
        let mut broke = false;
        for m in ms {
            let c = pos.play(m, &p);
            let b = evaluate::<f32>(&c, &p);
            evals += 1;
            tb += b.bind.abs() as f64;
            ts += b.space.abs() as f64;
            tr += b.roche.abs() as f64;
            for k in &b.kings {
                etas.push(k.eta);
                if k.eta > p.rho {
                    broke = true;
                }
            }
            vals.push(b.bind + b.space + b.roche);
        }
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let sd = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
        let m = vals.iter().map(|v| (v - mean).abs()).fold(0f32, f32::max);
        sstd += sd as f64;
        if !broke {
            smax += m as f64;
            mx = mx.max(m);
        }
        n += 1;
    }
    etas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| etas[((etas.len() - 1) as f64 * f) as usize];
    println!(
        "params bind={} space={} beta={}",
        p.lambda_bind, p.lambda_space, p.beta
    );
    println!(
        "sibling std mean={:.3}  max-dev(no-break) mean={:.3} worst={:.3}",
        sstd / n as f64,
        smax / n as f64,
        mx
    );
    println!(
        "mean |bind|={:.3} |space|={:.3} |roche|={:.3}",
        tb / evals as f64,
        ts / evals as f64,
        tr / evals as f64
    );
    println!(
        "eta quantiles p10={:.3} p50={:.3} p90={:.3} p99={:.3} max={:.3}",
        q(0.1),
        q(0.5),
        q(0.9),
        q(0.99),
        q(1.0)
    );
    println!(
        "eval speed: {:.0} evals/s",
        evals as f64 / t0.elapsed().as_secs_f64()
    );
}
