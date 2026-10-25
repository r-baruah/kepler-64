fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    kepler_uci::run(stdin.lock(), &mut out);
}
