<script lang="ts">
	import { onDestroy } from 'svelte';
	import Board from '$lib/components/Board.svelte';
	import { parsePlacement, sqIndex } from '$lib/chess';
	import { EngineClient } from '$lib/engine/client';
	import type { LandingFacts, LedgerCard, Masses } from '$lib/types';

	interface Props {
		landing: LandingFacts;
		/** True once the page's own engine has loaded; the demo board then asks the engine for the rook's squares. */
		ready: boolean;
		/** Generation file the demo engine loads (any generation works; the hop squares come from the rules). */
		generationUrl: string;
		onplayrejected: () => void;
	}
	let { landing, ready, generationUrl, onplayrejected }: Props = $props();

	// ---- 1. the one law --------------------------------------------------------------
	// The board shows only the rook and the pawn. The engine needs both kings to accept a position, so
	// they stand far from the rook's lines (a8, h8) and do not change its squares.
	const SHOWN_FEN = '8/8/8/8/3P4/8/3R4/8 w - - 0 1';
	const DEMO_FEN = 'K6k/8/8/8/3P4/8/3R4/8 w - - 0 1';
	const ROOK_SQ = sqIndex('d2');
	const demoPlacement = parsePlacement(SHOWN_FEN);
	let hop1 = $state.raw<number[] | null>(null);
	const cones = $derived(hop1 ? { origin: ROOK_SQ, hop1, hop2: [] } : null);
	let demoClient: EngineClient | null = null;
	let demoStarted = false;

	$effect(() => {
		if (!ready || demoStarted) return;
		demoStarted = true;
		void (async () => {
			try {
				const res = await fetch(generationUrl);
				if (!res.ok) return;
				const client = new EngineClient();
				demoClient = client;
				await client.start(await res.text());
				const snap = await client.reset(DEMO_FEN, []);
				const rook = snap.breakdown.pieces.find((p) => p.square === 'd2');
				if (rook) hop1 = rook.hops[0].map(sqIndex);
				client.stop();
			} catch (e) {
				console.warn('landing demo', e);
			}
		})();
	});
	onDestroy(() => demoClient?.stop());

	// ---- 2. the piece-weight chart ---------------------------------------------------
	const W = 520;
	const Y_TOP = 28;
	const Y_BOTTOM = 237;
	const chart = $derived(landing.chart);
	const all = $derived([...chart.runs.a, ...chart.runs.b].flat());
	const vmin = $derived(Math.min(...all));
	const vmax = $derived(Math.max(...all));
	const yOf = (v: number) => Y_BOTTOM - ((v - vmin) / (vmax - vmin)) * (Y_BOTTOM - Y_TOP);
	const rounds = $derived(chart.runs.a.length - 1);
	const points = (series: Masses[], i: number) => series.map((m, r) => `${((r / rounds) * W).toFixed(1)},${yOf(m[i]).toFixed(1)}`).join(' ');
	// Draw order: queen last-but-one so the accent line sits on top; knight at 55%.
	const LINES = [
		{ name: 'Queen', short: 'Q', i: 3, color: 'var(--accent)', opacity: 1 },
		{ name: 'Rook', short: 'R', i: 2, color: 'var(--ink)', opacity: 1 },
		{ name: 'Bishop', short: 'B', i: 1, color: 'var(--ink)', opacity: 1 },
		{ name: 'Knight', short: 'N', i: 0, color: 'var(--ink)', opacity: 0.55 }
	];
	const found = $derived(landing.found);
	const games = $derived(chart.games.toLocaleString('en-US'));
	const aria = $derived(
		`Piece weights over ${games} games of self-play, in two runs, from all equal to queen ${found.q.toFixed(1)}, rook ${found.r.toFixed(1)}, bishop ${found.b.toFixed(1)}, knight ${found.n.toFixed(1)}`
	);

	// ---- 3. the ledger ---------------------------------------------------------------
	const BADGE: Record<LedgerCard['verdict'], string> = { kept: 'Kept', back: 'Came back', rejected: 'Rejected', 'rejected-twice': 'Rejected twice' };
</script>

<div class="landing">
	<section class="sec split">
		<div class="copy">
			<div class="label">The one law</div>
			<h2>Pull follows the moves.</h2>
			<p>A piece pulls only where it could move, and stops at the first piece in its way. This rook stops at its own pawn.</p>
		</div>
		<div class="plate board-plate">
			<Board
				placement={demoPlacement}
				bottom="white"
				dots={[]}
				{cones}
				lastMove={null}
				selected={null}
				targets={new Map()}
				label="A white rook on d2 behind its own pawn on d4. The squares the rook pulls on are highlighted."
				onselectown={() => false}
				ontap={() => {}}
				onmove={() => {}}
			/>
		</div>
	</section>

	<section class="sec split">
		<div class="copy">
			<div class="label">Learned, not told</div>
			<h2>It found the piece values by itself.</h2>
			<p>Every piece started with the same weight. After {games} games against itself, it had ranked them. A second run found the same.</p>
			<p class="strong">
				Chess players say <b>3, 3, 5, 9</b>. This universe found <b>{found.n.toFixed(1)}, {found.b.toFixed(1)}, {found.r.toFixed(1)}, {found.q.toFixed(1)}</b>.
			</p>
		</div>
		<figure class="plate chart">
			<svg viewBox="-10 -10 640 300" role="img" aria-label={aria}>
				{#each LINES as l (l.name)}
					<polyline fill="none" stroke="#9aa1b0" stroke-width="1.2" stroke-dasharray="2 4" points={points(chart.runs.b, l.i)} />
				{/each}
				{#each LINES as l (l.name)}
					<polyline fill="none" stroke={l.color} stroke-opacity={l.opacity} stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points={points(chart.runs.a, l.i)} />
				{/each}
				<circle cx="0" cy={yOf(chart.start[0])} r="5" fill="var(--ink)" />
				<text class="axis" x="-6" y={yOf(chart.start[0]) - 22}>all equal</text>
				{#each LINES as l (l.name)}
					{@const y = yOf(chart.runs.a[rounds][l.i]) + 5}
					<text class="end full" x="534" {y} fill={l.name === 'Queen' ? 'var(--accent)' : l.name === 'Knight' ? 'var(--muted)' : 'var(--ink)'}>{l.name}</text>
					<text class="end abbr" x="534" y={y + 3} fill={l.name === 'Queen' ? 'var(--accent)' : l.name === 'Knight' ? 'var(--muted)' : 'var(--ink)'}>{l.short}</text>
				{/each}
			</svg>
			<figcaption>Piece weight over {games} games. The second run, dotted, lies almost exactly underneath.</figcaption>
		</figure>
	</section>

	<section class="sec stack">
		<div class="ledger-head">
			<div class="copy">
				<div class="label">The Law Ledger</div>
				<h2>Every law had to earn its place.</h2>
			</div>
			<p>We switched each one off and let it play itself. Laws that helped stayed. The rest you can still play with.</p>
		</div>
		<div class="cards">
			{#each landing.ledger as c (c.id)}
				<div class="lcard" class:cut={c.verdict === 'rejected' || c.verdict === 'rejected-twice'}>
					<span class="badge {c.verdict}">{BADGE[c.verdict]}</span>
					<div class="ct">{c.title}</div>
					<div class="cn">{c.note}</div>
				</div>
			{/each}
			<button class="lcard play" onclick={onplayrejected}>
				<span class="ct">Play a rejected universe &rarr;</span>
				<span class="cn">Switch a cut law back on and play that universe.</span>
			</button>
		</div>
	</section>

	<section class="open">
		<div class="opentext">
			<svg width="32" height="32" viewBox="0 0 20 20" aria-hidden="true"
				><circle cx="10" cy="10" r="4" fill="currentColor" /><ellipse cx="10" cy="10" rx="9" ry="3.4" fill="none" stroke="currentColor" stroke-width="1.2" transform="rotate(-20 10 10)" /></svg
			>
			<h2>Open source, all of it.</h2>
			<p>The engine, the training runs and every number on this page are on GitHub.</p>
		</div>
		<div class="openbtns">
			<a class="b1" href="https://github.com/r-baruah/kepler-64">Star on GitHub</a>
			<a class="b2" href="https://github.com/r-baruah/kepler-64">View the code</a>
		</div>
	</section>

	<footer class="foot">
		<span>Made by <a href="https://ripuranjan.com">Ripuranjan Baruah</a> &middot; also building Gambits</span>
		<a href="https://github.com/r-baruah/kepler-64">GitHub</a>
	</footer>
</div>

<style>
	.landing {
		max-width: 1280px;
		margin: 0 auto;
		padding: 0 var(--gutter);
	}
	.sec {
		padding: 40px 0;
		display: flex;
		flex-direction: column;
		gap: 22px;
	}
	.copy {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.label {
		font-size: 13px;
		font-weight: 500;
		color: var(--accent);
	}
	h2 {
		margin: 0;
		font-size: 32px;
		font-weight: 300;
		letter-spacing: -0.035em;
		line-height: 1.08;
	}
	p {
		margin: 0;
		font-size: 16px;
		line-height: 1.5;
		color: var(--ink-2);
	}
	p.strong {
		color: var(--ink);
	}
	b {
		font-weight: 600;
	}
	.plate {
		margin: 0;
		padding: 18px;
		background: var(--card);
		border-radius: 18px;
		box-shadow: var(--shadow-card);
	}
	.board-plate {
		align-self: center;
		width: min(100%, 300px);
	}
	.chart svg {
		display: block;
		width: 100%;
		height: auto;
		font-family: var(--font-sans);
	}
	.chart figcaption {
		margin-top: 8px;
		font-size: 12px;
		color: var(--muted);
	}
	.axis {
		font-size: 13px;
		fill: var(--muted);
	}
	.end {
		font-size: 16px;
	}
	.abbr {
		display: none;
		font-size: 26px;
	}
	@media (max-width: 899px) {
		.full {
			display: none;
		}
		.abbr {
			display: block;
		}
		.axis {
			font-size: 20px;
		}
	}

	/* ledger */
	.ledger-head {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.cards {
		display: grid;
		grid-template-columns: 1fr;
		gap: 10px;
	}
	.lcard {
		background: var(--card);
		border-radius: 14px;
		padding: 16px 18px;
		display: flex;
		flex-direction: column;
		gap: 6px;
		box-shadow: 0 6px 24px rgba(20, 33, 61, 0.06);
		text-align: left;
	}
	.lcard.cut {
		background: #eef0f4;
		box-shadow: none;
	}
	.lcard.cut .ct {
		color: var(--ink-2);
	}
	.ct {
		font-size: 17px;
	}
	.cn {
		font-size: 13px;
		color: var(--muted);
	}
	.badge {
		align-self: flex-start;
		font-size: 12px;
		font-weight: 500;
		border-radius: var(--radius-pill);
		padding: 4px 10px;
		color: #fff;
		background: var(--ink);
	}
	.badge.back {
		background: var(--accent);
	}
	.badge.rejected,
	.badge.rejected-twice {
		background: none;
		color: var(--muted);
		border: 1px solid var(--line-strong);
		padding: 3px 10px;
	}
	.lcard.play {
		background: none;
		box-shadow: none;
		border: 1px dashed #9aa1b0;
		justify-content: center;
		min-height: var(--tap);
		font: inherit;
	}
	.lcard.play .ct {
		color: var(--accent);
	}
	.lcard.play:hover {
		border-color: var(--accent);
	}

	/* open source band */
	.open {
		margin: 16px 0;
		padding: 28px 22px;
		border-radius: 24px;
		background: var(--ink);
		color: var(--bg);
		display: flex;
		flex-direction: column;
		gap: 18px;
	}
	.opentext {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.open h2 {
		font-size: 28px;
		letter-spacing: -0.03em;
	}
	.open p {
		color: #c9d0dd;
		font-size: 16px;
		max-width: 460px;
	}
	.openbtns {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.openbtns a {
		display: flex;
		align-items: center;
		justify-content: center;
		text-decoration: none;
		font-size: 15px;
		border-radius: var(--radius-pill);
		min-height: 48px;
	}
	.b1 {
		background: var(--bg);
		color: var(--ink);
		min-height: 52px !important;
	}
	.b1:hover {
		color: var(--ink);
		background: #fff;
	}
	.b2 {
		border: 1px solid rgba(246, 247, 249, 0.4);
		color: var(--bg);
	}
	.b2:hover {
		color: #fff;
		border-color: #fff;
	}

	.foot {
		padding: 16px 0 32px;
		font-size: 13px;
		color: var(--muted);
		line-height: 1.7;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.foot a {
		color: var(--muted);
	}
	.foot a:hover {
		color: var(--accent);
	}

	@media (min-width: 900px) {
		.landing {
			padding: 0 48px;
		}
		.sec {
			padding: 80px 0;
		}
		.split {
			flex-direction: row;
			align-items: center;
			gap: 72px;
		}
		.split .copy {
			flex: 1 1 0;
			gap: 18px;
		}
		.split p {
			font-size: 18px;
			line-height: 1.55;
			max-width: 460px;
		}
		h2 {
			font-size: 48px;
			letter-spacing: -0.04em;
			line-height: 1.05;
		}
		.plate {
			padding: 24px;
			border-radius: 20px;
		}
		.board-plate {
			width: 408px;
			flex: none;
		}
		.chart {
			width: 600px;
			flex: none;
			padding: 28px 28px 20px;
		}
		.ledger-head {
			flex-direction: row;
			justify-content: space-between;
			align-items: flex-end;
			gap: 48px;
		}
		.ledger-head > p {
			font-size: 18px;
			line-height: 1.55;
			max-width: 440px;
		}
		.ledger-head .copy {
			gap: 18px;
		}
		.cards {
			grid-template-columns: repeat(3, minmax(0, 1fr));
			gap: 16px;
		}
		.lcard {
			border-radius: 16px;
			padding: 22px;
			gap: 8px;
		}
		.ct {
			font-size: 18px;
		}
		.open {
			margin: 16px 0;
			padding: 56px;
			border-radius: 28px;
			flex-direction: row;
			justify-content: space-between;
			align-items: center;
			gap: 48px;
		}
		.open h2 {
			font-size: 40px;
			letter-spacing: -0.035em;
		}
		.open p {
			font-size: 17px;
			line-height: 1.55;
		}
		.openbtns {
			width: 260px;
			flex: none;
		}
		.foot {
			flex-direction: row;
			justify-content: space-between;
			align-items: center;
			height: 120px;
			padding: 0;
		}
	}
</style>
