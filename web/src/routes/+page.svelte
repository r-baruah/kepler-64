<script lang="ts">
	import { onMount } from 'svelte';
	import Board from '$lib/components/Board.svelte';
	import Dock from '$lib/components/Dock.svelte';
	import Header from '$lib/components/Header.svelte';
	import KingReadout from '$lib/components/KingReadout.svelte';
	import MoveList from '$lib/components/MoveList.svelte';
	import Sheet from '$lib/components/Sheet.svelte';
	import WeightsFigure from '$lib/components/WeightsFigure.svelte';
	import { fieldDots, oneDecimal } from '$lib/field';
	import { GameStore, resultText } from '$lib/game.svelte';
	import type { PresetFacts, Side } from '$lib/types';

	let { data } = $props();
	// svelte-ignore state_referenced_locally
	const game = new GameStore(data.facts);

	onMount(() => {
		// `?fen=` starts from a given position (used for testing, e.g. a mate in one).
		const fen = new URLSearchParams(location.search).get('fen');
		void game.boot(fen);
		return () => game.dispose();
	});

	const play = $derived(game.mode === 'play');
	const dots = $derived(game.fieldMode === 'pull' && game.snap ? fieldDots(game.snap.breakdown) : []);
	const targets = $derived(game.targets);
	const cones = $derived(game.cones);
	const top = $derived<Side>(game.bottom === 'white' ? 'black' : 'white');
	const whoIs = (s: Side) => (s === game.human ? 'You' : 'Kepler');
	const gamesText = $derived(
		game.preset.selfPlayGames > 0 ? `${game.preset.selfPlayGames.toLocaleString('en-US')} games of self-play` : 'No self-play yet'
	);
	const view = $derived(game.keplerView);
	const viewText = $derived(view === null ? '—' : (view >= 0 ? '+' : '−') + Math.abs(view).toFixed(2));
	const over = $derived(game.over);
	const boardLabel = $derived(`Chess board. ${game.snap ? game.snap.state.fen : ''}`);

	let now = $state(0);
	$effect(() => {
		if (!game.thinking) return;
		now = performance.now();
		const t = setInterval(() => (now = performance.now()), 100);
		return () => clearInterval(t);
	});
	const elapsed = $derived(game.thinking ? Math.max(0, now - game.thinkStart) / 1000 : 0);

	function diffChips(p: PresetFacts) {
		const keys = Object.keys(p.diff);
		const order = (k: string) => (k.startsWith('m_') ? 0 : 1);
		keys.sort((a, b) => order(a) - order(b));
		const fmt = (n: number) => (Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(Math.abs(n) < 1 ? 3 : 2).replace(/\.?0+$/, ''));
		const chips = keys.map((k) => `${k} ${fmt(p.diff[k].champion)} → ${fmt(p.diff[k].preset)}`);
		return { shown: chips.slice(0, 6), more: Math.max(0, chips.length - 6) };
	}
</script>

<svelte:head>
	<title>Kepler-64</title>
	<meta name="description" content="A tiny universe that learned to play chess. Play against it in your browser." />
</svelte:head>

<Header />

<main class="stage" data-mode={game.mode}>
	<!-- Hero text: the landing's first screen. Hidden once a game starts. -->
	<h1 class="h1">A tiny universe that learned to play chess.</h1>
	<p class="sub">
		<span class="long">It knows no chess strategy. Pieces pull along the lines they can move, and it learned the rest by playing itself.</span>
		<span class="short">It knows no chess strategy. It learned by playing itself.</span>
	</p>
	<div class="cta">
		<button class="primary" onclick={() => game.start('white')} disabled={game.status === 'nowasm'}>Play White</button>
		<button class="secondary" onclick={() => game.start('black')} disabled={game.status === 'nowasm'}>Play Black</button>
	</div>
	<div class="chips">
		<span>Runs in your browser</span>
		<span>{gamesText}</span>
		<span>Open source</span>
	</div>

	<!-- Left (desktop) / top (phone) while playing -->
	<section class="view" aria-label="Kepler's evaluation">
		<div class="viewnum">
			<div class="cap mono">KEPLER'S VIEW</div>
			<div class="num">{viewText}</div>
			<div class="cap mono">KU &middot; PAWN WEIGHT = 1</div>
		</div>
		<div class="meta mono">{game.preset.name.toUpperCase()}<br />{game.level.name.toUpperCase()} LEVEL</div>
		<div class="legend">
			<div class="cap mono">FIELD</div>
			<div class="lg"><span class="sw solid"></span>White pulls harder</div>
			<div class="lg"><span class="sw ring"></span>Black pulls harder</div>
			<p>Bigger dot, stronger pull. Pull follows the lines a piece can move and stops at the first piece in the way.</p>
		</div>
	</section>

	<section class="center">
		{#if game.status === 'nowasm'}
			<div class="note card" role="alert">
				<h2>Kepler needs WebAssembly</h2>
				<p>This browser can't run it, so the engine can't start here. Kepler-64 is open source: you can run the same engine from the command line.</p>
				<a href="https://github.com/r-baruah/kepler-64">Kepler-64 on GitHub</a>
			</div>
		{:else}
			<div class="row top" class:hide={!play}>
				<span class="who"
					>{whoIs(top)}
					{#if game.thinking && top === game.keplerSide}<span class="status mono" role="status">thinking {elapsed.toFixed(1)} s</span>{/if}</span
				>
				<KingReadout king={game.king(top)} who={whoIs(top) === 'You' ? 'You' : 'Kepler'} />
			</div>

			<div class="boardwrap">
				<Board
					placement={game.placement}
					bottom={game.bottom}
					{dots}
					{cones}
					lastMove={game.lastMove}
					selected={game.selected}
					{targets}
					label={boardLabel}
					onselectown={(sq) => game.selectOwn(sq)}
					ontap={(sq) => game.tap(sq)}
					onmove={(f, t) => game.tryMove(f, t)}
				/>
				{#if game.promotion}
					<div class="promo" role="dialog" aria-label="Choose a piece">
						{#each [['q', 'Queen'], ['r', 'Rook'], ['b', 'Bishop'], ['n', 'Knight']] as [l, name] (l)}
							<button onclick={() => game.choosePromotion(l as 'q' | 'r' | 'b' | 'n')}>
								<img src="/pieces/{game.human === 'white' ? 'w' : 'b'}{l.toUpperCase()}.svg" alt={name} width="44" height="44" />
							</button>
						{/each}
					</div>
				{/if}
			</div>

			<div class="row bottom" class:hide={!play}>
				<span class="who"
					>{whoIs(game.bottom)}
					{#if game.myTurn && game.bottom === game.human}<span class="status mono accent">YOUR MOVE</span
						>{:else if game.thinking && game.bottom === game.keplerSide}<span class="status mono" role="status">thinking {elapsed.toFixed(1)} s</span
						>{/if}</span
				>
				<KingReadout king={game.king(game.bottom)} who={whoIs(game.bottom) === 'You' ? 'You' : 'Kepler'} />
			</div>

			{#if !play}
				<div class="figure">
					<WeightsFigure constants={game.preset.constants} universe={game.preset.name} onchange={() => (game.sheet = 'universe')} />
				</div>
				<div class="restlegend">
					<span class="lg"><span class="sw solid"></span>White pulls harder</span>
					<span class="lg"><span class="sw ring"></span>Black pulls harder</span>
				</div>
			{:else}
				<div class="line" aria-live="polite">
					{#if game.status === 'loading'}
						<span class="muted">Kepler is waking up.</span>
					{:else if game.status === 'error'}
						<span>The engine stopped. Reload the page to start again.</span>
					{:else if over && game.snap}
						<div class="over">
							<strong>{resultText(game.snap.state, game.human)}</strong>
							<button class="primary sm" onclick={() => game.start(game.human)}>Play again</button>
						</div>
					{:else if game.thinking}
						<span class="muted">Kepler is thinking.</span>
						<button class="secondary sm" onclick={() => game.moveNow()}>Move now</button>
					{:else if game.line}
						<span class="meta2 mono">{game.lineMeta}</span> {game.line}
					{:else if game.snap?.state.inCheck}
						<span class="muted">Check.</span>
					{:else}
						<span class="muted">{game.myTurn ? 'Your move.' : ''}</span>
					{/if}
				</div>
				<div class="strip"><MoveList history={game.history} strip /></div>
				<Dock {game} />
			{/if}
		{/if}
	</section>

	<aside class="side" aria-label="Game details">
		<div class="cap mono">MOVES</div>
		<MoveList history={game.history} />
		<div class="cap mono spaced">UNIVERSE</div>
		<div class="val">{game.preset.name}</div>
		<div class="cap mono spaced">LEVEL</div>
		<div class="val">{game.level.name} &middot; {game.level.nodes.toLocaleString('en-US')} positions per move</div>
		<div class="strength">Strength at this level: not measured yet</div>
	</aside>
</main>

{#if game.sheet === 'new'}
	<Sheet title="New game" onclose={() => (game.sheet = null)}>
		<button class="primary big" onclick={() => game.start('white')}>Play White</button>
		<button class="secondary big" onclick={() => game.start('black')}>Play Black</button>
		<p class="fine">This starts from the beginning in the current universe: {game.preset.name}.</p>
	</Sheet>
{:else if game.sheet === 'universe'}
	<Sheet title="Universe" onclose={() => (game.sheet = null)}>
		<p class="fine">Switching universe starts a new game.</p>
		{#each data.facts.presets as p (p.slug)}
			{@const chips = diffChips(p)}
			<button class="opt" class:sel={p.slug === game.presetSlug} aria-pressed={p.slug === game.presetSlug} onclick={() => game.setPreset(p.slug)}>
				<span class="optname">{p.name}</span>
				<span class="optlabel">{p.label}</span>
				{#if chips.shown.length}
					<span class="chipsrow mono">
						{#each chips.shown as c (c)}<span class="chip">{c}</span>{/each}
						{#if chips.more}<span class="chip">+{chips.more} more</span>{/if}
					</span>
				{/if}
			</button>
		{/each}
	</Sheet>
{:else if game.sheet === 'level'}
	<Sheet title="Level" onclose={() => (game.sheet = null)}>
		{#each data.facts.levels as l (l.id)}
			<button class="opt" class:sel={l.id === game.levelId} aria-pressed={l.id === game.levelId} onclick={() => game.setLevel(l.id)}>
				<span class="optname">{l.name}</span>
				<span class="optlabel">{l.nodes.toLocaleString('en-US')} positions per move, about {l.about} on a phone.</span>
			</button>
		{/each}
		<p class="fine">Strength at this level: not measured yet. The same level gives the same move on every device.</p>
	</Sheet>
{:else if game.sheet === 'more'}
	<Sheet title="More" onclose={() => (game.sheet = null)}>
		<button class="secondary big" onclick={() => game.flip()}>Flip board</button>
		<button class="secondary big" onclick={() => (game.sheet = 'level')}>Level: {game.level.name}</button>
		<p class="fine">Field layer: tap the Field button to cycle Pull, Light cones, Off. Light cones: tap a piece.</p>
	</Sheet>
{/if}

<style>
	.stage {
		display: flex;
		flex-direction: column;
		gap: 18px;
		padding: 20px var(--gutter) 40px;
		max-width: 1280px;
		margin: 0 auto;
	}

	/* ---- hero text ---- */
	.h1 {
		order: 1;
		margin: 0;
		font-size: 42px;
		font-weight: 300;
		letter-spacing: -0.045em;
		line-height: 1.02;
	}
	.sub {
		order: 2;
		margin: 0;
		font-size: 17px;
		line-height: 1.5;
		color: var(--ink-2);
	}
	.sub .long {
		display: none;
	}
	.center {
		order: 4;
	}
	/* phone: the buttons sit above the board so they are on the first screen */
	.cta {
		order: 3;
		display: flex;
		gap: 10px;
	}
	.chips {
		order: 5;
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		font-size: 12px;
		color: var(--muted);
	}
	.chips span {
		padding: 6px 10px;
		border-radius: var(--radius-pill);
		background: var(--soft);
	}
	.view,
	.side {
		display: none;
	}

	.primary,
	.secondary {
		min-height: 54px;
		padding: 0 28px;
		font-size: 16px;
		font-weight: 500;
		border-radius: var(--radius-pill);
		border: 1px solid var(--ink);
	}
	.cta button {
		flex: 1 1 0;
	}
	.primary {
		background: var(--ink);
		color: #fff;
	}
	.primary:hover:not(:disabled) {
		background: var(--ink-2);
	}
	.secondary {
		background: var(--card);
		border-color: var(--line-strong);
		color: var(--ink);
	}
	.secondary:hover:not(:disabled) {
		border-color: var(--ink);
	}
	button:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.sm {
		min-height: var(--tap);
		padding: 0 18px;
		font-size: 14px;
	}
	.big {
		width: 100%;
	}

	/* ---- the centre column ---- */
	.center {
		--board: 100%;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 8px;
		min-width: 0;
	}
	.boardwrap {
		position: relative;
		width: var(--board);
	}
	.row {
		width: var(--board);
		display: flex;
		justify-content: space-between;
		align-items: flex-end;
		gap: 8px;
		min-height: 30px;
	}
	.row.hide {
		display: none;
	}
	.who {
		font-weight: 600;
		font-size: 14px;
		white-space: nowrap;
	}
	.status {
		margin-left: 8px;
		font-size: 11px;
		font-weight: 500;
		color: var(--muted);
	}
	.status.accent {
		color: var(--accent);
	}
	.figure {
		width: var(--board);
		margin-top: 10px;
	}
	.restlegend {
		display: flex;
		gap: 16px;
		font-size: 12px;
		color: var(--muted);
		width: var(--board);
	}
	.lg {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 13px;
	}
	.restlegend .lg {
		font-size: 12px;
	}
	.sw {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		display: inline-block;
		flex: none;
	}
	.sw.solid {
		background: var(--ink);
		opacity: 0.5;
	}
	.sw.ring {
		border: 1.5px solid var(--ink);
		opacity: 0.6;
	}
	.line {
		width: var(--board);
		min-height: 52px;
		margin-top: 2px;
		padding-top: 8px;
		border-top: 1px solid var(--line);
		font-size: 14px;
		line-height: 1.5;
		color: var(--ink-2);
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 12px;
	}
	.meta2 {
		font-size: 11px;
		color: var(--accent);
	}
	.muted {
		color: var(--muted);
	}
	.over {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		width: 100%;
		font-size: 16px;
		color: var(--ink);
	}
	.strip {
		width: var(--board);
	}
	.center :global(.dock) {
		margin-top: 6px;
	}

	.promo {
		position: absolute;
		left: 50%;
		top: 50%;
		transform: translate(-50%, -50%);
		display: flex;
		gap: 6px;
		padding: 8px;
		background: var(--card);
		border: 1px solid var(--line);
		border-radius: var(--radius-card-sm);
		box-shadow: var(--shadow-card);
	}
	.promo button {
		width: 56px;
		height: 56px;
		display: grid;
		place-items: center;
		background: var(--soft);
		border: none;
		border-radius: 12px;
	}
	.promo button:hover {
		background: var(--line);
	}

	.card {
		background: var(--card);
		border-radius: var(--radius-card);
		box-shadow: var(--shadow-card);
		padding: 24px;
	}
	.note h2 {
		margin: 0 0 8px;
		font-size: 20px;
	}
	.note p {
		margin: 0 0 12px;
		color: var(--ink-2);
	}

	/* ---- sheets ---- */
	.fine {
		margin: 4px 0 0;
		font-size: 13px;
		color: var(--muted);
	}
	.opt {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 4px;
		width: 100%;
		min-height: var(--tap);
		padding: 14px 16px;
		text-align: left;
		background: var(--card);
		border: 1px solid var(--line);
		border-radius: var(--radius-card-sm);
	}
	.opt:hover {
		border-color: var(--ink);
	}
	.opt.sel {
		border-color: var(--accent);
		box-shadow: inset 0 0 0 1px var(--accent);
	}
	.optname {
		font-weight: 600;
		font-size: 15px;
	}
	.optlabel {
		font-size: 13px;
		color: var(--ink-2);
		line-height: 1.45;
	}
	.chipsrow {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 4px;
	}
	.chip {
		padding: 2px 8px;
		font-size: 11px;
		background: var(--soft);
		border-radius: var(--radius-pill);
		color: var(--ink-2);
	}

	/* ---- play layout, phone ---- */
	.stage[data-mode='play'] .h1,
	.stage[data-mode='play'] .sub,
	.stage[data-mode='play'] .cta,
	.stage[data-mode='play'] .chips {
		display: none;
	}
	.stage[data-mode='play'] {
		gap: 10px;
		padding-top: 12px;
		padding-bottom: 12px;
	}
	.stage[data-mode='play'] .view {
		order: 0;
		display: flex;
		justify-content: space-between;
		align-items: flex-end;
	}
	.stage[data-mode='play'] .center {
		--board: min(100%, max(260px, calc(100dvh - 400px)));
	}
	.view .cap {
		font-size: 10px;
		color: var(--muted);
	}
	.view .num {
		font-size: 36px;
		font-weight: 300;
		letter-spacing: -0.04em;
		line-height: 1;
	}
	.view .meta {
		text-align: right;
		font-size: 10px;
		color: var(--muted);
		line-height: 1.6;
	}
	.view .legend {
		display: none;
	}
	.spaced {
		margin-top: 22px;
	}

	/* ---- desktop ---- */
	@media (min-width: 900px) {
		.stage {
			padding: 40px 48px;
		}
		.stage[data-mode='rest'] {
			display: grid;
			grid-template-columns: minmax(0, 1fr) auto;
			grid-template-areas: 'h1 center' 'sub center' 'cta center' 'chips center';
			grid-template-rows: 1fr auto auto 1fr;
			column-gap: 72px;
			row-gap: 26px;
			align-items: center;
			min-height: calc(100dvh - 56px);
		}
		.stage[data-mode='rest'] .h1 {
			grid-area: h1;
			align-self: end;
			font-size: 72px;
			letter-spacing: -0.05em;
			line-height: 1;
		}
		.stage[data-mode='rest'] .sub {
			grid-area: sub;
			font-size: 19px;
			line-height: 1.55;
			max-width: 470px;
		}
		.sub .long {
			display: inline;
		}
		.sub .short {
			display: none;
		}
		.stage[data-mode='rest'] .cta {
			grid-area: cta;
		}
		.stage[data-mode='rest'] .cta button {
			flex: none;
		}
		.stage[data-mode='rest'] .chips {
			grid-area: chips;
			align-self: start;
			font-size: 13px;
		}
		.stage[data-mode='rest'] .chips span {
			padding: 6px 12px;
		}
		.stage[data-mode='rest'] .center {
			grid-area: center;
			--board: min(500px, calc(100dvh - 240px));
		}

		.stage[data-mode='play'] {
			display: grid;
			grid-template-columns: 240px auto 200px;
			justify-content: center;
			column-gap: 56px;
			align-items: start;
			padding: 14px 32px 16px;
		}
		.stage[data-mode='play'] .view {
			order: 0;
			flex-direction: column;
			align-items: flex-start;
			gap: 6px;
			padding-top: 40px;
		}
		.stage[data-mode='play'] .center {
			order: 1;
			--board: min(560px, max(300px, calc(100dvh - 350px)));
		}
		.stage[data-mode='play'] .side {
			order: 2;
			display: flex;
			flex-direction: column;
			gap: 8px;
			padding-top: 40px;
			font-size: 13px;
		}
		.view .meta {
			display: none;
		}
		.view .num {
			font-size: 64px;
		}
		.view .cap {
			font-size: 11px;
		}
		.view .viewnum {
			display: flex;
			flex-direction: column;
			gap: 6px;
		}
		.view .legend {
			display: flex;
			flex-direction: column;
			gap: 6px;
			margin-top: 22px;
		}
		.view .legend p {
			margin: 4px 0 0;
			font-size: 12px;
			line-height: 1.55;
			color: var(--muted);
		}
		.side .cap {
			font-size: 11px;
			color: var(--muted);
		}
		.side .val {
			font-size: 14px;
		}
		.side .strength {
			margin-top: 6px;
			font-size: 12px;
			color: var(--muted);
		}
		.strip {
			display: none;
		}
		.line {
			justify-content: center;
			text-align: center;
			width: min(620px, 100%);
			font-size: 15px;
		}
	}
	.side .cap {
		color: var(--muted);
		font-size: 11px;
	}
</style>
