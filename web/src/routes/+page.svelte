<script lang="ts">
	import { onMount, tick } from 'svelte';
	import Board from '$lib/components/Board.svelte';
	import ConstantsPlate from '$lib/components/ConstantsPlate.svelte';
	import Dock from '$lib/components/Dock.svelte';
	import FieldLegend from '$lib/components/FieldLegend.svelte';
	import Header from '$lib/components/Header.svelte';
	import Landing from '$lib/components/Landing.svelte';
	import KingReadout from '$lib/components/KingReadout.svelte';
	import PgnActions from '$lib/components/PgnActions.svelte';
	import MoveList from '$lib/components/MoveList.svelte';
	import Sheet from '$lib/components/Sheet.svelte';
	import UniverseSystem from '$lib/components/UniverseSystem.svelte';
	import WeightsFigure from '$lib/components/WeightsFigure.svelte';
	import { diffLines } from '$lib/constants';
	import { GameStore, resultText, type FieldMode } from '$lib/game.svelte';
	import { motionOK } from '$lib/theme';
	import type { Side } from '$lib/types';

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
	const field = $derived(game.fieldMode === 'pull' && game.snap ? game.snap.breakdown.field : null);
	const top = $derived<Side>(game.bottom === 'white' ? 'black' : 'white');
	const whoIs = (s: Side) => (s === game.human ? 'You' : 'Kepler');
	const fmt = (n: number) => n.toLocaleString('en-US');
	/** Variants are Home's constants with one law changed: never trained, so no game count. */
	const gamesOf = (p: { variant: boolean; selfPlayGames: number }) =>
		p.variant ? 'Variant of Home, not trained' : p.selfPlayGames > 0 ? `${fmt(p.selfPlayGames)} games of self-play` : 'No self-play yet';
	const gamesText = $derived(gamesOf(game.preset));
	let showConstants = $state(false);
	const view = $derived(game.keplerView);
	// Rounded first, so a tiny negative score never shows as "−0.00".
	const viewText = $derived.by(() => {
		if (view === null) return '—';
		const s = Math.abs(view).toFixed(2);
		return s === '0.00' ? s : (view > 0 ? '+' : '−') + s;
	});
	const over = $derived(game.over);
	const boardLabel = $derived(`Chess board. ${game.snap ? game.snap.state.fen : ''}`);
	const moves = $derived(Math.ceil(game.history.length / 2));
	const maxNodes = $derived(Math.max(...data.facts.levels.map((l) => l.nodes)));
	const FIELD_MODES: [FieldMode, string][] = [
		['pull', 'Pull'],
		['cones', 'Light cones'],
		['off', 'Off']
	];

	let now = $state(0);
	$effect(() => {
		if (!game.thinking) return;
		now = performance.now();
		const t = setInterval(() => (now = performance.now()), 100);
		return () => clearInterval(t);
	});
	/** Header nav: landing sections live below the hero, so go back to it first if a game is open. */
	async function nav(id: string) {
		if (game.mode === 'play') {
			await game.goHome();
			if (game.mode === 'play') return;
			await tick();
		}
		document.getElementById(id)?.scrollIntoView({ behavior: motionOK() ? 'smooth' : 'auto', block: 'start' });
	}
	async function home() {
		if (game.mode === 'play') await game.goHome();
		else window.scrollTo({ top: 0, behavior: motionOK() ? 'smooth' : 'auto' });
	}

	/** From the landing: start a game in that universe, back at the top of the page. */
	async function playUniverse(slug: string) {
		await game.setPreset(slug);
		window.scrollTo({ top: 0, behavior: 'auto' });
		await game.start('white');
	}

	const elapsed = $derived(game.thinking ? Math.max(0, now - game.thinkStart) / 1000 : 0);

	// Phone: the PGN export menu at the right end of the move strip (shown once a move has been played).
	let pgnOpen = $state(false);
	let pgnMenu = $state<HTMLElement | null>(null);
	let pgnButton = $state<HTMLButtonElement | null>(null);
	$effect(() => {
		if (!game.history.length || game.sheet || !play) pgnOpen = false;
	});
	function pgnOutside(e: PointerEvent) {
		if (pgnOpen && pgnMenu && !pgnMenu.contains(e.target as Node)) pgnOpen = false;
	}
	function pgnKey(e: KeyboardEvent) {
		if (pgnOpen && e.key === 'Escape') {
			pgnOpen = false;
			pgnButton?.focus();
		}
	}
</script>

<svelte:window onpointerdown={pgnOutside} onkeydown={pgnKey} />

<svelte:head>
	<title>Kepler-64</title>
	<meta name="description" content="A tiny universe that learned to play chess. Play against it in your browser." />
</svelte:head>

<Header onhome={home} onnav={nav} playing={play} />

<main class="stage" data-mode={game.mode}>
	<!-- Hero text: the landing's first screen. Hidden once a game starts. -->
	<h1 class="h1">A tiny universe that learned to play chess.</h1>
	<p class="sub">We gave chess its own physics. Pieces have mass and pull along the lines they can move. It learned the rest by playing itself.</p>
	<div class="cta">
		<button class="primary" onclick={() => game.start('white')} disabled={game.status === 'nowasm'}>Play White</button>
		<button class="secondary" onclick={() => game.start('black')} disabled={game.status === 'nowasm'}>Play Black</button>
	</div>

	<!-- Left (desktop) / top (phone) while playing -->
	<section class="view" aria-label="Kepler's evaluation">
		<div class="viewnum">
			<div class="cap">Kepler's view</div>
			<div class="num" class:pos={view !== null && view > 0}>{viewText}</div>
			<div class="cap" title="Kepler units: one pawn's weight is 1">ku, pawn = 1</div>
		</div>
		<button class="meta" onclick={() => (game.sheet = 'universe')}>{game.preset.name}<br />{game.level.name} level</button>
		<div class="legendcol">
			<div class="cap">Field</div>
			<FieldLegend mode={game.fieldMode} c={game.preset.constants.c} />
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
					{#if game.thinking && top === game.keplerSide}<span class="status" role="status"><span class="pip"></span>Thinking {elapsed.toFixed(1)} s</span>{/if}</span
				>
				<KingReadout king={game.king(top)} who={whoIs(top) === 'You' ? 'You' : 'Kepler'} />
			</div>

			<div class="boardwrap">
				<Board
					placement={game.placement}
					bottom={game.bottom}
					{field}
					cones={game.cones}
					pulse={game.pulse}
					lastMove={game.lastMove}
					moveId={game.history.length}
					selected={game.selected}
					targets={game.targets}
					check={game.checkSquare}
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
					{#if game.myTurn && game.bottom === game.human}<span class="status accent">Your move</span
						>{:else if game.thinking && game.bottom === game.keplerSide}<span class="status" role="status"
							><span class="pip"></span>Thinking {elapsed.toFixed(1)} s</span
						>{/if}</span
				>
				<KingReadout king={game.king(game.bottom)} who={whoIs(game.bottom) === 'You' ? 'You' : 'Kepler'} />
			</div>

			{#if !play}
				<div class="restlegend"><FieldLegend mode="pull" c={game.preset.constants.c} compact /></div>
				<div class="figure">
					<WeightsFigure constants={game.preset.constants} universe={game.preset.name} onchange={() => (game.sheet = 'universe')} />
				</div>
			{:else}
				<div class="line" class:isover={over} aria-live="polite">
					{#if game.status === 'loading'}
						<span class="muted">Kepler is waking up.</span>
					{:else if game.status === 'error'}
						<span>The engine stopped. Reload the page to start again.</span>
					{:else if over && game.snap}
						<div class="over">
							<div class="res">
								<strong>{resultText(game.snap.state, game.human)}</strong>
								<span class="muted num">{moves} {moves === 1 ? 'move' : 'moves'} &middot; {game.preset.name}</span>
							</div>
							<div class="overpgn"><PgnActions {game} /></div>
							<div class="overbtns">
								<button class="secondary sm" onclick={() => (game.sheet = 'universe')}>Universe</button>
								<button class="primary sm" onclick={() => game.start(game.human)}>Play again</button>
							</div>
						</div>
					{:else if game.thinking}
						<span class="muted">Kepler is thinking.</span>
						<button class="secondary sm" onclick={() => game.moveNow()}>Move now</button>
					{:else if game.line}
						<span><span class="meta2 num">{game.lineMeta}</span> {game.line}</span>
					{:else if game.snap?.state.inCheck}
						<span class="muted">Check.</span>
					{:else}
						<span class="muted">{game.myTurn ? 'Your move. Tap a piece, then a square.' : ''}</span>
					{/if}
				</div>
				<div class="strip">
					<div class="stripmoves"><MoveList history={game.history} strip /></div>
					{#if game.history.length}
						<div class="pgnmenu" bind:this={pgnMenu}>
							<button bind:this={pgnButton} class="pgnbtn" aria-label="Export PGN" aria-expanded={pgnOpen} aria-controls="pgn-menu" onclick={() => (pgnOpen = !pgnOpen)}>
								<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3.5v9M6.5 9 10 12.5 13.5 9" /><path d="M4 13.5v2.5h12v-2.5" /></svg>
								<span>PGN</span>
							</button>
							{#if pgnOpen}
								<div class="pgnpop" id="pgn-menu">
									<PgnActions {game} stack ondownload={() => (pgnOpen = false)} />
								</div>
							{/if}
						</div>
					{/if}
				</div>
				<div class="dockwrap"><Dock {game} /></div>
			{/if}
		{/if}
	</section>

	<aside class="side" aria-label="Game details">
		<div class="block">
			<div class="cap">Moves</div>
			{#if game.history.length}
				<MoveList history={game.history} />
				<PgnActions {game} />
			{:else}
				<p class="empty">No moves yet.</p>
			{/if}
		</div>
		<div class="block">
			<div class="cap">Universe</div>
			<button class="pick upick" onclick={() => (game.sheet = 'universe')}>
				<span class="sidesys"><UniverseSystem constants={game.preset.constants} slug={game.preset.slug} name={game.preset.name} size="mini" /></span>
				<span><span class="val">{game.preset.name}</span><span class="sm2">{gamesText}</span></span>
			</button>
		</div>
		<div class="block">
			<div class="cap">Level</div>
			<button class="pick" onclick={() => (game.sheet = 'level')}>
				<span><span class="val">{game.level.name}</span><span class="sm2">{fmt(game.level.nodes)} positions per move</span></span>
			</button>
			<div class="strength">Strength at this level: not measured yet</div>
		</div>
	</aside>
</main>

{#if !play}
	<Landing facts={data.facts} ready={game.status === 'ready'} generationUrl={data.facts.presets[0].file} onplay={playUniverse} />
{/if}

{#if game.sheet === 'new'}
	<Sheet title="New game" onclose={() => (game.sheet = null)}>
		<div class="two">
			<button class="primary big" onclick={() => game.start('white')}>Play White</button>
			<button class="secondary big" onclick={() => game.start('black')}>Play Black</button>
		</div>
		<p class="fine">Starts from the beginning in the current universe: {game.preset.name}, {game.level.name} level.</p>
	</Sheet>
{:else if game.sheet === 'universe'}
	<Sheet title="Universe" onclose={() => (game.sheet = null)}>
		<p class="fine">Each universe is one set of constants. Switching starts a new game.</p>
		{#each data.facts.presets as p (p.slug)}
			{@const d = diffLines(p)}
			{@const sel = p.slug === game.presetSlug}
			<button class="opt uopt" class:sel aria-pressed={sel} onclick={() => game.setPreset(p.slug)}>
				<span class="usys"><UniverseSystem constants={p.constants} slug={p.slug} name={p.name} size="mini" /></span>
				<span class="utext">
					<span class="opthead"><span class="optname">{p.name}</span><span class="games num">{gamesOf(p)}</span></span>
					<span class="optlabel">{p.label}</span>
					{#if d.lines.length || d.more}
						<span class="diffs">
							{#each d.lines as l (l)}<span>{l}</span>{/each}
							{#if d.more}<span class="more">{d.more} other {d.more === 1 ? 'constant differs' : 'constants differ'}</span>{/if}
						</span>
					{/if}
				</span>
			</button>
		{/each}
		<button class="disclose" aria-expanded={showConstants} onclick={() => (showConstants = !showConstants)}>
			<span aria-hidden="true">{showConstants ? '−' : '+'}</span> Constants of {game.preset.name}
		</button>
		{#if showConstants}<ConstantsPlate preset={game.preset} />{/if}
	</Sheet>
{:else if game.sheet === 'level'}
	<Sheet title="Level" onclose={() => (game.sheet = null)}>
		<p class="fine">A level is a fixed number of positions Kepler looks at per move, so the same level gives the same move on every device.</p>
		{#each data.facts.levels as l (l.id)}
			<button class="opt" class:sel={l.id === game.levelId} aria-pressed={l.id === game.levelId} onclick={() => game.setLevel(l.id)}>
				<span class="opthead"><span class="optname">{l.name}</span><span class="num nodes">{fmt(l.nodes)}</span></span>
				<span class="scale" aria-hidden="true"><span style:width="{(l.nodes / maxNodes) * 100}%"></span></span>
				<span class="optlabel">Positions per move, about {l.about} on a phone.</span>
			</button>
		{/each}
		<p class="fine">Strength at this level: not measured yet.</p>
	</Sheet>
{:else if game.sheet === 'more'}
	<Sheet title="More" onclose={() => (game.sheet = null)}>
		<div class="cap">Field layer</div>
		<div class="seg" role="group" aria-label="Field layer">
			{#each FIELD_MODES as [m, name] (m)}
				<button class:on={game.fieldMode === m} aria-pressed={game.fieldMode === m} onclick={() => game.setField(m)}>{name}</button>
			{/each}
		</div>
		<FieldLegend mode={game.fieldMode} c={game.preset.constants.c} />
		<div class="two">
			<button class="secondary big" onclick={() => game.flip()}>Flip board</button>
			<button class="secondary big" onclick={() => (game.sheet = 'level')}>Level: {game.level.name}</button>
		</div>
		{#if game.history.length}
			<div class="cap">Save this game</div>
			<PgnActions {game} />
		{/if}
	</Sheet>
{/if}

<style>
	.stage {
		display: flex;
		flex-direction: column;
		gap: 18px;
		padding: 22px var(--gutter) 40px;
		max-width: 1280px;
		margin: 0 auto;
	}

	/* ---- hero text (phone first) ---- */
	.h1 {
		order: 1;
		margin: 0;
		font-size: clamp(34px, 10vw, 42px);
		font-weight: 300;
		letter-spacing: -0.045em;
		line-height: 1.03;
		text-wrap: balance;
	}
	.sub {
		order: 2;
		margin: 0;
		font-size: 16px;
		line-height: 1.5;
		color: var(--ink-2);
		max-width: 34em;
	}
	.cta {
		order: 3;
		display: flex;
		gap: 10px;
	}
	.center {
		order: 4;
	}
	.view,
	.side {
		display: none;
	}

	/* ---- buttons ---- */
	.primary,
	.secondary {
		min-height: 52px;
		padding: 0 26px;
		font-size: 16px;
		font-weight: 500;
		border-radius: var(--radius-pill);
		border: 1px solid var(--ink);
		transition:
			background-color 0.15s,
			border-color 0.15s;
	}
	.primary:focus-visible,
	.secondary:focus-visible {
		outline-offset: 3px;
	}
	.cta button {
		flex: 1 1 0;
	}
	.primary {
		background: var(--ink);
		color: var(--on-ink);
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
		font-size: 15px;
	}
	.big {
		width: 100%;
	}
	.two {
		display: flex;
		gap: 10px;
	}
	.two > * {
		flex: 1 1 0;
		padding: 0 12px;
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
		display: flex;
		align-items: baseline;
		font-weight: 500;
		font-size: var(--t-md);
		white-space: nowrap;
	}
	.status {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		margin-left: 10px;
		font-size: var(--t-xs);
		font-weight: 500;
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}
	.status.accent {
		color: var(--accent);
	}
	.pip {
		width: 6px;
		height: 6px;
		background: var(--accent);
		border-radius: 50%;
		animation: breathe 1.2s ease-in-out infinite;
	}
	@keyframes breathe {
		50% {
			opacity: 0.25;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.pip {
			animation: none;
		}
	}
	.restlegend {
		width: var(--board);
		margin-top: 4px;
	}
	.figure {
		width: var(--board);
		margin-top: 8px;
		padding-top: 8px;
		border-top: 1px solid var(--line);
	}
	.line {
		width: var(--board);
		min-height: 54px;
		padding-top: 8px;
		border-top: 1px solid var(--line);
		font-size: 15px;
		line-height: 1.45;
		color: var(--ink-2);
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 12px;
	}
	.meta2 {
		margin-right: 6px;
		font-weight: 500;
		color: var(--accent);
	}
	.muted {
		color: var(--muted);
	}
	.over {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 10px 12px;
		width: 100%;
		padding: 12px 14px;
		border-radius: var(--radius-card-sm);
		background: var(--card);
		border: 1px solid var(--line);
		color: var(--ink);
	}
	.res {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.res strong {
		font-size: var(--t-lg);
		font-weight: 500;
	}
	.res .num {
		font-size: var(--t-xs);
	}
	.overpgn {
		order: 3;
		flex: 1 0 100%;
		margin-top: -6px;
	}
	.overbtns {
		display: flex;
		gap: 8px;
	}
	.strip {
		width: var(--board);
		min-height: 26px;
		display: flex;
		align-items: center;
		gap: 12px;
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
		margin: 0;
		font-size: 13px;
		line-height: 1.5;
		color: var(--muted);
	}
	.opt {
		display: flex;
		flex-direction: column;
		align-items: stretch;
		gap: 6px;
		width: 100%;
		min-height: var(--tap);
		padding: 14px 16px;
		text-align: left;
		background: var(--card);
		border: 1px solid var(--line);
		border-radius: var(--radius-card-sm);
		transition: border-color 0.15s;
	}
	.opt:hover {
		border-color: var(--ink);
	}
	.opt.sel {
		border-color: var(--accent);
		box-shadow: inset 0 0 0 1px var(--accent);
	}
	.opthead {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
	}
	.optname {
		font-weight: 500;
		font-size: 15px;
	}
	.optlabel {
		font-size: 13px;
		color: var(--ink-2);
		line-height: 1.45;
	}
	.uopt {
		flex-direction: row;
		align-items: flex-start;
		gap: 14px;
	}
	.usys {
		flex: none;
		width: 120px;
		padding-top: 2px;
	}
	.utext {
		display: flex;
		flex-direction: column;
		gap: 4px;
		min-width: 0;
		flex: 1 1 auto;
	}
	.uopt .opthead {
		flex-wrap: wrap;
		gap: 0 12px;
	}
	.games {
		font-size: var(--t-xs);
		color: var(--muted);
		white-space: nowrap;
	}
	.diffs {
		display: flex;
		flex-direction: column;
		gap: 2px;
		margin-top: 4px;
		padding-top: 6px;
		border-top: 1px dashed var(--line-strong);
		font-size: var(--t-xs);
		line-height: 1.5;
		color: var(--ink);
		font-family: var(--font-mono);
	}
	.diffs .more {
		color: var(--muted);
	}
	.disclose {
		align-self: flex-start;
		min-height: var(--tap);
		padding: 0 4px;
		border: none;
		background: none;
		font-size: var(--t-sm);
		color: var(--accent);
	}
	.disclose span {
		display: inline-block;
		width: 1em;
		font-family: var(--font-mono);
	}
	.nodes {
		font-size: 13px;
		color: var(--ink-2);
	}
	.scale {
		display: block;
		height: 4px;
		border-radius: 2px;
		background: var(--soft);
		overflow: hidden;
	}
	.scale span {
		display: block;
		height: 100%;
		min-width: 3px;
		background: var(--ink);
	}
	.opt.sel .scale span {
		background: var(--accent);
	}
	.seg {
		display: flex;
		gap: 2px;
		padding: 3px;
		border-radius: var(--radius-pill);
		background: var(--soft);
	}
	.seg button {
		flex: 1 1 0;
		min-height: 42px;
		border: none;
		border-radius: var(--radius-pill);
		background: none;
		font-size: 15px;
	}
	.seg button.on {
		background: var(--card);
		box-shadow: var(--shadow-soft);
		font-weight: 500;
	}

	.cap {
		font-size: var(--t-xs);
		font-weight: 500;
		color: var(--muted);
	}

	/* ---- play layout, phone ---- */
	.stage[data-mode='play'] .h1,
	.stage[data-mode='play'] .sub,
	.stage[data-mode='play'] .cta {
		display: none;
	}
	.stage[data-mode='play'] {
		gap: 8px;
		padding-top: 10px;
		padding-bottom: calc(74px + env(safe-area-inset-bottom));
	}
	.stage[data-mode='play'] .view {
		order: 0;
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	/* On phones the board takes the full width between the gutters. It used to shrink with the
	   viewport height (so a phone with browser bars showed a small centred board); the page
	   scrolls instead, and the fixed dock has its own bottom padding. Wider, shorter screens
	   (tablet portrait, phone landscape) still fit the board to the height. */
	.stage[data-mode='play'] .center {
		--board: 100%;
	}
	/* Phones in play: the board runs edge to edge, like a chess app; the rows above and below keep the gutters. */
	@media (max-width: 559px) {
		.stage[data-mode='play'] .boardwrap {
			width: calc(100% + 2 * var(--gutter));
		}
		.stage[data-mode='play'] .boardwrap :global(.board) {
			border-radius: 0;
			border-left: none;
			border-right: none;
		}
	}
	@media (min-width: 560px) {
		.stage[data-mode='play'] .center {
			--board: min(100%, max(260px, calc(100dvh - 380px)));
		}
	}
	/* Tall phones: centre the block between the header and the dock instead of leaving a gap. */
	@media (max-width: 899px) {
		.stage[data-mode='play'] {
			min-height: calc(100dvh - 57px);
			justify-content: center;
		}
	}
	.view .viewnum {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.view .num {
		font-size: 30px;
		font-weight: 300;
		transition: color 0.3s var(--ease);
		letter-spacing: -0.04em;
		line-height: 1;
		font-variant-numeric: tabular-nums;
	}
	.view .meta {
		text-align: right;
		font-size: var(--t-xs);
		color: var(--muted);
		line-height: 1.6;
		background: none;
		border: none;
		padding: 4px 0;
		min-height: var(--tap);
	}
	.view .meta:hover {
		color: var(--accent);
	}
	.view .legendcol {
		display: none;
	}
	/* The move strip scrolls sideways on its own; the PGN button stays pinned at the row's right end. */
	.stripmoves {
		flex: 1 1 auto;
		min-width: 0;
	}
	.pgnmenu {
		position: relative;
		flex: none;
	}
	.pgnbtn {
		display: flex;
		align-items: center;
		gap: 5px;
		min-height: var(--tap);
		margin: -9px -6px -9px 0;
		padding: 0 6px 0 8px;
		border: none;
		border-radius: 8px;
		background: none;
		font-size: var(--t-xs);
		font-weight: 500;
		color: var(--muted);
		transition:
			border-color 0.15s,
			color 0.15s;
	}
	.pgnbtn svg {
		width: 15px;
		height: 15px;
		fill: none;
		stroke: currentColor;
		stroke-width: 1.4;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.pgnbtn:hover,
	.pgnbtn[aria-expanded='true'] {
		color: var(--ink);
	}
	/* Opens upward: the dock sits just below the strip. */
	.pgnpop {
		position: absolute;
		bottom: calc(100% + 6px);
		right: 0;
		z-index: 20;
		padding: 4px 0;
		background: var(--card);
		border: 1px solid var(--line);
		border-radius: var(--radius-card-sm);
		box-shadow: var(--shadow-card);
	}
	/* Phone: the dock is a slim bar along the bottom edge. */
	.dockwrap {
		position: fixed;
		left: 0;
		right: 0;
		bottom: 0;
		z-index: 10;
	}

	/* ---- desktop ---- */
	@media (min-width: 900px) {
		.stage {
			padding: 40px 48px;
		}
		.stage[data-mode='rest'] {
			display: grid;
			grid-template-columns: minmax(0, 1fr) auto;
			grid-template-areas: 'h1 center' 'sub center' 'cta center';
			grid-template-rows: 1fr auto 1fr;
			column-gap: 72px;
			row-gap: 24px;
			align-items: center;
			min-height: calc(100dvh - 57px);
		}
		.stage[data-mode='rest'] .h1 {
			grid-area: h1;
			align-self: end;
			font-size: clamp(56px, 5.6vw, 76px);
			letter-spacing: -0.05em;
			line-height: 1;
		}
		.stage[data-mode='rest'] .sub {
			grid-area: sub;
			font-size: 19px;
			line-height: 1.55;
			max-width: 470px;
		}
		.stage[data-mode='rest'] .cta {
			grid-area: cta;
			align-self: start;
		}
		.stage[data-mode='rest'] .cta button {
			flex: none;
			min-width: 160px;
		}
		.stage[data-mode='rest'] .center {
			grid-area: center;
			--board: min(480px, calc(100dvh - 340px));
		}

		.stage[data-mode='play'] {
			/* The side columns and the gap give way first on narrow desktops (900 to 1100px), then the
			   board, so the page never scrolls sideways; at 1100px and up this is the full layout. */
			--bw: min(600px, max(320px, calc(100dvh - 300px)), calc(100vw - 460px));
			display: grid;
			grid-template-columns: minmax(150px, 236px) var(--bw) minmax(150px, 236px);
			justify-content: center;
			column-gap: clamp(20px, 4vw, 48px);
			align-items: start;
			padding: 18px 32px 24px;
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
			--board: 100%;
			width: var(--bw);
		}
		.stage[data-mode='play'] .side {
			order: 2;
			display: flex;
			flex-direction: column;
			gap: 26px;
			padding-top: 40px;
			font-size: 13px;
		}
		.view .meta {
			display: none;
		}
		.view .num {
			font-size: 60px;
		}
		.view .viewnum {
			gap: 8px;
			padding-bottom: 22px;
			border-bottom: 1px solid var(--line);
			width: 100%;
		}
		.view .legendcol {
			display: flex;
			flex-direction: column;
			gap: 10px;
			margin-top: 16px;
		}
		.side .block {
			display: flex;
			flex-direction: column;
			gap: 8px;
		}
		.side .empty {
			margin: 0;
			color: var(--muted);
		}
		.pick {
			display: flex;
			align-items: center;
			gap: 12px;
			margin: 0 -10px;
			padding: 8px 10px;
			min-height: var(--tap);
			border: 1px solid transparent;
			border-radius: 12px;
			background: none;
			text-align: left;
		}
		.pick:hover {
			border-color: var(--line-strong);
			background: var(--card);
		}
		.pick > span {
			display: flex;
			flex-direction: column;
		}
		.upick {
			flex-direction: column;
			align-items: flex-start;
			gap: 6px;
		}
		.pick > .sidesys {
			display: block;
			width: 150px;
		}
		.side .val {
			font-size: 15px;
			font-weight: 500;
		}
		.sm2 {
			font-size: 12px;
			color: var(--muted);
		}
		.side .strength {
			font-size: 12px;
			color: var(--muted);
		}
		.strip {
			display: none;
		}
		.line {
			justify-content: center;
			text-align: center;
			font-size: 15px;
		}
		.over {
			text-align: left;
		}
		.dockwrap {
			position: static;
			margin-top: 6px;
		}
		.stage[data-mode='play'] {
			padding-bottom: 24px;
		}
	}
</style>
