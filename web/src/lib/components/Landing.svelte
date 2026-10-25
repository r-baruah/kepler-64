<script lang="ts">
	import { onDestroy } from 'svelte';
	import Board from '$lib/components/Board.svelte';
	import PhysicsPlate from '$lib/components/PhysicsPlate.svelte';
	import Sheet from '$lib/components/Sheet.svelte';
	import UniverseSystem from '$lib/components/UniverseSystem.svelte';
	import { parsePlacement, sqIndex } from '$lib/chess';
	import { EngineClient } from '$lib/engine/client';
	import { reveal } from '$lib/reveal';
	import type { Breakdown, LedgerCard, SiteFacts } from '$lib/types';

	interface Props {
		facts: SiteFacts;
		/** True once the page's own engine has loaded; the demo boards then ask the engine for their numbers. */
		ready: boolean;
		/** Home's generation file, loaded by the demo engine. */
		generationUrl: string;
		/** Start a game in this universe. */
		onplay: (slug: string) => void;
	}
	let { facts, ready, generationUrl, onplay }: Props = $props();

	const landing = $derived(facts.landing);
	const home = $derived(facts.presets[0]);
	const fmt = (n: number) => n.toLocaleString('en-US');

	// ---- 1. the worked example: a rook stops at its own pawn ------------------------------
	// The board shows only the rook and the pawn. The engine needs both kings to accept a position, so
	// they stand far from the rook's lines (a8, h8) and do not change its squares.
	const SHOWN_FEN = '8/8/8/8/3P4/8/3R4/8 w - - 0 1';
	const DEMO_FEN = 'K6k/8/8/8/3P4/8/3R4/8 w - - 0 1';
	// The field example in the plate: a white knight on c3 and a black bishop on f5 (kings in far corners).
	const FIELD_FEN = 'k7/8/8/8/5b2/2N5/8/7K w - - 0 1';
	const ROOK_SQ = sqIndex('d2');
	const demoPlacement = parsePlacement(SHOWN_FEN);
	let hop1 = $state.raw<number[] | null>(null);
	let plateField = $state.raw<Breakdown['field'] | null>(null);
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
				plateField = (await client.reset(FIELD_FEN, [])).breakdown.field;
				client.stop();
			} catch (e) {
				console.warn('landing demo', e);
			}
		})();
	});
	onDestroy(() => demoClient?.stop());

	// ---- 2. no chess knowledge inside ------------------------------------------------
	const proof = $derived(landing.proof);
	const ABSENT = ['Piece-square tables', 'King-safety rules', 'Pawn-structure rules', 'An opening book', 'A neural network trained on chess games'];

	// ---- 3. the ledger ---------------------------------------------------------------
	const BADGE: Record<LedgerCard['verdict'], string> = { kept: 'Kept', back: 'Came back', rejected: 'Rejected', 'rejected-twice': 'Rejected twice' };
	const kept = $derived(landing.ledger.filter((c) => c.universes.length === 0));
	const cut = $derived(landing.ledger.filter((c) => c.universes.length > 0));
	const presetBy = $derived(Object.fromEntries(facts.presets.map((p) => [p.slug, p])));

	// ---- 4. cite ---------------------------------------------------------------------
	let citeOpen = $state(false);
	let copied = $state(false);
	async function copyCite() {
		try {
			await navigator.clipboard.writeText(facts.cite.bibtex);
			copied = true;
			setTimeout(() => (copied = false), 1800);
		} catch {
			copied = false;
		}
	}
</script>

<div class="landing">
	<section class="sec" id="how" use:reveal>
		<header class="head">
			<div class="eyebrow mono"><span class="dot"></span>01 &middot; How it works</div>
			<h2>Physics, rebuilt for a chessboard.</h2>
			<p class="lede">We wrote the laws. It found the constants itself, in {fmt(facts.selfPlayGames)} games of self-play.</p>
		</header>
		<div class="how">
			<PhysicsPlate constants={home.constants} field={plateField} />
			<figure class="example">
				<div class="exboard">
					<Board
						placement={demoPlacement}
						bottom="white"
						field={null}
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
				<figcaption><span class="mono k">Worked example</span> The rook pulls up to its own pawn, and no further.</figcaption>
			</figure>
		</div>
		<p class="namenote">Named after Johannes Kepler, who found the planets' laws in observations. Kepler-64 finds its constants in its own games.</p>
	</section>

	<section class="sec" id="inside" use:reveal>
		<header class="head">
			<div class="eyebrow mono"><span class="dot"></span>02 &middot; What's inside</div>
			<h2>It knows space, not chess.</h2>
		</header>
		<div class="inside">
			<div class="cols">
				<div class="col">
					<div class="colhead mono">Not inside</div>
					<ul class="absent">
						{#each ABSENT as a (a)}
							<li><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5.5" /><path d="M3.2 10.8l7.6-7.6" /></svg>{a}</li>
						{/each}
					</ul>
				</div>
				<div class="col">
					<div class="colhead mono">Inside</div>
					<ul class="has">
						<li>
							<svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true"
								><rect x="2" y="2" width="24" height="24" rx="2" class="g" /><path d="M2 10h24M2 18h24M10 2v24M18 2v24" class="g" /><rect x="10" y="10" width="8" height="8" class="o" /><path d="M14 14h12M14 14V2M14 14H2" class="r" /></svg
							>One field
						</li>
						<li>
							<svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true"
								><ellipse cx="14" cy="15" rx="12" ry="5" class="g" /><circle cx="4" cy="16" r="1.5" class="b" /><circle cx="10" cy="19.6" r="2.3" class="b" /><circle cx="17" cy="19.8" r="2.6" class="b" /><circle cx="24.5" cy="16.5" r="3.4" class="q" /></svg
							>About a dozen learned numbers
						</li>
					</ul>
				</div>
			</div>
			<div class="proof">
				<div class="big">{proof.scorePercent}<span>%</span></div>
				<div class="prooftext">
					<p class="proofline">
						Gen {proof.generation} scored {proof.scorePercent}% against {proof.opponent} limited to its {proof.setting} setting, at {proof.baseSeconds} seconds plus
						{proof.incrementSeconds} per move, over {proof.games} games.
					</p>
					<div class="wdl" role="img" aria-label="{proof.w} wins, {proof.d} draws, {proof.l} losses">
						<span class="w" style:flex-grow={proof.w}></span><span class="d" style:flex-grow={proof.d}></span><span class="l" style:flex-grow={proof.l}></span>
					</div>
					<div class="wdlkeys mono" aria-hidden="true"><span>{proof.w} won</span><span>{proof.d} drawn</span><span>{proof.l} lost</span></div>
				</div>
				<p class="fine">{proof.setting} is Stockfish's own setting, not a human rating. The opponent on this page plays a fixed search size; its strength isn't measured yet.</p>
			</div>
			<button class="try" onclick={() => onplay(home.slug)}>Try beating it <span aria-hidden="true">&rarr;</span></button>
		</div>
	</section>

	<section class="sec" id="ledger" use:reveal>
		<header class="head">
			<div class="eyebrow mono"><span class="dot"></span>03 &middot; The Law Ledger</div>
			<h2>Every law has to earn its place.</h2>
			<p class="lede">Each law played against a copy of the engine without it. Laws that helped stayed. The rest became alternate universes.</p>
		</header>
		<div class="keptrow">
			{#each kept as c (c.id)}
				<div class="kcard">
					<span class="badge {c.verdict}">{BADGE[c.verdict]}</span>
					<div class="ct">{c.title}</div>
					<div class="cn">{c.note}</div>
				</div>
			{/each}
		</div>
		<div class="cutrow">
			{#each cut as c (c.id)}
				<div class="ucard" class:wide={c.universes.length > 1}>
					<div class="uhead">
						<span class="badge {c.verdict}">{BADGE[c.verdict]}</span>
						<div class="ct">{c.title}</div>
						<div class="cn">{c.note}</div>
					</div>
					<div class="unis">
						{#each c.universes as slug (slug)}
							{@const p = presetBy[slug]}
							<div class="uni">
								<UniverseSystem constants={p.constants} slug={p.slug} name={p.name} />
								<div class="unifoot">
									<span class="uname">{p.name}</span>
									<button class="play" onclick={() => onplay(p.slug)}>Play {p.name} <span aria-hidden="true">&rarr;</span></button>
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/each}
		</div>
	</section>

	<section class="open" use:reveal aria-labelledby="open-h">
		<div class="plate">
			<div class="opentext">
				<svg width="32" height="32" viewBox="0 0 20 20" aria-hidden="true"
					><circle cx="10" cy="10" r="4" fill="currentColor" /><ellipse cx="10" cy="10" rx="9" ry="3.4" fill="none" stroke="currentColor" stroke-width="1.2" transform="rotate(-20 10 10)" /></svg
				>
				<h2 id="open-h">Open source, all of it.</h2>
				<p>The engine, the training runs and every number on this page are on GitHub.</p>
				<a class="problems" href="https://github.com/r-baruah/kepler-64/blob/v1/docs/v1/11_OPEN_PROBLEMS.md">Open problems <span aria-hidden="true">&rarr;</span></a>
				<div class="openbtns">
					<a class="b1" href="https://github.com/r-baruah/kepler-64">Star on GitHub</a>
					<a class="b2" href="https://github.com/r-baruah/kepler-64">View the code</a>
					<button class="b2" onclick={() => (citeOpen = true)}>Cite</button>
				</div>
			</div>
			<!-- The photograph, in the plate's two tones (see .duo). Its sky is the plate's own colour, so the
			     photo has no visible edge on the sky side; the text never sits over the Moon or the Earth. -->
			<div class="photo">
				<div class="duo">
					<picture>
						<source type="image/webp" srcset="/images/earthrise-as17-152-23272-800.webp 800w, /images/earthrise-as17-152-23272.webp 1280w" sizes="(min-width: 700px) 60vw, 130vw" />
						<img
							src="/images/earthrise-as17-152-23272.jpg"
							alt="A crescent Earth rising over the grey, cratered horizon of the Moon, seen from Apollo 17 in lunar orbit."
							width="1280"
							height="1280"
							loading="lazy"
							decoding="async"
						/>
					</picture>
				</div>
			</div>
		</div>
		<p class="platecap mono">Earth rising over the Moon, Apollo 17, 1972&nbsp;&middot;&nbsp;NASA</p>
	</section>

	<footer class="foot">
		<span>Made by <a href="https://ripuranjan.com">Ripuranjan Baruah</a> &middot; also building Gambits</span>
		<a href="https://github.com/r-baruah/kepler-64">GitHub</a>
	</footer>
</div>

{#if citeOpen}
	<Sheet title="Cite Kepler-64" onclose={() => (citeOpen = false)}>
		<pre class="bib mono">{facts.cite.bibtex}</pre>
		<button class="copy" onclick={copyCite} aria-live="polite">{copied ? 'Copied' : 'Copy BibTeX'}</button>
	</Sheet>
{/if}

<style>
	.landing {
		max-width: 1280px;
		margin: 0 auto;
		padding: 0 var(--gutter);
	}
	.sec {
		scroll-margin-top: 64px;
		padding: 40px 0;
		display: flex;
		flex-direction: column;
		gap: 20px;
		border-top: 1px solid var(--line);
	}
	.head {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.eyebrow {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: var(--t-xs);
		font-weight: 500;
		color: var(--accent);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.dot {
		width: 9px;
		height: 9px;
		border-radius: 50%;
		border: 1.5px solid currentColor;
		position: relative;
	}
	.dot::after {
		content: '';
		position: absolute;
		inset: 1.5px;
		border-radius: 50%;
		background: currentColor;
	}
	h2 {
		margin: 0;
		font-size: 32px;
		font-weight: 300;
		letter-spacing: -0.035em;
		line-height: 1.08;
		text-wrap: balance;
	}
	p {
		margin: 0;
		font-size: 16px;
		line-height: 1.5;
		color: var(--ink-2);
	}
	.lede {
		max-width: 34em;
	}

	/* 1. how it works */
	.how {
		display: flex;
		flex-direction: column;
		gap: 18px;
	}
	.example {
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 14px;
		background: var(--card);
		border-radius: var(--radius-card);
		box-shadow: var(--shadow-soft);
	}
	.exboard {
		width: min(100%, 260px);
		align-self: center;
	}
	.example figcaption {
		font-size: var(--t-sm);
		color: var(--ink-2);
		line-height: 1.45;
		text-align: center;
	}
	.k {
		display: block;
		font-size: 11px;
		color: var(--accent);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		margin-bottom: 2px;
	}
	.namenote {
		font-size: var(--t-sm);
		color: var(--muted);
		max-width: 40em;
	}

	/* 2. inside */
	.inside {
		display: flex;
		flex-direction: column;
		gap: 20px;
		padding: 20px;
		background: var(--card);
		border-radius: var(--radius-card);
		box-shadow: var(--shadow-card);
	}
	.cols {
		display: grid;
		grid-template-columns: 1fr;
		gap: 20px;
	}
	.colhead {
		font-size: 11px;
		font-weight: 500;
		color: var(--muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		margin-bottom: 10px;
	}
	ul {
		margin: 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 9px;
	}
	.absent li {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: var(--t-md);
		color: var(--muted);
	}
	.absent svg {
		flex: none;
		fill: none;
		stroke: var(--faint);
		stroke-width: 1.3;
	}
	.has {
		gap: 12px;
	}
	.has li {
		display: flex;
		align-items: center;
		gap: 12px;
		font-size: var(--t-lg);
		color: var(--ink);
	}
	.has svg {
		flex: none;
	}
	.has .g {
		fill: none;
		stroke: var(--orn);
		stroke-width: 1;
	}
	.has .o {
		fill: var(--pull-w);
		opacity: 0.45;
	}
	.has .r {
		stroke: var(--accent);
		stroke-width: 1.4;
		fill: none;
	}
	.has .b {
		fill: var(--ink);
	}
	.has .q {
		fill: var(--accent);
	}
	.proof {
		display: grid;
		grid-template-columns: auto 1fr;
		column-gap: 18px;
		row-gap: 12px;
		align-items: center;
		padding-top: 18px;
		border-top: 1px solid var(--line);
	}
	.big {
		font-size: 56px;
		font-weight: 300;
		letter-spacing: -0.04em;
		line-height: 1;
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.big span {
		font-size: 0.5em;
		margin-left: 2px;
		color: var(--ink-2);
	}
	.prooftext {
		display: flex;
		flex-direction: column;
		gap: 8px;
		min-width: 0;
	}
	.proofline {
		font-size: var(--t-md);
		line-height: 1.45;
		color: var(--ink);
	}
	.wdl {
		display: flex;
		gap: 2px;
		height: 6px;
	}
	.wdl span {
		flex-basis: 0;
		border-radius: 2px;
		min-width: 3px;
	}
	.wdl .w {
		background: var(--accent);
	}
	.wdl .d {
		background: var(--orn);
	}
	.wdl .l {
		background: var(--ink);
		opacity: 0.8;
	}
	.wdlkeys {
		display: flex;
		justify-content: space-between;
		font-size: 11px;
		color: var(--muted);
	}
	.fine {
		grid-column: 1 / -1;
		font-size: var(--t-sm);
		line-height: 1.5;
		color: var(--muted);
	}
	.try {
		align-self: flex-start;
		min-height: 48px;
		padding: 0 22px;
		border-radius: var(--radius-pill);
		border: 1px solid var(--ink);
		background: var(--ink);
		color: var(--on-ink);
		font-size: 15px;
		font-weight: 500;
	}
	.try:hover {
		background: var(--ink-2);
	}

	/* 3. ledger */
	.keptrow {
		display: grid;
		grid-template-columns: 1fr;
		gap: 10px;
	}
	.kcard {
		background: var(--card);
		border-radius: var(--radius-card-sm);
		padding: 12px 16px;
		display: flex;
		flex-direction: column;
		gap: 6px;
		box-shadow: var(--shadow-soft);
	}
	.ct {
		font-size: var(--t-lg);
		color: var(--ink);
		line-height: 1.3;
	}
	.cn {
		font-size: var(--t-sm);
		color: var(--muted);
	}
	.badge {
		align-self: flex-start;
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		border-radius: var(--radius-pill);
		padding: 3px 9px;
		color: var(--on-ink);
		background: var(--ink);
	}
	.badge.back {
		background: var(--accent);
	}
	.badge.rejected,
	.badge.rejected-twice {
		background: none;
		color: var(--muted);
		border: 1px dashed var(--faint);
		padding: 2px 8px;
	}
	.cutrow {
		display: grid;
		grid-template-columns: 1fr;
		gap: 10px;
	}
	.ucard {
		border: 1px dashed var(--line-strong);
		border-radius: var(--radius-card);
		padding: 16px 18px 14px;
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.uhead {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.unis {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: 10px;
	}
	.uni {
		display: flex;
		flex-direction: column;
		gap: 8px;
		padding: 8px 10px 2px;
		border-radius: var(--radius-card-sm);
		background: var(--card);
	}
	.unifoot {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 0 8px;
		border-top: 1px solid var(--line);
	}
	.uname {
		font-size: var(--t-md);
		font-weight: 500;
	}
	.play {
		min-height: var(--tap);
		padding: 0 2px;
		border: none;
		background: none;
		color: var(--accent);
		font-size: var(--t-md);
		font-weight: 500;
	}
	.play:hover {
		text-decoration: underline;
	}

	/* 4. open source: a photo plate. The NASA photograph (Earthrise, AS17-152-23272) is set in two
	   tones from the tokens: black maps to --plate-lo, white to --plate-hi. The sky is pure black in the
	   original, so it comes out exactly the plate's background and the text sits on it, never on the
	   Moon or the Earth. Phone: text above the photo. From 700px: text left, photo right. */
	.open {
		margin: 16px 0;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.plate {
		position: relative;
		overflow: hidden;
		border-radius: 24px;
		background: var(--plate-lo);
		color: var(--band-ink);
		/* the plate mount: a thin rule just outside the plate */
		outline: 1px solid var(--line-strong);
		outline-offset: 5px;
		display: flex;
		flex-direction: column;
	}
	/* the plate's edge rule, drawn over the photo too */
	.plate::after {
		content: '';
		position: absolute;
		inset: 0;
		z-index: 2;
		border-radius: inherit;
		box-shadow: inset 0 0 0 1px var(--line);
		pointer-events: none;
	}
	.opentext {
		position: relative;
		z-index: 1;
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding: 28px 22px 4px;
	}
	.open h2 {
		font-size: 28px;
		letter-spacing: -0.03em;
	}
	.opentext p {
		color: var(--band-muted);
		font-size: 16px;
		max-width: 460px;
	}
	.problems {
		align-self: flex-start;
		color: var(--band-ink);
		font-size: var(--t-md);
		min-height: var(--tap);
		display: inline-flex;
		align-items: center;
	}
	.problems:hover {
		color: var(--band-ink);
		opacity: 0.8;
	}
	.openbtns {
		display: flex;
		flex-direction: column;
		gap: 10px;
		margin-top: 6px;
	}
	.openbtns a,
	.openbtns button {
		display: flex;
		align-items: center;
		justify-content: center;
		text-decoration: none;
		font-size: 15px;
		border-radius: var(--radius-pill);
		min-height: 48px;
		padding: 0 22px;
		background: none;
	}
	.b1 {
		background: var(--band-ink) !important;
		color: var(--plate-lo);
		min-height: 52px !important;
	}
	.b1:hover {
		color: var(--plate-lo);
		opacity: 0.9;
	}
	.b2 {
		border: 1px solid color-mix(in srgb, var(--band-ink) 40%, transparent);
		color: var(--band-ink);
	}
	.b2:hover {
		color: var(--band-ink);
		border-color: var(--band-ink);
	}
	/* The photo window. --ex/--ey place the Earth's centre in the window; --s is the photo's width. */
	.photo {
		position: relative;
		overflow: hidden;
		aspect-ratio: 4 / 3;
		--s: 125%;
		--ex: 46%;
		--ey: 62%;
	}
	.duo {
		position: absolute;
		left: var(--ex);
		top: var(--ey);
		width: var(--s);
		aspect-ratio: 1;
		/* the Earth's centre in the photograph, as a fraction of its width and height */
		transform: translate(-47.5%, -58%);
		/* Duotone: a grey photo multiplied onto the light tone, then the dark tone screened on top.
		   Black becomes --plate-lo, white about --plate-hi. */
		background: var(--plate-hi);
		isolation: isolate;
	}
	.duo::after {
		content: '';
		position: absolute;
		inset: 0;
		background: var(--plate-lo);
		mix-blend-mode: screen;
		pointer-events: none;
	}
	.duo img {
		display: block;
		width: 100%;
		height: 100%;
		/* contrast() also takes the film's near-black sky (at most 11/255) to pure black, so the sky
		   meets the plate with no seam */
		filter: grayscale(1) contrast(1.14);
		mix-blend-mode: multiply;
	}
	.platecap {
		margin: 4px 0 0;
		padding: 0 6px;
		font-size: 11.5px;
		line-height: 1.5;
		color: var(--muted);
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

	.bib {
		margin: 0;
		padding: 14px;
		border-radius: var(--radius-card-sm);
		background: var(--bg-2);
		border: 1px solid var(--line);
		font-size: 12px;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
		color: var(--ink);
	}
	.copy {
		min-height: 48px;
		border-radius: var(--radius-pill);
		border: 1px solid var(--ink);
		background: var(--ink);
		color: var(--on-ink);
		font-size: 15px;
		font-weight: 500;
	}

	@media (min-width: 700px) {
		.plate {
			display: grid;
			grid-template-columns: minmax(0, 5fr) minmax(0, 6fr);
		}
		.opentext {
			padding: 36px 28px 36px 32px;
		}
		.photo {
			aspect-ratio: auto;
			/* sized by the text beside it; cqw/cqh below are this window's own width and height */
			container-type: size;
			--s: max(110cqw, 130cqh);
			--ex: 45%;
			--ey: 60%;
		}
		.keptrow {
			grid-template-columns: repeat(3, minmax(0, 1fr));
		}
		.unis {
			grid-template-columns: 1fr 1fr;
		}
	}

	@media (min-width: 900px) {
		.landing {
			padding: 0 48px;
		}
		.sec {
			padding: 88px 0;
			gap: 36px;
		}
		.head {
			display: grid;
			grid-template-columns: minmax(0, 1fr) minmax(0, 440px);
			grid-template-areas: 'eyebrow eyebrow' 'h2 lede';
			column-gap: 64px;
			row-gap: 16px;
			align-items: end;
		}
		.eyebrow {
			grid-area: eyebrow;
		}
		.head h2 {
			grid-area: h2;
		}
		.lede {
			grid-area: lede;
			font-size: 18px;
			line-height: 1.55;
		}
		h2 {
			font-size: 48px;
			letter-spacing: -0.04em;
			line-height: 1.05;
		}
		.how {
			display: grid;
			grid-template-columns: minmax(0, 1fr) 360px;
			gap: 20px;
			align-items: stretch;
		}
		.exboard {
			width: min(100%, 300px);
		}
		.example {
			padding: 22px;
			justify-content: center;
		}
		.inside {
			display: grid;
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
			grid-template-areas: 'cols proof' 'try proof';
			column-gap: 48px;
			padding: 32px;
		}
		.cols {
			grid-area: cols;
			grid-template-columns: 1fr 1fr;
			gap: 28px;
		}
		.proof {
			grid-area: proof;
			align-self: start;
			padding-top: 0;
			padding-left: 40px;
			border-top: none;
			border-left: 1px solid var(--line);
		}
		.big {
			font-size: 76px;
		}
		.try {
			grid-area: try;
			align-self: end;
			justify-self: start;
		}
		.keptrow {
			gap: 16px;
		}
		.kcard {
			padding: 22px;
		}
		.cutrow {
			grid-template-columns: 1fr 2fr;
			gap: 16px;
		}
		.ucard {
			padding: 22px;
		}
		.ucard:not(.wide) .unis {
			grid-template-columns: 1fr;
		}
		.plate {
			border-radius: 28px;
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		}
		.opentext {
			padding: 56px 24px 56px 56px;
		}
		.open h2 {
			font-size: 40px;
			letter-spacing: -0.035em;
		}
		.opentext p {
			font-size: 17px;
			line-height: 1.55;
		}
		.openbtns {
			flex-direction: row;
			flex-wrap: wrap;
			margin-top: 14px;
		}
		.openbtns a,
		.openbtns button {
			padding: 0 20px;
		}
		.platecap {
			font-size: 12px;
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
