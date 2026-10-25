<script lang="ts">
	// The translation plate: four real physics ideas, each beside its chess version. Every chess-side
	// drawing is either the rules (which squares a piece reaches, where it is blocked) or the engine's
	// own numbers (the Home weights; the field crop comes from the engine's breakdown of FIELD_FEN).
	import { oneDecimal } from '$lib/field';
	import type { Breakdown, Constants } from '$lib/types';

	interface Props {
		/** Home's constants: the queen's weight and light speed are read from here. */
		constants: Constants;
		/** The engine's field for the field example (Landing's FIELD_FEN), once the engine has answered. */
		field: Breakdown['field'] | null;
	}
	let { constants, field }: Props = $props();

	const S = 12.5; // one square in the chess drawings
	const X0 = 16.75;
	const Y0 = 4;
	const sq = (f: number, r: number) => ({ x: X0 + f * S, y: Y0 + (4 - r) * S });
	const cells = Array.from({ length: 25 }, (_, i) => ({ f: i % 5, r: Math.floor(i / 5) }));
	const dark = (f: number, r: number) => (f + r) % 2 === 0;
	const centre = (f: number, r: number) => ({ x: X0 + (f + 0.5) * S, y: Y0 + (4 - r + 0.5) * S });

	// Gravity: a rook on c2 of the 5x5 crop, its own pawn on c4. The ray up stops on the pawn.
	const ROOK = { f: 2, r: 1 };
	const PAWN = { f: 2, r: 3 };
	const rookReach = [
		...[0, 1, 3, 4].map((f) => ({ f, r: 1 })),
		{ f: 2, r: 0 },
		{ f: 2, r: 2 },
		{ f: 2, r: 3 }
	];
	const rayEnds = [
		{ f: 0, r: 1 },
		{ f: 4, r: 1 },
		{ f: 2, r: 0 },
		{ f: 2, r: 3 }
	];

	// Light speed: a king in the middle; one move reaches the 8 squares around it, two moves the ring beyond.
	const ring = (f: number, r: number) => Math.max(Math.abs(f - 2), Math.abs(r - 2));

	// Field: the engine's field on files b-f, ranks 2-6 of FIELD_FEN (white knight c3, black bishop f5).
	const CROP_F0 = 1;
	const CROP_R0 = 1;
	// A sparse position has far less pull than a full board, so the tint is scaled to the largest
	// difference in this crop (the board's own scale would leave it nearly blank).
	const cropTints = $derived.by(() => {
		if (!field) return [];
		const cells: { sq: number; f: number; r: number; d: number }[] = [];
		for (let r = 0; r < 5; r++)
			for (let f = 0; f < 5; f++) {
				const sq = (r + CROP_R0) * 8 + f + CROP_F0;
				cells.push({ sq, f, r, d: field.white[sq] - field.black[sq] });
			}
		const max = Math.max(...cells.map((c) => Math.abs(c.d)));
		if (!(max > 0)) return [];
		return cells
			.filter((c) => Math.abs(c.d) >= 0.04 * max)
			.map((c) => ({ ...c, side: c.d > 0 ? 'white' : 'black', strength: Math.abs(c.d) / max }));
	});

	const qR = $derived(6.2 * Math.sqrt(constants.m_q));

	// Fallback for the progress thumb where CSS scroll timelines are missing: 0 at the first card, 1 at the last.
	const cssTimeline = typeof CSS !== 'undefined' && CSS.supports('animation-timeline: scroll()');
	let progress = $state(0);
	let rowFrame = 0;
	function onRowScroll(e: Event) {
		if (cssTimeline || rowFrame) return;
		const el = e.currentTarget as HTMLElement;
		rowFrame = requestAnimationFrame(() => {
			rowFrame = 0;
			const max = el.scrollWidth - el.clientWidth;
			progress = max > 0 ? Math.min(1, Math.max(0, el.scrollLeft / max)) : 0;
		});
	}
</script>

<div class="pp" style:--p={progress}>
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div class="plate" role="region" aria-label="Physics ideas and their chess versions" tabindex="0" data-pin-track onscroll={onRowScroll}>
	<!-- 1. Mass -->
	<figure class="pair">
		<div class="half">
			<svg viewBox="0 0 96 72" aria-hidden="true">
				<ellipse class="orbit" cx="44" cy="38" rx="36" ry="12" transform="rotate(-10 44 38)" />
				<circle class="body" cx="44" cy="38" r="12" />
				<circle class="body" cx="79" cy="31.5" r="3.6" />
			</svg>
			<div class="term">Mass</div>
		</div>
		<span class="to" aria-hidden="true"></span>
		<div class="half">
			<svg viewBox="0 0 96 72" role="img" aria-label="A pawn weighs 1, a queen {oneDecimal(constants.m_q)}">
				<circle class="ring" cx="22" cy="40" r="6.2" />
				<image href="/pieces/wP.svg" x="15" y="33" width="14" height="14" />
				<circle class="ring" cx="64" cy="36" r={qR} />
				<image href="/pieces/wQ.svg" x="53" y="25" width="22" height="22" />
				<text class="val" x="22" y="66" text-anchor="middle">1</text>
				<text class="val" x="64" y="66" text-anchor="middle">{oneDecimal(constants.m_q)}</text>
			</svg>
			<div class="term">Piece weight</div>
		</div>
		<figcaption>Learned by the engine. Pawn = 1.</figcaption>
	</figure>

	<!-- 2. Gravity -->
	<figure class="pair">
		<div class="half">
			<svg viewBox="0 0 96 72" aria-hidden="true">
				{#each Array.from({ length: 12 }, (_, i) => (i * Math.PI) / 6) as a (a)}
					<line class="fl" x1={48 + 10 * Math.cos(a)} y1={36 + 10 * Math.sin(a)} x2={48 + 30 * Math.cos(a)} y2={36 + 30 * Math.sin(a)} />
				{/each}
				<circle class="body" cx="48" cy="36" r="6" />
			</svg>
			<div class="term">Gravity</div>
		</div>
		<span class="to" aria-hidden="true"></span>
		<div class="half">
			<svg viewBox="0 0 96 72" role="img" aria-label="A rook pulls along its lines; the line up stops at its own pawn">
				{#each cells as c (c.f * 5 + c.r)}
					{@const p = sq(c.f, c.r)}
					<rect x={p.x} y={p.y} width={S} height={S} class={dark(c.f, c.r) ? 'sqd' : 'sql'} />
				{/each}
				{#each rookReach as c (c.f * 5 + c.r)}
					{@const p = sq(c.f, c.r)}
					<rect x={p.x} y={p.y} width={S} height={S} class="reach1" />
				{/each}
				{#each rayEnds as e (e.f * 5 + e.r)}
					{@const a = centre(ROOK.f, ROOK.r)}
					{@const b = centre(e.f, e.r)}
					<line class="ray" x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
				{/each}
				<image href="/pieces/wR.svg" x={sq(ROOK.f, ROOK.r).x} y={sq(ROOK.f, ROOK.r).y} width={S} height={S} />
				<image href="/pieces/wP.svg" x={sq(PAWN.f, PAWN.r).x} y={sq(PAWN.f, PAWN.r).y} width={S} height={S} />
			</svg>
			<div class="term">Pull along its moves</div>
		</div>
		<figcaption>Stops at the first piece in the way.</figcaption>
	</figure>

	<!-- 3. Light speed -->
	<figure class="pair">
		<div class="half">
			<svg viewBox="0 0 96 72" aria-hidden="true">
				<circle class="wave" cx="48" cy="36" r="14" />
				<circle class="wave far" cx="48" cy="36" r="27" />
				<circle class="src" cx="48" cy="36" r="3" />
			</svg>
			<div class="term">Light speed</div>
		</div>
		<span class="to" aria-hidden="true"></span>
		<div class="half">
			<svg viewBox="0 0 96 72" role="img" aria-label="A king's pull: one move to the squares around it, two moves to the ring beyond, nothing further">
				{#each cells as c (c.f * 5 + c.r)}
					{@const p = sq(c.f, c.r)}
					<rect x={p.x} y={p.y} width={S} height={S} class={dark(c.f, c.r) ? 'sqd' : 'sql'} />
					{#if ring(c.f, c.r) === 1}<rect x={p.x} y={p.y} width={S} height={S} class="reach1" />{/if}
					{#if ring(c.f, c.r) === 2}<rect x={p.x} y={p.y} width={S} height={S} class="reach2" />{/if}
				{/each}
				<image href="/pieces/wK.svg" x={sq(2, 2).x} y={sq(2, 2).y} width={S} height={S} />
				<text class="tag" x={sq(3, 2).x + S / 2} y={sq(3, 2).y + S / 2 + 2.6} text-anchor="middle">1</text>
				<text class="tag" x={sq(4, 2).x + S / 2} y={sq(4, 2).y + S / 2 + 2.6} text-anchor="middle">2</text>
			</svg>
			<div class="term">{constants.c} moves</div>
		</div>
		<figcaption>Pull reaches at most {constants.c} moves.</figcaption>
	</figure>

	<!-- 4. Field -->
	<figure class="pair">
		<div class="half">
			<svg viewBox="0 0 96 72" aria-hidden="true">
				<path class="side w" d="M4 6 H60 Q53 36 60 66 H4 Z" />
				<path class="side b" d="M60 6 H92 V66 H60 Q53 36 60 6 Z" />
				<path class="eq" d="M60 6 Q53 36 60 66" />
				<circle class="body" cx="28" cy="36" r="9" />
				<circle class="body" cx="76" cy="36" r="5" />
			</svg>
			<div class="term">Field</div>
		</div>
		<span class="to" aria-hidden="true"></span>
		<div class="half">
			<svg viewBox="0 0 96 72" role="img" aria-label="The engine's field around a white knight and a black bishop: each square tinted by the side that pulls harder">
				{#each cells as c (c.f * 5 + c.r)}
					{@const p = sq(c.f, c.r)}
					<rect x={p.x} y={p.y} width={S} height={S} class={dark(c.f + CROP_F0, c.r + CROP_R0) ? 'sqd' : 'sql'} />
				{/each}
				{#each cropTints as t (t.sq)}
					{@const p = sq(t.f, t.r)}
					<rect x={p.x} y={p.y} width={S} height={S} class="tint {t.side}" style:--s={t.strength} />
				{/each}
				<image href="/pieces/wN.svg" x={sq(1, 1).x} y={sq(1, 1).y} width={S} height={S} />
				<image href="/pieces/bB.svg" x={sq(4, 3).x} y={sq(4, 3).y} width={S} height={S} />
			</svg>
			<div class="term">Who owns a square</div>
		</div>
		<figcaption>Whoever pulls harder on it.</figcaption>
	</figure>
</div>
<!-- Phone: where the row is (four cards). A CSS scroll timeline drives it where supported. -->
<div class="track" aria-hidden="true"><span class="thumb"></span></div>
</div>

<style>
	/* Phone: one row of cards that scrolls sideways (the next card peeks in), so the four pairs
	 * take one card's height. From 600px: a 2 x 2 grid. */
	.plate {
		display: grid;
		grid-auto-flow: column;
		grid-auto-columns: 84%;
		gap: 10px;
		overflow-x: auto;
		scroll-snap-type: x mandatory;
		overscroll-behavior-x: contain;
		margin: 0 calc(-1 * var(--gutter));
		padding: 2px var(--gutter) 10px;
		scroll-padding: 0 var(--gutter);
		scrollbar-width: thin;
	}
	.pair {
		scroll-snap-align: start;
	}
	/* While pinned the page drives the row, so no snapping (it would fight every step). */
	:global(.pinned) .plate {
		scroll-snap-type: none;
	}
	/* The progress track: a short line with a thumb a quarter of its length. */
	.track {
		position: relative;
		width: 64px;
		height: 2px;
		margin: 4px auto 0;
		border-radius: 1px;
		background: var(--line-strong);
	}
	.thumb {
		position: absolute;
		left: 0;
		top: 0;
		width: 16px;
		height: 100%;
		border-radius: 1px;
		background: var(--accent);
		transform: translateX(calc(var(--p, 0) * 48px));
	}
	@supports (animation-timeline: scroll()) {
		.pp {
			timeline-scope: --cards;
		}
		.plate {
			scroll-timeline: --cards inline;
		}
		.thumb {
			animation: cards-thumb linear both;
			animation-timeline: --cards;
		}
	}
	@keyframes cards-thumb {
		from {
			transform: translateX(0);
		}
		to {
			transform: translateX(48px);
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.track {
			display: none;
		}
	}
	.pair {
		margin: 0;
		display: grid;
		grid-template-columns: 1fr 22px 1fr;
		align-items: start;
		padding: 14px 14px 12px;
		border-radius: var(--radius-card-sm);
		background: var(--card);
		box-shadow: var(--shadow-soft);
		background-image:
			linear-gradient(var(--grid) 1px, transparent 1px),
			linear-gradient(90deg, var(--grid) 1px, transparent 1px);
		background-size: 16px 16px;
		background-position: -1px -1px;
	}
	.half {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		min-width: 0;
	}
	svg {
		display: block;
		width: 100%;
		max-width: 132px;
		height: auto;
		overflow: visible;
	}
	.term {
		font-size: var(--t-md);
		font-weight: 500;
		color: var(--ink);
		text-align: center;
		line-height: 1.25;
	}
	.to {
		align-self: center;
		margin-top: -18px;
		height: 1px;
		background: var(--orn);
		position: relative;
	}
	.to::after {
		content: '';
		position: absolute;
		right: 0;
		top: -3px;
		border: 3.5px solid transparent;
		border-left: 5px solid var(--orn);
		border-right: 0;
	}
	figcaption {
		grid-column: 1 / -1;
		margin-top: 8px;
		padding-top: 8px;
		border-top: 1px solid var(--line);
		font-size: var(--t-sm);
		color: var(--muted);
		text-align: center;
	}

	.orbit {
		fill: none;
		stroke: var(--orn);
		stroke-width: 1;
	}
	.body {
		fill: var(--ink);
	}
	.ring {
		fill: var(--bg-2);
		stroke: var(--orn);
		stroke-width: 1;
	}
	.val {
		font-family: var(--font-mono);
		font-size: 9px;
		font-weight: 500;
		fill: var(--ink);
	}
	.fl {
		stroke: var(--orn);
		stroke-width: 1;
		stroke-dasharray: 2.5 2.5;
	}
	.wave {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1;
	}
	.wave.far {
		stroke-dasharray: 3 3;
		opacity: 0.7;
	}
	.src {
		fill: var(--accent);
	}
	.side.w {
		fill: var(--pull-w);
		opacity: calc(var(--pull-w-max) * 0.6);
	}
	.side.b {
		fill: var(--pull-b);
		opacity: calc(var(--pull-b-max) * 0.6);
	}
	.eq {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 1;
		stroke-dasharray: 3 2.5;
	}
	.sql {
		fill: var(--sq-light);
	}
	.sqd {
		fill: var(--sq-dark);
	}
	.reach1 {
		fill: var(--cone);
		opacity: 0.24;
	}
	.reach2 {
		fill: var(--cone);
		opacity: 0.09;
	}
	.ray {
		stroke: var(--cone);
		stroke-width: 1.1;
		opacity: 0.85;
	}
	.tag {
		font-family: var(--font-mono);
		font-size: 7px;
		font-weight: 500;
		fill: var(--ink);
	}
	.tint.white {
		fill: var(--pull-w);
		opacity: calc(0.1 + 0.42 * var(--s));
	}
	.tint.black {
		fill: var(--pull-b);
		opacity: calc(0.08 + 0.3 * var(--s));
	}

	@media (min-width: 600px) {
		.track {
			display: none;
		}
		.plate {
			grid-auto-flow: row;
			grid-template-columns: 1fr 1fr;
			grid-auto-columns: auto;
			gap: 14px;
			overflow: visible;
			margin: 0;
			padding: 0;
		}
	}
</style>
