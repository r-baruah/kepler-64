<script lang="ts">
	import { FILES, type Occupant } from '$lib/chess';
	import { fieldTints } from '$lib/field';
	import type { Breakdown, Side } from '$lib/types';

	interface Props {
		placement: (Occupant | null)[];
		bottom: Side;
		/** The engine's field, or null to draw no field. */
		field: Breakdown['field'] | null;
		cones: { origin: number; hop1: number[]; hop2: number[] } | null;
		/** Hop-1 squares of the piece that just moved; its pull lines fade in once. */
		pulse?: { key: string; origin: number; hop1: number[] } | null;
		lastMove: { from: number; to: number } | null;
		/** Increases by one with every move played; the moved piece glides only when it goes up. */
		moveId?: number;
		selected: number | null;
		targets: Map<number, 'move' | 'capture'>;
		/** Square of a king in check. */
		check?: number | null;
		label: string;
		onselectown: (sq: number) => boolean;
		ontap: (sq: number) => void;
		onmove: (from: number, to: number) => void;
	}
	let {
		placement,
		bottom,
		field,
		cones,
		pulse = null,
		lastMove,
		moveId = 0,
		selected,
		targets,
		check = null,
		label,
		onselectown,
		ontap,
		onmove
	}: Props = $props();

	const SQ = 100;
	const flip = $derived(bottom === 'black');
	/** Top-left of a square in board units. */
	function pos(i: number): [number, number] {
		const f = i % 8;
		const r = Math.floor(i / 8);
		return flip ? [(7 - f) * SQ, r * SQ] : [f * SQ, (7 - r) * SQ];
	}
	const centre = (i: number): [number, number] => {
		const [x, y] = pos(i);
		return [x + SQ / 2, y + SQ / 2];
	};
	const hop1 = $derived(new Set(cones?.hop1 ?? []));
	const hop2 = $derived(new Set(cones?.hop2 ?? []));
	const squares = Array.from({ length: 64 }, (_, i) => i);
	const tints = $derived(field ? fieldTints({ field }) : []);

	// The moved piece glides from its old square. Only forward moves animate (not undo or a new game).
	let lastSeen = 0;
	let glide = $state<{ id: number; to: number; dx: number; dy: number } | null>(null);
	$effect.pre(() => {
		const id = moveId;
		const lm = lastMove;
		if (id > lastSeen && lm) {
			const [fx, fy] = pos(lm.from);
			const [tx, ty] = pos(lm.to);
			glide = { id, to: lm.to, dx: fx - tx, dy: fy - ty };
		} else if (id <= lastSeen) {
			glide = null;
		}
		lastSeen = id;
	});
	const pieces = $derived(
		placement.flatMap((o, i) => (o ? [{ i, key: glide && glide.to === i ? `g${glide.id}` : `s${i}`, img: o.color[0] + o.piece }] : []))
	);

	let svg: SVGSVGElement;
	let drag = $state<{ from: number; x: number; y: number; moved: boolean } | null>(null);
	let downSq = -1;
	let downPicked = false;
	let startX = 0;
	let startY = 0;

	function toSquare(e: PointerEvent): { sq: number; x: number; y: number } | null {
		const rect = svg.getBoundingClientRect();
		const x = ((e.clientX - rect.left) / rect.width) * 8 * SQ;
		const y = ((e.clientY - rect.top) / rect.height) * 8 * SQ;
		let f = Math.floor(x / SQ);
		const rr = Math.floor(y / SQ);
		if (f < 0 || f > 7 || rr < 0 || rr > 7) return { sq: -1, x, y };
		if (flip) f = 7 - f;
		const r = flip ? rr : 7 - rr;
		return { sq: r * 8 + f, x, y };
	}

	function down(e: PointerEvent) {
		const t = toSquare(e);
		if (!t || t.sq < 0) return;
		downSq = t.sq;
		startX = e.clientX;
		startY = e.clientY;
		downPicked = onselectown(t.sq);
		if (downPicked) {
			drag = { from: t.sq, x: t.x, y: t.y, moved: false };
			svg.setPointerCapture(e.pointerId);
		}
	}

	function move(e: PointerEvent) {
		if (!drag) return;
		const t = toSquare(e);
		if (!t) return;
		const far = Math.hypot(e.clientX - startX, e.clientY - startY) > 6;
		drag = { from: drag.from, x: t.x, y: t.y, moved: drag.moved || far };
	}

	function up(e: PointerEvent) {
		const t = toSquare(e);
		const wasDrag = drag?.moved ?? false;
		drag = null;
		if (!t || t.sq < 0) return;
		if (wasDrag && downPicked) {
			if (t.sq !== downSq) {
				glide = null;
				onmove(downSq, t.sq);
			}
		} else if (!downPicked) {
			ontap(t.sq);
		} else if (targets.has(t.sq) && t.sq !== downSq) {
			ontap(t.sq);
		}
	}

	function cancel() {
		drag = null;
	}
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<svg
	bind:this={svg}
	class="board"
	viewBox="0 0 800 800"
	role="img"
	aria-label={label}
	onpointerdown={down}
	onpointermove={move}
	onpointerup={up}
	onpointercancel={cancel}
>
	{#each squares as i (i)}
		{@const [x, y] = pos(i)}
		{@const dark = (Math.floor(i / 8) + (i % 8)) % 2 === 0}
		<rect {x} {y} width={SQ} height={SQ} fill={dark ? 'var(--sq-dark)' : 'var(--sq-light)'} />
	{/each}

	<!-- The field: each square tinted by which side pulls harder, deeper for a bigger difference.
	     Never dots: dots are move hints. -->
	{#each tints as t (t.sq)}
		{@const [x, y] = pos(t.sq)}
		<rect class="tint {t.side}" {x} {y} width={SQ} height={SQ} style:--s={t.strength} />
	{/each}

	{#if lastMove}
		{#each [lastMove.from, lastMove.to] as s (s)}
			{@const [x, y] = pos(s)}
			<rect {x} {y} width={SQ} height={SQ} class="last" />
		{/each}
	{/if}

	{#if check !== null}
		{@const [x, y] = pos(check)}
		<rect {x} {y} width={SQ} height={SQ} class="check" />
	{/if}

	{#if cones}
		{#each [...hop2].filter((s) => !hop1.has(s)) as s (s)}
			{@const [x, y] = pos(s)}
			<rect {x} {y} width={SQ} height={SQ} class="cone2" />
		{/each}
		{#each [...hop1] as s (s)}
			{@const [x, y] = pos(s)}
			<rect {x} {y} width={SQ} height={SQ} class="cone1" />
		{/each}
		{@const [ox, oy] = centre(cones.origin)}
		<g class="rays">
			{#each [...hop1] as s (s)}
				{@const [cx, cy] = centre(s)}
				<line x1={ox} y1={oy} x2={cx} y2={cy} />
			{/each}
		</g>
		{@const [sx, sy] = pos(cones.origin)}
		<rect x={sx} y={sy} width={SQ} height={SQ} class="sel" />
	{/if}

	{#if selected !== null && !cones}
		{@const [sx, sy] = pos(selected)}
		<rect x={sx} y={sy} width={SQ} height={SQ} class="sel" />
	{/if}

	{#if pulse && !cones}
		{#key pulse.key}
			{@const [ox, oy] = centre(pulse.origin)}
			<g class="pulse" aria-hidden="true">
				{#each pulse.hop1 as s (s)}
					{@const [cx, cy] = centre(s)}
					<line x1={ox} y1={oy} x2={cx} y2={cy} />
				{/each}
			</g>
		{/key}
	{/if}

	{#each pieces as p (p.key)}
		{@const [x, y] = pos(p.i)}
		<image
			class="piece"
			class:glide={glide !== null && p.key === `g${glide.id}`}
			style:--dx="{glide?.dx ?? 0}px"
			style:--dy="{glide?.dy ?? 0}px"
			href="/pieces/{p.img}.svg"
			x={x + SQ * 0.06}
			y={y + SQ * 0.05}
			width={SQ * 0.88}
			height={SQ * 0.88}
			opacity={drag?.moved && drag.from === p.i ? 0.35 : 1}
			pointer-events="none"
		/>
	{/each}

	{#each [...targets] as [s, kind] (s)}
		{@const [x, y] = pos(s)}
		{#if kind === 'move'}
			<circle cx={x + SQ / 2} cy={y + SQ / 2} r="14" class="hint" />
		{:else}
			<circle cx={x + SQ / 2} cy={y + SQ / 2} r="43" class="hint ring" />
		{/if}
	{/each}

	{#each squares as i (i)}
		{@const [x, y] = pos(i)}
		{@const dark = (Math.floor(i / 8) + (i % 8)) % 2 === 0}
		{#if x === 0}
			<text class="coord" class:ondark={dark} x={x + 7} y={y + 24}>{Math.floor(i / 8) + 1}</text>
		{/if}
		{#if y === 700}
			<text class="coord" class:ondark={dark} x={x + SQ - 7} y={y + SQ - 8} text-anchor="end">{FILES[i % 8]}</text>
		{/if}
	{/each}

	{#if drag?.moved}
		{@const o = placement[drag.from]}
		{#if o}
			<image href="/pieces/{o.color[0] + o.piece}.svg" x={drag.x - SQ * 0.55} y={drag.y - SQ * 0.55} width={SQ * 1.1} height={SQ * 1.1} pointer-events="none" />
		{/if}
	{/if}
</svg>

<style>
	.board {
		display: block;
		width: 100%;
		height: auto;
		aspect-ratio: 1;
		border-radius: 6px;
		box-shadow: 0 0 0 1px var(--board-edge);
		touch-action: pan-y;
		user-select: none;
		-webkit-user-select: none;
		cursor: default;
	}

	/* field */
	.tint {
		pointer-events: none;
		transition: opacity 0.35s var(--ease);
		animation: appear 0.35s var(--ease);
	}
	.tint.white {
		fill: var(--pull-w);
		opacity: calc(0.04 + (var(--pull-w-max) - 0.04) * var(--s));
	}
	.tint.black {
		fill: var(--pull-b);
		opacity: calc(0.03 + (var(--pull-b-max) - 0.03) * var(--s));
	}
	@keyframes appear {
		from {
			opacity: 0;
		}
	}

	/* markers: one soft fill each, the usual chess conventions */
	.last {
		fill: var(--last);
		opacity: var(--last-alpha);
	}
	.sel {
		fill: var(--select);
		opacity: 0.22;
	}
	.check {
		fill: var(--alert);
		opacity: 0.32;
	}
	.hint {
		fill: var(--hint);
		opacity: var(--hint-alpha);
	}
	.hint.ring {
		fill: none;
		stroke: var(--hint);
		stroke-width: 7;
	}
	.cone1 {
		fill: var(--cone);
		opacity: 0.16;
	}
	.cone2 {
		fill: var(--cone);
		opacity: 0.06;
	}
	.rays line,
	.pulse line {
		stroke: var(--cone);
		stroke-width: 3;
		stroke-linecap: round;
	}
	/* Dashed, like a drawn trajectory: the rays say where pull goes without competing with the pieces. */
	.rays line {
		opacity: 0.42;
		stroke-width: 2.5;
		stroke-dasharray: 9 9;
	}
	.pulse line {
		opacity: 0;
		animation: pulse 1.4s var(--ease) 1;
	}
	@keyframes pulse {
		25% {
			opacity: 0.5;
		}
		100% {
			opacity: 0;
		}
	}
	.piece.glide {
		animation: glide 0.2s var(--ease);
	}
	@keyframes glide {
		from {
			transform: translate(var(--dx), var(--dy));
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.pulse {
			display: none;
		}
		.piece.glide,
		.tint {
			animation: none;
			transition: none;
		}
	}
	.coord {
		font-family: var(--font-sans);
		font-size: 20px;
		font-weight: 500;
		fill: var(--coord-on-light);
		pointer-events: none;
	}
	.coord.ondark {
		fill: var(--coord-on-dark);
	}
</style>
