<script lang="ts">
	import { FILES, type Occupant } from '$lib/chess';
	import type { Dot } from '$lib/field';
	import type { Side } from '$lib/types';

	interface Props {
		placement: (Occupant | null)[];
		bottom: Side;
		dots: Dot[];
		cones: { origin: number; hop1: number[]; hop2: number[] } | null;
		lastMove: { from: number; to: number } | null;
		selected: number | null;
		targets: Map<number, 'move' | 'capture'>;
		label: string;
		onselectown: (sq: number) => boolean;
		ontap: (sq: number) => void;
		onmove: (from: number, to: number) => void;
	}
	let { placement, bottom, dots, cones, lastMove, selected, targets, label, onselectown, ontap, onmove }: Props = $props();

	const SQ = 100;
	const flip = $derived(bottom === 'black');
	/** Pixel-free position of a square (top-left) in board units. */
	function pos(i: number): [number, number] {
		const f = i % 8;
		const r = Math.floor(i / 8);
		return flip ? [(7 - f) * SQ, r * SQ] : [f * SQ, (7 - r) * SQ];
	}
	const hop1 = $derived(new Set(cones?.hop1 ?? []));
	const hop2 = $derived(new Set(cones?.hop2 ?? []));
	const squares = Array.from({ length: 64 }, (_, i) => i);
	const pieces = $derived(
		placement.flatMap((o, i) => (o ? [{ i, key: o.color[0] + o.piece, ...o }] : []))
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
		let rr = Math.floor(y / SQ);
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
			if (t.sq !== downSq) onmove(downSq, t.sq);
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

	{#if lastMove}
		{#each [lastMove.from, lastMove.to] as s (s)}
			{@const [x, y] = pos(s)}
			<rect {x} {y} width={SQ} height={SQ} fill="var(--accent)" opacity="0.1" />
		{/each}
	{/if}

	{#each dots as d (d.sq)}
		{@const [x, y] = pos(d.sq)}
		{@const r = (d.size * SQ) / 2}
		{#if d.kind === 'solid'}
			<circle class="dot" cx={x + SQ / 2} cy={y + SQ / 2} {r} fill="var(--ink)" opacity={d.opacity} />
		{:else}
			<circle class="dot" cx={x + SQ / 2} cy={y + SQ / 2} r={r - 1.5} fill="none" stroke="var(--ink)" stroke-width="3" opacity={d.opacity + 0.04} />
		{/if}
	{/each}

	{#if cones}
		{#each [...hop2].filter((s) => !hop1.has(s)) as s (s)}
			{@const [x, y] = pos(s)}
			<rect class="cone2" {x} {y} width={SQ} height={SQ} fill="var(--accent)" opacity="0.06" />
		{/each}
		{#each [...hop1] as s (s)}
			{@const [x, y] = pos(s)}
			<rect class="cone1" {x} {y} width={SQ} height={SQ} fill="var(--accent)" opacity="0.2" />
		{/each}
		{@const [ox, oy] = pos(cones.origin)}
		<rect x={ox + 2} y={oy + 2} width={SQ - 4} height={SQ - 4} fill="none" stroke="var(--accent)" stroke-width="4" />
	{/if}

	{#if selected !== null && !cones}
		{@const [sx, sy] = pos(selected)}
		<rect x={sx + 2} y={sy + 2} width={SQ - 4} height={SQ - 4} fill="none" stroke="var(--accent)" stroke-width="4" />
	{/if}

	{#each pieces as p (p.i)}
		{@const [x, y] = pos(p.i)}
		<image
			href="/pieces/{p.key}.svg"
			x={x + SQ * 0.06}
			y={y + SQ * 0.04}
			width={SQ * 0.88}
			height={SQ * 0.88}
			opacity={drag?.moved && drag.from === p.i ? 0.35 : 1}
			pointer-events="none"
		/>
	{/each}

	{#each [...targets] as [s, kind] (s)}
		{@const [x, y] = pos(s)}
		{#if kind === 'move'}
			<circle cx={x + SQ / 2} cy={y + SQ / 2} r="15" fill="var(--accent)" opacity="0.85" />
		{:else}
			<circle cx={x + SQ / 2} cy={y + SQ / 2} r="44" fill="none" stroke="var(--accent)" stroke-width="6" opacity="0.85" />
		{/if}
	{/each}

	{#each squares as i (i)}
		{@const [x, y] = pos(i)}
		{#if pos(i)[0] === 0}
			<text class="coord" x={x + 7} y={y + 24}>{Math.floor(i / 8) + 1}</text>
		{/if}
		{#if pos(i)[1] === 700}
			<text class="coord" x={x + SQ - 7} y={y + SQ - 8} text-anchor="end">{FILES[i % 8]}</text>
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
		border: 1px solid var(--ink);
		box-sizing: border-box;
		touch-action: pan-y;
		user-select: none;
		-webkit-user-select: none;
		cursor: default;
	}
	.coord {
		font-family: var(--font-mono);
		font-size: 18px;
		fill: var(--coord);
		pointer-events: none;
	}
</style>
