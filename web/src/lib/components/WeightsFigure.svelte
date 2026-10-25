<script lang="ts">
	import { bodyRadius, oneDecimal, pieceWeights } from '$lib/field';
	import type { Constants } from '$lib/types';

	let { constants, universe, onchange }: { constants: Constants; universe: string; onchange: () => void } = $props();

	let measured = $state(500);
	// clientWidth is 0 before layout (and while hidden); keep the geometry finite.
	const width = $derived(measured > 40 ? measured : 500);
	const bodies = $derived(pieceWeights(constants));
	// Pawn radius scales with the figure so the queen (area about 7x) fits at any width.
	const pawn = $derived(Math.max(4, Math.min(5.5, width / 85)));
	const lineY = $derived(pawn * 3.6);
	const height = $derived(lineY + pawn * 3.2 + 34);
	const xOf = (i: number) => ((i + 0.5) / 5) * width;
	// The bodies sit on a shallow orbit arc (a circle through both ends of the figure).
	const sag = $derived(Math.min(22, width / 22));
	const arcR = $derived((width * width) / (8 * sag) + sag / 2);
	const arcY = (x: number) => lineY + (arcR - Math.sqrt(Math.max(0, arcR * arcR - (x - width / 2) ** 2)));
	const label = $derived('Learned weights: ' + bodies.map((b) => `${b.name.toLowerCase()} ${oneDecimal(b.weight)}`).join(', '));
</script>

<figure class="fig" bind:clientWidth={measured}>
	<figcaption class="cap">
		<span class="title mono">Weights in this universe</span>
		<button class="change" onclick={onchange}>{universe} <span aria-hidden="true">&rsaquo;</span></button>
	</figcaption>
	<svg {width} height={height + sag} viewBox="0 0 {width} {height + sag}" role="img" aria-label={label}>
		<path d="M0 {arcY(0)} A {arcR} {arcR} 0 0 1 {width} {arcY(width)}" class="orbit" />
		{#each bodies as b, i (b.letter)}
			{@const cy = arcY(xOf(i))}
			<circle cx={xOf(i)} cy={cy} r={bodyRadius(b.weight, pawn)} class="body" class:q={b.letter === 'Q'} />
			<text x={xOf(i)} y={cy + pawn * 3.2 + 12} text-anchor="middle" class="name">{b.name}</text>
			<text x={xOf(i)} y={cy + pawn * 3.2 + 28} text-anchor="middle" class="val">{oneDecimal(b.weight)}</text>
		{/each}
	</svg>
	<div class="reach">Area shows weight, pawn = 1. Light speed: pull travels at most {constants.c} moves.</div>
</figure>

<style>
	.fig {
		margin: 0;
		width: 100%;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.cap {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		font-size: 12px;
		color: var(--muted);
	}
	.title {
		font-size: var(--t-xs);
		font-weight: 500;
	}
	.change {
		min-height: var(--tap);
		padding: 0 2px;
		border: none;
		background: none;
		font-size: 13px;
		color: var(--accent);
		white-space: nowrap;
	}
	.change:hover {
		text-decoration: underline;
	}
	svg {
		display: block;
		overflow: visible;
	}
	.orbit {
		fill: none;
		stroke: var(--orn);
		stroke-width: 1;
	}
	.body {
		fill: var(--ink);
	}
	.body.q {
		fill: var(--accent);
	}
	.name {
		font-family: var(--font-sans);
		font-size: 12px;
		fill: var(--muted);
	}
	.val {
		font-family: var(--font-mono);
		font-size: 12px;
		font-weight: 500;
		fill: var(--ink);
	}
	.reach {
		font-size: 12px;
		color: var(--muted);
	}
</style>
