<script lang="ts">
	import { bodyRadius, oneDecimal, pieceWeights } from '$lib/field';
	import type { Constants } from '$lib/types';

	let { constants, universe, onchange }: { constants: Constants; universe: string; onchange: () => void } = $props();

	let width = $state(500);
	const bodies = $derived(pieceWeights(constants));
	// Pawn radius scales with the figure so the queen (area 7.4x) fits at any width.
	const pawn = $derived(Math.max(4, Math.min(6, width / 83)));
	const lineY = $derived(pawn * 3.4);
	const height = $derived(lineY + 38);
	const label = $derived(
		'Learned weights: ' + bodies.map((b) => `${b.name.toLowerCase()} ${oneDecimal(b.weight)}`).join(', ')
	);
</script>

<figure class="fig" bind:clientWidth={width}>
	<figcaption class="cap">
		<span>How much each piece weighs in this universe</span>
		<button class="change" onclick={onchange}>Universe: {universe}</button>
	</figcaption>
	<svg {width} {height} viewBox="0 0 {width} {height}" role="img" aria-label={label}>
		<line x1="0" y1={lineY} x2={width} y2={lineY} stroke="var(--line-strong)" stroke-width="1" />
		{#each bodies as b, i (b.letter)}
			{@const cx = ((i + 0.5) / bodies.length) * width}
			<circle {cx} cy={lineY} r={bodyRadius(b.weight, pawn)} fill={b.letter === 'Q' ? 'var(--accent)' : 'var(--ink)'} />
			<text x={cx} y={lineY + pawn * 3.4 + 12} text-anchor="middle" class="val">{b.name} <tspan class="num">{oneDecimal(b.weight)}</tspan></text>
		{/each}
	</svg>
	<div class="reach">Pull reaches {constants.c} moves ahead; the dots on the board show it.</div>
</figure>

<style>
	.fig {
		margin: 0;
		width: 100%;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.cap {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: center;
		gap: 0 12px;
		font-size: 12px;
		color: var(--muted);
	}
	.change {
		min-height: var(--tap);
		padding: 0;
		border: none;
		background: none;
		font-size: 12px;
		color: var(--accent);
	}
	svg {
		display: block;
		overflow: visible;
	}
	.val {
		font-family: var(--font-sans);
		font-size: 12.5px;
		fill: var(--ink);
	}
	.num {
		font-family: var(--font-mono);
		font-size: 12px;
	}
	.reach {
		font-size: 12px;
		color: var(--muted);
	}
</style>
