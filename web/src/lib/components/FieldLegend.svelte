<script lang="ts">
	// Key for the field layer. Swatches are drawn the same way the board draws the field.
	import type { FieldMode } from '$lib/game.svelte';

	let { mode, c, compact = false }: { mode: FieldMode; c: number; compact?: boolean } = $props();
</script>

<div class="legend" class:compact>
	{#if mode === 'pull'}
		<div class="keys">
			<span class="k"
				><svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"
					><rect width="18" height="18" class="sq" /><rect width="18" height="18" class="t w" /></svg
				>White pulls harder</span
			>
			<span class="k"
				><svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"
					><rect width="18" height="18" class="sq" /><rect width="18" height="18" class="t b" /></svg
				>Black pulls harder</span
			>
		</div>
		{#if !compact}
			<p>Deeper tint, stronger pull.</p>
		{/if}
	{:else if mode === 'cones'}
		<div class="keys">
			<span class="k"><svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><rect width="18" height="18" class="sq" /><rect width="18" height="18" class="c1" /></svg>1 move</span>
			<span class="k"><svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><rect width="18" height="18" class="sq" /><rect width="18" height="18" class="c2" /></svg>2 moves</span>
		</div>
		{#if !compact}<p>Tap a piece to see where its pull arrives. Light speed: {c} moves.</p>{/if}
	{:else}
		<p class="offline">Field hidden. The engine still uses it.</p>
	{/if}
</div>

<style>
	.legend {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.keys {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 16px;
	}
	.k {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		font-size: var(--t-sm);
		color: var(--ink-2);
	}
	.compact .k {
		font-size: var(--t-xs);
		color: var(--muted);
	}
	svg {
		flex: none;
		border-radius: 4px;
		box-shadow: 0 0 0 1px var(--line-strong);
	}
	.sq {
		fill: var(--sq-light);
	}
	.t.w {
		fill: var(--pull-w);
		opacity: var(--pull-w-max);
	}
	.t.b {
		fill: var(--pull-b);
		opacity: var(--pull-b-max);
	}
	.c1 {
		fill: var(--cone);
		opacity: 0.22;
	}
	.c2 {
		fill: var(--cone);
		opacity: 0.08;
	}
	p {
		margin: 0;
		font-size: var(--t-xs);
		line-height: 1.55;
		color: var(--muted);
	}
	.compact p {
		font-size: 12px;
	}
</style>
