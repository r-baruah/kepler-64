<script lang="ts">
	import { oneDecimal } from '$lib/field';
	import type { KingReport } from '$lib/types';

	let { king, who }: { king: KingReport | null; who: 'Kepler' | 'You' } = $props();
	const possessive = $derived(who === 'Kepler' ? "Kepler's" : 'your');
	const total = $derived(king ? king.stress + king.binding : 0);
	const share = $derived(king && total > 0 ? (king.stress / total) * 100 : 0);
</script>

<div class="readout" aria-label="Pull on {possessive} king">
	{#if king}
		<span class="text">Pull on {possessive} king: enemy {oneDecimal(king.stress)} <span class="dot" aria-hidden="true">&middot;</span> own {oneDecimal(king.binding)}</span>
		<span class="bar" aria-hidden="true"><span class="enemy" style:width="{share}%"></span></span>
	{:else}
		<span class="text">&nbsp;</span>
	{/if}
</div>

<style>
	.readout {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 4px;
		font-family: var(--font-mono);
		font-size: 10.5px;
		color: var(--muted);
		min-width: 0;
	}
	.text {
		white-space: nowrap;
	}
	.bar {
		display: block;
		width: 100%;
		max-width: 160px;
		height: 2px;
		background: var(--line);
	}
	.enemy {
		display: block;
		height: 2px;
		background: var(--ink);
		min-width: 1px;
	}
</style>
