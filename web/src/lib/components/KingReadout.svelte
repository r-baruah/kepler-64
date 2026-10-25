<script lang="ts">
	// Pull on one king: enemy pull across its body against its own side's pull holding it
	// (KingReport.stress and .binding). A readout, not a meter: no threshold, no event.
	import { oneDecimal } from '$lib/field';
	import type { KingReport } from '$lib/types';

	let { king, who }: { king: KingReport | null; who: 'Kepler' | 'You' } = $props();
	const possessive = $derived(who === 'Kepler' ? "Kepler's" : 'your');
	const total = $derived(king ? king.stress + king.binding : 0);
	const share = $derived(king && total > 0 ? (king.stress / total) * 100 : 0);
</script>

<div class="readout" title="Pull on {possessive} king: enemy pull across its body, and its own side's pull holding it">
	{#if king}
		<span class="sr-only">Pull on {possessive} king: enemy {oneDecimal(king.stress)}, own {oneDecimal(king.binding)}.</span>
		<span class="vals" aria-hidden="true">
			<span class="k">King pull</span>
			<span class="pair"><span class="l">enemy</span><span class="n">{oneDecimal(king.stress)}</span></span>
			<span class="pair"><span class="l">own</span><span class="n">{oneDecimal(king.binding)}</span></span>
		</span>
		<span class="bar" aria-hidden="true"><span class="enemy" style:width="{share}%"></span></span>
	{/if}
</div>

<style>
	.readout {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 5px;
		min-width: 0;
		min-height: 28px;
	}
	.vals {
		display: flex;
		align-items: baseline;
		gap: 12px;
		white-space: nowrap;
	}
	.k {
		font-size: var(--t-xs);
		color: var(--faint);
	}
	.pair {
		display: inline-flex;
		align-items: baseline;
		gap: 5px;
	}
	.l {
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.n {
		font-size: var(--t-md);
		font-weight: 500;
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.bar {
		display: block;
		width: 100%;
		max-width: 168px;
		height: 3px;
		border-radius: 2px;
		background: var(--line-strong);
		overflow: hidden;
	}
	.enemy {
		display: block;
		height: 3px;
		background: var(--ink);
		min-width: 2px;
		transition: width 0.35s var(--ease);
	}
	@media (max-width: 380px) {
		.k {
			display: none;
		}
	}
</style>
