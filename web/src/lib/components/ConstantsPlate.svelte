<script lang="ts">
	// "Constants of this universe": the preset's real numbers with plain names (src/lib/constants.ts).
	import { constantRows } from '$lib/constants';
	import type { PresetFacts } from '$lib/types';

	let { preset }: { preset: PresetFacts } = $props();
	const rows = $derived(constantRows(preset));
</script>

<div class="plate">
	<div class="head mono"><span>Constants of this universe</span><span>{preset.name}</span></div>
	<dl>
		{#each rows as r (r.key)}
			<div class="row" class:off={r.value === '0'}>
				<dt><span class="n">{r.name}</span><span class="m">{r.meaning}</span></dt>
				<dd class="mono">{r.value}</dd>
			</div>
		{/each}
	</dl>
</div>

<style>
	.plate {
		border: 1px solid var(--line);
		border-radius: var(--radius-card-sm);
		padding: 12px 14px 6px;
		background: var(--bg-2);
	}
	.head {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		font-size: var(--t-xs);
		font-weight: 500;
		color: var(--muted);
		padding-bottom: 8px;
		border-bottom: 1px solid var(--line);
	}
	dl {
		margin: 0;
	}
	.row {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 12px;
		padding: 7px 0;
		border-bottom: 1px solid var(--line);
	}
	.row:last-child {
		border-bottom: none;
	}
	dt {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.n {
		font-size: 13px;
		color: var(--ink);
	}
	.m {
		font-size: 11.5px;
		color: var(--muted);
		line-height: 1.35;
	}
	dd {
		margin: 0;
		font-size: 13px;
		font-weight: 500;
		white-space: nowrap;
	}
	.off dd {
		color: var(--muted);
		font-weight: 400;
	}
</style>
