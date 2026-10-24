<script lang="ts">
	import type { HistoryItem } from '$lib/game.svelte';

	let { history, strip = false }: { history: HistoryItem[]; strip?: boolean } = $props();

	function score(h: HistoryItem): string {
		if (h.mover !== 'kepler') return '';
		if (h.mateIn !== null) return `#${Math.abs(h.mateIn)}`;
		if (h.scoreKu === null) return '';
		const v = h.scoreKu;
		return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
	}
</script>

{#if strip}
	<ol class="strip mono" aria-label="Moves">
		{#each history as h, i (i)}
			<li class:k={h.mover === 'kepler'}>
				{#if i % 2 === 0}<span class="n">{i / 2 + 1}</span>{/if}{h.san}
			</li>
		{/each}
	</ol>
{:else}
	<ol class="list mono" aria-label="Moves">
		{#each history as h, i (i)}
			{#if i % 2 === 0}<li class="n">{i / 2 + 1}</li>{/if}
			<li class:k={h.mover === 'kepler'} title={h.mover === 'kepler' ? 'Kepler’s score, in ku (Kepler units)' : undefined}>
				{h.san}{#if score(h)}<span class="sc">{score(h)}</span>{/if}
			</li>
		{/each}
	</ol>
{/if}

<style>
	ol {
		margin: 0;
		padding: 0;
		list-style: none;
		font-size: 13px;
	}
	.list {
		display: grid;
		grid-template-columns: 24px 1fr 1fr;
		row-gap: 6px;
		max-height: 300px;
		overflow: auto;
		align-content: start;
	}
	.list .n {
		color: #9aa1b0;
	}
	.k {
		color: var(--accent);
	}
	.sc {
		margin-left: 6px;
		font-size: 10.5px;
		color: var(--muted);
	}
	.strip {
		display: flex;
		gap: 12px;
		overflow-x: auto;
		white-space: nowrap;
		padding: 4px 0;
		font-size: 12.5px;
		scrollbar-width: none;
	}
	.strip .n {
		color: #9aa1b0;
		margin-right: 4px;
	}
</style>
