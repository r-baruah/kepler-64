<script lang="ts">
	import type { GameStore } from '$lib/game.svelte';

	let { game, align = 'start' }: { game: GameStore; align?: 'start' | 'end' } = $props();
</script>

<div class="pgn" class:end={align === 'end'}>
	<button onclick={() => game.copyPgn()}>{game.pgnCopied ? 'Copied' : 'Copy PGN'}</button>
	<button onclick={() => game.downloadPgn()}>Download .pgn</button>
</div>

<style>
	.pgn {
		display: flex;
		flex-wrap: wrap;
		gap: 0 14px;
	}
	.pgn.end {
		justify-content: flex-end;
	}
	button {
		min-height: var(--tap);
		padding: 0 2px;
		border: none;
		background: none;
		font-size: var(--t-sm);
		color: var(--accent);
	}
	button:hover {
		text-decoration: underline;
	}
</style>
