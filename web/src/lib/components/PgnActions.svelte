<script lang="ts">
	import type { GameStore } from '$lib/game.svelte';

	interface Props {
		game: GameStore;
		align?: 'start' | 'end';
		/** One action per line, as in a small menu. */
		stack?: boolean;
		/** Called after the file download starts (a menu can close itself). */
		ondownload?: () => void;
	}
	let { game, align = 'start', stack = false, ondownload }: Props = $props();
</script>

<div class="pgn" class:end={align === 'end'} class:stack>
	<button onclick={() => game.copyPgn()}>{game.pgnCopied ? 'Copied' : 'Copy PGN'}</button>
	<button
		onclick={() => {
			game.downloadPgn();
			ondownload?.();
		}}>Download .pgn</button
	>
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
	.pgn.stack {
		flex-direction: column;
		align-items: stretch;
		gap: 0;
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
	.stack button {
		padding: 0 14px;
		text-align: left;
		white-space: nowrap;
	}
</style>
