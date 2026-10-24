<script lang="ts">
	import type { GameStore } from '$lib/game.svelte';

	let { game }: { game: GameStore } = $props();
	const fieldLabel = $derived({ off: 'Off', pull: 'Pull', cones: 'Light cones' }[game.fieldMode]);
	const fieldShort = $derived({ off: 'Off', pull: 'Pull', cones: 'Cones' }[game.fieldMode]);
</script>

<nav class="dock" aria-label="Game controls">
	<button onclick={() => (game.sheet = 'new')}>New<span class="long">&nbsp;game</span></button>
	<button onclick={() => game.undo()} disabled={!game.canUndo}>Undo</button>
	<button class="wide" onclick={() => game.flip()}>Flip</button>
	<button class="on field" onclick={() => game.cycleField()} aria-label="Field layer: {fieldLabel}. Tap to change.">
		<span class="desk">Field: {fieldLabel}</span>
		<span class="phone"><small>Field</small>{fieldShort}</span>
	</button>
	<button onclick={() => (game.sheet = 'universe')}>Universe</button>
	<button class="wide" onclick={() => (game.sheet = 'level')}>Level</button>
	<button class="narrow" onclick={() => (game.sheet = 'more')}>More</button>
</nav>

<style>
	.dock {
		display: flex;
		gap: 2px;
		padding: 4px;
		width: 100%;
		max-width: 100%;
		background: var(--card);
		border: 1px solid var(--line);
		border-radius: var(--radius-pill);
		box-shadow: var(--shadow-dock);
	}
	button {
		flex: 1 1 0;
		min-width: 0;
		min-height: var(--tap);
		padding: 0 6px;
		border: none;
		border-radius: var(--radius-pill);
		background: none;
		font-size: 12.5px;
		color: var(--ink);
		white-space: nowrap;
	}
	button:hover:not(:disabled) {
		background: var(--soft);
	}
	button:disabled {
		color: var(--line-strong);
		cursor: default;
	}
	button.on {
		background: var(--ink);
		color: #fff;
	}
	button.on:hover {
		background: var(--ink-2) !important;
	}
	.wide,
	.long,
	.desk {
		display: none;
	}
	.phone {
		display: flex;
		flex-direction: column;
		line-height: 1.15;
	}
	.phone small {
		font-size: 9.5px;
		opacity: 0.7;
	}
	@media (min-width: 900px) {
		.dock {
			width: auto;
		}
		button {
			flex: none;
			padding: 0 16px;
			font-size: 13px;
		}
		.wide {
			display: inline;
		}
		.long,
		.desk {
			display: inline;
		}
		.phone {
			display: none;
		}
		.narrow {
			display: none;
		}
	}
</style>
