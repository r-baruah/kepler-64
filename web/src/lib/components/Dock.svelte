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
	<button class="field" class:off={game.fieldMode === 'off'} onclick={() => game.cycleField()} aria-label="Field layer: {fieldLabel}. Press to change.">
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
		min-height: 48px;
		padding: 0 6px;
		border: none;
		border-radius: var(--radius-pill);
		background: none;
		font-size: 13px;
		color: var(--ink);
		white-space: nowrap;
		transition: background-color 0.15s;
	}
	button:hover:not(:disabled) {
		background: var(--soft);
	}
	button:focus-visible {
		outline-offset: -2px;
	}
	button:disabled {
		color: var(--faint);
		opacity: 0.55;
		cursor: default;
	}
	button.field {
		background: var(--ink);
		color: var(--on-ink);
	}
	button.field:hover {
		background: var(--ink-2) !important;
	}
	button.field.off {
		background: none;
		color: var(--ink);
		box-shadow: inset 0 0 0 1px var(--line-strong);
	}
	button.field:focus-visible {
		outline-offset: 2px;
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
		font-size: 10px;
		opacity: 0.75;
	}
	@media (min-width: 900px) {
		.dock {
			width: auto;
		}
		button {
			flex: none;
			min-height: 44px;
			padding: 0 16px;
		}
		.wide,
		.long,
		.desk {
			display: inline;
		}
		.phone,
		.narrow {
			display: none;
		}
	}
</style>
