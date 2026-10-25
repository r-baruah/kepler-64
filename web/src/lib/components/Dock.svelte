<script lang="ts">
	import type { GameStore } from '$lib/game.svelte';

	let { game }: { game: GameStore } = $props();
	const fieldLabel = $derived({ off: 'Off', pull: 'Pull', cones: 'Light cones' }[game.fieldMode]);
	const fieldShort = $derived({ off: 'Off', pull: 'Pull', cones: 'Cones' }[game.fieldMode]);
</script>

<!-- Phone: a slim bar of five icons with short labels (Field shows its mode inline).
     Desktop (900px and up): the text pill, with Flip and Level and without More. -->
<nav class="dock" aria-label="Game controls">
	<button aria-label="New game" onclick={() => (game.sheet = 'new')}>
		<svg class="ic" viewBox="0 0 20 20" aria-hidden="true"><path d="M10 4v12M4 10h12" /></svg>
		<span class="lab">New</span><span class="desk">New game</span>
	</button>
	<button aria-label="Undo" onclick={() => game.undo()} disabled={!game.canUndo}>
		<svg class="ic" viewBox="0 0 20 20" aria-hidden="true"><path d="M7.5 4.5 4 8l3.5 3.5" /><path d="M4 8h7.5a4.25 4.25 0 0 1 0 8.5H9" /></svg>
		<span class="lab">Undo</span><span class="desk">Undo</span>
	</button>
	<button class="wide" aria-label="Flip board" onclick={() => game.flip()}>Flip</button>
	<button class="field" class:off={game.fieldMode === 'off'} onclick={() => game.cycleField()} aria-label="Field layer: {fieldLabel}. Press to change.">
		<svg class="ic" viewBox="0 0 20 20" aria-hidden="true"
			><rect x="3.5" y="3.5" width="13" height="13" rx="1.5" /><path d="M10 3.5v13M3.5 10h13" /><rect class="fill" x="3.5" y="10" width="6.5" height="6.5" /></svg
		>
		<span class="lab">Field<span class="mode">&nbsp;&middot; {fieldShort}</span></span>
		<span class="desk">Field: {fieldLabel}</span>
	</button>
	<button aria-label="Universe" onclick={() => (game.sheet = 'universe')}>
		<svg class="ic" viewBox="0 0 20 20" aria-hidden="true"
			><ellipse cx="10" cy="10" rx="7.5" ry="3.6" transform="rotate(-20 10 10)" /><circle class="dot" cx="10" cy="10" r="2.2" /></svg
		>
		<span class="lab">Universe</span><span class="desk">Universe</span>
	</button>
	<button class="wide" aria-label="Level" onclick={() => (game.sheet = 'level')}>Level</button>
	<button class="narrow" aria-label="More" onclick={() => (game.sheet = 'more')}>
		<svg class="ic" viewBox="0 0 20 20" aria-hidden="true"
			><circle class="dot" cx="4.5" cy="10" r="1.3" /><circle class="dot" cx="10" cy="10" r="1.3" /><circle class="dot" cx="15.5" cy="10" r="1.3" /></svg
		>
		<span class="lab">More</span>
	</button>
</nav>

<style>
	/* ---- phone: the slim bar ---- */
	.dock {
		display: grid;
		grid-template-columns: 1fr 1fr 1.3fr 1fr 1fr;
		width: 100%;
		max-width: 100%;
		height: calc(58px + env(safe-area-inset-bottom));
		padding: 0 4px env(safe-area-inset-bottom);
		background: var(--card);
		border-top: 1px solid var(--line);
	}
	button {
		min-width: 0;
		min-height: 48px;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 4px;
		padding: 0 2px;
		border: none;
		border-radius: var(--radius-card-sm);
		background: none;
		color: var(--ink);
		white-space: nowrap;
		transition: background-color 0.15s;
	}
	/* hover only where there is a pointer that hovers: on touch, a tapped button would stay grey */
	@media (hover: hover) {
		button:hover:not(:disabled) {
			background: var(--soft);
		}
	}
	button:active:not(:disabled) {
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
	.ic {
		width: 20px;
		height: 20px;
		flex: none;
		fill: none;
		stroke: currentColor;
		stroke-width: 1.4;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.ic .dot {
		fill: currentColor;
		stroke: none;
	}
	.ic .fill {
		fill: currentColor;
		stroke: none;
		opacity: 0.35;
	}
	.lab {
		font-size: 11.5px;
		font-weight: 500;
		line-height: 1;
		color: var(--ink-2);
	}
	button:disabled .lab {
		color: inherit;
	}
	.field .mode {
		color: var(--accent);
	}
	.field .ic {
		color: var(--accent);
	}
	.field.off .ic {
		color: var(--ink);
	}
	.field.off .ic .fill {
		opacity: 0;
	}
	.field.off .mode {
		color: var(--muted);
	}
	.wide,
	.desk {
		display: none;
	}

	/* ---- desktop: the text pill, as before ---- */
	@media (min-width: 900px) {
		.dock {
			display: flex;
			gap: 2px;
			width: auto;
			height: auto;
			padding: 4px;
			border: 1px solid var(--line);
			border-radius: var(--radius-pill);
			box-shadow: var(--shadow-dock);
		}
		button {
			display: block;
			flex: none;
			min-height: 44px;
			padding: 0 16px;
			border-radius: var(--radius-pill);
			font-size: 13px;
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
		.desk {
			display: inline;
		}
		.ic,
		.lab,
		.narrow {
			display: none;
		}
	}
</style>
