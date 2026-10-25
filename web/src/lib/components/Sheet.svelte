<script lang="ts">
	import type { Snippet } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import { motionOK } from '$lib/theme';

	const ms = (n: number) => (motionOK() ? n : 0);

	let { title, onclose, children }: { title: string; onclose: () => void; children: Snippet } = $props();
	let panel: HTMLDivElement;

	$effect(() => {
		const prev = document.activeElement as HTMLElement | null;
		panel?.focus();
		return () => prev?.focus?.();
	});

	function key(e: KeyboardEvent) {
		if (e.key === 'Escape') onclose();
	}
</script>

<svelte:window onkeydown={key} />

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div class="scrim" onclick={onclose} transition:fade|global={{ duration: ms(180) }}>
	<div class="panel" transition:fly|global={{ y: 28, duration: ms(240), opacity: 0.4 }} role="dialog" aria-modal="true" aria-label={title} tabindex="-1" bind:this={panel} onclick={(e) => e.stopPropagation()}>
		<div class="head">
			<h2>{title}</h2>
			<button class="x" onclick={onclose} aria-label="Close">Close</button>
		</div>
		<div class="body">{@render children()}</div>
	</div>
</div>

<style>
	.scrim {
		position: fixed;
		inset: 0;
		z-index: 20;
		background: var(--scrim);
		display: flex;
		align-items: flex-end;
		justify-content: center;
	}
	.panel {
		width: 100%;
		max-width: 520px;
		max-height: min(86dvh, 720px);
		display: flex;
		flex-direction: column;
		background: var(--card);
		border-radius: 20px 20px 0 0;
		box-shadow: var(--shadow-card);
		outline: none;
		padding-bottom: env(safe-area-inset-bottom);
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 14px 20px 10px;
		border-bottom: 1px solid var(--line);
	}
	h2 {
		margin: 0;
		font-size: var(--t-lg);
		font-weight: 500;
		letter-spacing: -0.01em;
	}
	.x:hover {
		background: var(--line);
	}
	.x {
		min-height: 44px;
		padding: 0 14px;
		border: none;
		border-radius: 999px;
		background: var(--soft);
		font: inherit;
		font-size: 13px;
		color: var(--ink);
		cursor: pointer;
	}
	.body > :global(*) {
		flex-shrink: 0;
	}
	.body {
		overflow: auto;
		overscroll-behavior: contain;
		padding: 14px 20px 22px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	@media (min-width: 700px) {
		.scrim {
			align-items: center;
		}
		.panel {
			border-radius: 20px;
		}
	}
</style>
