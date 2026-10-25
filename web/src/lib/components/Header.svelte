<script lang="ts">
	import { onMount } from 'svelte';
	import { currentTheme, setTheme, type Theme } from '$lib/theme';

	interface Props {
		/** Back to the start page (asks first if a game is in progress). */
		onhome?: () => void;
		/** Go to a landing section by id (returns to the start page first if needed). */
		onnav?: (id: string) => void;
		playing?: boolean;
	}
	let { onhome, onnav, playing = false }: Props = $props();

	let theme = $state<Theme>('light');
	onMount(() => (theme = currentTheme()));
	function toggle() {
		theme = theme === 'dark' ? 'light' : 'dark';
		setTheme(theme);
	}
	const NAV = [
		['how', 'How it works'],
		['ledger', 'Law Ledger']
	] as const;
</script>

<header class="top">
	<button class="home" onclick={() => onhome?.()} aria-label="Kepler-64, start page">
		<svg class="mark" width="22" height="22" viewBox="0 0 20 20" aria-hidden="true"
			><circle cx="10" cy="10" r="4" fill="currentColor" /><ellipse
				cx="10"
				cy="10"
				rx="9"
				ry="3.4"
				fill="none"
				stroke="currentColor"
				stroke-width="1.2"
				transform="rotate(-20 10 10)"
			/></svg
		>
		<span class="name">Kepler-64</span>
	</button>

	<nav class="nav" aria-label="Site">
		<button class="link" aria-current={playing ? 'page' : undefined} onclick={() => onhome?.()}>Play</button>
		{#each NAV as [id, text] (id)}
			<button class="link" onclick={() => onnav?.(id)}>{text}</button>
		{/each}
		<span class="link soon" title="Not built yet">Lab <small>Soon</small></span>
	</nav>

	<div class="actions">
		<button class="icon" onclick={toggle} aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} title="Theme">
			{#if theme === 'dark'}
				<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"
					><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" stroke-width="1.6" />{#each [0, 45, 90, 135, 180, 225, 270, 315] as a (a)}<line
							x1="12"
							y1="2.5"
							x2="12"
							y2="5"
							stroke="currentColor"
							stroke-width="1.6"
							stroke-linecap="round"
							transform="rotate({a} 12 12)"
						/>{/each}</svg
				>
			{:else}
				<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"
					><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" /></svg
				>
			{/if}
		</button>
		<a class="star" href="https://github.com/r-baruah/kepler-64">
			<svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"
				><path
					d="M12 3.2l2.6 5.5 6 .8-4.4 4.1 1.1 5.9L12 16.6l-5.3 2.9 1.1-5.9-4.4-4.1 6-.8Z"
					fill="none"
					stroke="currentColor"
					stroke-width="1.6"
					stroke-linejoin="round"
				/></svg
			>
			<span>Star<span class="long">&nbsp;on GitHub</span></span>
		</a>
	</div>
</header>

<style>
	.top {
		position: sticky;
		top: 0;
		z-index: 15;
		height: 56px;
		padding: 0 var(--gutter);
		display: flex;
		align-items: center;
		gap: 8px;
		background: color-mix(in srgb, var(--bg) 86%, transparent);
		backdrop-filter: saturate(1.2) blur(10px);
		-webkit-backdrop-filter: saturate(1.2) blur(10px);
		border-bottom: 1px solid var(--line);
	}
	.home {
		display: flex;
		align-items: center;
		gap: 10px;
		background: none;
		border: none;
		padding: 0 4px;
		min-height: var(--tap);
		color: var(--ink);
		border-radius: 8px;
	}
	.name {
		font-weight: 500;
		font-size: 16px;
		letter-spacing: -0.01em;
	}
	.nav {
		display: none;
	}
	.actions {
		margin-left: auto;
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.icon {
		width: var(--tap);
		height: var(--tap);
		display: grid;
		place-items: center;
		border: none;
		border-radius: var(--radius-pill);
		background: none;
		color: var(--ink);
		transition: background-color 0.15s;
	}
	.icon:hover {
		background: var(--soft);
	}
	.star {
		min-height: 38px;
		padding: 0 14px;
		display: inline-flex;
		align-items: center;
		gap: 7px;
		font-size: var(--t-sm);
		font-weight: 500;
		text-decoration: none;
		color: var(--ink);
		border: 1px solid var(--line-strong);
		border-radius: var(--radius-pill);
		transition:
			border-color 0.15s,
			background-color 0.15s;
	}
	.star:hover {
		color: var(--ink);
		border-color: var(--ink);
	}
	.long {
		display: none;
	}
	@media (min-width: 900px) {
		.top {
			padding: 0 32px;
			gap: 28px;
		}
		.nav {
			display: flex;
			align-items: center;
			gap: 4px;
		}
		.link {
			min-height: 36px;
			padding: 0 12px;
			border: none;
			border-radius: var(--radius-pill);
			background: none;
			font-size: var(--t-sm);
			color: var(--muted);
			display: inline-flex;
			align-items: center;
			gap: 6px;
			transition:
				color 0.15s,
				background-color 0.15s;
		}
		button.link:hover {
			color: var(--ink);
			background: var(--soft);
		}
		.link[aria-current='page'] {
			color: var(--ink);
		}
		.soon {
			cursor: default;
			color: var(--faint);
		}
		.soon small {
			font-size: 11px;
			padding: 1px 6px;
			border-radius: var(--radius-pill);
			background: var(--soft);
		}
		.long {
			display: inline;
		}
	}
</style>
