<script lang="ts">
	// The "Star on GitHub" icon: a filled gold star with three small four-point sparkles. The sparkles
	// twinkle once shortly after load and again when the link around the icon is hovered or focused.
	// Nothing moves under reduced motion; the star simply stays gold.
	import { onMount } from 'svelte';

	let { size = 16 }: { size?: number } = $props();
	// The load twinkle runs only while `intro` is set; dropping it afterwards means leaving the
	// link (which removes the hover animation) can't restart the load animation.
	let intro = $state(true);
	onMount(() => {
		const t = setTimeout(() => (intro = false), 2200);
		return () => clearTimeout(t);
	});
	const spark = (x: number, y: number, r: number) =>
		`M${x} ${y - r}Q${x} ${y} ${x + r} ${y}Q${x} ${y} ${x} ${y + r}Q${x} ${y} ${x - r} ${y}Q${x} ${y} ${x} ${y - r}Z`;
	const SPARKS = [
		{ d: spark(22.4, 3.2, 3), delay: 0 },
		{ d: spark(1.6, 6.4, 2), delay: 120 },
		{ d: spark(22.2, 19.6, 1.8), delay: 220 }
	];
</script>

<svg class="ico" class:intro width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
	<path class="st" d="M12 3.2l2.6 5.5 6 .8-4.4 4.1 1.1 5.9L12 16.6l-5.3 2.9 1.1-5.9-4.4-4.1 6-.8Z" />
	{#each SPARKS as s (s.d)}
		<path class="spark" d={s.d} style:--d="{s.delay}ms" />
	{/each}
</svg>

<style>
	.ico {
		display: block;
		flex: none;
		overflow: visible;
	}
	.st {
		fill: var(--star);
		stroke: var(--star-edge);
		stroke-width: 1.3;
		stroke-linejoin: round;
	}
	.spark {
		fill: var(--star);
		opacity: 0;
		transform-box: fill-box;
		transform-origin: center;
	}
	@media (prefers-reduced-motion: no-preference) {
		.intro .spark {
			animation: twinkle-load 0.8s ease-out calc(900ms + var(--d)) both;
		}
		/* A different animation name restarts it, so hover and focus replay the twinkle once. */
		:global(:is(a, button):is(:hover, :focus-visible)) > .ico .spark {
			animation: twinkle-again 0.8s ease-out var(--d) both;
		}
	}
	@keyframes twinkle-load {
		0% {
			opacity: 0;
			transform: scale(0.2) rotate(-20deg);
		}
		40% {
			opacity: 1;
			transform: scale(1) rotate(0deg);
		}
		100% {
			opacity: 0;
			transform: scale(0.5) rotate(15deg);
		}
	}
	@keyframes twinkle-again {
		0% {
			opacity: 0;
			transform: scale(0.2) rotate(-20deg);
		}
		40% {
			opacity: 1;
			transform: scale(1) rotate(0deg);
		}
		100% {
			opacity: 0;
			transform: scale(0.5) rotate(15deg);
		}
	}
</style>
