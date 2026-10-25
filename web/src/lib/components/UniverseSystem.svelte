<script lang="ts">
	// One universe drawn as its own small system: the five piece weights as bodies on one orbit
	// (area proportional to weight, pawn = 1, from the universe's generation file), around a
	// barycentre mark. A law that is switched on in this universe adds one mark:
	//   accretion (f_acc > 0): a captured body falling into a bigger one along a dashed path;
	//   tidal load (lambda_tide > 0): the king's body stretched by enemy pull, arrows along the stretch;
	//   strong Roche (only the preset that turns it up): the king inside its dashed Roche limit, shedding pieces.
	// The Roche mark is never drawn for other universes, even though their files carry a small Roche
	// weight, so Home never suggests a Roche limit drives its play (claims register).
	import { bodyRadius, pieceWeights } from '$lib/field';
	import type { Constants } from '$lib/types';

	interface Props {
		constants: Constants;
		slug: string;
		/** Name for the accessible label. */
		name: string;
		/** 'card' draws letters under the bodies and a faint grid; 'mini' is the bare system. */
		size?: 'card' | 'mini';
	}
	let { constants, slug, name, size = 'card' }: Props = $props();

	const W = 180;
	const H = 104;
	const CX = 90;
	const CY = 58;
	const RX = 74;
	const RY = 24;
	const TILT = -7;
	const pawn = $derived(size === 'mini' ? 3.1 : 2.8);
	// Law marks are drawn larger and bolder at picker size so the one difference reads.
	const k = $derived(size === 'mini' ? 1.6 : 1);
	const bodies = $derived(pieceWeights(constants));
	// Bodies on the near half of the orbit, pawn on the left, queen on the right.
	const ANGLES = [168, 128, 90, 52, 12];
	const rad = (d: number) => (d * Math.PI) / 180;
	function onOrbit(t: number): [number, number] {
		const x = RX * Math.cos(rad(t));
		const y = RY * Math.sin(rad(t));
		const a = rad(TILT);
		return [CX + x * Math.cos(a) - y * Math.sin(a), CY + x * Math.sin(a) + y * Math.cos(a)];
	}
	const placed = $derived(bodies.map((b, i) => ({ ...b, at: onOrbit(ANGLES[i]), r: bodyRadius(b.weight, pawn) })));

	const accretion = $derived(constants.f_acc > 0);
	const tidal = $derived(constants.lambda_tide > 0);
	const roche = $derived(slug === 'strong-roche');
	// The king sits on the far side of the orbit when a king law is drawn.
	const king = $derived(onOrbit(-90));
	const rook = $derived(placed[3]);

	const label = $derived(
		`${name}: piece weights as bodies, ` +
			bodies.map((b) => `${b.name.toLowerCase()} ${b.weight.toFixed(1)}`).join(', ') +
			(accretion ? '. Captures absorb weight.' : '') +
			(tidal ? '. Enemy pull drags on the king.' : '') +
			(roche ? '. The king has a strong Roche limit.' : '')
	);
</script>

<svg class="sys {size}" viewBox="0 0 {W} {H}" role="img" aria-label={label}>
	{#if size === 'card'}
		<g class="grid" aria-hidden="true">
			{#each [1, 2, 3, 4, 5, 6, 7, 8] as i (i)}<line x1={i * 20} y1="0" x2={i * 20} y2={H} />{/each}
			{#each [1, 2, 3, 4] as i (i)}<line x1="0" y1={i * 20 + 2} x2={W} y2={i * 20 + 2} />{/each}
		</g>
	{/if}
	<ellipse class="orbit" cx={CX} cy={CY} rx={RX} ry={RY} transform="rotate({TILT} {CX} {CY})" />
	<g class="bary" aria-hidden="true"><line x1={CX - 4} y1={CY} x2={CX + 4} y2={CY} /><line x1={CX} y1={CY - 4} x2={CX} y2={CY + 4} /></g>

	{#if accretion}
		<!-- A captured body falls into the rook along a dashed path. -->
		{@const [rx, ry] = rook.at}
		{@const sx = rx - 34 * Math.min(k, 1.25)}
		{@const sy = ry - 34 * Math.min(k, 1.25) + (k > 1 ? 6 : 0)}
		<path class="trail" style:stroke-width={k * 1} d="M{sx} {sy} Q {rx - 4} {sy - 2} {rx - rook.r * 0.6} {ry - rook.r * 0.9}" />
		<circle class="crumb" cx={sx} cy={sy} r={1.8 * k} />
	{/if}

	{#if roche}
		{@const [kx, ky] = king}
		<circle class="limit" style:stroke-width={k} cx={kx} cy={ky} r={13 * k} />
		<circle class="king" cx={kx} cy={ky} r={4.2 * k} />
		<circle class="crumb" cx={kx + 9 * k} cy={ky - 3 * k} r={1.2 * k} />
		<circle class="crumb" cx={kx + 14.5 * k} cy={ky - 5.5 * k} r={1 * k} />
		<circle class="crumb" cx={kx + 19.5 * k} cy={ky - 7.5 * k} r={0.8 * k} />
	{/if}

	{#if tidal}
		{@const [kx, ky] = king}
		<ellipse class="king" cx={kx} cy={ky} rx={6.4 * k} ry={3.4 * k} />
		<path class="arrow" style:stroke-width={1.1 * k} d="M{kx - 9 * k} {ky} H{kx - 19 * k} M{kx - 16 * k} {ky - 2.5 * k} L{kx - 19 * k} {ky} L{kx - 16 * k} {ky + 2.5 * k}" />
		<path class="arrow" style:stroke-width={1.1 * k} d="M{kx + 9 * k} {ky} H{kx + 19 * k} M{kx + 16 * k} {ky - 2.5 * k} L{kx + 19 * k} {ky} L{kx + 16 * k} {ky + 2.5 * k}" />
	{/if}

	{#each placed as b (b.letter)}
		<circle class="body" class:q={b.letter === 'Q'} cx={b.at[0]} cy={b.at[1]} r={b.r} />
		{#if size === 'card'}
			<text class="lt" x={b.at[0]} y={b.at[1] + b.r + 10} text-anchor="middle">{b.letter}</text>
		{/if}
	{/each}
</svg>

<style>
	.sys {
		display: block;
		width: 100%;
		height: auto;
		overflow: visible;
	}
	.grid line {
		stroke: var(--grid);
		stroke-width: 1;
	}
	.orbit {
		fill: none;
		stroke: var(--orn);
		stroke-width: 1;
	}
	.bary line {
		stroke: var(--faint);
		stroke-width: 1;
	}
	.body {
		fill: var(--ink);
	}
	.body.q {
		fill: var(--accent);
	}
	.lt {
		font-family: var(--font-mono);
		font-size: 8px;
		fill: var(--muted);
	}
	.trail {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 1;
		stroke-dasharray: 2.5 2.5;
	}
	.crumb {
		fill: var(--ink-2);
	}
	.limit {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1;
		stroke-dasharray: 3 2.5;
	}
	.king {
		fill: var(--card);
		stroke: var(--ink);
		stroke-width: 1.4;
	}
	.arrow {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.1;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
</style>
