<header class="top">
	<svg class="mark" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"
		><circle cx="10" cy="10" r="4" fill="currentColor" /><ellipse cx="10" cy="10" rx="9" ry="3.4" fill="none" stroke="currentColor" stroke-width="1.2" transform="rotate(-20 10 10)" /></svg
	>
	<span class="name">Kepler-64</span>
	<span class="tag mono">/ a tiny universe that learned to play chess</span>
	<a class="star" href="https://github.com/r-baruah/kepler-64">Star on GitHub</a>
</header>

<style>
	.top {
		height: 56px;
		padding: 0 var(--gutter);
		display: flex;
		align-items: center;
		gap: 10px;
		border-bottom: 1px solid var(--line);
	}
	.mark {
		flex: none;
	}
	.name {
		font-weight: 600;
		font-size: 16px;
		letter-spacing: -0.01em;
	}
	.tag {
		display: none;
		flex-grow: 1;
		font-size: 11px;
		color: var(--muted);
	}
	.name {
		flex-grow: 1;
	}
	.star {
		min-height: 40px;
		padding: 0 14px;
		display: inline-flex;
		align-items: center;
		font-size: 13px;
		text-decoration: none;
		background: var(--ink);
		color: #fff;
		border-radius: var(--radius-pill);
	}
	.star:hover {
		color: #fff;
		background: var(--ink-2);
	}
	@media (min-width: 900px) {
		.top {
			padding: 0 32px;
			gap: 14px;
		}
		.name {
			flex-grow: 0;
		}
		.tag {
			display: block;
		}
	}
</style>
