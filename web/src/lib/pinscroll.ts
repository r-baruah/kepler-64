// Svelte action for the "How it works" cards on phones: a pinned sideways scroll. The node is a
// wrapper around a sticky box (`[data-pin-box]`) that holds a sideways card row (`[data-pin-track]`).
// While the box is stuck under the header, vertical scrolling moves the row sideways one pixel per
// pixel, so the pinned distance is exactly the row's own overflow; then the page carries on.
//
// The row stays a normal sideways scroller, so a swipe still works: a swipe while pinned moves the
// page to the matching point, keeping the two in step. Nothing is hidden at rest, and with no JS,
// reduced motion or a width of 600px and up there is no pin at all (the plain swipe row or grid).
const QUERY = '(max-width: 599px) and (prefers-reduced-motion: no-preference)';

export function pinScroll(node: HTMLElement) {
	if (typeof matchMedia === 'undefined') return;
	const box = node.querySelector<HTMLElement>('[data-pin-box]');
	const track = node.querySelector<HTMLElement>('[data-pin-track]');
	if (!box || !track) return;
	const mq = matchMedia(QUERY);

	let on = false;
	let dist = 0;
	let frame = 0;
	let lastLeft = -1;
	let ro: ResizeObserver | null = null;

	const stickTop = () => parseFloat(getComputedStyle(box).top) || 0;

	function measure() {
		dist = Math.max(0, track!.scrollWidth - track!.clientWidth);
		// A box taller than the screen can't stay stuck; fall back to the plain row.
		const fits = box!.offsetHeight <= innerHeight - stickTop();
		if (!fits) dist = 0;
		node.classList.toggle('pinned', dist > 0);
		node.style.setProperty('--pin-dist', `${dist}px`);
		sync();
	}

	/** Vertical position -> sideways position. */
	function sync() {
		frame = 0;
		if (!dist) return;
		const p = Math.min(1, Math.max(0, (stickTop() - node.getBoundingClientRect().top) / dist));
		const left = Math.round(p * dist);
		if (left === lastLeft) return;
		lastLeft = left;
		track!.scrollLeft = left;
	}
	function onScroll() {
		if (!frame) frame = requestAnimationFrame(sync);
	}
	/** A swipe on the row while the box is stuck: move the page to the matching point. */
	function onTrack() {
		if (!dist) return;
		const left = Math.round(track!.scrollLeft);
		if (Math.abs(left - lastLeft) <= 1) return; // our own write
		const top = node.getBoundingClientRect().top;
		const t = stickTop();
		if (top > t + 1 || top < t - dist - 1) return; // not stuck: leave the page alone
		lastLeft = left;
		// (t - top) is how far the page has scrolled into the pin; move it to `left`.
		window.scrollTo({ top: window.scrollY + left - (t - top), behavior: 'instant' });
	}

	function enable() {
		if (on) return;
		on = true;
		ro = new ResizeObserver(measure);
		ro.observe(track!);
		ro.observe(box!);
		addEventListener('scroll', onScroll, { passive: true });
		addEventListener('resize', measure, { passive: true });
		track!.addEventListener('scroll', onTrack, { passive: true });
		measure();
	}
	function disable() {
		if (!on) return;
		on = false;
		ro?.disconnect();
		ro = null;
		removeEventListener('scroll', onScroll);
		removeEventListener('resize', measure);
		track!.removeEventListener('scroll', onTrack);
		if (frame) cancelAnimationFrame(frame);
		frame = 0;
		dist = 0;
		lastLeft = -1;
		node.classList.remove('pinned');
		node.style.removeProperty('--pin-dist');
	}
	const onChange = () => (mq.matches ? enable() : disable());
	mq.addEventListener('change', onChange);
	onChange();

	return {
		destroy() {
			mq.removeEventListener('change', onChange);
			disable();
		}
	};
}
