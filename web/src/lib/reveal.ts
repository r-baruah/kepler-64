// Svelte action for the landing's section reveal. It never hides anything up front: a section is
// fully visible until the moment it first scrolls into view, then gets `.in` (a short fade and
// rise, see base.css). Sections already on screen when the page loads are left alone. Nothing
// happens without IntersectionObserver or with reduced motion.
import { motionOK } from './theme';

export function reveal(node: HTMLElement) {
	if (typeof IntersectionObserver === 'undefined' || !motionOK()) return;
	let first = true;
	const io = new IntersectionObserver(
		(entries) => {
			for (const e of entries) {
				if (first) {
					first = false;
					// On screen (or above it) at load: never animate.
					if (e.isIntersecting || e.boundingClientRect.top < 0) {
						io.disconnect();
						return;
					}
					continue;
				}
				if (e.isIntersecting) {
					node.classList.add('in');
					io.disconnect();
				}
			}
		},
		{ threshold: 0.08 }
	);
	io.observe(node);
	return { destroy: () => io.disconnect() };
}
