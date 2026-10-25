// Light or dark. The first value is set before paint by src/app.html (saved choice, else the
// system preference); the header toggle flips it and remembers the choice in this browser.
export type Theme = 'light' | 'dark';

export function currentTheme(): Theme {
	if (typeof document === 'undefined') return 'light';
	return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

export function setTheme(t: Theme) {
	document.documentElement.setAttribute('data-theme', t);
	try {
		localStorage.setItem('kepler-theme', t);
	} catch {
		// private mode: the choice lasts for this page only
	}
}

/** True unless the visitor asked for reduced motion. */
export function motionOK(): boolean {
	return typeof matchMedia === 'undefined' || !matchMedia('(prefers-reduced-motion: reduce)').matches;
}
