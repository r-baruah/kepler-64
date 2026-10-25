import type { SiteFacts } from '$lib/types';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => {
	const res = await fetch('/data/site-facts.json');
	if (!res.ok) throw new Error('site-facts.json is missing: run npm run sync');
	return { facts: (await res.json()) as SiteFacts };
};
