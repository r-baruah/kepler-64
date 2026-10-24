import adapter from '@sveltejs/adapter-vercel';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	plugins: [
		sveltekit({
			adapter: adapter(),
			alias: { $lib: 'src/lib' },
			prerender: { handleHttpError: 'fail' }
		})
	],
	worker: { format: 'es' },
	test: { include: ['src/**/*.test.ts', 'scripts/**/*.test.mjs'] }
});
