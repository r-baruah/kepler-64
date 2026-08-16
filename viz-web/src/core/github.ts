/**
 * Kepler-64 GitHub Contributors Client
 * Rate-limit-safe: short-TTL localStorage cache + ETag conditional requests,
 * and graceful fallback when the unauthenticated GitHub API quota is exhausted.
 */

export interface GitHubContributor {
  login: string;
  avatarUrl: string;
  htmlUrl: string;
  contributions: number;
}

interface ContributorCacheEntry {
  fetchedAt: number;
  etag: string | null;
  data: GitHubContributor[];
}

export const GITHUB_REPO = 'r-baruah/kepler-64';
export const GITHUB_REPO_URL = `https://github.com/${GITHUB_REPO}`;

const CACHE_KEY = 'k64.contributors.v2';
const TTL_MS = 5 * 60 * 1000; // 5 min between live refreshes — ETag 304s don't count against the 60/hr limit
const RATE_LIMIT_TTL_MS = 60 * 60 * 1000; // on 403/429, don't retry for 1 hour

function readCache(): ContributorCacheEntry | null {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    return raw ? (JSON.parse(raw) as ContributorCacheEntry) : null;
  } catch {
    return null;
  }
}

function writeCache(entry: ContributorCacheEntry): void {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(entry));
  } catch {
    /* storage unavailable — treat as a cache miss next time */
  }
}

/**
 * Fetch contributors for GITHUB_REPO. Returns null when no data is available
 * (network failure or rate-limited with no cached copy). Never throws.
 */
export async function fetchContributors(force = false): Promise<GitHubContributor[] | null> {
  const cached = readCache();

  // Fresh cache wins unless a forced refresh is requested.
  if (!force && cached && Date.now() - cached.fetchedAt < TTL_MS) {
    return cached.data;
  }

  // If we recently hit the rate limit, stay quiet and serve cache (if any).
  if (!force && cached && Date.now() - cached.fetchedAt < RATE_LIMIT_TTL_MS && cached.data.length === 0) {
    return null;
  }

  const headers: Record<string, string> = {
    Accept: 'application/vnd.github+json',
  };
  if (cached?.etag) {
    // Conditional request: a 304 response does not consume rate-limit quota.
    headers['If-None-Match'] = cached.etag;
  }

  try {
    const res = await fetch(`https://api.github.com/repos/${GITHUB_REPO}/contributors?per_page=100`, {
      headers,
    });

    if (res.status === 304) {
      if (cached) {
        writeCache({ ...cached, fetchedAt: Date.now() });
        return cached.data;
      }
      return null;
    }

    const remaining = res.headers.get('x-ratelimit-remaining');
    if (res.status === 403 || res.status === 429 || remaining === '0') {
      // Rate limited: keep any cached data and push the next attempt out.
      if (cached) {
        writeCache({ ...cached, fetchedAt: Date.now() });
        return cached.data;
      }
      return null;
    }

    if (!res.ok) {
      return cached?.data ?? null;
    }

    const raw = (await res.json()) as Array<{
      login: string;
      avatar_url: string;
      html_url: string;
      contributions: number;
    }>;

    const data: GitHubContributor[] = raw.map((c) => ({
      login: c.login,
      avatarUrl: c.avatar_url,
      htmlUrl: c.html_url,
      contributions: c.contributions,
    }));

    writeCache({ fetchedAt: Date.now(), etag: res.headers.get('etag'), data });
    return data;
  } catch {
    return cached?.data ?? null;
  }
}
