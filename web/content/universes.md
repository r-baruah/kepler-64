# Universes and levels on the website

`web/scripts/sync-data.mjs` reads the two tables below at build time, so the site never retypes them. Every preset is a committed file under `engine/data/`, and the labels are shown on the site word for word.

## Universe presets

| slug | name | file | label shown |
|---|---|---|---|
| `champion` | Home | `generations/gen-004.json` | "Our universe. The strongest so far (Gen 4)." |
| `equal-start` | Genesis | `ablations/control-equal-masses-a-candidate.json` | "Every piece was born the same weight. It found its own values by playing itself, with accretion on." |
| `gen-0` | First Light | `generations/gen-000.json` | "Before it learned anything: human piece values, laws set by hand." |
| `accretion` | Hunger | gen-004 with `f_acc = 0.8` | "Captures absorb weight: a piece grows by eating. Rejected, it played better without. Home's constants with this law switched on." |
| `strong-roche` | Roche | gen-004 with `beta = 10` | "A king can be torn apart by enemy pull, like a moon too close to its planet. Two versions were tried; neither made the engine stronger. Home's constants with this law turned up." |
| `tidal-load` | Undertow | gen-004 with `lambda_tide = 0.1` | "Enemy pull drags on the king's whole body. Rejected, it made the engine weaker. Home's constants with this law switched on." |

## Levels

| level | nodes per move | about, on a phone (F34) |
|---|---|---|
| Quick | 60,000 | 0.15 s |
| Default | 400,000 | 1 s |
| Strong | 1,500,000 | 3.5 s |

The same node budget gives the same move on every device, because the search is deterministic for a given node count.
