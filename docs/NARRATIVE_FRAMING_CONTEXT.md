# Kepler-64: Master Narrative Framing & Launch Strategy Context

This document is the targeted context, voice guide, and strategic roadmap for crafting public-facing posts, hackathon submissions, and storytelling for **Kepler-64 (The Roche Engine)**.

---

## 👤 The Creator Persona & Voice Guidelines

**Creator & Lead Architect:** **Ripuranjan Baruah**

### ✍️ Voice & Tone Directives (CRITICAL: Human Authenticity)
The posts must sound like an authentic, obsessive independent builder and physics/chess enthusiast—**never like generic AI marketing copy**.

- **Tone:** Thoughtful, witty, technically rigorous, slightly self-aware of the glorious absurdity of the project.
- **Banned AI Tropes & Slop:**
  - ❌ *Never use:* "game-changer", "revolutionary paradigm", "delve into", "testament to", "unleash", "in today's fast-paced world", "a symphony of", "seamlessly blends".
  - ✅ *Use natural human phrasing:* "The question that started this was simple enough to be stupid...", "What surprised me was...", "It's not Stockfish. It was never trying to be.", "All my loves in one project: chess, space, maths, physics."
- **Intellectual Honesty:** Never claim it's a 3500-Elo engine that will beat Stockfish. Acknowledge what it is: *a differentiable physics simulation that happens to play legal chess, where the evaluation is pure gravity and tensor calculus.*

---

## 🪐 The Core Narrative Pillars

### 1. The Philosophical Hook (The Feynman Reversal)
> *"Richard Feynman once said that figuring out the laws of physics is like watching a chess game without knowing the rules—you observe long enough, patterns emerge, and you start to guess the laws underneath.*  
> *Kepler-64 does the exact opposite. It already knows the rules of chess. It's the universe that has to figure out how to play."*

### 2. The Unhinged Yet 100% Real Physics
This is not a cosmetic theme. Every evaluation is a legitimate calculation from astrophysics:
- **Pieces as Point Masses:** Pawns ($1\text{m}$), Knights/Bishops ($3\text{m}$), Rooks ($5\text{m}$), Queen ($9\text{m}$), King ($1000\text{m}$) warping 2D discrete spacetime.
- **Plummer Potential Field $\Phi(p)$:** Softened Newtonian gravity preventing infinite singularities:
  $$\Phi(p) = -G \sum_{j=1}^{64} \frac{|m_j| \cdot \sigma(c - \lVert p - r_j \rVert)}{\sqrt{\lVert p - r_j \rVert^2 + \varepsilon^2}}$$
- **Tidal Tensor & Line of Failure:** The Hessian $\mathbf{A} = \nabla\nabla\Phi$ at the King computes directional stretching eigenvalues $(\lambda_1, \lambda_2)$ and the geometric line of failure.
- **The Dimensionless Roche Limit ($\eta$):** The King doesn't get checkmated; it suffers **Roche Disruption** when tidal tearing overcomes self-gravitational cohesion ($\eta > \rho_{\text{roche}}$).
- **Accretion Queens (Black Holes):** Capturing pieces absorb **80% of the victim's mass**. A Queen that eats multiple pieces becomes an ultra-dense well that physically pulls the board toward it.
- **Learnable Universe Constants via JAX:** Physical constants ($G, \varepsilon, c, \rho$) are **learnable leaves** optimized via gradient descent on master games.

### 3. The Emergent Revelation
When gravity learns chess, it independently rediscovers **romantic, pre-computer chess** (concentrating mass, building pressure, and crushing the enemy King under physical weight) from first principles, without human heuristics.

---

## 🎯 Target Platforms & Strategic Angles

### 1. Manware's "Unhinged Project Hackathon" (Discord)
- **Channel:** `#unhinged-project-submission` in Manware's Discord community.
- **Goal:** Maximizing **💀 reaction votes** and earning a spotlight review in Manware's upcoming YouTube video.
- **Angle:** *"I replaced chess engine heuristics with actual differentiable astrophysics and Black Hole Queens."*
- **Key Assets:** Direct link to the [Live Web Simulator](https://r-baruah.github.io/kepler-64/) + animated match GIF.

### 2. Reddit Campaigns
- **r/chess (1.3M):** *"I built a chess engine that doesn't know any chess heuristics—it knows gravity instead."* (Focus on how gravity rediscovers romantic attacking chess).
- **r/MachineLearning (3.0M):** `[P] Kepler-64: Differentiable N-Body Gravitational Chess Engine in JAX` (Focus on the 218-pad trick, XLA compilation, analytical Plummer autograd, and learned $G$).
- **r/Physics & r/astrophysics:** *"Feynman said you can discover physics by watching chess. I did the opposite."* (Focus on genuine Hill sphere / Roche lobe scaling and tensor math).
- **r/programming (6.0M):** *"How I built a differentiable chess engine where the King loses by tidal disruption."* (Focus on the JAX tensor pipeline, closed-form $2\times2$ eigensolvers, and performance).
- **r/dataisbeautiful (20M):** High-resolution visual potential heatmap and equipotential contour GIF.

### 3. Hacker News (Show HN)
- **Title:** `Show HN: Kepler-64 – A differentiable gravitational chess engine built with JAX`
- **Tone:** Direct, technical, intellectually honest, engineer-to-engineer.

### 4. Personal Portfolio Blog Post
- **Purpose:** Long-form reflective essay for Ripuranjan's personal website detailing the journey, the Feynman quote, the mathematical derivations, and why building absurd things matters.

---

## 🔗 Live Artifacts & References to Include

- **Live Web Observatory:** `https://r-baruah.github.io/kepler-64/`
- **GitHub Repository:** `https://github.com/r-baruah/kepler-64`
- **Scientific References:** `REFERENCES.md` (Plummer 1911, Roche 1849, Hill 1878, JAX 2018)
- **Citation Spec:** `CITATION.cff`
- **Existing Drafts for Reference:**
  - [`Planning/kepler64_promo_drafts.md`](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/Planning/kepler64_promo_drafts.md)
  - [`Planning/Kapler-64 Blog Writeup.md`](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/Planning/Kapler-64%20Blog%20Writeup.md)
  - [`Planning/HACKATHON_AND_SOCIAL_PROMOTION.md`](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/Planning/HACKATHON_AND_SOCIAL_PROMOTION.md)

---

## 🛡️ Instructions for the Narrative Agent

When working with Ripuranjan on narrative framing:
1. **Never sound like a corporate press release or ChatGPT template.** Speak like a passionate builder chatting with fellow developers.
2. **Emphasize the visual and interactive proof:** Always point readers to the live zero-friction web simulator.
3. **Iterate collaboratively:** Present copy options with distinct hooks (e.g. The Philosophical Hook vs. The Crazy Engineering Hook vs. The Visual Absurdity Hook) and refine based on Ripuranjan's authentic style.
