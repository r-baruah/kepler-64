# Visualizer Design Philosophy: Tidal Stress and Gravitational Gradients

This document captures the core aesthetic and technical direction for the Kepler-64 physics visualizer, heavily based on the idea of using natural physics gradients over harsh geometric overlays.

## The Core Concept (Tidal Stress & Conflict Zones)

The original idea for representing Tidal Stress (the tearing force on the King) avoids using harsh, unnatural geometric shapes (like red ellipses or arrows). Instead, it relies on shading and gradients to convey the "mood" or danger of the position:

1. **The King's Shadow (Trouble vs. Safe):** 
   When a King is under extreme gravitational stress (tidal tearing), its square and immediate surroundings become dark (e.g., a "sad" or "troubled" state). 
2. **Smooth Radiance:** 
   This darkness doesn't stop abruptly. It radiates outward, transitioning perfectly smoothly and gradually getting lighter the further away it gets from the intense gravity well.
3. **Natural Conflict Zones:** 
   Because both Kings exist on the board and both generate/attract gravitational forces, their radiating gradients will eventually collide. Rather than manually programming how these gradients interact, we allow the underlying physics of the tidal stress and gravity to naturally resolve the conflict zones. The gradient automatically darkens or lightens based on the true net gravitational pull in those contested areas.

This ensures the visualization remains clean, professional, and entirely driven by the real underlying math, rather than arbitrary UI elements.

## Current Technical Implementation

To achieve this clean look without breaking the usability of the chessboard, the visualizer uses a dual-layer approach.

### 1. The Interactive Board (React / chessiro-canvas)
- The frontend is a React (Vite) application.
- The actual chess board is rendered using the `chessiro-canvas` library. 
- The board renders its normal, fully opaque light and dark squares, ensuring the pieces and the game itself remain completely legible and interactive.

### 2. The Physics Overlay (Transparent PNG)
- The physics engine (`generate_overlay.py`) calculates the potential field `U` using the Kepler-64 JAX internals.
- Instead of generating a solid image that replaces the board, it outputs a **transparent PNG** (`overlay.png`).
- This overlay contains:
  - A low-opacity (15-25%) `imshow` heatmap using the `magma` colormap. Because `magma` maps the lowest values (deepest gravity wells) to black/dark purple, it perfectly achieves the "King is dark" effect naturally.
  - Very subtle, semi-transparent white/silver topological contour lines (`#ffffff` at `0.4` opacity) to show the shape of the field.

### 3. The CSS Sandwich
In the frontend (`App.tsx` and `App.css`), the overlay is placed exactly on top of the chessboard:
```css
.physics-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: url('/overlay.png') center/cover no-repeat;
  z-index: 10;
  pointer-events: none; /* Crucial: Allows the user to click THROUGH the overlay to move pieces */
}
```

By placing the perfectly aligned transparent PNG directly over the `chessiro-canvas`, the dark `magma` gradient physically darkens the squares beneath it exactly in proportion to the gravitational field, achieving the user's exact vision of a natural, physics-driven shadow.
