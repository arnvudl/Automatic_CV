# Design System Specification: The Analytical Luminary

## 1. Overview & Creative North Star
This design system is built to transform the standard Applicant Tracking System (ATS) from a utility tool into a high-end editorial experience. Our Creative North Star is **"The Analytical Luminary."** 

We are moving away from the rigid, boxy constraints of traditional enterprise software. Instead, we embrace an "airy" Google-esque philosophy—not through generic templates, but through **intentional asymmetry and tonal depth.** The goal is to make data feel weightless yet authoritative. We achieve this by using extreme white space (breathing room), oversized typographic scales, and layered surfaces that mimic the physical world's subtle interplay of light and shadow.

## 2. Colors & Surface Logic
The palette leverages the iconic Google primary spectrum but applies it with professional restraint. We prioritize the "chromatic white" experience, where the background isn't just white, but a carefully selected `surface` (`#f9f9ff`) that feels softer on the eyes.

### The "No-Line" Rule
**Strict Mandate:** Designers are prohibited from using 1px solid borders to section off content. In this design system, boundaries are defined exclusively through background color shifts or tonal transitions.
- Use `surface_container_low` to define a sidebar or a header against the main `surface` background.
- Use `surface_container_lowest` (pure white) for the most prominent content cards to make them "pop" against the tinted background.

### The Glass & Gradient Rule
To prevent the UI from feeling flat or "out-of-the-box," use **Glassmorphism** for floating elements like navigation bars or modal overlays. 
- **Recipe:** Use a semi-transparent version of `surface` with a `20px` to `40px` backdrop-blur. 
- **Signature Textures:** Apply a subtle linear gradient to primary CTAs, transitioning from `primary` (#0058bd) to `primary_container` (#2771df). This adds a "soul" to the action items that flat colors cannot replicate.

### Core Token Mapping
- **Primary:** `primary` (#0058bd) — Used for high-emphasis actions.
- **AI/Status Success:** `tertiary` (#006b2b) — For high-match AI scores.
- **AI/Status Warning:** Use yellow tones (Google Yellow) for mid-tier candidates.
- **AI/Status Error:** `error` (#ba1a1a) — For low-match or rejected states.

## 3. Typography
We utilize **Inter** to achieve a clean, modern, and highly readable sans-serif look that feels precision-engineered. 

- **Display & Headline:** Use `display-lg` (3.5rem) or `headline-lg` (2rem) for dashboard overviews (e.g., "Total Candidates"). The generous scale creates an editorial feel that commands attention.
- **Title:** `title-lg` (1.375rem) is the standard for card headers and section titles. It provides enough weight to anchor the layout without the need for a divider line.
- **Body:** `body-md` (0.875rem) is our workhorse for candidate details. Ensure a line-height of at least 1.5 to maintain the "airy" feel.
- **Label:** `label-sm` (0.6875rem) in `on_surface_variant` is used for metadata, ensuring it remains legible but secondary in the visual hierarchy.

## 4. Elevation & Depth
Depth in this system is achieved through **Tonal Layering** rather than structural lines.

### The Layering Principle
Think of the UI as stacked sheets of fine paper. 
1. **Base Layer:** `surface` (#f9f9ff).
2. **Section Layer:** `surface_container_low` (#f2f3fd).
3. **Content Layer (Cards):** `surface_container_lowest` (#ffffff).

### Ambient Shadows
When a "floating" effect is required (e.g., a dragged candidate card), use an **Ambient Shadow**:
- **X/Y Offset:** 0px 8px
- **Blur:** 24px
- **Color:** A tinted version of the surface color (e.g., `on_surface` at 6% opacity). 
Avoid dark grey shadows; they feel "dirty" and break the Luminary aesthetic.

### The "Ghost Border" Fallback
If a container sits on a background of the same color, you may use a **Ghost Border**: a 1px stroke using `outline_variant` at **20% opacity**. It should be felt more than seen.

## 5. Components

### Buttons
- **Primary:** Fully rounded (`full` / 9999px). Uses the `primary` gradient. High-contrast white text (`on_primary`).
- **Secondary:** Fully rounded. Uses `secondary_container` background with `on_secondary_container` text. No border.
- **State Changes:** On hover, increase the elevation or shift the gradient intensity slightly. Never use a "glow" effect.

### Candidate Cards
Cards must never have a border. Use `surface_container_lowest` and a subtle `sm` or `md` corner radius. Separation between cards is achieved through vertical white space (use the **1.5rem / 24px** spacing token).
- **AI Score Chip:** Position in the top-right corner using a semi-transparent `tertiary_container` (Green) background to signify a high match.

### Inputs & Search
Use a `surface_container_high` background for input fields. Forgo the traditional "box" look; use a `none` border and a slightly larger `md` roundedness. When focused, use a 2px `primary` bottom-glow or a very soft shadow to indicate activity.

### AI Data Visualization
Avoid standard bar charts. Use "Soft Glow" indicators—circles or rings that use `tertiary` (Green) for high scores, with a blurred outer ring to suggest "Intelligence" and "Insight" rather than just a hard data point.

## 6. Do's and Don'ts

### Do:
- **Embrace White Space:** If a section feels crowded, double the padding.
- **Use Nested Surfaces:** Layer a pure white card inside a soft grey-blue section.
- **Align to a Soft Grid:** Use an 8px grid, but allow "Hero" elements (like a candidate's name) to break the grid slightly to create a bespoke, editorial feel.

### Don't:
- **Don't use 100% Black:** Always use `on_surface` (#191b22) for text.
- **Don't use Dividers:** Never use a horizontal line to separate two list items. Use 16px–24px of empty space instead.
- **Don't use "Default" Shadows:** Avoid the standard CSS `box-shadow: 0 2px 4px rgba(0,0,0,0.5)`. It is too heavy for this system.
- **Don't use Hard Corners:** Every interactive element must have at least a `sm` (0.25rem) radius to maintain the approachable, "Google-esque" friendliness.