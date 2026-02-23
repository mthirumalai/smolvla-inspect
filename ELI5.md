# How this works (ELI5)

## The big picture

SmolVLA is an AI that looks at camera images and decides how to move a robot arm. This tool lets you **see what the AI is looking at** in each image, so you can tell if it's paying attention to the right things.

Think of it like eye-tracking for a robot brain.

---

## Step by step

### 1. The AI sees in tiles

When SmolVLA looks at a camera image, it doesn't see the whole picture at once. It chops the image into a grid of small square tiles (like graph paper laid over a photo). Each tile is 16x16 pixels, so a 512x512 image becomes a 32x32 grid of tiles.

```
Original image          What the AI sees
┌─────────────┐         ┌──┬──┬──┬──┐
│             │         │  │  │  │  │
│  [robot     │   →     ├──┼──┼──┼──┤
│   arm]      │         │  │  │  │  │
│             │         ├──┼──┼──┼──┤
│     [cup]   │         │  │  │  │  │
└─────────────┘         └──┴──┴──┴──┘
                        (each square = one tile)
```

### 2. Tiles talk to each other

The AI runs something called **self-attention**: every tile "looks at" every other tile and decides how important the others are. After this process, each tile knows not just what's in its own square, but has gathered information from across the whole image.

This is like asking every person in a room to rate how relevant everyone else is to them. Popular people (tiles containing the gripper, the object to grab) get high ratings from lots of others.

### 3. We measure popularity

For each tile, we add up how much attention it received from all the other tiles. Tiles that got a lot of attention are "important" to the AI. We assign each tile a score from 0 (ignored) to 1 (highly attended).

### 4. We blow it back up into a heatmap

Those scores sit on the 32x32 tile grid, which is tiny compared to the original image. So we stretch them back to the full image size using smooth interpolation (like zooming in on a low-res photo). The result is a **heatmap** the same size as the original image where:

- **Bright/warm colors** = the AI pays a lot of attention here
- **Dark/cool colors** = the AI mostly ignores this area

### 5. We overlay it on the original

The heatmap is blended on top of the camera frame so you can see exactly which parts of the scene the AI is focused on.

---

## What about cross-attention?

SmolVLA has two brains working together:

1. **The vision encoder** (SigLIP) — looks at images and summarizes them into a set of tokens
2. **The action expert** — takes those visual tokens plus the language instruction ("pick up the cup") and decides what motor commands to send

**Self-attention** (described above) tells you what the vision encoder finds interesting internally.

**Cross-attention** tells you which visual tokens the action expert actually reads when making its decision. This is arguably more useful: it's the direct link between "what the AI sees" and "what the AI does."

The trick is that SmolVLA doesn't have a separate cross-attention module. Instead, the action expert queries a shared memory bank that contains the vision tokens. We intercept that query to see which vision tokens get the most attention from the action decoder.

```
Vision encoder:  "Here are 64 visual tokens summarizing the image"
                          ↓
Action expert:   "I'll look mostly at tokens 12 and 37"  ← we capture this
                          ↓
                 "Move arm left 2cm, close gripper"
```

---

## What about attention rollout?

The vision encoder has many layers stacked on top of each other. Each layer runs its own self-attention. By default we only look at the **last layer**, but attention rollout **chains all layers together** to show the cumulative effect of attention through the full network.

It's like the difference between asking "who did you talk to last?" (last layer) vs. "who influenced your final opinion, even indirectly through a chain of people?" (rollout).

---

## What about per-head attention?

Each attention layer actually runs multiple attention computations in parallel, called **heads**. Different heads often specialize: one might track the gripper, another might track the target object, another might attend to edges.

The `--show-heads` flag shows each head's attention separately so you can see this specialization (or lack of it).

---

## What should I look for?

**Healthy model:** Bright spots on the gripper, the object being manipulated, and the goal location. Dark background.

**Overfitting to background:** Bright spots on the table texture, shelves, cables, or other scene elements that shouldn't matter. This means the model is using background cues instead of task-relevant features, and it will likely fail when the background changes.

**Diffuse attention:** Everything is roughly the same brightness. The model hasn't learned to focus on anything specific yet.
