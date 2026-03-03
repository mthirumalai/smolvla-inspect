/**
 * Client-side heatmap rendering on HTML Canvas.
 *
 * Takes patch-resolution float arrays, bilinear-interpolates to pixel resolution,
 * applies a colormap, and alpha-blends over the original frame.
 */

type RGBA = [number, number, number, number];

// Pre-computed 256-entry colormap tables (matching matplotlib)
function buildJet(): RGBA[] {
  const table: RGBA[] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let r: number, g: number, b: number;
    if (t < 0.125) { r = 0; g = 0; b = 0.5 + t * 4; }
    else if (t < 0.375) { r = 0; g = (t - 0.125) * 4; b = 1; }
    else if (t < 0.625) { r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4; }
    else if (t < 0.875) { r = 1; g = 1 - (t - 0.625) * 4; b = 0; }
    else { r = 1 - (t - 0.875) * 4; g = 0; b = 0; }
    table.push([
      Math.round(Math.max(0, Math.min(1, r)) * 255),
      Math.round(Math.max(0, Math.min(1, g)) * 255),
      Math.round(Math.max(0, Math.min(1, b)) * 255),
      200,
    ]);
  }
  return table;
}

function buildMagma(): RGBA[] {
  // Simplified magma approximation
  const table: RGBA[] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const r = Math.min(1, t * 3.5 - 0.5);
    const g = Math.max(0, Math.min(1, t * 2 - 0.3));
    const b = Math.max(0.05, Math.min(1, 0.9 - t * 0.5 + Math.sin(t * Math.PI) * 0.5));
    table.push([
      Math.round(Math.max(0, Math.min(1, r)) * 255),
      Math.round(Math.max(0, Math.min(1, g)) * 255),
      Math.round(Math.max(0, Math.min(1, b)) * 255),
      200,
    ]);
  }
  return table;
}

function buildGreens(): RGBA[] {
  const table: RGBA[] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    table.push([
      Math.round((1 - t * 0.8) * 255),
      Math.round((0.4 + t * 0.6) * 255),
      Math.round((1 - t * 0.8) * 255),
      Math.round(t * 200),
    ]);
  }
  return table;
}

function buildInferno(): RGBA[] {
  const table: RGBA[] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const r = Math.min(1, 1.5 * t + 0.1 * Math.sin(t * 6));
    const g = Math.max(0, Math.min(1, 3 * (t - 0.4)));
    const b = Math.max(0, Math.min(1, 1 - 2 * (t - 0.2) * (t - 0.2)));
    table.push([
      Math.round(Math.max(0, Math.min(1, r)) * 255),
      Math.round(Math.max(0, Math.min(1, g)) * 255),
      Math.round(Math.max(0, Math.min(1, b)) * 255),
      200,
    ]);
  }
  return table;
}

function buildRdBu(): RGBA[] {
  const table: RGBA[] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    // Red at 0, white at 0.5, blue at 1
    let r: number, g: number, b: number;
    if (t < 0.5) {
      const s = t / 0.5;
      r = 0.4 + 0.6 * (1 - s);
      g = 0.2 + 0.8 * s;
      b = 0.2 + 0.8 * s;
    } else {
      const s = (t - 0.5) / 0.5;
      r = 1 - 0.8 * s;
      g = 1 - 0.8 * s;
      b = 0.4 + 0.6 * s;
    }
    table.push([
      Math.round(r * 255),
      Math.round(g * 255),
      Math.round(b * 255),
      200,
    ]);
  }
  return table;
}

const COLORMAPS: Record<string, RGBA[]> = {
  jet: buildJet(),
  magma: buildMagma(),
  inferno: buildInferno(),
  greens: buildGreens(),
  rdbu: buildRdBu(),
};

/**
 * Bilinear interpolation of a 2D float array to target dimensions.
 */
function bilinearInterpolate(
  data: number[][],
  targetH: number,
  targetW: number
): Float32Array {
  const srcH = data.length;
  const srcW = data[0]?.length || 0;
  const result = new Float32Array(targetH * targetW);

  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const srcY = (y / targetH) * srcH;
      const srcX = (x / targetW) * srcW;
      const y0 = Math.floor(srcY);
      const x0 = Math.floor(srcX);
      const y1 = Math.min(y0 + 1, srcH - 1);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const fy = srcY - y0;
      const fx = srcX - x0;

      const val =
        data[y0][x0] * (1 - fy) * (1 - fx) +
        data[y1][x0] * fy * (1 - fx) +
        data[y0][x1] * (1 - fy) * fx +
        data[y1][x1] * fy * fx;

      result[y * targetW + x] = val;
    }
  }
  return result;
}

export type ColormapName = "jet" | "magma" | "inferno" | "greens" | "rdbu";

/**
 * Render a heatmap overlay on a canvas.
 *
 * @param canvas  Target canvas element
 * @param heatmap 2D float array (patch resolution, values in [0, 1])
 * @param frameImage Optional HTMLImageElement for the background frame
 * @param colormap Colormap name
 * @param alpha Overlay alpha (0-1)
 */
export function renderHeatmap(
  canvas: HTMLCanvasElement,
  heatmap: number[][],
  frameImage?: HTMLImageElement | null,
  colormap: ColormapName = "jet",
  alpha: number = 0.6
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;

  // Draw frame image as background (this may taint the canvas, but we never
  // call getImageData on it — we use an offscreen canvas for the heatmap layer
  // to avoid SecurityError from tainted canvases entirely).
  if (frameImage) {
    ctx.drawImage(frameImage, 0, 0, w, h);
  } else {
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, w, h);
  }

  // Render colormap into an offscreen canvas so we never read pixels back
  // from the (potentially tainted) main canvas.
  const offscreen = document.createElement("canvas");
  offscreen.width = w;
  offscreen.height = h;
  const offCtx = offscreen.getContext("2d");
  if (!offCtx) return;

  const interpolated = bilinearInterpolate(heatmap, h, w);
  const cmap = COLORMAPS[colormap] || COLORMAPS.jet;
  const offData = offCtx.createImageData(w, h);
  const pixels = offData.data;

  for (let i = 0; i < interpolated.length; i++) {
    const val = Math.max(0, Math.min(1, interpolated[i]));
    const cmapIdx = Math.round(val * 255);
    const [cr, cg, cb] = cmap[cmapIdx];
    const px = i * 4;
    pixels[px]     = cr;
    pixels[px + 1] = cg;
    pixels[px + 2] = cb;
    pixels[px + 3] = Math.round(val * 255); // transparent where attention is low
  }

  offCtx.putImageData(offData, 0, 0);

  // Composite the heatmap layer over the frame using globalAlpha
  ctx.globalAlpha = alpha;
  ctx.drawImage(offscreen, 0, 0);
  ctx.globalAlpha = 1.0;
}

/**
 * Render a raw heatmap (no background image) on a canvas.
 */
export function renderRawHeatmap(
  canvas: HTMLCanvasElement,
  heatmap: number[][],
  colormap: ColormapName = "jet"
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;

  const interpolated = bilinearInterpolate(heatmap, h, w);
  const cmap = COLORMAPS[colormap] || COLORMAPS.jet;
  const imageData = ctx.createImageData(w, h);
  const pixels = imageData.data;

  for (let i = 0; i < interpolated.length; i++) {
    const val = Math.max(0, Math.min(1, interpolated[i]));
    const cmapIdx = Math.round(val * 255);
    const [cr, cg, cb] = cmap[cmapIdx];
    const px = i * 4;
    pixels[px] = cr;
    pixels[px + 1] = cg;
    pixels[px + 2] = cb;
    pixels[px + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

export const COLORMAP_NAMES: ColormapName[] = ["jet", "magma", "inferno", "greens", "rdbu"];
