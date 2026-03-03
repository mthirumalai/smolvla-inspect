import { useRef, useEffect, useState } from "react";
import {
  renderHeatmap,
  renderRawHeatmap,
  type ColormapName,
} from "../utils/heatmapRenderer";
import ZoomableWrapper from "./ZoomableWrapper";

interface Props {
  heatmap: number[][];
  frameImageUrl?: string;
  colormap?: ColormapName;
  alpha?: number;
  width?: number;
  height?: number;
  label?: string;
}

export default function HeatmapCanvas({
  heatmap,
  frameImageUrl,
  colormap = "jet",
  alpha = 0.6,
  width = 320,
  height = 240,
  label,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frameLoadFailed, setFrameLoadFailed] = useState(false);

  useEffect(() => {
    setFrameLoadFailed(false);
    const canvas = canvasRef.current;
    if (!canvas || !heatmap || heatmap.length === 0) return;

    canvas.width = width;
    canvas.height = height;

    if (frameImageUrl) {
      const img = new Image();
      // Do NOT set crossOrigin — the Vite proxy makes these same-origin requests,
      // and crossOrigin="anonymous" can cause load failures if CORS headers
      // aren't present on the response.
      img.onload = () => {
        renderHeatmap(canvas, heatmap, img, colormap, alpha);
      };
      img.onerror = () => {
        renderRawHeatmap(canvas, heatmap, colormap);
        setFrameLoadFailed(true);
      };
      img.src = frameImageUrl;
    } else {
      renderRawHeatmap(canvas, heatmap, colormap);
    }
  }, [heatmap, frameImageUrl, colormap, alpha, width, height]);

  return (
    <div className="heatmap-cell">
      <ZoomableWrapper>
        <canvas ref={canvasRef} style={{ width: "100%", height: "auto" }} />
      </ZoomableWrapper>
      {label && <span className="frame-label">{label}</span>}
      {frameLoadFailed && (
        <span style={{ fontSize: 10, color: "var(--text-body)", opacity: 0.7 }}>
          Frame image unavailable
        </span>
      )}
    </div>
  );
}
