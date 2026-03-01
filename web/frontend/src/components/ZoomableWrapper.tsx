import { useRef, type ReactNode } from "react";
import useZoomPan from "../hooks/useZoomPan";

interface Props {
  children: ReactNode;
}

export default function ZoomableWrapper({ children }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const {
    scale,
    translateX,
    translateY,
    isZoomed,
    onMouseDown,
    zoomIn,
    zoomOut,
    resetZoom,
  } = useZoomPan(containerRef);

  return (
    <div
      ref={containerRef}
      className="zoomable-container"
      onMouseDown={onMouseDown}
      style={{ cursor: isZoomed ? "grab" : "default" }}
    >
      <div
        className="zoomable-content"
        style={{
          transform: `translate(${translateX}px, ${translateY}px) scale(${scale})`,
        }}
      >
        {children}
      </div>
      <div className="zoom-toolbar">
        <button className="zoom-btn" onClick={zoomIn} title="Zoom in">+</button>
        <span className="zoom-level">{Math.round(scale * 100)}%</span>
        <button className="zoom-btn" onClick={zoomOut} title="Zoom out">&minus;</button>
        <button className="zoom-btn zoom-btn-fit" onClick={resetZoom} title="Reset zoom">
          Fit
        </button>
      </div>
    </div>
  );
}
