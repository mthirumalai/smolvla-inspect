import { useState, useRef, useEffect, useCallback } from "react";

const MIN_SCALE = 1;
const MAX_SCALE = 8;
const ZOOM_STEP = 0.15; // 15% per tick

interface ZoomPanState {
  scale: number;
  translateX: number;
  translateY: number;
}

export default function useZoomPan(containerRef: React.RefObject<HTMLDivElement | null>) {
  const [state, setState] = useState<ZoomPanState>({
    scale: 1,
    translateX: 0,
    translateY: 0,
  });

  const stateRef = useRef(state);
  stateRef.current = state;

  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const translateStart = useRef({ x: 0, y: 0 });

  // Clamp translation so content stays within bounds
  const clampTranslate = useCallback(
    (tx: number, ty: number, scale: number): { x: number; y: number } => {
      const el = containerRef.current;
      if (!el || scale <= 1) return { x: 0, y: 0 };
      const maxTx = ((scale - 1) / 2) * el.clientWidth;
      const maxTy = ((scale - 1) / 2) * el.clientHeight;
      return {
        x: Math.max(-maxTx, Math.min(maxTx, tx)),
        y: Math.max(-maxTy, Math.min(maxTy, ty)),
      };
    },
    [containerRef],
  );

  // Wheel zoom — native listener so we can preventDefault
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const { scale, translateX, translateY } = stateRef.current;
      const direction = e.deltaY < 0 ? 1 : -1;
      const newScale = Math.min(
        MAX_SCALE,
        Math.max(MIN_SCALE, scale * (1 + direction * ZOOM_STEP)),
      );
      const clamped = clampTranslate(translateX, translateY, newScale);
      setState({ scale: newScale, translateX: clamped.x, translateY: clamped.y });
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [containerRef, clampTranslate]);

  // Mouse drag for panning
  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (stateRef.current.scale <= 1) return;
      isDragging.current = true;
      dragStart.current = { x: e.clientX, y: e.clientY };
      translateStart.current = {
        x: stateRef.current.translateX,
        y: stateRef.current.translateY,
      };
      e.preventDefault();
    },
    [],
  );

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;
      const clamped = clampTranslate(
        translateStart.current.x + dx,
        translateStart.current.y + dy,
        stateRef.current.scale,
      );
      setState((prev) => ({ ...prev, translateX: clamped.x, translateY: clamped.y }));
    };

    const onMouseUp = () => {
      isDragging.current = false;
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [clampTranslate]);

  const zoomIn = useCallback(() => {
    setState((prev) => {
      const newScale = Math.min(MAX_SCALE, prev.scale * (1 + ZOOM_STEP));
      const clamped = clampTranslate(prev.translateX, prev.translateY, newScale);
      return { scale: newScale, translateX: clamped.x, translateY: clamped.y };
    });
  }, [clampTranslate]);

  const zoomOut = useCallback(() => {
    setState((prev) => {
      const newScale = Math.max(MIN_SCALE, prev.scale * (1 - ZOOM_STEP));
      const clamped = clampTranslate(prev.translateX, prev.translateY, newScale);
      return { scale: newScale, translateX: clamped.x, translateY: clamped.y };
    });
  }, [clampTranslate]);

  const resetZoom = useCallback(() => {
    setState({ scale: 1, translateX: 0, translateY: 0 });
  }, []);

  return {
    scale: state.scale,
    translateX: state.translateX,
    translateY: state.translateY,
    isDragging: isDragging.current,
    isZoomed: state.scale > 1,
    onMouseDown,
    zoomIn,
    zoomOut,
    resetZoom,
  };
}
