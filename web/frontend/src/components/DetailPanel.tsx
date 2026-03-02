import { useEffect, useState } from "react";
import { useAppStore } from "../stores/appStore";
import { getVizData, frameUrl, type VizData } from "../services/api";
import FrameSelector from "./FrameSelector";
import DisplayModeToggle from "./DisplayModeToggle";
import HeatmapCanvas from "./HeatmapCanvas";
import ImageViz from "./ImageViz";
import ZoomableWrapper from "./ZoomableWrapper";
import VisionVsStateChart from "./VisionVsStateChart";
import CompareView from "./CompareView";
import HealthView from "./HealthView";
import RunInsightsView from "./RunInsightsView";
import RunNotesEditor from "./RunNotesEditor";
import LLMPanel from "./LLMPanel";

const VIZ_TYPES: { key: string; label: string }[] = [
  { key: "self_attention", label: "Self-Attention Heatmaps" },
  { key: "cross_attention", label: "Cross-Attention (Action Expert)" },
  { key: "per_head", label: "Per-Head Analysis" },
  { key: "per_step_cross_attention", label: "Per-Step Cross-Attention" },
  { key: "saliency", label: "Saliency Maps" },
  { key: "gradcam_siglip", label: "GradCAM (SigLIP)" },
  { key: "gradcam_connector", label: "GradCAM (Connector)" },
  { key: "gradcam_vlm_layers", label: "VLM Layer GradCAM" },
  { key: "per_action_dim", label: "Per-Action Dimension" },
  { key: "language_diff", label: "Language Conditional Comparison" },
  { key: "vision_vs_state", label: "Vision vs State Attribution" },
];

const HEATMAP_VIZ_TYPES = new Set([
  "self_attention",
  "cross_attention",
  "saliency",
  "gradcam_siglip",
  "gradcam_connector",
]);

const SKIP_VIZ_FETCH = new Set(["compare", "health", "legacy_images", "run_insights"]);

export default function DetailPanel() {
  const { selectedRunId, selectedRunDetail, selectedVizType, displayMode } = useAppStore();
  const [vizData, setVizData] = useState<VizData | null>(null);
  const [selectedFrames, setSelectedFrames] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);

  const manifest = selectedRunDetail?.manifest as Record<string, unknown> | undefined;
  const modelInfo = (manifest?.model_info as Record<string, unknown>) || {};
  const datasetInfo = (manifest?.dataset_info as Record<string, unknown>) || {};
  const imagePaths = (manifest?.images as string[]) || [];
  const numFrames = (datasetInfo.num_frames as number) || 8;

  // Initialize frame selection when run changes
  useEffect(() => {
    setSelectedFrames(Array.from({ length: numFrames }, (_, i) => i));
  }, [numFrames]);

  // Load viz data when selected viz type changes
  useEffect(() => {
    if (!selectedRunId || !selectedVizType) {
      setVizData(null);
      return;
    }

    if (SKIP_VIZ_FETCH.has(selectedVizType)) {
      setVizData(null);
      return;
    }

    setLoading(true);
    getVizData(selectedRunId, selectedVizType)
      .then(setVizData)
      .catch(() => setVizData(null))
      .finally(() => setLoading(false));
  }, [selectedRunId, selectedVizType]);

  if (!selectedRunId || !selectedRunDetail) {
    return <div className="empty-state">Select a run from the sidebar.</div>;
  }

  if (!selectedVizType) {
    return (
      <div className="detail-panel">
        <RunHeader
          runId={selectedRunId}
          modelInfo={modelInfo}
          datasetInfo={datasetInfo}
          numFrames={numFrames}
        />
        <div className="empty-state">
          <h3>Select a visualization</h3>
          <p>Pick a visualization type from the sidebar to view it here.</p>
        </div>
      </div>
    );
  }

  const vizLabel =
    VIZ_TYPES.find((v) => v.key === selectedVizType)?.label ||
    (selectedVizType === "legacy_images" ? "Generated Images" : selectedVizType);

  const showFrameSelector = HEATMAP_VIZ_TYPES.has(selectedVizType);

  return (
    <div className="detail-panel">
      <RunHeader
        runId={selectedRunId}
        modelInfo={modelInfo}
        datasetInfo={datasetInfo}
        numFrames={numFrames}
      />

      {/* Special views */}
      {selectedVizType === "compare" && <CompareView />}
      {selectedVizType === "health" && <HealthView />}
      {selectedVizType === "run_insights" && <RunInsightsView />}

      {/* Legacy images */}
      {selectedVizType === "legacy_images" && (
        <>
          <h3 className="detail-viz-title">{vizLabel}</h3>
          <ImageViz runId={selectedRunId} imagePaths={imagePaths} />
        </>
      )}

      {/* Standard viz types */}
      {!SKIP_VIZ_FETCH.has(selectedVizType) && (
          <>
            <h3 className="detail-viz-title">{vizLabel}</h3>

            {showFrameSelector && (
              <div className="card" style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                <FrameSelector
                  totalFrames={numFrames}
                  selectedFrames={selectedFrames}
                  onChange={setSelectedFrames}
                />
                <DisplayModeToggle />
              </div>
            )}

            {loading && <div className="empty-state">Loading visualization...</div>}

            {!loading && vizData && (
              <VizContent
                vizType={selectedVizType}
                data={vizData}
                selectedFrames={selectedFrames}
                runId={selectedRunId}
                imagePaths={imagePaths}
                displayMode={displayMode}
              />
            )}

            {!loading && !vizData && (
              <div className="empty-state" style={{ height: 200 }}>
                <p>No data available for this visualization.</p>
              </div>
            )}

            {selectedRunId && (
              <LLMPanel
                analysisType={`single_viz_${selectedVizType}`}
                runId={selectedRunId}
                vizType={selectedVizType}
              />
            )}
          </>
        )}
    </div>
  );
}

function RunHeader({
  runId,
  modelInfo,
  datasetInfo,
  numFrames,
}: {
  runId: string;
  modelInfo: Record<string, unknown>;
  datasetInfo: Record<string, unknown>;
  numFrames: number;
}) {
  return (
    <div className="run-header">
      <div className="meta-item">
        <span className="meta-label">Model</span>
        <span className="meta-value">
          {(modelInfo.model_id as string) || "unknown"}
        </span>
      </div>
      <div className="meta-item">
        <span className="meta-label">Dataset</span>
        <span className="meta-value">
          {(datasetInfo.dataset_id as string) || "unknown"}
        </span>
      </div>
      <div className="meta-item">
        <span className="meta-label">Episode</span>
        <span className="meta-value">{datasetInfo.episode_idx as number}</span>
      </div>
      <div className="meta-item">
        <span className="meta-label">Frames</span>
        <span className="meta-value">{numFrames}</span>
      </div>
      <div className="meta-item">
        <span className="meta-label">Task</span>
        <span className="meta-value" style={{ maxWidth: 300 }}>
          {(datasetInfo.task_string as string) || "\u2014"}
        </span>
      </div>
      <div className="meta-item" style={{ marginLeft: "auto" }}>
        <span className="meta-label">Note</span>
        <RunNotesEditor runId={runId} />
      </div>
    </div>
  );
}

function VizContent({
  vizType,
  data,
  selectedFrames,
  runId,
  imagePaths,
  displayMode,
}: {
  vizType: string;
  data: VizData;
  selectedFrames: number[];
  runId: string;
  imagePaths: string[];
  displayMode: "map" | "overlay" | "original";
}) {
  if (data.image_urls && data.image_urls.length > 0) {
    return <ImageViz runId={runId} imagePaths={data.image_urls} />;
  }

  if (data.heatmaps && data.heatmaps.length > 0) {
    const colormap =
      vizType === "cross_attention"
        ? "greens"
        : vizType.includes("gradcam")
          ? "magma"
          : vizType === "saliency"
            ? "inferno"
            : "jet";

    if (displayMode === "original") {
      return (
        <div className="heatmap-grid">
          {data.heatmaps.map((_, i) =>
            selectedFrames.includes(i) ? (
              <div key={i} className="heatmap-cell">
                <ZoomableWrapper>
                  <img
                    src={frameUrl(runId, i)}
                    alt={`Frame ${i}`}
                    draggable={false}
                    style={{ width: "100%", height: "auto", display: "block", borderRadius: 4 }}
                  />
                </ZoomableWrapper>
                <span className="frame-label">Frame {i}</span>
              </div>
            ) : null
          )}
        </div>
      );
    }

    return (
      <div className="heatmap-grid">
        {data.heatmaps.map((hm, i) =>
          selectedFrames.includes(i) ? (
            <HeatmapCanvas
              key={`${i}-${displayMode}`}
              heatmap={hm}
              frameImageUrl={displayMode === "overlay" ? frameUrl(runId, i) : undefined}
              colormap={colormap as "jet"}
              label={`Frame ${i}`}
            />
          ) : null
        )}
      </div>
    );
  }

  if (vizType === "vision_vs_state" && data.chart_data) {
    return <VisionVsStateChart data={data.chart_data} />;
  }

  const matchingImages = imagePaths.filter((p) => {
    const name = p.split("/").pop()?.toLowerCase() || "";
    const prefixes: Record<string, string[]> = {
      per_head: ["per_head"],
      per_step_cross_attention: ["per_step_cross_attn"],
      gradcam_vlm_layers: ["vlm_layers"],
      per_action_dim: ["per_action_dim"],
      language_diff: ["language_diff"],
    };
    return (prefixes[vizType] || []).some((pfx) => name.startsWith(pfx));
  });

  if (matchingImages.length > 0) {
    return <ImageViz runId={runId} imagePaths={matchingImages} />;
  }

  return (
    <p style={{ fontSize: 13, color: "var(--text-body)" }}>
      Data available but no renderer for this view yet.
    </p>
  );
}
