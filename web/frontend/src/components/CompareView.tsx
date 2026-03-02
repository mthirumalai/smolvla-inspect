import { useState } from "react";
import { useAppStore } from "../stores/appStore";
import { getVizData, frameUrl, type VizData } from "../services/api";
import DisplayModeToggle from "./DisplayModeToggle";
import HeatmapCanvas from "./HeatmapCanvas";
import ImageViz from "./ImageViz";
import ZoomableWrapper from "./ZoomableWrapper";
import VisionVsStateChart from "./VisionVsStateChart";
import LLMPanel from "./LLMPanel";

const COMPARABLE_TYPES = [
  "self_attention",
  "cross_attention",
  "saliency",
  "gradcam_siglip",
  "vision_vs_state",
];

export default function CompareView() {
  const { runs, selectedRunId, displayMode } = useAppStore();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectedVizTypes, setSelectedVizTypes] = useState<string[]>([
    "self_attention",
  ]);
  const [compareData, setCompareData] = useState<Record<string, Record<string, VizData>>>({});
  const [loading, setLoading] = useState(false);

  const toggleRun = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id)
        ? prev.filter((x) => x !== id)
        : prev.length < 3
          ? [...prev, id]
          : prev
    );
  };

  const handleCompare = async () => {
    if (selectedIds.length < 2) return;
    setLoading(true);
    const result: Record<string, Record<string, VizData>> = {};
    // Fetch ALL comparable types in parallel so toggling checkboxes is instant
    const jobs = selectedIds.flatMap((runId) =>
      COMPARABLE_TYPES.map(async (vt) => {
        try {
          const data = await getVizData(runId, vt);
          return { runId, vt, data };
        } catch {
          return { runId, vt, data: null as VizData | null };
        }
      })
    );
    const settled = await Promise.all(jobs);
    for (const { runId, vt, data } of settled) {
      if (!result[runId]) result[runId] = {};
      if (data) result[runId][vt] = data;
    }
    setCompareData(result);
    setLoading(false);
  };

  return (
    <div>
      <div className="card">
        <h3 style={{ marginBottom: 12 }}>Select 2-3 runs to compare</h3>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
          {runs.map((run) => (
            <label
              key={run.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: 13,
                cursor: "pointer",
                padding: "4px 10px",
                borderRadius: 6,
                background: selectedIds.includes(run.id)
                  ? "var(--secondary-bg)"
                  : "transparent",
                border: `1px solid ${
                  selectedIds.includes(run.id)
                    ? "var(--secondary-border)"
                    : "var(--border)"
                }`,
              }}
            >
              <input
                type="checkbox"
                checked={selectedIds.includes(run.id)}
                onChange={() => toggleRun(run.id)}
              />
              {run.name}
            </label>
          ))}
        </div>

        <h4 style={{ marginBottom: 8 }}>Visualization types</h4>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
          {COMPARABLE_TYPES.map((vt) => (
            <label
              key={vt}
              style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}
            >
              <input
                type="checkbox"
                checked={selectedVizTypes.includes(vt)}
                onChange={() =>
                  setSelectedVizTypes((prev) =>
                    prev.includes(vt)
                      ? prev.filter((x) => x !== vt)
                      : [...prev, vt]
                  )
                }
              />
              {vt.replace(/_/g, " ")}
            </label>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button
            className="btn-primary"
            onClick={handleCompare}
            disabled={selectedIds.length < 2 || loading}
          >
            {loading ? "Loading..." : "Compare"}
          </button>
          {Object.keys(compareData).length > 0 && <DisplayModeToggle />}
        </div>
      </div>

      {/* Side by side results */}
      {Object.keys(compareData).length > 0 &&
        selectedVizTypes.map((vt) => (
          <div key={vt} className="card">
            <h3 style={{ marginBottom: 12 }}>{vt.replace(/_/g, " ")}</h3>
            <div className="side-by-side">
              {selectedIds.map((runId) => {
                const run = runs.find((r) => r.id === runId);
                const data = compareData[runId]?.[vt];
                return (
                  <div key={runId} className="compare-run-column">
                    <h4>{run?.name || runId}</h4>
                    {data?.heatmaps ? (
                      <CompareHeatmaps
                        runId={runId}
                        vizType={vt}
                        heatmaps={data.heatmaps}
                        displayMode={displayMode}
                      />
                    ) : data?.chart_data ? (
                      <VisionVsStateChart data={data.chart_data} />
                    ) : data?.image_urls ? (
                      <ImageViz runId={runId} imagePaths={data.image_urls} />
                    ) : (
                      <p style={{ fontSize: 12, color: "var(--text-body)" }}>
                        Not available
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}

      {Object.keys(compareData).length > 0 && selectedRunId && (
        <div className="card">
          <LLMPanel
            analysisType="comparison"
            runId={selectedRunId}
          />
        </div>
      )}
    </div>
  );
}

function CompareHeatmaps({
  runId,
  vizType,
  heatmaps,
  displayMode,
}: {
  runId: string;
  vizType: string;
  heatmaps: number[][][];
  displayMode: "map" | "overlay" | "original";
}) {
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
        {heatmaps.slice(0, 4).map((_, i) => (
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
        ))}
      </div>
    );
  }

  return (
    <div className="heatmap-grid">
      {heatmaps.slice(0, 4).map((hm, i) => (
        <HeatmapCanvas
          key={`${i}-${displayMode}`}
          heatmap={hm}
          frameImageUrl={displayMode === "overlay" ? frameUrl(runId, i) : undefined}
          colormap={colormap as "jet"}
          label={`Frame ${i}`}
        />
      ))}
    </div>
  );
}
