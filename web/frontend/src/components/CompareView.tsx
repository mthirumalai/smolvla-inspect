import { useState } from "react";
import { useAppStore } from "../stores/appStore";
import { getVizData, type VizData } from "../services/api";
import HeatmapCanvas from "./HeatmapCanvas";
import ImageViz from "./ImageViz";
import LLMPanel from "./LLMPanel";

const COMPARABLE_TYPES = [
  "self_attention",
  "cross_attention",
  "saliency",
  "gradcam_siglip",
  "vision_vs_state",
];

export default function CompareView() {
  const { runs, selectedRunId } = useAppStore();
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
    for (const runId of selectedIds) {
      result[runId] = {};
      for (const vt of selectedVizTypes) {
        try {
          result[runId][vt] = await getVizData(runId, vt);
        } catch {
          // skip
        }
      }
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

        <button
          className="btn-primary"
          onClick={handleCompare}
          disabled={selectedIds.length < 2 || loading}
        >
          {loading ? "Loading..." : "Compare"}
        </button>
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
                      <div className="heatmap-grid">
                        {data.heatmaps.slice(0, 4).map((hm, i) => (
                          <HeatmapCanvas
                            key={i}
                            heatmap={hm}
                            label={`Frame ${i}`}
                          />
                        ))}
                      </div>
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
