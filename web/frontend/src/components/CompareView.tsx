import { useState, useEffect, useMemo } from "react";
import { useAppStore } from "../stores/appStore";
import { getVizData, getRunDetail, frameUrl, type VizData, type RunDetail } from "../services/api";
import DisplayModeToggle from "./DisplayModeToggle";
import HeatmapCanvas from "./HeatmapCanvas";
import ImageViz from "./ImageViz";
import ZoomableWrapper from "./ZoomableWrapper";
import VisionVsStateChart from "./VisionVsStateChart";
import ConfigDiffCard from "./ConfigDiffCard";
import RunNotesEditor from "./RunNotesEditor";
import LLMPanel from "./LLMPanel";
import InfoPopover from "./InfoPopover";

const COMPARISON_TERMS_ITEMS = [
  {
    label: "Coverage / Spatial coverage",
    desc: "What fraction of the image has meaningful attention (above 30% of peak weight). A positive change means the model is looking at a broader area; negative means it's narrowing in.",
  },
  {
    label: "Temporal stability",
    desc: "How consistent the attention focus is across consecutive frames. Higher is more stable — the model's gaze doesn't jump around between frames.",
  },
  {
    label: "Centroid drift / Drift",
    desc: "How much the center of attention shifts between the two runs. Lower drift means the model focuses on the same region as before; higher drift means it's attending to a different part of the image.",
  },
  {
    label: "Gini",
    desc: "How concentrated the attention is. Near 1 = almost all weight on one small region. Near 0 = evenly spread. A rising Gini after fine-tuning can mean the model is over-fixating on one spot.",
  },
];

const COMPARABLE_TYPES = [
  "self_attention",
  "cross_attention",
  "saliency",
  "gradcam_siglip",
  "gradcam_connector",
  "gradcam_vlm_layers",
  "language_diff",
  "vision_vs_state",
];

export default function CompareView() {
  const { runs, selectedRunId, displayMode, runNotes } = useAppStore();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectedVizTypes, setSelectedVizTypes] = useState<string[]>([
    "self_attention",
  ]);
  const [compareData, setCompareData] = useState<Record<string, Record<string, VizData>>>({});
  const [runDetails, setRunDetails] = useState<Record<string, RunDetail>>({});
  const [loading, setLoading] = useState(false);
  // Track which runs have frame images available (for overlay mode)
  const [frameAvailability, setFrameAvailability] = useState<Record<string, boolean>>({});

  const toggleRun = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id)
        ? prev.filter((x) => x !== id)
        : prev.length < 3
          ? [...prev, id]
          : prev
    );
  };

  // Fetch run details for selected runs (for config diff)
  useEffect(() => {
    for (const id of selectedIds) {
      if (!runDetails[id]) {
        getRunDetail(id)
          .then((detail) => setRunDetails((prev) => ({ ...prev, [id]: detail })))
          .catch(() => {});
      }
    }
  }, [selectedIds, runDetails]);

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

    // Probe frame availability for each run so overlay mode can be consistent.
    // Use an Image object (same mechanism as HeatmapCanvas) so the result is
    // accurate and the loaded image is pre-cached by the browser.
    const availability: Record<string, boolean> = {};
    await Promise.all(
      selectedIds.map(
        (runId) =>
          new Promise<void>((resolve) => {
            const img = new window.Image();
            img.onload = () => { availability[runId] = true; resolve(); };
            img.onerror = () => { availability[runId] = false; resolve(); };
            img.src = frameUrl(runId, 0);
          })
      )
    );
    setFrameAvailability(availability);
  };

  // Build config diff data
  const configDiffData = useMemo(() => {
    if (selectedIds.length < 2 || Object.keys(compareData).length === 0) return [];
    return selectedIds
      .filter((id) => runDetails[id])
      .map((id) => ({
        id,
        name: runs.find((r) => r.id === id)?.name || id,
        detail: runDetails[id],
      }));
  }, [selectedIds, runDetails, compareData, runs]);

  // Overlay mode is only consistent when every selected run has frame images.
  // If any run is missing frames, force all columns to map mode.
  const allRunsHaveFrames =
    Object.keys(frameAvailability).length > 0 &&
    selectedIds.every((id) => frameAvailability[id] === true);
  const effectiveDisplayMode =
    displayMode === "overlay" && !allRunsHaveFrames ? "map" : displayMode;

  // Collect run notes for selected runs
  const selectedRunNotes = useMemo(() => {
    const notes: Record<string, string> = {};
    for (const id of selectedIds) {
      if (runNotes[id]) notes[id] = runNotes[id];
    }
    return notes;
  }, [selectedIds, runNotes]);

  return (
    <div>
      <div className="card">
        <h3 style={{ marginBottom: 12 }}>Select runs to compare</h3>
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
          {Object.values(compareData).some((runData) =>
            Object.values(runData).some((vd) => vd.heatmaps && vd.heatmaps.length > 0)
          ) && <DisplayModeToggle hideOverlay={!allRunsHaveFrames} />}
        </div>
      </div>

      {/* Config diff card */}
      {configDiffData.length >= 2 && <ConfigDiffCard runDetails={configDiffData} />}

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
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
                      <h4 style={{ margin: 0 }}>{run?.name || runId}</h4>
                      <RunNotesEditor runId={runId} compact />
                    </div>
                    {frameAvailability[runId] === false && (
                      <p style={{ fontSize: 11, color: "var(--text-body)", marginBottom: 8, opacity: 0.8 }}>
                        Frame images unavailable — overlay mode disabled for consistency.
                      </p>
                    )}
                    {data?.heatmaps ? (
                      <CompareHeatmaps
                        runId={runId}
                        vizType={vt}
                        heatmaps={data.heatmaps}
                        displayMode={effectiveDisplayMode}
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
          <h4 style={{ marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
            AI Analysis
            <InfoPopover items={COMPARISON_TERMS_ITEMS} />
          </h4>
          <LLMPanel
            analysisType="multi_run_comparison"
            runId={selectedRunId}
            compareRunIds={selectedIds}
            runNotes={selectedRunNotes}
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
