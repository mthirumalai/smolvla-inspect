import { useState, useMemo, useEffect } from "react";
import { useAppStore } from "../stores/appStore";
import { listRuns } from "../services/api";

interface VizNavItem {
  key: string;
  label: string;
}

interface VizNavGroup {
  category: string;
  items: VizNavItem[];
}

const VIZ_NAV_GROUPS: VizNavGroup[] = [
  {
    category: "GRADIENT",
    items: [
      { key: "saliency", label: "Saliency" },
      { key: "gradcam_siglip", label: "GradCAM SigLIP" },
      { key: "gradcam_connector", label: "GradCAM Connector" },
      { key: "gradcam_vlm_layers", label: "VLM Layers" },
      { key: "per_action_dim", label: "Per-Action Dim" },
    ],
  },
  {
    category: "ATTENTION",
    items: [
      { key: "self_attention", label: "Self-Attention" },
      { key: "cross_attention", label: "Cross-Attention" },
      { key: "per_head", label: "Per-Head" },
      { key: "per_step_cross_attention", label: "Per-Step Cross" },
    ],
  },
  {
    category: "ANALYSIS",
    items: [
      { key: "language_diff", label: "Language Diff" },
      { key: "vision_vs_state", label: "Vision vs State" },
    ],
  },
  {
    category: "INSIGHTS",
    items: [
      { key: "run_insights", label: "Run Insights" },
    ],
  },
  {
    category: "TOOLS",
    items: [
      { key: "compare", label: "Compare Runs" },
      { key: "health", label: "Model Health" },
    ],
  },
];

export default function RunSelector() {
  const {
    runs,
    setRuns,
    selectedRunId,
    setSelectedRunId,
    selectedVizType,
    setSelectedVizType,
    baseDir,
    setBaseDir,
  } = useAppStore();
  const [dirInput, setDirInput] = useState(baseDir);
  const [scanning, setScanning] = useState(false);

  const handleScan = async () => {
    setScanning(true);
    try {
      setBaseDir(dirInput);
      const r = await listRuns(dirInput);
      setRuns(r);
    } catch {
      setRuns([]);
    } finally {
      setScanning(false);
    }
  };

  // Get available viz types from the selected run
  const selectedRun = runs.find((r) => r.id === selectedRunId);
  const availViz = selectedRun?.available_visualizations || {};
  const isLegacy = selectedRun?.is_legacy ?? false;

  // Filter nav groups to only show available viz types
  const filteredGroups = useMemo(() => {
    if (!selectedRunId) return [];

    return VIZ_NAV_GROUPS.map((group) => ({
      ...group,
      items: group.items.filter((item) => {
        // Tools and insights are always available
        if (item.key === "compare" || item.key === "health" || item.key === "run_insights") return true;
        return availViz[item.key];
      }),
    })).filter((group) => group.items.length > 0);
  }, [selectedRunId, availViz]);

  // Auto-select first available viz when a run is selected but no viz type is set
  // (e.g. on initial load when App.tsx auto-selects the first run)
  useEffect(() => {
    if (!selectedRunId || selectedVizType) return;
    const run = runs.find((r) => r.id === selectedRunId);
    if (!run) return;
    if (run.is_legacy) {
      setSelectedVizType("legacy_images");
      return;
    }
    const avail = run.available_visualizations || {};
    for (const group of VIZ_NAV_GROUPS) {
      for (const item of group.items) {
        if (avail[item.key]) {
          setSelectedVizType(item.key);
          return;
        }
      }
    }
  }, [selectedRunId, selectedVizType, runs, setSelectedVizType]);

  const handleSelectRun = (id: string) => {
    setSelectedRunId(id);

    // Auto-select the first viz type that has data
    const run = runs.find((r) => r.id === id);
    const avail = run?.available_visualizations || {};
    const isLeg = run?.is_legacy ?? false;

    if (isLeg) {
      setSelectedVizType("legacy_images");
      return;
    }

    // Walk nav groups in display order to find the first available item
    for (const group of VIZ_NAV_GROUPS) {
      for (const item of group.items) {
        if (avail[item.key]) {
          setSelectedVizType(item.key);
          return;
        }
      }
    }
    setSelectedVizType(null);
  };

  return (
    <aside className="sidebar">
      {/* Directory input */}
      <div style={{ padding: "12px 12px 8px" }}>
        <label
          style={{
            fontSize: 11,
            color: "var(--text-body)",
            display: "block",
            marginBottom: 4,
          }}
        >
          Output directory
        </label>
        <div style={{ display: "flex", gap: 4 }}>
          <input
            type="text"
            value={dirInput}
            onChange={(e) => setDirInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleScan()}
            style={{ flex: 1, fontSize: 12, padding: "6px 8px" }}
          />
          <button
            className="btn-primary"
            onClick={handleScan}
            disabled={scanning}
            style={{ fontSize: 11, padding: "6px 10px" }}
          >
            {scanning ? "..." : "Scan"}
          </button>
        </div>
      </div>

      {/* Scrollable area: runs + viz nav */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {/* Runs section */}
        <div className="sidebar-section-header">Runs</div>
        <div style={{ padding: "4px 8px" }}>
          {runs.length === 0 && (
            <p
              style={{
                fontSize: 12,
                textAlign: "center",
                padding: "20px 0",
                color: "var(--text-body)",
              }}
            >
              No runs found
            </p>
          )}
          {runs.map((run) => (
            <div
              key={run.id}
              onClick={() => handleSelectRun(run.id)}
              style={{
                padding: "8px 10px",
                borderRadius: 6,
                cursor: "pointer",
                marginBottom: 2,
                background:
                  selectedRunId === run.id ? "var(--secondary-bg)" : "transparent",
                border:
                  selectedRunId === run.id
                    ? "1px solid var(--secondary-border)"
                    : "1px solid transparent",
              }}
            >
              <div
                style={{
                  fontSize: 13,
                  fontWeight: selectedRunId === run.id ? 600 : 400,
                  color: "var(--text-heading)",
                  marginBottom: 2,
                }}
              >
                {run.name}
                {run.is_legacy && (
                  <span
                    style={{
                      fontSize: 10,
                      background: "#fff3e0",
                      color: "#b36b00",
                      padding: "1px 5px",
                      borderRadius: 8,
                      marginLeft: 6,
                    }}
                  >
                    legacy
                  </span>
                )}
              </div>
              {run.model_id && (
                <div style={{ fontSize: 11, color: "var(--text-body)" }}>
                  {run.model_id}
                </div>
              )}
              {run.created_at && (
                <div style={{ fontSize: 10, color: "var(--text-body)" }}>
                  {new Date(run.created_at).toLocaleDateString()}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Viz type navigation */}
        {selectedRunId && (
          <>
            <div className="sidebar-section-header">Visualizations</div>
            <div className="viz-nav">
              {/* Legacy runs: single Images item */}
              {isLegacy && (
                <div className="viz-nav-group">
                  <div className="viz-nav-category">LEGACY</div>
                  <div
                    className={`viz-nav-item ${selectedVizType === "legacy_images" ? "active" : ""}`}
                    onClick={() => setSelectedVizType("legacy_images")}
                  >
                    Images
                  </div>
                </div>
              )}

              {/* Standard viz nav groups */}
              {filteredGroups.map((group) => (
                <div key={group.category} className="viz-nav-group">
                  <div className="viz-nav-category">{group.category}</div>
                  {group.items.map((item) => (
                    <div
                      key={item.key}
                      className={`viz-nav-item ${selectedVizType === item.key ? "active" : ""}`}
                      onClick={() => setSelectedVizType(item.key)}
                    >
                      {item.label}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
