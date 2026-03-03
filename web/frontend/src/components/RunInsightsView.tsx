import { useAppStore } from "../stores/appStore";
import StatsOverviewCards from "./StatsOverviewCards";
import LLMPanel from "./LLMPanel";
import InfoPopover from "./InfoPopover";

const COMMON_METRICS_ITEMS = [
  {
    label: "Entropy",
    desc: "How spread out the attention is. Near 0 = tightly focused on one region, near 1 = uniform across the whole image. High entropy can mean confusion or broad scene awareness.",
  },
  {
    label: "Coverage",
    desc: "What fraction of the image has meaningful attention (pixels above 30% of the peak weight). Low = model is zeroing in on a small area, high = model is using broad context.",
  },
  {
    label: "Stability",
    desc: "How consistent the attention focus is across consecutive frames (cosine similarity). Near 1 = focus barely moves between frames, near 0 = attention jumps around erratically.",
  },
  {
    label: "Gini",
    desc: "How concentrated the attention is. Near 1 = almost all weight on one small region (laser-focused), near 0 = evenly spread across all patches.",
  },
];

export default function RunInsightsView() {
  const { selectedRunId, selectedRunDetail } = useAppStore();

  if (!selectedRunId || !selectedRunDetail) {
    return (
      <div className="empty-state">
        <h3>Run insights</h3>
        <p>Select a run from the sidebar to view insights.</p>
      </div>
    );
  }

  const manifest = selectedRunDetail.manifest as Record<string, unknown>;
  const avail = (manifest.available_visualizations as Record<string, boolean>) || {};
  const hasData = Object.values(avail).some(Boolean);

  if (!hasData) {
    return (
      <div className="empty-state">
        <h3>No visualization data</h3>
        <p>Run the CLI with visualization flags to generate data for insights.</p>
      </div>
    );
  }

  return (
    <div>
      <h3 className="detail-viz-title" style={{ marginBottom: 16 }}>Run Insights</h3>

      <div className="card">
        <h4 style={{ marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
          Statistics overview
          <InfoPopover items={COMMON_METRICS_ITEMS} />
        </h4>
        <StatsOverviewCards runId={selectedRunId} />
      </div>

      <div className="card">
        <LLMPanel
          analysisType="run_insights"
          runId={selectedRunId}
        />
      </div>
    </div>
  );
}
