import { useAppStore } from "../stores/appStore";
import StatsOverviewCards from "./StatsOverviewCards";
import LLMPanel from "./LLMPanel";

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
        <h4 style={{ marginBottom: 12 }}>Statistics overview</h4>
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
