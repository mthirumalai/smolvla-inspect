import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { useAppStore } from "../stores/appStore";
import { getHealthData, type HealthData } from "../services/api";
import LLMPanel from "./LLMPanel";

export default function HealthView() {
  const { selectedRunId, selectedRunDetail } = useAppStore();
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const avail =
    (
      selectedRunDetail?.manifest as Record<string, unknown> | undefined
    )?.available_visualizations as Record<string, boolean> | undefined;

  useEffect(() => {
    if (!selectedRunId || !avail?.model_health) {
      setHealth(null);
      return;
    }
    setLoading(true);
    setError("");
    getHealthData(selectedRunId)
      .then(setHealth)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [selectedRunId, avail]);

  if (!selectedRunId) {
    return <div className="empty-state">Select a run.</div>;
  }

  if (!avail?.model_health) {
    return (
      <div className="empty-state">
        <h3>No health data</h3>
        <p>
          Run the CLI with <code>--model-health</code> and{" "}
          <code>--export-data</code> to generate health diagnostics.
        </p>
      </div>
    );
  }

  if (loading) return <div className="empty-state">Loading health data...</div>;
  if (error)
    return <div className="empty-state">Error loading health data: {error}</div>;
  if (!health) return null;

  return (
    <div>
      {/* Summary badges */}
      <div className="card" style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <HealthBadge label="Entropy" data={health.entropy} />
        <HealthBadge label="Redundancy" data={health.redundancy} />
        <HealthBadge label="WeightWatcher" data={health.weightwatcher} />
      </div>

      <div className="health-grid">
        {/* Entropy chart */}
        {health.entropy && (
          <div className="health-card">
            <h3 style={{ marginBottom: 12 }}>Attention Entropy</h3>
            <GenericBarChart
              data={health.entropy}
              dataKey="entropy_ratio"
              label="Entropy Ratio"
              warnLine={0.8}
              critLine={0.95}
            />
          </div>
        )}

        {/* Redundancy chart */}
        {health.redundancy && (
          <div className="health-card">
            <h3 style={{ marginBottom: 12 }}>Head Redundancy</h3>
            <GenericBarChart
              data={health.redundancy}
              dataKey="max_cosine_sim"
              label="Max Cosine Similarity"
              warnLine={0.7}
              critLine={0.9}
            />
          </div>
        )}

        {/* Spectral alpha chart */}
        {health.weightwatcher && (
          <div className="health-card">
            <h3 style={{ marginBottom: 12 }}>Spectral Alpha (WeightWatcher)</h3>
            <GenericBarChart
              data={health.weightwatcher}
              dataKey="alpha"
              label="Alpha"
            />
          </div>
        )}
      </div>

      {selectedRunId && (
        <div className="card" style={{ marginTop: 16 }}>
          <LLMPanel analysisType="health" runId={selectedRunId} />
        </div>
      )}
    </div>
  );
}

function HealthBadge({
  label,
  data,
}: {
  label: string;
  data: Record<string, unknown> | null;
}) {
  if (!data) {
    return (
      <div>
        <span className="health-badge" style={{ background: "#f0f0f0", color: "#666" }}>
          {label}: N/A
        </span>
      </div>
    );
  }

  // Simple heuristic: check if any "status" field exists
  const status =
    (data as Record<string, unknown>).overall_status ||
    (data as Record<string, unknown>).status ||
    "ok";
  const statusStr = String(status).toLowerCase();
  const severity = statusStr.includes("critical")
    ? "critical"
    : statusStr.includes("warn")
      ? "warning"
      : "healthy";

  return (
    <div>
      <span className={`health-badge ${severity}`}>
        {label}: {statusStr}
      </span>
    </div>
  );
}

function GenericBarChart({
  data,
  dataKey,
  label,
  warnLine,
  critLine,
}: {
  data: Record<string, unknown>;
  dataKey: string;
  label: string;
  warnLine?: number;
  critLine?: number;
}) {
  // Try to extract per-layer or per-component data
  let chartData: { name: string; value: number }[] = [];

  if (Array.isArray(data)) {
    chartData = data.map((d: Record<string, unknown>, i: number) => ({
      name: (d.name as string) || (d.layer as string) || `${i}`,
      value: (d[dataKey] as number) || (d.value as number) || 0,
    }));
  } else if (typeof data === "object") {
    // Try "layers" or "components" key
    const items =
      (data.layers as unknown[]) ||
      (data.components as unknown[]) ||
      (data.heads as unknown[]) ||
      [];
    if (Array.isArray(items)) {
      chartData = items.map((d: unknown, i: number) => {
        const obj = d as Record<string, unknown>;
        return {
          name: (obj.name as string) || (obj.layer as string) || `${i}`,
          value: (obj[dataKey] as number) || (obj.value as number) || 0,
        };
      });
    }
  }

  if (chartData.length === 0) {
    return (
      <p style={{ fontSize: 12, color: "var(--text-body)" }}>
        No chart data available. Raw: {JSON.stringify(data).slice(0, 200)}
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
        <XAxis dataKey="name" fontSize={10} />
        <YAxis fontSize={10} />
        <Tooltip />
        <Bar dataKey="value" fill="#3B3BD3" name={label} />
        {warnLine !== undefined && (
          <ReferenceLine y={warnLine} stroke="#b36b00" strokeDasharray="5 5" />
        )}
        {critLine !== undefined && (
          <ReferenceLine y={critLine} stroke="#c41e1e" strokeDasharray="5 5" />
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}
