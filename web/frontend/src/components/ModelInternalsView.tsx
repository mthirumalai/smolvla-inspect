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
import {
  getModelInternalsData,
  type ModelInternalsData,
} from "../services/api";
import LLMPanel from "./LLMPanel";

export default function ModelInternalsView() {
  const { selectedRunId, selectedRunDetail } = useAppStore();
  const [internals, setInternals] = useState<ModelInternalsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const avail =
    (
      selectedRunDetail?.manifest as Record<string, unknown> | undefined
    )?.available_visualizations as Record<string, boolean> | undefined;

  useEffect(() => {
    if (!selectedRunId || !avail?.model_internals) {
      setInternals(null);
      return;
    }
    setLoading(true);
    setError("");
    getModelInternalsData(selectedRunId)
      .then(setInternals)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [selectedRunId, avail]);

  if (!selectedRunId) {
    return <div className="empty-state">Select a run.</div>;
  }

  if (!avail?.model_internals) {
    return (
      <div className="empty-state">
        <h3>No model internals data</h3>
        <p>
          Run the CLI with <code>--internals-only</code> or{" "}
          <code>--with-internals</code>, plus <code>--export-data</code>.
        </p>
      </div>
    );
  }

  if (loading) return <div className="empty-state">Loading model internals...</div>;
  if (error) {
    return <div className="empty-state">Error loading model internals: {error}</div>;
  }
  if (!internals) return null;

  return (
    <div>
      <h3 className="detail-viz-title" style={{ marginBottom: 16 }}>
        Model Internals
      </h3>

      <div className="card" style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <InternalsBadge label="Entropy" data={internals.entropy} />
        <InternalsBadge label="Redundancy" data={internals.redundancy} />
        <InternalsBadge label="WeightWatcher" data={internals.weightwatcher} />
      </div>

      <div className="internals-grid">
        {internals.entropy && (
          <div className="internals-card">
            <h3 style={{ marginBottom: 12 }}>Attention Entropy</h3>
            <GenericBarChart
              data={internals.entropy}
              dataKey="entropy_ratio"
              label="Entropy Ratio"
              warnLine={0.8}
              critLine={0.95}
            />
          </div>
        )}

        {internals.redundancy && (
          <div className="internals-card">
            <h3 style={{ marginBottom: 12 }}>Head Redundancy</h3>
            <GenericBarChart
              data={internals.redundancy}
              dataKey="max_redundancy"
              label="Max Cosine Similarity"
              warnLine={0.7}
              critLine={0.9}
            />
          </div>
        )}

        {internals.weightwatcher && (
          <div className="internals-card">
            <h3 style={{ marginBottom: 12 }}>Spectral Alpha (WeightWatcher)</h3>
            <GenericBarChart
              data={internals.weightwatcher}
              dataKey="alpha"
              label="Alpha"
            />
          </div>
        )}
      </div>

      {selectedRunId && (
        <div className="card" style={{ marginTop: 16 }}>
          <LLMPanel analysisType="model_internals" runId={selectedRunId} />
        </div>
      )}
    </div>
  );
}

function InternalsBadge({
  label,
  data,
}: {
  label: string;
  data: Record<string, unknown> | null;
}) {
  if (!data) {
    return (
      <div>
        <span className="internals-badge" style={{ background: "#f0f0f0", color: "#666" }}>
          {label}: N/A
        </span>
      </div>
    );
  }

  const status =
    (data as Record<string, unknown>).overall_status ||
    (data as Record<string, unknown>).status ||
    "ok";
  const statusStr = String(status).toLowerCase();
  const severity = statusStr.includes("critical")
    ? "critical"
    : statusStr.includes("warn")
      ? "warning"
      : "ok";

  return (
    <div>
      <span className={`internals-badge ${severity}`}>
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
  let chartData: { name: string; value: number }[] = [];

  if (Array.isArray(data)) {
    chartData = data.map((d: Record<string, unknown>, i: number) => ({
      name: (d.name as string) || String(d.layer ?? i),
      value: Number(d[dataKey] ?? d.value ?? 0),
    }));
  } else if (typeof data === "object") {
    const groupedEntries = Object.entries(data).filter(([, value]) => Array.isArray(value));
    if (groupedEntries.length > 0) {
      chartData = groupedEntries.flatMap(([groupName, values]) =>
        (values as unknown[]).map((d: unknown, i: number) => {
          const obj = d as Record<string, unknown>;
          const layerLabel =
            obj.layer !== undefined ? `L${String(obj.layer)}` : String(i);
          return {
            name: `${shortLabel(groupName)} ${layerLabel}`,
            value: Number(obj[dataKey] ?? obj.value ?? 0),
          };
        })
      );
    }

    const items =
      (data.layers as unknown[]) ||
      (data.components as unknown[]) ||
      (data.heads as unknown[]) ||
      [];
    if (chartData.length === 0 && Array.isArray(items)) {
      chartData = items.map((d: unknown, i: number) => {
        const obj = d as Record<string, unknown>;
        return {
          name: (obj.name as string) || String(obj.layer ?? i),
          value: Number(obj[dataKey] ?? obj.value ?? 0),
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

function shortLabel(label: string): string {
  if (label.startsWith("SigLIP")) return "SigLIP";
  if (label.startsWith("VLM+Expert")) return "VLM+Expert";
  if (label.startsWith("Expert-to-VLM")) return "Expert XA";
  if (label.startsWith("Vision Encoder")) return "Vision";
  if (label.startsWith("VLM Text Model")) return "VLM";
  if (label.startsWith("Expert")) return "Expert";
  if (label.startsWith("Connector")) return "Connector";
  return label;
}
