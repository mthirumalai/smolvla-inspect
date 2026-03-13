const API_BASE = import.meta.env.VITE_API_URL || "";

interface Props {
  runId: string;
  testType: string;
  hypothesisId: string;
  actionDelta: number;
  confirmed: boolean;
  metrics?: Record<string, number | string | boolean | number[] | string[]>;
}

export default function CounterfactualComparison({
  runId,
  testType,
  hypothesisId,
  actionDelta,
  confirmed,
  metrics,
}: Props) {
  const imgUrl = `${API_BASE}/api/diagnostic/${runId}/counterfactual/${testType}/comparison`;
  const metricEntries = Object.entries(metrics || {}).filter(([, value]) => value !== null && value !== undefined);

  return (
    <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
      <img
        src={imgUrl}
        alt={`${testType} comparison`}
        style={{
          maxWidth: 300,
          borderRadius: 4,
          border: "1px solid #DBE4E8",
        }}
        onError={(e) => {
          (e.target as HTMLImageElement).style.display = "none";
        }}
      />
      <div style={{ fontSize: 13 }}>
        <div style={{ fontWeight: 500, color: "var(--text-heading)", marginBottom: 4 }}>
          {testType.replace(/_/g, " ")}
          <span style={{ fontWeight: 400, color: "var(--text-body)", marginLeft: 8 }}>
            (for {hypothesisId})
          </span>
        </div>
        <div style={{ color: "var(--text-body)", marginBottom: 4 }}>
          Action delta: <code style={{ fontWeight: 600 }}>{actionDelta.toFixed(4)}</code>
        </div>
        <div>
          <span
            style={{
              borderRadius: 10,
              padding: "1px 8px",
              fontSize: 11,
              fontWeight: 600,
              background: confirmed ? "#dc3545" : "#28a745",
              color: "#fff",
            }}
          >
            {confirmed ? "Confirmed" : "Not confirmed"}
          </span>
        </div>
        {metricEntries.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div style={{ color: "var(--text-heading)", fontWeight: 600, fontSize: 12, marginBottom: 6 }}>
              Probe metrics
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
                gap: 6,
                maxWidth: 420,
              }}
            >
              {metricEntries.map(([key, value]) => (
                <div
                  key={key}
                  style={{
                    background: "#F8FAFB",
                    border: "1px solid #DBE4E8",
                    borderRadius: 6,
                    padding: "6px 8px",
                  }}
                >
                  <div style={{ color: "var(--text-body)", fontSize: 11, textTransform: "uppercase", letterSpacing: 0.3 }}>
                    {key.replace(/_/g, " ")}
                  </div>
                  <div style={{ color: "var(--text-heading)", fontSize: 12, fontWeight: 600 }}>
                    {formatMetricValue(value)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function formatMetricValue(value: number | string | boolean | number[] | string[]) {
  if (typeof value === "number") {
    if (Math.abs(value) >= 1) return value.toFixed(3);
    return value.toFixed(4);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  return value;
}
