const API_BASE = import.meta.env.VITE_API_URL || "";

interface Props {
  runId: string;
  testType: string;
  hypothesisId: string;
  actionDelta: number;
  confirmed: boolean;
}

export default function CounterfactualComparison({
  runId,
  testType,
  hypothesisId,
  actionDelta,
  confirmed,
}: Props) {
  const imgUrl = `${API_BASE}/api/diagnostic/${runId}/counterfactual/${testType}/comparison`;

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
      </div>
    </div>
  );
}
