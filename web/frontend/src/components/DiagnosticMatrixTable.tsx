interface Props {
  signalTypes: string[];
  regions: string[];
  attributionMass: Record<string, Record<string, number>>;
  scalars: Record<string, number>;
}

export default function DiagnosticMatrixTable({ signalTypes, regions, attributionMass, scalars }: Props) {
  const cellColor = (value: number) => {
    // Blue (low) → Red (high) gradient
    const r = Math.round(255 * Math.min(value * 2.5, 1));
    const b = Math.round(255 * Math.max(1 - value * 2.5, 0));
    return `rgba(${r}, 60, ${b}, 0.15)`;
  };

  return (
    <div className="card" style={{ padding: 16, overflowX: "auto" }}>
      <h4 style={{ color: "var(--text-heading)", marginBottom: 8, fontSize: 14 }}>
        Diagnostic Matrix
      </h4>

      {/* Scalar badges */}
      {Object.keys(scalars).length > 0 && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
          {Object.entries(scalars).map(([key, value]) => (
            <span
              key={key}
              title={`${key}: ${value}`}
              style={{
                background: "#EDECFB",
                color: "#1820A0",
                borderRadius: 10,
                padding: "2px 10px",
                fontSize: 11,
                fontWeight: 500,
              }}
            >
              {key.replace(/_/g, " ")}: {typeof value === "number" ? value.toFixed(3) : value}
            </span>
          ))}
        </div>
      )}

      {/* Matrix table */}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "6px 8px", borderBottom: "2px solid #DBE4E8", color: "var(--text-heading)" }}>
              Signal
            </th>
            {regions.map((r) => (
              <th
                key={r}
                style={{
                  textAlign: "center",
                  padding: "6px 8px",
                  borderBottom: "2px solid #DBE4E8",
                  color: "var(--text-heading)",
                  whiteSpace: "nowrap",
                }}
              >
                {r}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {signalTypes.map((signal) => (
            <tr key={signal}>
              <td
                style={{
                  padding: "6px 8px",
                  borderBottom: "1px solid #DBE4E8",
                  color: "var(--text-heading)",
                  fontWeight: 500,
                  whiteSpace: "nowrap",
                }}
              >
                {signal.replace(/_/g, " ")}
              </td>
              {regions.map((region) => {
                const val = attributionMass[signal]?.[region] ?? 0;
                return (
                  <td
                    key={region}
                    title={`${signal} → ${region}: ${(val * 100).toFixed(1)}%`}
                    style={{
                      textAlign: "center",
                      padding: "6px 8px",
                      borderBottom: "1px solid #DBE4E8",
                      background: cellColor(val),
                      color: "var(--text-heading)",
                      fontFamily: "monospace",
                    }}
                  >
                    {(val * 100).toFixed(1)}%
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
