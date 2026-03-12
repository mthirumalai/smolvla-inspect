import { useState } from "react";

interface Finding {
  id: string;
  severity: string;
  title: string;
  observation: string;
  test_description: string;
  test_result: string;
  interpretation: string;
  fix: string;
  expected_impact: string;
  evidence_refs: string[];
}

interface Props {
  finding: Finding;
}

const SEVERITY_STYLES: Record<string, { border: string; bg: string; badge: string; badgeText: string }> = {
  critical: { border: "#dc3545", bg: "#fff5f5", badge: "#dc3545", badgeText: "#fff" },
  warning: { border: "#ffc107", bg: "#fffdf0", badge: "#ffc107", badgeText: "#333" },
  info: { border: "#17a2b8", bg: "#f0fafe", badge: "#17a2b8", badgeText: "#fff" },
};

export default function FindingCard({ finding }: Props) {
  const [expanded, setExpanded] = useState(false);
  const style = SEVERITY_STYLES[finding.severity] || SEVERITY_STYLES.info;

  return (
    <div
      className="card"
      style={{
        borderLeft: `4px solid ${style.border}`,
        background: expanded ? style.bg : undefined,
        padding: 0,
        overflow: "hidden",
      }}
    >
      {/* Header — always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "10px 14px",
          background: "none",
          border: "none",
          cursor: "pointer",
          textAlign: "left",
        }}
      >
        <span
          style={{
            background: style.badge,
            color: style.badgeText,
            borderRadius: 10,
            padding: "1px 8px",
            fontSize: 11,
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          {finding.severity}
        </span>
        <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-heading)", flex: 1 }}>
          {finding.title}
        </span>
        <span style={{ fontSize: 12, color: "var(--text-body)" }}>
          {expanded ? "▾" : "▸"}
        </span>
      </button>

      {/* Details — shown when expanded */}
      {expanded && (
        <div style={{ padding: "0 14px 14px", fontSize: 13, lineHeight: 1.6 }}>
          <Section label="Observation" text={finding.observation} />
          <Section label="Test" text={finding.test_description} />
          {finding.test_result && finding.test_result !== "N/A" && (
            <Section label="Result" text={finding.test_result} />
          )}
          <Section label="Interpretation" text={finding.interpretation} />
          <Section label="Fix" text={finding.fix} isCode />
          <Section label="Expected Impact" text={finding.expected_impact} />
        </div>
      )}
    </div>
  );
}

function Section({ label, text, isCode }: { label: string; text: string; isCode?: boolean }) {
  if (!text) return null;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontWeight: 600, color: "var(--text-heading)", fontSize: 12, marginBottom: 2 }}>
        {label}
      </div>
      <div
        style={{
          color: "var(--text-body)",
          ...(isCode
            ? { fontFamily: "monospace", background: "#f5f5f5", padding: "8px 10px", borderRadius: 4, whiteSpace: "pre-wrap", fontSize: 12 }
            : {}),
        }}
      >
        {text}
      </div>
    </div>
  );
}
