import { useEffect, useState } from "react";
import { useAppStore } from "../stores/appStore";
import DiagnosticMatrixTable from "./DiagnosticMatrixTable";
import FindingCard from "./FindingCard";
import CounterfactualComparison from "./CounterfactualComparison";

interface DiagnosticReport {
  metadata: Record<string, unknown>;
  anomalies: Array<{
    type: string;
    severity: string;
    description: string;
    evidence: Record<string, unknown>;
  }>;
  findings: Array<{
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
  }>;
  matrix: {
    signal_types: string[];
    regions: string[];
    attribution_mass: Record<string, Record<string, number>>;
    scalars: Record<string, number>;
  };
  counterfactual_results: Array<{
    hypothesis_id: string;
    test_type: string;
    action_delta_l2: number;
    confirmed: boolean;
  }>;
  llm_synthesis: string;
  scene?: {
    objects: Array<{ label: string; score: number }>;
  };
}

const API_BASE = import.meta.env.VITE_API_URL || "";

export default function DiagnosticPanel() {
  const { selectedRunId } = useAppStore();
  const [report, setReport] = useState<DiagnosticReport | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "running" | "complete" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [showFullReport, setShowFullReport] = useState(false);

  // Check if diagnostic exists
  useEffect(() => {
    if (!selectedRunId) return;
    setStatus("loading");
    fetch(`${API_BASE}/api/diagnostic/${selectedRunId}`)
      .then((res) => {
        if (res.ok) return res.json();
        if (res.status === 404) return null;
        throw new Error(`API ${res.status}`);
      })
      .then((data) => {
        if (data) {
          setReport(data);
          setStatus("complete");
        } else {
          setStatus("idle");
        }
      })
      .catch(() => setStatus("idle"));
  }, [selectedRunId]);

  const handleRunDiagnostic = async () => {
    if (!selectedRunId) return;
    setStatus("running");
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/diagnostic/${selectedRunId}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ skip_counterfactuals: false }),
      });

      if (!res.ok) throw new Error(`API ${res.status}`);
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: [DONE]")) {
            break;
          }
          if (line.startsWith("event: complete")) {
            // Next data line has the report
          }
          if (line.startsWith("data: ")) {
            try {
              const parsed = JSON.parse(line.slice(6));
              if (parsed.report) {
                setReport(parsed.report);
                setStatus("complete");
              }
              if (parsed.message) {
                setError(parsed.message);
                setStatus("error");
              }
            } catch {
              // ignore parse errors
            }
          }
        }
      }

      if (status !== "complete" && status !== "error") {
        // Reload report
        const reloadRes = await fetch(`${API_BASE}/api/diagnostic/${selectedRunId}`);
        if (reloadRes.ok) {
          setReport(await reloadRes.json());
          setStatus("complete");
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setStatus("error");
    }
  };

  if (!selectedRunId) return null;

  // Empty state
  if (status === "idle") {
    return (
      <div className="card" style={{ padding: 32, textAlign: "center" }}>
        <h3 style={{ color: "var(--text-heading)", marginBottom: 8 }}>
          Diagnostic Analysis
        </h3>
        <p style={{ color: "var(--text-body)", marginBottom: 16, maxWidth: 480, margin: "0 auto 16px" }}>
          Run an automated diagnostic to analyze this model's behavior — detecting issues like
          spatial shortcuts, background reliance, and weak object grounding — with specific fixes.
        </p>
        <button
          onClick={handleRunDiagnostic}
          style={{
            background: "#3B3BD3",
            color: "#FFFFFF",
            border: "none",
            borderRadius: 6,
            padding: "10px 24px",
            fontSize: 14,
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Run Diagnostic
        </button>
      </div>
    );
  }

  // Loading / Running
  if (status === "loading" || status === "running") {
    return (
      <div className="card" style={{ padding: 32, textAlign: "center" }}>
        <h3 style={{ color: "var(--text-heading)", marginBottom: 8 }}>
          {status === "loading" ? "Loading diagnostic..." : "Running diagnostic analysis..."}
        </h3>
        {status === "running" && (
          <p style={{ color: "var(--text-body)", fontSize: 13 }}>
            This may take a few minutes. Analyzing scene, building matrix, forming hypotheses...
          </p>
        )}
        <div style={{ margin: "16px auto", width: 200, height: 4, background: "#DBE4E8", borderRadius: 2, overflow: "hidden" }}>
          <div style={{
            width: status === "running" ? "60%" : "30%",
            height: "100%",
            background: "#3B3BD3",
            borderRadius: 2,
            transition: "width 2s ease",
          }} />
        </div>
      </div>
    );
  }

  // Error
  if (status === "error") {
    return (
      <div className="card" style={{ padding: 24, borderLeft: "4px solid #dc3545" }}>
        <h3 style={{ color: "#dc3545", marginBottom: 8 }}>Diagnostic Failed</h3>
        <p style={{ color: "var(--text-body)", fontSize: 13 }}>{error}</p>
        <button
          onClick={handleRunDiagnostic}
          style={{
            marginTop: 12,
            background: "#EDECFB",
            color: "#1820A0",
            border: "1px solid #C9C5F2",
            borderRadius: 6,
            padding: "8px 16px",
            fontSize: 13,
            cursor: "pointer",
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  // Complete — show report
  if (!report) return null;

  const critical = report.findings.filter((f) => f.severity === "critical");
  const warnings = report.findings.filter((f) => f.severity === "warning");
  const info = report.findings.filter((f) => f.severity === "info");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Status bar */}
      <div className="card" style={{ padding: "12px 16px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-heading)" }}>
            Diagnostic Complete
          </span>
          {critical.length > 0 && (
            <span style={{ background: "#dc3545", color: "#fff", borderRadius: 10, padding: "2px 8px", fontSize: 11, fontWeight: 600 }}>
              {critical.length} critical
            </span>
          )}
          {warnings.length > 0 && (
            <span style={{ background: "#ffc107", color: "#333", borderRadius: 10, padding: "2px 8px", fontSize: 11, fontWeight: 600 }}>
              {warnings.length} warning{warnings.length > 1 ? "s" : ""}
            </span>
          )}
          {info.length > 0 && (
            <span style={{ background: "#17a2b8", color: "#fff", borderRadius: 10, padding: "2px 8px", fontSize: 11, fontWeight: 600 }}>
              {info.length} info
            </span>
          )}
        </div>
        <button
          onClick={handleRunDiagnostic}
          style={{
            background: "#EDECFB",
            color: "#1820A0",
            border: "1px solid #C9C5F2",
            borderRadius: 6,
            padding: "6px 12px",
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          Re-run
        </button>
      </div>

      {/* Scene Understanding */}
      {report.scene && report.scene.objects && report.scene.objects.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <h4 style={{ color: "var(--text-heading)", marginBottom: 8, fontSize: 14 }}>
            Scene Understanding
          </h4>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <img
              src={`${API_BASE}/api/diagnostic/${selectedRunId}/scene/annotated-frame`}
              alt="Annotated scene"
              style={{ maxWidth: 300, borderRadius: 4, border: "1px solid #DBE4E8" }}
              onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
            />
            <div style={{ fontSize: 13 }}>
              <p style={{ color: "var(--text-body)", marginBottom: 8 }}>
                Detected {report.scene.objects.length} objects:
              </p>
              {report.scene.objects.map((obj, i) => (
                <div key={i} style={{ marginBottom: 4 }}>
                  <span style={{ fontWeight: 500, color: "var(--text-heading)" }}>
                    {obj.label}
                  </span>
                  <span style={{ color: "var(--text-body)", marginLeft: 8 }}>
                    ({(obj.score * 100).toFixed(0)}% confidence)
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Diagnostic Matrix */}
      {report.matrix && (
        <DiagnosticMatrixTable
          signalTypes={report.matrix.signal_types}
          regions={report.matrix.regions}
          attributionMass={report.matrix.attribution_mass}
          scalars={report.matrix.scalars}
        />
      )}

      {/* Anomalies */}
      {report.anomalies && report.anomalies.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <h4 style={{ color: "var(--text-heading)", marginBottom: 8, fontSize: 14 }}>
            Detected Anomalies ({report.anomalies.length})
          </h4>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {report.anomalies.map((a, i) => (
              <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start", fontSize: 13 }}>
                <SeverityBadge severity={a.severity} />
                <span style={{ color: "var(--text-heading)" }}>{a.description}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Findings */}
      {report.findings.length > 0 && (
        <div>
          <h4 style={{ color: "var(--text-heading)", marginBottom: 8, fontSize: 14, padding: "0 4px" }}>
            Findings ({report.findings.length})
          </h4>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {report.findings.map((f) => (
              <FindingCard key={f.id} finding={f} />
            ))}
          </div>
        </div>
      )}

      {/* Counterfactual Evidence */}
      {report.counterfactual_results && report.counterfactual_results.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <h4 style={{ color: "var(--text-heading)", marginBottom: 8, fontSize: 14 }}>
            Counterfactual Evidence ({report.counterfactual_results.length} tests)
          </h4>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {report.counterfactual_results.map((cf, i) => (
              <CounterfactualComparison
                key={i}
                runId={selectedRunId!}
                testType={cf.test_type}
                hypothesisId={cf.hypothesis_id}
                actionDelta={cf.action_delta_l2}
                confirmed={cf.confirmed}
              />
            ))}
          </div>
        </div>
      )}

      {/* Full Report */}
      {report.llm_synthesis && (
        <div className="card" style={{ padding: 16 }}>
          <button
            onClick={() => setShowFullReport(!showFullReport)}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "#3B3BD3",
              fontSize: 14,
              fontWeight: 500,
              padding: 0,
            }}
          >
            {showFullReport ? "Hide" : "Show"} Full Report
          </button>
          {showFullReport && (
            <div
              style={{
                marginTop: 12,
                fontSize: 13,
                lineHeight: 1.6,
                color: "var(--text-body)",
                whiteSpace: "pre-wrap",
              }}
            >
              {report.llm_synthesis}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const styles: Record<string, { bg: string; color: string }> = {
    critical: { bg: "#dc3545", color: "#fff" },
    warning: { bg: "#ffc107", color: "#333" },
    info: { bg: "#17a2b8", color: "#fff" },
  };
  const s = styles[severity] || styles.info;
  return (
    <span
      style={{
        background: s.bg,
        color: s.color,
        borderRadius: 10,
        padding: "1px 8px",
        fontSize: 11,
        fontWeight: 600,
        flexShrink: 0,
      }}
    >
      {severity}
    </span>
  );
}
