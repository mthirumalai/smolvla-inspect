import { useEffect, useState } from "react";
import { getAllVizStats } from "../services/api";

interface Props {
  runId: string;
}

const CATEGORY_MAP: Record<string, string> = {
  self_attention: "ATTENTION",
  cross_attention: "ATTENTION",
  per_head: "ATTENTION",
  saliency: "GRADIENT",
  gradcam_siglip: "GRADIENT",
  gradcam_connector: "GRADIENT",
  per_action_dim: "GRADIENT",
  vision_vs_state: "ANALYSIS",
};

const LABEL_MAP: Record<string, string> = {
  self_attention: "Self-Attention",
  cross_attention: "Cross-Attention",
  per_head: "Per-Head",
  saliency: "Saliency",
  gradcam_siglip: "GradCAM SigLIP",
  gradcam_connector: "GradCAM Connector",
  per_action_dim: "Per-Action Dim",
  vision_vs_state: "Vision vs State",
};

export default function StatsOverviewCards({ runId }: Props) {
  const [stats, setStats] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setLoading(true);
    getAllVizStats(runId)
      .then((data) => setStats(data as Record<string, Record<string, unknown>>))
      .catch(() => setStats(null))
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 24 }}>
        <p style={{ fontSize: 13, color: "var(--text-body)" }}>Loading stats...</p>
      </div>
    );
  }

  if (!stats || Object.keys(stats).length === 0) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 24 }}>
        <p style={{ fontSize: 13, color: "var(--text-body)" }}>
          No stats available. Run the CLI with visualization flags to generate data.
        </p>
      </div>
    );
  }

  // Group by category
  const grouped: Record<string, string[]> = {};
  for (const vizType of Object.keys(stats)) {
    const cat = CATEGORY_MAP[vizType] || "OTHER";
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(vizType);
  }

  const categoryOrder = ["GRADIENT", "ATTENTION", "ANALYSIS", "OTHER"];

  return (
    <div>
      {categoryOrder.map((cat) => {
        const types = grouped[cat];
        if (!types || types.length === 0) return null;
        return (
          <div key={cat} style={{ marginBottom: 16 }}>
            <div style={{
              fontSize: 10,
              fontWeight: 600,
              color: "var(--text-body)",
              letterSpacing: "0.8px",
              marginBottom: 8,
              opacity: 0.7,
            }}>
              {cat}
            </div>
            <div className="stats-card-grid">
              {types.map((vizType) => (
                <StatsCard
                  key={vizType}
                  vizType={vizType}
                  data={stats[vizType]}
                  isExpanded={expanded[vizType] ?? false}
                  onToggle={() =>
                    setExpanded((prev) => ({ ...prev, [vizType]: !prev[vizType] }))
                  }
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StatsCard({
  vizType,
  data,
  isExpanded,
  onToggle,
}: {
  vizType: string;
  data: Record<string, unknown>;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const label = LABEL_MAP[vizType] || vizType.replace(/_/g, " ");
  const aggregate = (data.aggregate || data) as Record<string, unknown>;

  // Pick 3-4 key metrics to show as badges
  const badges = extractBadges(vizType, aggregate);

  return (
    <div className="stats-card">
      <div
        className="stats-card-header"
        onClick={onToggle}
        style={{ cursor: "pointer" }}
      >
        <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-heading)" }}>
          {label}
        </span>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          {badges.map((b) => (
            <span key={b.label} className="stats-badge" title={b.tooltip}>
              {b.label}: {b.value}
            </span>
          ))}
          <span style={{ fontSize: 11, color: "var(--text-body)", marginLeft: 4 }}>
            {isExpanded ? "\u25B2" : "\u25BC"}
          </span>
        </div>
      </div>

      {isExpanded && (
        <div className="stats-card-body">
          {Object.entries(aggregate).map(([key, val]) => {
            if (key === "per_frame") return null;
            return (
              <div key={key} style={{ fontSize: 12, marginBottom: 2 }}>
                <span style={{ color: "var(--text-body)" }}>
                  {key.replace(/_/g, " ")}:
                </span>{" "}
                <span style={{ color: "var(--text-heading)", fontWeight: 500 }}>
                  {formatValue(val)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

interface Badge {
  label: string;
  value: string;
  tooltip: string;
}

function extractBadges(
  vizType: string,
  aggregate: Record<string, unknown>
): Badge[] {
  const badges: Badge[] = [];

  if ("mean_entropy_ratio" in aggregate) {
    badges.push({
      label: "Entropy",
      value: String(aggregate.mean_entropy_ratio),
      tooltip: "Mean Shannon entropy ratio (0=peaked, 1=uniform)",
    });
  }
  if ("mean_coverage_pct" in aggregate) {
    badges.push({
      label: "Coverage",
      value: `${aggregate.mean_coverage_pct}%`,
      tooltip: "Mean % of pixels above 30% of max",
    });
  }
  if ("temporal_stability" in aggregate && aggregate.temporal_stability !== null) {
    badges.push({
      label: "Stability",
      value: String(aggregate.temporal_stability),
      tooltip: "Mean cosine similarity between consecutive frames",
    });
  }
  if ("mean_gini" in aggregate) {
    badges.push({
      label: "Gini",
      value: String(aggregate.mean_gini),
      tooltip: "Gini coefficient (0=uniform, 1=concentrated)",
    });
  }
  // Per-head specific
  if ("mean_inter_head_similarity" in aggregate) {
    badges.push({
      label: "Head sim.",
      value: String(aggregate.mean_inter_head_similarity),
      tooltip: "Mean inter-head cosine similarity",
    });
  }
  if ("dead_heads" in aggregate) {
    const dh = aggregate.dead_heads as number[];
    badges.push({
      label: "Dead",
      value: `${dh.length}`,
      tooltip: `Dead heads (near-uniform entropy): ${dh.join(", ") || "none"}`,
    });
  }
  // Vision vs state
  if ("mean_vision_share" in aggregate) {
    badges.push({
      label: "Vision",
      value: `${Math.round((aggregate.mean_vision_share as number) * 100)}%`,
      tooltip: "Mean vision share of total gradient norm",
    });
  }
  if ("trend_direction" in aggregate) {
    badges.push({
      label: "Trend",
      value: String(aggregate.trend_direction).replace(/_/g, " "),
      tooltip: "Vision share trend across frames",
    });
  }
  // Per-action dim
  if ("strength_ranking" in aggregate) {
    const sr = aggregate.strength_ranking as { dim: number; mean_strength: number }[];
    if (sr.length > 0) {
      badges.push({
        label: "Top dim",
        value: `dim ${sr[0].dim}`,
        tooltip: `Strongest dim: ${sr[0].dim} (strength: ${sr[0].mean_strength})`,
      });
    }
  }

  return badges.slice(0, 4);
}

function formatValue(val: unknown): string {
  if (val === null || val === undefined) return "\u2014";
  if (typeof val === "number") return String(Math.round(val * 10000) / 10000);
  if (Array.isArray(val)) {
    if (val.length === 0) return "none";
    if (val.length <= 5) return val.map(String).join(", ");
    return `${val.length} items`;
  }
  if (typeof val === "object") return JSON.stringify(val);
  return String(val);
}
