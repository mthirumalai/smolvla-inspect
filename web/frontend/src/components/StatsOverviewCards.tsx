import { useEffect, useState } from "react";
import { getAllVizStats } from "../services/api";
import InfoPopover, { type InfoPopoverItem } from "./InfoPopover";

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
  vision_vs_state: "ANALYSIS",
};

const LABEL_MAP: Record<string, string> = {
  self_attention: "Self-Attention",
  cross_attention: "Cross-Attention",
  per_head: "Per-Head",
  saliency: "Saliency",
  gradcam_siglip: "GradCAM SigLIP",
  gradcam_connector: "GradCAM Connector",
  vision_vs_state: "Vision vs State",
};

const CATEGORY_POPOVER_ITEMS: Record<string, InfoPopoverItem[]> = {
  ATTENTION: [
    {
      label: "Head similarity",
      desc: "How alike the attention heads are to each other. High values mean heads are redundant — doing the same thing and wasting capacity. Low values mean good head diversity.",
    },
    {
      label: "Dead heads",
      desc: "Heads with near-uniform attention that aren't specializing on anything useful. Ideally this count is zero.",
    },
  ],
  ANALYSIS: [
    {
      label: "Vision share",
      desc: "What fraction of the action prediction is driven by visual input vs. proprioceptive state (joint positions/velocities). 70% means the model is primarily reacting to what it sees.",
    },
  ],
};

export default function StatsOverviewCards({ runId }: Props) {
  const [stats, setStats] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [loading, setLoading] = useState(false);

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

  const HIDDEN = new Set(["per_action_dim"]);

  // Group by category
  const grouped: Record<string, string[]> = {};
  for (const vizType of Object.keys(stats)) {
    if (HIDDEN.has(vizType)) continue;
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
        const popoverItems = CATEGORY_POPOVER_ITEMS[cat];
        return (
          <div key={cat} style={{ marginBottom: 16 }}>
            <div style={{
              fontSize: 10,
              fontWeight: 600,
              color: "var(--text-body)",
              letterSpacing: "0.8px",
              marginBottom: 8,
              opacity: 0.7,
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}>
              {cat}
              {popoverItems && <InfoPopover items={popoverItems} />}
            </div>
            <div className="stats-card-grid">
              {types.map((vizType) => (
                <StatsCard
                  key={vizType}
                  vizType={vizType}
                  data={stats[vizType]}
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
}: {
  vizType: string;
  data: Record<string, unknown>;
}) {
  const label = LABEL_MAP[vizType] || vizType.replace(/_/g, " ");
  const aggregate = (data.aggregate || data) as Record<string, unknown>;
  const badges = extractBadges(aggregate);

  return (
    <div className="stats-card">
      <div className="stats-card-header">
        <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-heading)" }}>
          {label}
        </span>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          {badges.map((b) => (
            <span key={b.label} className="stats-badge">
              {b.label}: {b.value}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

interface Badge {
  label: string;
  value: string;
}

function extractBadges(aggregate: Record<string, unknown>): Badge[] {
  const badges: Badge[] = [];

  if ("mean_entropy_ratio" in aggregate) {
    badges.push({ label: "Entropy", value: String(aggregate.mean_entropy_ratio) });
  }
  if ("mean_coverage_pct" in aggregate) {
    badges.push({ label: "Coverage", value: `${aggregate.mean_coverage_pct}%` });
  }
  if ("temporal_stability" in aggregate && aggregate.temporal_stability !== null) {
    badges.push({ label: "Stability", value: String(aggregate.temporal_stability) });
  }
  if ("mean_gini" in aggregate) {
    badges.push({ label: "Gini", value: String(aggregate.mean_gini) });
  }
  if ("mean_inter_head_similarity" in aggregate) {
    badges.push({ label: "Head sim.", value: String(aggregate.mean_inter_head_similarity) });
  }
  if ("dead_heads" in aggregate) {
    const dh = aggregate.dead_heads as number[];
    badges.push({ label: "Dead heads", value: `${dh.length}` });
  }
  if ("mean_vision_share" in aggregate) {
    badges.push({
      label: "Vision share",
      value: `${Math.round((aggregate.mean_vision_share as number) * 100)}%`,
    });
  }
  if ("strength_ranking" in aggregate) {
    const sr = aggregate.strength_ranking as { dim: number; mean_strength: number }[];
    if (sr.length > 0) {
      badges.push({ label: "Top dim", value: `dim ${sr[0].dim}` });
    }
  }

  return badges.slice(0, 4);
}
