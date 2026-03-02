import { useMemo } from "react";
import type { RunDetail } from "../services/api";

interface Props {
  runDetails: { id: string; name: string; detail: RunDetail }[];
}

export default function ConfigDiffCard({ runDetails }: Props) {
  const diff = useMemo(() => computeConfigDiff(runDetails), [runDetails]);

  if (Object.keys(diff).length === 0) {
    return null;
  }

  return (
    <div className="card">
      <h4 style={{ marginBottom: 12 }}>Configuration differences</h4>
      <div style={{ overflowX: "auto" }}>
        <table className="config-diff-table">
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>Field</th>
              {runDetails.map((r) => (
                <th key={r.id} style={{ textAlign: "left" }}>
                  {r.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(diff).map(([field, values]) => (
              <tr key={field}>
                <td style={{ fontWeight: 500, color: "var(--text-heading)" }}>
                  {field}
                </td>
                {values.map((val, i) => (
                  <td key={i} style={{ color: "var(--text-body)" }}>
                    {formatVal(val)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function computeConfigDiff(
  runDetails: { id: string; name: string; detail: RunDetail }[]
): Record<string, unknown[]> {
  if (runDetails.length < 2) return {};

  const configKeys = ["cli_args", "model_info", "dataset_info"];

  function flatten(obj: Record<string, unknown>, prefix = ""): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === "object" && !Array.isArray(v)) {
        Object.assign(result, flatten(v as Record<string, unknown>, key));
      } else {
        result[key] = v;
      }
    }
    return result;
  }

  const flattened = runDetails.map((r) => {
    const manifest = r.detail.manifest;
    const flat: Record<string, unknown> = {};
    for (const ck of configKeys) {
      const section = manifest[ck];
      if (section && typeof section === "object") {
        Object.assign(flat, flatten(section as Record<string, unknown>, ck));
      }
    }
    return flat;
  });

  // Collect all keys
  const allKeys = new Set<string>();
  for (const f of flattened) {
    for (const k of Object.keys(f)) allKeys.add(k);
  }

  // Find fields that differ
  const diffs: Record<string, unknown[]> = {};
  for (const key of Array.from(allKeys).sort()) {
    const values = flattened.map((f) => f[key]);
    const strs = values.map((v) => JSON.stringify(v));
    if (new Set(strs).size > 1) {
      diffs[key] = values;
    }
  }

  return diffs;
}

function formatVal(val: unknown): string {
  if (val === null || val === undefined) return "\u2014";
  if (typeof val === "string") return val;
  if (typeof val === "number") return String(val);
  if (typeof val === "boolean") return val ? "true" : "false";
  return JSON.stringify(val);
}
