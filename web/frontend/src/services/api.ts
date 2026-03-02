const API_BASE = import.meta.env.VITE_API_URL || "";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// -- Runs --
export interface RunSummary {
  id: string;
  name: string;
  created_at: string | null;
  model_id: string | null;
  dataset_id: string | null;
  episode_idx: number | null;
  num_frames: number | null;
  available_visualizations: Record<string, boolean>;
  is_legacy: boolean;
}

export interface RunDetail {
  id: string;
  name: string;
  manifest: Record<string, unknown>;
}

export const listRuns = (baseDir?: string) =>
  fetchJson<RunSummary[]>(
    `/api/runs${baseDir ? `?base_dir=${encodeURIComponent(baseDir)}` : ""}`
  );

export const scanRuns = (baseDir?: string) =>
  fetchJson<RunSummary[]>("/api/runs/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(baseDir ? { base_dir: baseDir } : {}),
  });

export const getRunDetail = (runId: string) =>
  fetchJson<RunDetail>(`/api/runs/${runId}`);

// -- Viz --
export interface VizData {
  viz_type: string;
  frames: number[];
  heatmaps: number[][][] | null;
  image_urls: string[] | null;
  chart_data: Record<string, unknown>[] | null;
  metadata: Record<string, unknown>;
}

export const getVizData = (runId: string, vizType: string) =>
  fetchJson<VizData>(`/api/runs/${runId}/viz/${vizType}`);

// -- Stats --
export interface VizStats {
  viz_type: string;
  per_frame: Record<string, unknown>[];
  aggregate: Record<string, unknown>;
}

export const getVizStats = (runId: string, vizType: string) =>
  fetchJson<VizStats>(`/api/runs/${runId}/stats/${vizType}`);

export const getAllVizStats = (runId: string) =>
  fetchJson<Record<string, unknown>>(`/api/runs/${runId}/stats`);

// -- Notes --
export interface RunNotes {
  notes: string;
  updated_at: string | null;
}

export const getRunNotes = (runId: string) =>
  fetchJson<RunNotes>(`/api/runs/${runId}/notes`);

export const setRunNotes = (runId: string, notes: string) =>
  fetchJson<RunNotes>(`/api/runs/${runId}/notes`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ notes }),
  });

// -- Health --
export interface HealthData {
  weightwatcher: Record<string, unknown> | null;
  entropy: Record<string, unknown> | null;
  redundancy: Record<string, unknown> | null;
}

export const getHealthData = (runId: string) =>
  fetchJson<HealthData>(`/api/runs/${runId}/health`);

// -- Compare --
export const compareRuns = (
  runIds: string[],
  vizTypes: string[],
  frameIndices?: number[]
) =>
  fetchJson<{ runs: unknown[]; diffs: Record<string, unknown> }>("/api/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_ids: runIds, viz_types: vizTypes, frame_indices: frameIndices }),
  });

// -- LLM --
export interface LLMConfig {
  provider: string;
  model: string;
  api_key: string;
  base_url: string;
}

export const getLLMConfig = () => fetchJson<LLMConfig>("/api/llm/config");

export const setLLMConfig = (config: LLMConfig) =>
  fetchJson<{ status: string }>("/api/llm/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });

export const getPromptTemplate = (analysisType: string) =>
  fetchJson<{ analysis_type: string; template: string }>(
    `/api/llm/prompts/${analysisType}`
  );

export async function* streamAnalysis(req: {
  run_id: string;
  analysis_type: string;
  prompt?: string;
  viz_type?: string;
  include_images?: boolean;
  include_stats?: boolean;
  compare_run_ids?: string[];
  run_notes?: Record<string, string>;
}): AsyncGenerator<string> {
  const res = await fetch(`${API_BASE}/api/llm/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!res.ok) throw new Error(`LLM API ${res.status}`);
  if (!res.body) return;

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
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") return;
        // Tokens are JSON-encoded to preserve newlines through SSE
        try {
          yield JSON.parse(data);
        } catch {
          yield data;
        }
      }
    }
  }
}

export const imageUrl = (runId: string, path: string) =>
  `${API_BASE}/api/runs/${runId}/image/${path}`;

export const frameUrl = (runId: string, frameIdx: number) =>
  `${API_BASE}/api/runs/${runId}/frame/${frameIdx}`;
