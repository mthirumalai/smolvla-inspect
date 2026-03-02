import { create } from "zustand";
import type { RunSummary, RunDetail, VizStats } from "../services/api";

interface AppState {
  // Runs
  runs: RunSummary[];
  setRuns: (runs: RunSummary[]) => void;

  // Selected run
  selectedRunId: string | null;
  setSelectedRunId: (id: string | null) => void;
  selectedRunDetail: RunDetail | null;
  setSelectedRunDetail: (detail: RunDetail | null) => void;

  // Selected viz type in the sidebar nav
  selectedVizType: string | null;
  setSelectedVizType: (vt: string | null) => void;

  // Compare runs
  compareRunIds: string[];
  setCompareRunIds: (ids: string[]) => void;

  // Frame selection
  selectedFrames: number[];
  setSelectedFrames: (frames: number[]) => void;

  // Base directory
  baseDir: string;
  setBaseDir: (dir: string) => void;

  // Display mode for canvas-based heatmaps
  displayMode: "map" | "overlay" | "original";
  setDisplayMode: (mode: "map" | "overlay" | "original") => void;

  // LLM config
  llmConfigured: boolean;
  setLlmConfigured: (v: boolean) => void;

  // Run notes (runId -> note text)
  runNotes: Record<string, string>;
  setRunNote: (runId: string, note: string) => void;

  // Viz stats cache (runId -> vizType -> VizStats)
  vizStats: Record<string, Record<string, VizStats>>;
  setVizStats: (runId: string, vizType: string, stats: VizStats) => void;

  // LLM response cache (keyed by "runId:analysisType")
  llmResponses: Record<string, string>;
  setLlmResponse: (key: string, response: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  runs: [],
  setRuns: (runs) => set({ runs }),

  selectedRunId: null,
  setSelectedRunId: (id) => set({ selectedRunId: id }),
  selectedRunDetail: null,
  setSelectedRunDetail: (detail) => set({ selectedRunDetail: detail }),

  selectedVizType: null,
  setSelectedVizType: (vt) => set({ selectedVizType: vt }),

  compareRunIds: [],
  setCompareRunIds: (ids) => set({ compareRunIds: ids }),

  selectedFrames: [],
  setSelectedFrames: (frames) => set({ selectedFrames: frames }),

  baseDir: "./outputs",
  setBaseDir: (dir) => set({ baseDir: dir }),

  displayMode: "overlay",
  setDisplayMode: (mode) => set({ displayMode: mode }),

  llmConfigured: false,
  setLlmConfigured: (v) => set({ llmConfigured: v }),

  runNotes: {},
  setRunNote: (runId, note) =>
    set((state) => ({
      runNotes: { ...state.runNotes, [runId]: note },
    })),

  vizStats: {},
  setVizStats: (runId, vizType, stats) =>
    set((state) => ({
      vizStats: {
        ...state.vizStats,
        [runId]: { ...(state.vizStats[runId] || {}), [vizType]: stats },
      },
    })),

  llmResponses: {},
  setLlmResponse: (key, response) =>
    set((state) => ({
      llmResponses: { ...state.llmResponses, [key]: response },
    })),
}));
