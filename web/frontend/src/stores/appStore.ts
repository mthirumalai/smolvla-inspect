import { create } from "zustand";
import type { RunSummary, RunDetail } from "../services/api";

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

  // LLM config
  llmConfigured: boolean;
  setLlmConfigured: (v: boolean) => void;
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

  llmConfigured: false,
  setLlmConfigured: (v) => set({ llmConfigured: v }),
}));
