import { useEffect, useState } from "react";
import { useAppStore } from "./stores/appStore";
import { listRuns, getRunDetail, getLLMConfig } from "./services/api";
import Header from "./components/Header";
import RunSelector from "./components/RunSelector";
import DetailPanel from "./components/DetailPanel";
import LLMConfigModal from "./components/LLMConfigModal";
import "./App.css";

function App() {
  const {
    selectedRunId,
    setSelectedRunId,
    setSelectedRunDetail,
    runs,
    setRuns,
    baseDir,
    setLlmConfigured,
  } = useAppStore();

  const [showLLMConfig, setShowLLMConfig] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const r = await listRuns(baseDir);
        setRuns(r);
      } catch {
        setRuns([]);
      } finally {
        setLoading(false);
      }
    })();
  }, [baseDir, setRuns]);

  useEffect(() => {
    getLLMConfig()
      .then((c) => setLlmConfigured(!!c.api_key && c.api_key !== ""))
      .catch(() => {});
  }, [setLlmConfigured]);

  useEffect(() => {
    if (!selectedRunId) {
      setSelectedRunDetail(null);
      return;
    }
    getRunDetail(selectedRunId)
      .then(setSelectedRunDetail)
      .catch(() => setSelectedRunDetail(null));
  }, [selectedRunId, setSelectedRunDetail]);

  useEffect(() => {
    if (runs.length > 0 && !selectedRunId) {
      setSelectedRunId(runs[0].id);
    }
  }, [runs, selectedRunId, setSelectedRunId]);

  return (
    <div className="app">
      <Header onLLMConfig={() => setShowLLMConfig(true)} />
      <div className="app-body">
        <RunSelector />
        <main className="content-area">
          {loading ? (
            <div className="empty-state">Loading runs...</div>
          ) : runs.length === 0 ? (
            <div className="empty-state">
              <h3>No runs found</h3>
              <p>
                Run the CLI with <code>--export-data</code> to generate
                structured output, then point the viewer at the output directory.
              </p>
            </div>
          ) : (
            <DetailPanel />
          )}
        </main>
      </div>
      {showLLMConfig && (
        <LLMConfigModal onClose={() => setShowLLMConfig(false)} />
      )}
    </div>
  );
}

export default App;
