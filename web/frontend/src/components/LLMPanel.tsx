import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import { streamAnalysis, getPromptTemplate, getLLMAnalyses, saveLLMAnalysis } from "../services/api";
import { useAppStore } from "../stores/appStore";

interface Props {
  analysisType: string;
  runId: string;
  vizType?: string;
  compareRunIds?: string[];
  runNotes?: Record<string, string>;
}

type TabSections = { observations: string; recommendations: string };

function parseTabSections(text: string): TabSections | null {
  const lines = text.split("\n");
  let obsStart = -1;
  let recStart = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (/^##\s+Observations\s*$/.test(t)) obsStart = i;
    else if (/^##\s+Recommendations\s*$/.test(t)) recStart = i;
  }
  if (obsStart === -1 || recStart === -1 || recStart <= obsStart) return null;
  return {
    observations: lines.slice(obsStart + 1, recStart).join("\n").trim(),
    recommendations: lines.slice(recStart + 1).join("\n").trim(),
  };
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}

export default function LLMPanel({ analysisType, runId, vizType, compareRunIds, runNotes }: Props) {
  const llmConfigured = useAppStore((s) => s.llmConfigured);
  const setLlmResponse = useAppStore((s) => s.setLlmResponse);
  const llmCacheLoaded = useAppStore((s) => s.llmCacheLoaded);
  const setLlmCacheLoaded = useAppStore((s) => s.setLlmCacheLoaded);
  const hydrateLlmResponses = useAppStore((s) => s.hydrateLlmResponses);

  // Reactive selector for cached response — updates when hydration completes
  const cachedResponse = useAppStore(
    (s) => s.llmResponses[`${runId}:${analysisType}`] || ""
  );
  const cacheMeta = useAppStore(
    (s) => s.llmAnalysesMeta[`${runId}:${analysisType}`]
  );

  const [streamingResponse, setStreamingResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [promptEditable, setPromptEditable] = useState(false);
  const [defaultPrompt, setDefaultPrompt] = useState("");
  const [fullscreen, setFullscreen] = useState(false);
  const [activeTab, setActiveTab] = useState<"observations" | "recommendations">("observations");

  // Track the current analysisType to abort stale streams
  const activeAnalysisRef = useRef(analysisType);

  // Hydrate from disk cache once per run
  useEffect(() => {
    if (llmCacheLoaded[runId]) return;
    getLLMAnalyses(runId)
      .then((data) => {
        hydrateLlmResponses(runId, data.analyses);
        setLlmCacheLoaded(runId);
      })
      .catch(() => {
        // Silently ignore — cache is optional
        setLlmCacheLoaded(runId);
      });
  }, [runId, llmCacheLoaded, hydrateLlmResponses, setLlmCacheLoaded]);

  // Reset local state when the section changes
  useEffect(() => {
    activeAnalysisRef.current = analysisType;
    setStreamingResponse("");
    setShowPrompt(false);
    setPrompt("");
    setDefaultPrompt("");
    setPromptEditable(false);
    setLoading(false);
    setActiveTab("observations");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisType, runId]);

  // Close fullscreen on Escape
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  const loadPrompt = useCallback(async () => {
    try {
      const data = await getPromptTemplate(analysisType);
      setPrompt(data.template);
      setDefaultPrompt(data.template);
    } catch {
      setPrompt("(No template available for this analysis type)");
    }
  }, [analysisType]);

  const displayResponse = loading ? streamingResponse : cachedResponse;
  const tabSections = useMemo(
    () => (!loading && displayResponse ? parseTabSections(displayResponse) : null),
    [loading, displayResponse]
  );

  const handleAnalyze = async () => {
    const currentKey = `${runId}:${analysisType}`;
    const currentAnalysis = analysisType;
    const usedCustomPrompt = promptEditable && prompt !== defaultPrompt;
    setLoading(true);
    setStreamingResponse("");
    setLlmResponse(currentKey, "");

    let accumulated = "";
    let streamCompleted = false;
    try {
      const gen = streamAnalysis({
        run_id: runId,
        analysis_type: analysisType,
        prompt: usedCustomPrompt ? prompt : undefined,
        viz_type: vizType,
        include_images: true,
        include_stats: true,
        compare_run_ids: compareRunIds,
        run_notes: runNotes,
      });
      for await (const token of gen) {
        // Stop accumulating if user navigated away
        if (activeAnalysisRef.current !== currentAnalysis) break;
        accumulated += token;
        setStreamingResponse(accumulated);
      }
      if (activeAnalysisRef.current === currentAnalysis) {
        streamCompleted = true;
      }
    } catch (err) {
      accumulated = `Error: ${err}`;
      setStreamingResponse(accumulated);
    } finally {
      setLlmResponse(currentKey, accumulated);
      setLoading(false);

      // Persist to disk if stream completed normally
      if (streamCompleted && accumulated && !accumulated.startsWith("Error:")) {
        saveLLMAnalysis(runId, analysisType, accumulated, usedCustomPrompt).catch(
          () => {} // Fire-and-forget — persistence failure is non-critical
        );
      }
    }
  };

  if (!llmConfigured) {
    return (
      <div className="llm-panel">
        <p style={{ fontSize: 12, color: "var(--text-body)" }}>
          Configure an LLM API key to enable AI-powered analysis.
        </p>
      </div>
    );
  }

  const headerButtons = (
    <>
      <button
        className="btn-primary"
        onClick={handleAnalyze}
        disabled={loading}
        style={{ fontSize: 12 }}
      >
        {loading ? "Analyzing..." : displayResponse ? "Regenerate" : "Analyze with LLM"}
      </button>
      <button
        className="btn-tertiary"
        onClick={() => {
          if (!showPrompt && !prompt) loadPrompt();
          setShowPrompt(!showPrompt);
        }}
        style={{ fontSize: 12 }}
      >
        {showPrompt ? "Hide prompt" : "Edit prompt"}
      </button>
    </>
  );

  const promptEditor = showPrompt && (
    <div className="llm-prompt-editor">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 4,
        }}
      >
        <label style={{ fontSize: 11, color: "var(--text-body)" }}>
          Prompt template
        </label>
        <div style={{ display: "flex", gap: 8 }}>
          {!promptEditable && (
            <button
              className="btn-tertiary"
              onClick={() => setPromptEditable(true)}
              style={{ fontSize: 11, padding: "2px 8px" }}
            >
              Edit
            </button>
          )}
          {promptEditable && prompt !== defaultPrompt && (
            <button
              className="btn-tertiary"
              onClick={() => {
                setPrompt(defaultPrompt);
                setPromptEditable(false);
              }}
              style={{ fontSize: 11, padding: "2px 8px" }}
            >
              Reset to default
            </button>
          )}
        </div>
      </div>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        readOnly={!promptEditable}
        style={{
          opacity: promptEditable ? 1 : 0.7,
          background: promptEditable ? "white" : "#f8f9fa",
        }}
      />
    </div>
  );

  const metaIndicator = !loading && displayResponse && cacheMeta && (
    <div
      style={{
        borderTop: "1px solid var(--border, #DBE4E8)",
        marginTop: 8,
        paddingTop: 6,
        fontSize: 11,
        color: "var(--text-body)",
        display: "flex",
        gap: 4,
        flexWrap: "wrap",
      }}
    >
      {cacheMeta.model && <span>Model: {cacheMeta.model}</span>}
      {cacheMeta.model && cacheMeta.updated_at && <span>|</span>}
      {cacheMeta.updated_at && <span>Generated: {formatDate(cacheMeta.updated_at)}</span>}
      {cacheMeta.custom_prompt && <span>| Custom prompt</span>}
    </div>
  );

  return (
    <>
      <div className="llm-panel">
        <div className="llm-panel-header">
          {headerButtons}
          {displayResponse && (
            <button
              className="btn-tertiary llm-fullscreen-btn"
              onClick={() => setFullscreen(true)}
              title="View full screen"
              style={{ marginLeft: "auto", fontSize: 18, padding: "2px 8px", lineHeight: 1 }}
            >
              &#x26F6;
            </button>
          )}
        </div>

        {promptEditor}

        {displayResponse && (
          tabSections ? (
            <div className="llm-tabbed">
              <div className="llm-tabs">
                <button
                  className={`llm-tab${activeTab === "observations" ? " llm-tab--active" : ""}`}
                  onClick={() => setActiveTab("observations")}
                >
                  Observations
                </button>
                <button
                  className={`llm-tab${activeTab === "recommendations" ? " llm-tab--active" : ""}`}
                  onClick={() => setActiveTab("recommendations")}
                >
                  Recommendations
                </button>
              </div>
              <div className="llm-response">
                <ReactMarkdown>
                  {activeTab === "observations" ? tabSections.observations : tabSections.recommendations}
                </ReactMarkdown>
                {metaIndicator}
              </div>
            </div>
          ) : (
            <div className="llm-response">
              <ReactMarkdown>{displayResponse}</ReactMarkdown>
              {metaIndicator}
            </div>
          )
        )}
      </div>

      {fullscreen && (
        <div className="modal-overlay" onClick={() => setFullscreen(false)}>
          <div
            className="llm-fullscreen-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="llm-fullscreen-header">
              <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text-heading)" }}>
                LLM Analysis
              </span>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                {headerButtons}
                <button
                  className="btn-tertiary"
                  onClick={() => setFullscreen(false)}
                  title="Exit full screen"
                  style={{ fontSize: 16, padding: "4px 8px", lineHeight: 1 }}
                >
                  &#x2715;
                </button>
              </div>
            </div>
            {promptEditor}
            <div className="llm-fullscreen-body">
              {displayResponse ? (
                tabSections ? (
                  <div className="llm-tabbed llm-tabbed-fullscreen">
                    <div className="llm-tabs">
                      <button
                        className={`llm-tab${activeTab === "observations" ? " llm-tab--active" : ""}`}
                        onClick={() => setActiveTab("observations")}
                      >
                        Observations
                      </button>
                      <button
                        className={`llm-tab${activeTab === "recommendations" ? " llm-tab--active" : ""}`}
                        onClick={() => setActiveTab("recommendations")}
                      >
                        Recommendations
                      </button>
                    </div>
                    <div className="llm-response llm-response-fullscreen">
                      <ReactMarkdown>
                        {activeTab === "observations" ? tabSections.observations : tabSections.recommendations}
                      </ReactMarkdown>
                      {metaIndicator}
                    </div>
                  </div>
                ) : (
                  <div className="llm-response llm-response-fullscreen">
                    <ReactMarkdown>{displayResponse}</ReactMarkdown>
                    {metaIndicator}
                  </div>
                )
              ) : (
                <div style={{ color: "var(--text-body)", fontSize: 13, padding: 16 }}>
                  Click "Analyze with LLM" to generate analysis.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
