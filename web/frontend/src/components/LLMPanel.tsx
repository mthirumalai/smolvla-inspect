import { useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { streamAnalysis, getPromptTemplate } from "../services/api";
import { useAppStore } from "../stores/appStore";

interface Props {
  analysisType: string;
  runId: string;
  vizType?: string;
}

export default function LLMPanel({ analysisType, runId, vizType }: Props) {
  const llmConfigured = useAppStore((s) => s.llmConfigured);
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [promptEditable, setPromptEditable] = useState(false);
  const [defaultPrompt, setDefaultPrompt] = useState("");

  const loadPrompt = useCallback(async () => {
    try {
      const data = await getPromptTemplate(analysisType);
      setPrompt(data.template);
      setDefaultPrompt(data.template);
    } catch {
      setPrompt("(No template available for this analysis type)");
    }
  }, [analysisType]);

  const handleAnalyze = async () => {
    setLoading(true);
    setResponse("");
    try {
      const gen = streamAnalysis({
        run_id: runId,
        analysis_type: analysisType,
        prompt: promptEditable && prompt !== defaultPrompt ? prompt : undefined,
        viz_type: vizType,
        include_images: true,
      });
      for await (const token of gen) {
        setResponse((prev) => prev + token);
      }
    } catch (err) {
      setResponse(`Error: ${err}`);
    } finally {
      setLoading(false);
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

  return (
    <div className="llm-panel">
      <div className="llm-panel-header">
        <button
          className="btn-primary"
          onClick={handleAnalyze}
          disabled={loading}
          style={{ fontSize: 12 }}
        >
          {loading ? "Analyzing..." : response ? "Regenerate" : "Analyze with LLM"}
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
      </div>

      {showPrompt && (
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
      )}

      {response && (
        <div className="llm-response">
          <ReactMarkdown>{response}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}
