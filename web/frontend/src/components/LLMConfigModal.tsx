import { useState, useEffect } from "react";
import { getLLMConfig, setLLMConfig, type LLMConfig } from "../services/api";
import { useAppStore } from "../stores/appStore";

interface Props {
  onClose: () => void;
}

export default function LLMConfigModal({ onClose }: Props) {
  const setLlmConfigured = useAppStore((s) => s.setLlmConfigured);
  const [config, setConfig] = useState<LLMConfig>({
    provider: "anthropic",
    model: "claude-sonnet-4-20250514",
    api_key: "",
    base_url: "",
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getLLMConfig()
      .then((c) =>
        setConfig({
          provider: c.provider,
          model: c.model,
          api_key: c.api_key === "***" ? "" : c.api_key,
          base_url: c.base_url || "",
        })
      )
      .catch(() => {});
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await setLLMConfig(config);
      setLlmConfigured(!!config.api_key);
      onClose();
    } catch (err) {
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>LLM Configuration</h2>

        <div className="form-group">
          <label>Provider</label>
          <select
            value={config.provider}
            onChange={(e) =>
              setConfig({
                ...config,
                provider: e.target.value,
                model:
                  e.target.value === "anthropic"
                    ? "claude-sonnet-4-20250514"
                    : "gpt-4o",
              })
            }
          >
            <option value="anthropic">Anthropic (Claude)</option>
            <option value="openai">OpenAI</option>
          </select>
        </div>

        <div className="form-group">
          <label>Model</label>
          <input
            type="text"
            value={config.model}
            onChange={(e) => setConfig({ ...config, model: e.target.value })}
          />
        </div>

        {config.provider === "openai" && (
          <div className="form-group">
            <label>Base URL</label>
            <input
              type="text"
              value={config.base_url}
              onChange={(e) =>
                setConfig({ ...config, base_url: e.target.value })
              }
              placeholder="https://api.openai.com/v1 (leave empty for default)"
            />
            <p style={{ fontSize: 11, marginTop: 4 }}>
              For OpenAI-compatible servers (vLLM, Ollama, LM Studio, etc.)
            </p>
          </div>
        )}

        <div className="form-group">
          <label>API Key</label>
          <input
            type="text"
            value={config.api_key}
            onChange={(e) =>
              setConfig({ ...config, api_key: e.target.value })
            }
            placeholder={
              config.provider === "anthropic"
                ? "sk-ant-..."
                : "sk-..."
            }
          />
          <p style={{ fontSize: 11, marginTop: 4 }}>
            Stored in server memory only. Not persisted to disk.
          </p>
        </div>

        <div className="modal-actions">
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn-primary"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
