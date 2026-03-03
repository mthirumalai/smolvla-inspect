import { useAppStore } from "../stores/appStore";

interface Props {
  onLLMConfig: () => void;
}

export default function Header({ onLLMConfig }: Props) {
  const llmConfigured = useAppStore((s) => s.llmConfigured);

  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 24px",
        borderBottom: "1px solid var(--border)",
        background: "var(--bg-card)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <h1 style={{ fontSize: 18, fontWeight: 700 }}>smolvla-inspect</h1>
      </div>
      <button className="btn-secondary" onClick={onLLMConfig}>
        {llmConfigured ? "LLM configured" : "Configure LLM"}
      </button>
    </header>
  );
}
