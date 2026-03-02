import { useAppStore } from "../stores/appStore";

const MODES = [
  { key: "overlay", label: "Overlay" },
  { key: "map", label: "Map Only" },
  { key: "original", label: "Original" },
] as const;

export default function DisplayModeToggle() {
  const { displayMode, setDisplayMode } = useAppStore();

  return (
    <div className="display-mode-toggle">
      <span className="toggle-label">Display:</span>
      {MODES.map((m) => (
        <button
          key={m.key}
          className={displayMode === m.key ? "btn-secondary" : "btn-tertiary"}
          onClick={() => setDisplayMode(m.key)}
          style={{ padding: "4px 10px", fontSize: 12 }}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
