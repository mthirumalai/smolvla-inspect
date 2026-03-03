import { useAppStore } from "../stores/appStore";

const MODES = [
  { key: "overlay", label: "Overlay" },
  { key: "map", label: "Map Only" },
  { key: "original", label: "Original" },
] as const;

interface Props {
  hideOverlay?: boolean;
}

export default function DisplayModeToggle({ hideOverlay = false }: Props) {
  const { displayMode, setDisplayMode } = useAppStore();

  const visibleModes = hideOverlay ? MODES.filter((m) => m.key !== "overlay") : MODES;

  return (
    <div className="display-mode-toggle">
      <span className="toggle-label">Display:</span>
      {visibleModes.map((m) => (
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
