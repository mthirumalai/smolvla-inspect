interface Props {
  totalFrames: number;
  selectedFrames: number[];
  onChange: (frames: number[]) => void;
}

export default function FrameSelector({
  totalFrames,
  selectedFrames,
  onChange,
}: Props) {
  const allSelected = selectedFrames.length === totalFrames;

  const toggleAll = () => {
    if (allSelected) {
      onChange([]);
    } else {
      onChange(Array.from({ length: totalFrames }, (_, i) => i));
    }
  };

  const toggleFrame = (idx: number) => {
    if (selectedFrames.includes(idx)) {
      onChange(selectedFrames.filter((f) => f !== idx));
    } else {
      onChange([...selectedFrames, idx].sort((a, b) => a - b));
    }
  };

  return (
    <div className="frame-selector">
      <span style={{ fontSize: 12, color: "var(--text-body)", fontWeight: 500 }}>
        Frames:
      </span>
      <label>
        <input
          type="checkbox"
          checked={allSelected}
          onChange={toggleAll}
        />
        All
      </label>
      {Array.from({ length: totalFrames }, (_, i) => (
        <label key={i}>
          <input
            type="checkbox"
            checked={selectedFrames.includes(i)}
            onChange={() => toggleFrame(i)}
          />
          {i}
        </label>
      ))}
    </div>
  );
}
