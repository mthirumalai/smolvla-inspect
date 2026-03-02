import { useState, useEffect, useRef, useCallback } from "react";
import { getRunNotes, setRunNotes } from "../services/api";
import { useAppStore } from "../stores/appStore";

interface Props {
  runId: string;
  compact?: boolean;
}

export default function RunNotesEditor({ runId, compact = false }: Props) {
  const { runNotes, setRunNote } = useAppStore();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const saveTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const note = runNotes[runId] ?? "";

  // Load notes from backend on mount
  useEffect(() => {
    let cancelled = false;
    getRunNotes(runId)
      .then((data) => {
        if (!cancelled && data.notes) {
          setRunNote(runId, data.notes);
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [runId, setRunNote]);

  useEffect(() => {
    setDraft(note);
  }, [note]);

  const save = useCallback(
    (text: string) => {
      setLoading(true);
      setRunNotes(runId, text)
        .then(() => {
          setRunNote(runId, text);
        })
        .catch(() => {})
        .finally(() => setLoading(false));
    },
    [runId, setRunNote]
  );

  const handleChange = (text: string) => {
    setDraft(text);
    // Debounce save by 1 second
    if (saveTimeout.current) clearTimeout(saveTimeout.current);
    saveTimeout.current = setTimeout(() => save(text), 1000);
  };

  const handleBlur = () => {
    // Save immediately on blur
    if (saveTimeout.current) clearTimeout(saveTimeout.current);
    if (draft !== note) {
      save(draft);
    }
    if (!draft.trim()) {
      setEditing(false);
    }
  };

  if (!editing && !note) {
    return (
      <button
        className="btn-tertiary"
        onClick={() => setEditing(true)}
        style={{ fontSize: 11, padding: compact ? "2px 6px" : "2px 8px" }}
      >
        Add note
      </button>
    );
  }

  if (!editing && note) {
    return (
      <div
        className="run-note-preview"
        style={{
          fontSize: 12,
          color: "var(--text-body)",
          display: "flex",
          alignItems: "center",
          gap: 6,
          maxWidth: compact ? 200 : 400,
        }}
      >
        <span
          style={{
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={note}
        >
          {note}
        </span>
        <button
          className="btn-tertiary"
          onClick={() => setEditing(true)}
          style={{ fontSize: 11, padding: "2px 6px", flexShrink: 0 }}
        >
          Edit
        </button>
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <textarea
        value={draft}
        onChange={(e) => handleChange(e.target.value)}
        onBlur={handleBlur}
        autoFocus
        placeholder="Add a note about this run (e.g., what changed, why it was run)..."
        style={{
          width: "100%",
          minHeight: compact ? 48 : 64,
          fontSize: 12,
          resize: "vertical",
          fontFamily: "inherit",
        }}
      />
      {loading && (
        <span
          style={{
            position: "absolute",
            bottom: 6,
            right: 8,
            fontSize: 10,
            color: "var(--text-body)",
          }}
        >
          Saving...
        </span>
      )}
    </div>
  );
}
