import { imageUrl } from "../services/api";
import ZoomableWrapper from "./ZoomableWrapper";

interface Props {
  runId: string;
  imagePaths: string[];
  label?: string;
}

export default function ImageViz({ runId, imagePaths, label }: Props) {
  if (!imagePaths.length) {
    return (
      <div className="empty-state" style={{ height: 100 }}>
        No images available
      </div>
    );
  }

  return (
    <div>
      {label && (
        <p style={{ fontSize: 12, marginBottom: 8, color: "var(--text-body)" }}>
          {label}
        </p>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {imagePaths.map((path) => (
          <ZoomableWrapper key={path}>
            <img
              src={imageUrl(runId, path)}
              alt={path}
              draggable={false}
              style={{
                width: "100%",
                maxWidth: 900,
                borderRadius: 6,
                border: "1px solid var(--border)",
              }}
            />
          </ZoomableWrapper>
        ))}
      </div>
    </div>
  );
}
