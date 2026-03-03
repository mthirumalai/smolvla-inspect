import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface InfoPopoverItem {
  label: string;
  desc: string;
}

interface Props {
  items: InfoPopoverItem[];
}

export default function InfoPopover({ items }: Props) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const btnRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  const toggle = () => {
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setPos({
        top: rect.bottom + 6,
        left: rect.left,
      });
    }
    setOpen((v) => !v);
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onClickOutside = (e: MouseEvent) => {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(e.target as Node) &&
        btnRef.current &&
        !btnRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onClickOutside);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onClickOutside);
    };
  }, [open]);

  return (
    <>
      <button
        ref={btnRef}
        className={`info-popover-trigger${open ? " info-popover-trigger--active" : ""}`}
        onClick={toggle}
        aria-label="Show metric definitions"
        title=""
      >
        ⓘ
      </button>

      {open &&
        createPortal(
          <div
            ref={popoverRef}
            className="info-popover"
            style={{ top: pos.top, left: pos.left }}
          >
            <div className="info-popover-items">
              {items.map((item, i) => (
                <div key={item.label} className={`info-popover-item${i < items.length - 1 ? " info-popover-item--sep" : ""}`}>
                  <span className="info-popover-label">{item.label}</span>
                  <span className="info-popover-desc">{item.desc}</span>
                </div>
              ))}
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
