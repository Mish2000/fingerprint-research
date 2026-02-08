import {useEffect, useRef} from "react";
import type {OverlayMatch} from "../types";
import styles from "../App.module.css";

interface MatchCanvasProps {
    fileA: File;
    fileB: File;
    matches: OverlayMatch[];
    showOutliers: boolean;
    showTentative: boolean;
    maxMatches: number;
}

export function MatchCanvas({
                                fileA,
                                fileB,
                                matches,
                                showOutliers,
                                showTentative,
                                maxMatches,
                            }: MatchCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
        let cancelled = false;

        async function draw() {
            const canvas = canvasRef.current;
            if (!canvas) return;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            const box = 512;
            const gap = 28;

            const dpr = window.devicePixelRatio || 1;
            const cssW = box * 2 + gap;
            const cssH = box;

            canvas.width = Math.floor(cssW * dpr);
            canvas.height = Math.floor(cssH * dpr);
            canvas.style.width = `${cssW}px`;
            canvas.style.height = `${cssH}px`;

            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, cssW, cssH);

            // Background
            ctx.fillStyle = "rgba(128,128,128,0.05)";
            ctx.fillRect(0, 0, cssW, cssH);

            let bmpA: ImageBitmap | null = null;
            let bmpB: ImageBitmap | null = null;

            try {
                bmpA = await createImageBitmap(fileA);
                bmpB = await createImageBitmap(fileB);
            } catch {
                ctx.fillStyle = "#ef4444";
                ctx.font = "14px Inter, sans-serif";
                ctx.fillText("Failed to decode one of the images.", 12, 24);
                return;
            }

            try {
                if (cancelled) return;

                ctx.drawImage(bmpA, 0, 0, box, box);
                ctx.drawImage(bmpB, box + gap, 0, box, box);

                // Divider line
                ctx.strokeStyle = "rgba(128,128,128,0.2)";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(box + gap / 2, 0);
                ctx.lineTo(box + gap / 2, box);
                ctx.stroke();

                // Labels
                ctx.fillStyle = "rgba(128,128,128,0.8)";
                ctx.font = "600 12px Inter, sans-serif";
                ctx.fillText("Source A", 10, 20);
                ctx.fillText("Source B", box + gap + 10, 20);

                // Filter matches
                const filtered = matches.filter((m) => {
                    const k = (m.kind || "").toLowerCase();
                    if (k === "outlier" && !showOutliers) return false;
                    if ((k === "tentative" || (k !== "inlier" && k !== "outlier")) && !showTentative) return false;
                    return true;
                });

                const drawList = filtered.slice(0, Math.max(0, maxMatches));

                if (drawList.length === 0) {
                    ctx.fillStyle = "rgba(128,128,128,0.6)";
                    ctx.font = "14px Inter, sans-serif";
                    ctx.fillText("No matches found with current filters.", 12, box - 12);
                    return;
                }

                // Draw lines
                for (const m of drawList) {
                    const [ax, ay] = m.a;
                    const [bx, by] = m.b;
                    const rightX = bx + box + gap;

                    const k = (m.kind || "").toLowerCase();
                    ctx.strokeStyle = k === "inlier" ? "#10b981" : k === "outlier" ? "#ef4444" : "#9ca3af";
                    ctx.globalAlpha = k === "outlier" ? 0.3 : 0.8;
                    ctx.lineWidth = k === "inlier" ? 1.5 : 1;

                    ctx.beginPath();
                    ctx.moveTo(ax, ay);
                    ctx.lineTo(rightX, by);
                    ctx.stroke();
                }

                // Draw points
                ctx.globalAlpha = 1.0;
                const r = 3;
                for (const m of drawList) {
                    const [ax, ay] = m.a;
                    const [bx, by] = m.b;
                    const rightX = bx + box + gap;

                    const k = (m.kind || "").toLowerCase();
                    ctx.fillStyle = k === "inlier" ? "#10b981" : k === "outlier" ? "#ef4444" : "#9ca3af";

                    ctx.beginPath();
                    ctx.arc(ax, ay, r, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.beginPath();
                    ctx.arc(rightX, by, r, 0, Math.PI * 2);
                    ctx.fill();
                }
            } finally {
                bmpA?.close?.();
                bmpB?.close?.();
            }
        }

        draw();

        return () => {
            cancelled = true;
        };
    }, [fileA, fileB, matches, showOutliers, showTentative, maxMatches]);

    return (
        <div className={styles.visPanel}>
            <div className={styles.visHeader}>
                <div className={styles.visTitle}>Feature Matching Overlay</div>
                <div style={{ fontSize: 12, color: "var(--muted)" }}>Green: Inlier • Red: Outlier</div>
            </div>

            <div className={styles.canvasWrapper}>
                <canvas ref={canvasRef} className={styles.canvas} />
            </div>
        </div>
    );
}