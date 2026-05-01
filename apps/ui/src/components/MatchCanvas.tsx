import { useEffect, useRef } from "react";
import { useAppPreferences } from "../shared/preferences/useAppPreferences.ts";
import type { OverlayMatch } from "../types";

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
    const { preferences } = useAppPreferences();
    const isHebrew = preferences.language === "he";

    useEffect(() => {
        let cancelled = false;
        const canvasLabels = isHebrew
            ? {
                noMatches: "לא נמצאו התאמות עם המסננים הנוכחיים.",
                probe: "תמונת בדיקה",
                reference: "תמונת ייחוס",
            }
            : {
                noMatches: "No matches found with current filters.",
                probe: "Probe Image",
                reference: "Reference Image",
            };

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
            ctx.scale(dpr, dpr);

            ctx.clearRect(0, 0, cssW, cssH);

            const [imgA, imgB] = await Promise.all([
                loadImage(URL.createObjectURL(fileA)),
                loadImage(URL.createObjectURL(fileB)),
            ]);
            if (cancelled) return;

            ctx.drawImage(imgA, 0, 0, box, box);
            ctx.drawImage(imgB, box + gap, 0, box, box);

            // Divider line
            ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(box + gap / 2, 0);
            ctx.lineTo(box + gap / 2, box);
            ctx.stroke();

            ctx.fillStyle = "rgba(71, 85, 105, 0.9)";
            ctx.font = "600 14px system-ui, sans-serif";
            ctx.direction = isHebrew ? "rtl" : "ltr";
            ctx.textAlign = isHebrew ? "right" : "left";
            ctx.fillText(canvasLabels.probe, isHebrew ? box - 16 : 16, 24);
            ctx.fillText(canvasLabels.reference, isHebrew ? box + gap + box - 16 : box + gap + 16, 24);

            const filtered = matches.filter((m) => {
                const k = (m.kind || "").toLowerCase();
                if (k === "outlier" && !showOutliers) return false;
                if ((k === "tentative" || (k !== "inlier" && k !== "outlier")) && !showTentative) return false;
                return true;
            });

            const drawList = filtered.slice(0, Math.max(0, maxMatches));

            if (drawList.length === 0) {
                ctx.fillStyle = "rgba(100, 116, 139, 0.8)";
                ctx.font = "14px system-ui, sans-serif";
                ctx.fillText(canvasLabels.noMatches, isHebrew ? box - 16 : 16, box - 16);
                return;
            }

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
        }

        draw();

        return () => {
            cancelled = true;
        };
    }, [fileA, fileB, isHebrew, matches, maxMatches, showOutliers, showTentative]);

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden flex flex-col mb-6">
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 bg-slate-50/50">
                <div>
                    <h3 className="text-lg font-semibold text-slate-800">Match Visualization</h3>
                    <p className="text-sm text-slate-500">Connecting corresponding minutiae/patches</p>
                </div>
                <div className="flex items-center space-x-4 text-sm font-medium">
                    <div className="flex items-center text-slate-600"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500 mr-2 shadow-sm"></span>Inlier</div>
                    <div className="flex items-center text-slate-600"><span className="w-2.5 h-2.5 rounded-full bg-red-500 mr-2 shadow-sm"></span>Outlier</div>
                </div>
            </div>
            <div className="p-6 flex justify-center bg-slate-50/30 overflow-x-auto">
                <canvas ref={canvasRef} className="max-w-full rounded-lg shadow-sm border border-slate-200 bg-white" />
            </div>
        </div>
    );
}

function loadImage(src: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            URL.revokeObjectURL(src);
            resolve(img);
        };
        img.onerror = reject;
        img.src = src;
    });
}
