import { Activity, CheckCircle2, Clock, GitBranch, Layers3, XCircle, Zap } from "lucide-react";
import type { MatchMeta, MatchResponse } from "../types/index.ts";
import { formatMethodLabel } from "../shared/storytelling.ts";
import { MetricTile, StatusPill } from "../shared/ui/presentation.tsx";

interface ResultSummaryProps {
    resp: MatchResponse;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "-";
    }
    return value.toFixed(digits);
}

function readNumber(meta: MatchMeta, key: string): number | null {
    const value = meta[key];
    return typeof value === "number" && !Number.isNaN(value) ? value : null;
}

function readRecord(meta: MatchMeta, key: string): Record<string, number> {
    const value = meta[key];
    if (!value || typeof value !== "object") {
        return {};
    }

    return Object.fromEntries(
        Object.entries(value).filter(
            (entry): entry is [string, number] => typeof entry[1] === "number" && !Number.isNaN(entry[1]),
        ),
    );
}

function readDlBackbone(meta: MatchMeta): string {
    const config = meta.dl_config;
    if (!config || typeof config !== "object") {
        return "-";
    }

    const backbone = config.backbone;
    return typeof backbone === "string" ? backbone : "-";
}

function formatKeyValueRecord(record: Record<string, number>, digits = 1): string {
    const entries = Object.entries(record);
    if (entries.length === 0) {
        return "-";
    }

    return entries
        .map(([key, value]) => `${key}: ${value.toFixed(digits)}ms`)
        .join(" | ");
}

export function ResultSummary({ resp }: ResultSummaryProps) {
    const { method, score, decision, threshold, latency_ms, meta } = resp;
    const dedicatedStats = readRecord(meta, "stats");
    const dedicatedLatencyBreakdown = readRecord(meta, "latency_breakdown_ms");
    const classicInliers = readNumber(meta, "inliers");
    const classicMatches = readNumber(meta, "matches");
    const classicK1 = readNumber(meta, "k1");
    const classicK2 = readNumber(meta, "k2");
    const dedicatedTentative = readNumber(meta, "tentative_count");
    const dedicatedInliers = readNumber(meta, "inliers_count");

    const isClassic = method === "classic_orb" || method === "classic_gftt_orb" || method === "harris" || method === "sift";
    const isEmbeddingModel = method === "dl" || method === "vit";
    const isDedicated = method === "dedicated";

    return (
        <div className="surface-card mb-6 p-5">
            <div
                className={`rounded-xl border p-4 ${
                    decision
                        ? "border-[var(--app-success-border)] bg-[var(--app-success-surface)] text-[var(--app-success-text)]"
                        : "border-[var(--app-error-border)] bg-[var(--app-error-surface)] text-[var(--app-error-text)]"
                }`}
            >
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex min-w-0 items-center gap-4">
                        {decision ? (
                            <CheckCircle2 className="h-9 w-9 shrink-0" />
                        ) : (
                            <XCircle className="h-9 w-9 shrink-0" />
                        )}
                        <div className="min-w-0">
                            <StatusPill tone={decision ? "success" : "error"}>Decision</StatusPill>
                            <h3 className="mt-2 text-2xl font-bold leading-tight">
                                {decision ? "MATCH CONFIRMED" : "NO MATCH"}
                            </h3>
                        </div>
                    </div>
                    <MetricTile
                        label="Similarity score"
                        value={Math.min(Math.max(score, 0), 1).toFixed(4)}
                        tone={decision ? "success" : "error"}
                        className="min-w-40"
                    />
                </div>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <MetricTile icon={GitBranch} label="Method" value={formatMethodLabel(method)} title={formatMethodLabel(method)} />
                <MetricTile icon={Activity} label="Threshold" value={formatNumber(threshold, 2)} />
                <MetricTile icon={Clock} label="Latency" value={`${latency_ms.toFixed(0)} ms`} />
                <MetricTile icon={Layers3} label="Overlay" value={`${resp.overlay?.matches.length ?? 0} matches`} />

                {isClassic ? (
                    <>
                        <MetricTile label="Raw matches" value={formatNumber(classicMatches, 0)} />
                        <MetricTile label="Inliers" value={formatNumber(classicInliers, 0)} />
                        <MetricTile
                            label="Keypoints A/B"
                            value={`${formatNumber(classicK1, 0)} / ${formatNumber(classicK2, 0)}`}
                        />
                    </>
                ) : null}

                {isEmbeddingModel ? (
                    <>
                        <MetricTile
                            label="Backbone"
                            value={<span className="safe-truncate">{readDlBackbone(meta)}</span>}
                            title={readDlBackbone(meta)}
                        />
                        <MetricTile
                            icon={Zap}
                            label="Embed A/B"
                            value={`${formatNumber(readNumber(meta, "embed_ms_a"), 0)} / ${formatNumber(readNumber(meta, "embed_ms_b"), 0)} ms`}
                        />
                    </>
                ) : null}

                {isDedicated ? (
                    <>
                        <MetricTile
                            label="Tentative / Inliers"
                            value={`${formatNumber(dedicatedTentative, 0)} / ${formatNumber(dedicatedInliers, 0)}`}
                        />
                        <MetricTile label="Mean inlier sim" value={formatNumber(dedicatedStats.mean_inlier_sim, 4)} />
                        <MetricTile
                            label="Latency breakdown"
                            value={<span className="text-sm leading-6">{formatKeyValueRecord(dedicatedLatencyBreakdown)}</span>}
                            className="md:col-span-2"
                        />
                    </>
                ) : null}
            </div>
        </div>
    );
}
