import { Activity, CheckCircle2, Clock, GitBranch, Layers3, XCircle, Zap } from "lucide-react";
import type { MatchMeta, MatchResponse } from "../types/index.ts";
import { formatMethodLabel } from "../shared/storytelling.ts";

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
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col gap-6 mb-6">
            <div
                className={`flex items-center justify-between p-5 rounded-lg border ${
                    decision ? "bg-emerald-50 border-emerald-100" : "bg-red-50 border-red-100"
                }`}
            >
                <div className="flex items-center space-x-4">
                    {decision ? (
                        <CheckCircle2 className="w-10 h-10 text-emerald-600" />
                    ) : (
                        <XCircle className="w-10 h-10 text-red-500" />
                    )}
                    <div>
                        <p className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-0.5">
                            Decision
                        </p>
                        <h3
                            className={`text-2xl font-bold ${
                                decision ? "text-emerald-700" : "text-red-700"
                            }`}
                        >
                            {decision ? "MATCH CONFIRMED" : "NO MATCH"}
                        </h3>
                    </div>
                </div>
                <div className="text-right">
                    <p className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-0.5">
                        Similarity score
                    </p>
                    <h3 className="text-3xl font-bold text-slate-800">
                        {Math.min(Math.max(score, 0), 1).toFixed(4)}
                    </h3>
                </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                    <span className="text-slate-500 text-sm font-medium flex items-center mb-1">
                        <GitBranch className="w-4 h-4 mr-1.5" /> Method
                    </span>
                    <span className="text-slate-800 font-semibold text-lg">{formatMethodLabel(method)}</span>
                </div>
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                    <span className="text-slate-500 text-sm font-medium flex items-center mb-1">
                        <Activity className="w-4 h-4 mr-1.5" /> Threshold
                    </span>
                    <span className="text-slate-800 font-semibold text-lg">{formatNumber(threshold, 2)}</span>
                </div>
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                    <span className="text-slate-500 text-sm font-medium flex items-center mb-1">
                        <Clock className="w-4 h-4 mr-1.5" /> Latency
                    </span>
                    <span className="text-slate-800 font-semibold text-lg">{latency_ms.toFixed(0)} ms</span>
                </div>
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                    <span className="text-slate-500 text-sm font-medium flex items-center mb-1">
                        <Layers3 className="w-4 h-4 mr-1.5" /> Overlay
                    </span>
                    <span className="text-slate-800 font-semibold text-lg">
                        {resp.overlay?.matches.length ?? 0} matches
                    </span>
                </div>

                {isClassic && (
                    <>
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                            <span className="text-slate-500 text-sm font-medium mb-1">Raw matches</span>
                            <span className="text-slate-800 font-semibold text-lg">{formatNumber(classicMatches, 0)}</span>
                        </div>
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                            <span className="text-slate-500 text-sm font-medium mb-1">Inliers</span>
                            <span className="text-slate-800 font-semibold text-lg">{formatNumber(classicInliers, 0)}</span>
                        </div>
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                            <span className="text-slate-500 text-sm font-medium mb-1">Keypoints A/B</span>
                            <span className="text-slate-800 font-semibold text-lg">
                                {formatNumber(classicK1, 0)} / {formatNumber(classicK2, 0)}
                            </span>
                        </div>
                    </>
                )}

                {isEmbeddingModel && (
                    <>
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                            <span className="text-slate-500 text-sm font-medium mb-1">Backbone</span>
                            <span className="text-slate-800 font-semibold text-lg truncate" title={readDlBackbone(meta)}>
                                {readDlBackbone(meta)}
                            </span>
                        </div>
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                            <span className="text-slate-500 text-sm font-medium flex items-center mb-1">
                                <Zap className="w-4 h-4 mr-1.5" /> Embed A/B
                            </span>
                            <span className="text-slate-800 font-semibold text-lg">
                                {formatNumber(readNumber(meta, "embed_ms_a"), 0)} / {formatNumber(readNumber(meta, "embed_ms_b"), 0)} ms
                            </span>
                        </div>
                    </>
                )}

                {isDedicated && (
                    <>
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                            <span className="text-slate-500 text-sm font-medium mb-1">Tentative / Inliers</span>
                            <span className="text-slate-800 font-semibold text-lg">
                                {formatNumber(dedicatedTentative, 0)} / {formatNumber(dedicatedInliers, 0)}
                            </span>
                        </div>
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col">
                            <span className="text-slate-500 text-sm font-medium mb-1">Mean inlier sim</span>
                            <span className="text-slate-800 font-semibold text-lg">
                                {formatNumber(dedicatedStats.mean_inlier_sim, 4)}
                            </span>
                        </div>
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex flex-col md:col-span-2">
                            <span className="text-slate-500 text-sm font-medium mb-1">Latency breakdown</span>
                            <span className="text-slate-800 font-semibold text-sm leading-6">
                                {formatKeyValueRecord(dedicatedLatencyBreakdown)}
                            </span>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
