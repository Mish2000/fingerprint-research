import type { BenchmarkBestMetric, BenchmarkSortMode, ComparisonRow } from "../../types";

const METHOD_LABELS: Record<string, string> = {
    classic: "Classic (ORB)",
    classic_v2: "Classic (ORB)",
    harris: "Classic (Harris + ORB)",
    sift: "Classic (SIFT)",
    dl: "Deep Learning (ResNet50)",
    dl_quick: "Deep Learning (ResNet50)",
    dedicated: "Dedicated (Patch AI)",
    vit: "Deep Learning (ViT)",
};

export function formatMethodLabel(method: string | null | undefined, methodLabel?: string | null): string {
    if (methodLabel && methodLabel.trim()) {
        return methodLabel.trim();
    }

    const normalized = (method ?? "").trim().toLowerCase();
    return METHOD_LABELS[normalized] ?? method ?? "";
}

export function formatMetric(value: number | null | undefined, digits = 4): string {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "N/A";
    }
    return value.toFixed(digits);
}

export function formatLatency(value: number | null | undefined): string {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "N/A";
    }
    return `${value.toFixed(2)} ms`;
}

export function formatPairs(value: number | null | undefined): string {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "N/A";
    }
    return value.toLocaleString();
}

export function bestMetricLabel(metric: BenchmarkBestMetric): string {
    switch (metric) {
        case "best_auc":
            return "Best accuracy";
        case "best_eer":
            return "Lowest EER";
        case "best_latency":
            return "Fastest method";
        default:
            return metric;
    }
}

export function sortModeLabel(sortMode: BenchmarkSortMode): string {
    switch (sortMode) {
        case "lowest_eer":
            return "Lowest EER";
        case "lowest_latency":
            return "Lowest latency";
        case "best_accuracy":
        default:
            return "Best accuracy";
    }
}

export function sortModeForMetric(metric: BenchmarkBestMetric): BenchmarkSortMode {
    switch (metric) {
        case "best_eer":
            return "lowest_eer";
        case "best_latency":
            return "lowest_latency";
        case "best_auc":
        default:
            return "best_accuracy";
    }
}

export function statusLabel(status: string): string {
    switch (status) {
        case "validated":
            return "Validated";
        case "smoke":
            return "Smoke";
        case "archived":
            return "Archive";
        case "partial":
            return "Partial";
        default:
            return status;
    }
}

export function statusToneClassName(status: string): string {
    switch (status) {
        case "validated":
            return "border-emerald-500/30 bg-emerald-500/10 text-emerald-200";
        case "smoke":
            return "border-amber-500/30 bg-amber-500/10 text-amber-200";
        case "partial":
            return "border-rose-500/30 bg-rose-500/10 text-rose-200";
        case "archived":
        default:
            return "border-slate-600 bg-slate-900 text-slate-300";
    }
}

export function highlightClassName(sortMode: BenchmarkSortMode): string {
    switch (sortMode) {
        case "lowest_eer":
            return "border-blue-500/40 bg-blue-500/10";
        case "lowest_latency":
            return "border-amber-500/40 bg-amber-500/10";
        case "best_accuracy":
        default:
            return "border-emerald-500/40 bg-emerald-500/10";
    }
}

export function championValue(row: ComparisonRow, metric: BenchmarkBestMetric): number | null {
    if (metric === "best_latency") {
        return row.latency_ms ?? null;
    }
    if (metric === "best_eer") {
        return row.eer;
    }
    return row.auc;
}
