import type { BenchmarkBestMetric, BenchmarkSortMode, BenchmarkViewMode, ComparisonRow } from "../../types";

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
            return "Lowest error";
        case "best_latency":
            return "Fastest method";
        default:
            return metric;
    }
}

export function viewModeLabel(viewMode: BenchmarkViewMode | string): string {
    switch (viewMode) {
        case "canonical":
            return "Showcase";
        case "smoke":
            return "Smoke";
        case "archive":
            return "Archive";
        default:
            return viewMode;
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
            return "border-[var(--app-success-border)] bg-[var(--app-success-surface)] text-[var(--app-success-text)]";
        case "smoke":
            return "border-[var(--app-warning-border)] bg-[var(--app-warning-surface)] text-[var(--app-warning-text)]";
        case "partial":
            return "border-[var(--app-error-border)] bg-[var(--app-error-surface)] text-[var(--app-error-text)]";
        case "archived":
        default:
            return "border-[var(--app-border)] bg-[var(--app-surface-muted)] text-[var(--app-text-soft)]";
    }
}

export function highlightClassName(sortMode: BenchmarkSortMode): string {
    switch (sortMode) {
        case "lowest_eer":
            return "border-[var(--app-info-border)] bg-[var(--app-info-surface)]";
        case "lowest_latency":
            return "border-[var(--app-warning-border)] bg-[var(--app-warning-surface)]";
        case "best_accuracy":
        default:
            return "border-[var(--app-success-border)] bg-[var(--app-success-surface)]";
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

function sameMethod(a: ComparisonRow | null | undefined, b: { run: string; method: string; benchmark_method?: string | null; split: string }): boolean {
    if (!a) {
        return false;
    }
    return a.run === b.run
        && a.method === b.method
        && a.split === b.split
        && a.benchmark_method === (b.benchmark_method ?? a.benchmark_method);
}

export function championTradeoffText(entry: {
    metric: BenchmarkBestMetric;
    run: string;
    method: string;
    benchmark_method?: string | null;
    split: string;
}, rows: ComparisonRow[]): string {
    const bestAccuracy = rows.find((row) => row.auc_rank === 1) ?? null;
    const fastest = rows.find((row) => row.latency_rank === 1 && row.latency_ms != null) ?? null;

    if (sameMethod(bestAccuracy, entry) && sameMethod(fastest, entry)) {
        return "Best balance on this split.";
    }

    if (entry.metric === "best_latency") {
        if (sameMethod(bestAccuracy, entry)) {
            return "Fastest method and top accuracy.";
        }
        return "Fastest method, slightly lower AUC.";
    }

    if (entry.metric === "best_eer") {
        if (sameMethod(fastest, entry)) {
            return "Lowest error and fastest method.";
        }
        return "Lowest error, slower than the fastest.";
    }

    if (sameMethod(fastest, entry)) {
        return "Best accuracy and fastest method.";
    }
    return "Best accuracy, slower than the fastest.";
}
