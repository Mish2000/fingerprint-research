import type { BenchmarkBestMetric, BenchmarkOperatingPoint, BenchmarkSortMode, BenchmarkViewMode, ComparisonRow } from "../../types";

const METHOD_LABELS: Record<string, string> = {
    classic: "Classic (ORB)",
    classic_v2: "Classic (ROI GFTT+ORB)",
    classic_gftt_orb: "Classic (ROI GFTT+ORB)",
    classic_orb: "Classic (ORB)",
    minutiae: "Classic (Minutiae)",
    harris: "Classic (Harris + ORB)",
    sift: "Classic (SIFT)",
    sift_plain_roll_v2: "SIFT Plain/Roll v2 (Experimental)",
    sourceafis_open: "SourceAFIS Open Matcher",
    dl: "Deep Learning (ResNet18)",
    dl_quick: "Deep Learning (ResNet18)",
    dedicated: "Dedicated Patch AI",
    vit: "Deep Learning (ViT)",
};

export function formatMethodLabel(method: string | null | undefined, methodLabel?: string | null): string {
    const normalized = (method ?? "").trim().toLowerCase();
    if (methodLabel && methodLabel.trim()) {
        return methodLabel.trim();
    }
    if (normalized === "dedicated") {
        return "Dedicated Patch AI";
    }

    return METHOD_LABELS[normalized] ?? method ?? "";
}

export function formatMetric(value: number | null | undefined, digits = 4): string {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "N/A";
    }
    return value.toFixed(digits);
}

export function formatOperatingPoint(value: number | null | undefined): string {
    return formatMetric(value);
}

export function formatPercentFromFraction(value: number | null | undefined, digits = 2): string {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "N/A";
    }
    return `${(value * 100).toFixed(digits)}%`;
}

type FarTarget = "1e-2" | "1e-3";

const FAR_TARGET_PERCENT_LABELS: Record<FarTarget, string> = {
    "1e-2": "1%",
    "1e-3": "0.1%",
};

export function formatFarTargetLabel(value: FarTarget): string {
    return `FAR ${FAR_TARGET_PERCENT_LABELS[value]} (${value})`;
}

export function formatTarAtFarLabel(value: FarTarget): string {
    return `TAR @ ${formatFarTargetLabel(value)}`;
}

function legacyOperatingPoint(target: FarTarget, testTar: number | null | undefined): BenchmarkOperatingPoint | null {
    if (typeof testTar !== "number" || Number.isNaN(testTar)) {
        return null;
    }
    if (target === "1e-2") {
        return {
            target_far: 0.01,
            label: "1.00% FAR",
            test_tar: testTar,
        };
    }
    return {
        target_far: 0.001,
        label: "0.10% FAR",
        test_tar: testTar,
    };
}

export function operatingPointsForRow(row: Pick<ComparisonRow, "operating_points" | "tar_at_far_1e_2" | "tar_at_far_1e_3">): BenchmarkOperatingPoint[] {
    if (Array.isArray(row.operating_points) && row.operating_points.length > 0) {
        return row.operating_points;
    }

    return [
        legacyOperatingPoint("1e-2", row.tar_at_far_1e_2),
        legacyOperatingPoint("1e-3", row.tar_at_far_1e_3),
    ].filter((point): point is BenchmarkOperatingPoint => point != null);
}

export function formatOperatingPointTarget(point: BenchmarkOperatingPoint): string {
    const label = point.label?.trim();
    if (label) {
        return label;
    }
    return `${formatPercentFromFraction(point.target_far)} FAR`;
}

export function formatOperatingPointTar(point: BenchmarkOperatingPoint): string {
    return formatPercentFromFraction(point.test_tar);
}

export function formatOperatingPointActualFar(point: BenchmarkOperatingPoint): string {
    return formatPercentFromFraction(point.test_far);
}

export function formatOperatingPointFrr(point: BenchmarkOperatingPoint): string {
    return formatPercentFromFraction(point.test_frr);
}

export function formatOperatingPointThreshold(point: BenchmarkOperatingPoint): string {
    return formatMetric(point.threshold, 4);
}

export function formatOperatingPointCounts(point: BenchmarkOperatingPoint): string | null {
    const values = [point.ta, point.fr, point.fa, point.tr];
    if (values.some((value) => typeof value !== "number" || Number.isNaN(value))) {
        return null;
    }
    return `TA ${point.ta} / FR ${point.fr} / FA ${point.fa} / TR ${point.tr}`;
}

export function formatApproxEqualEer(value: number | null | undefined): string {
    return formatMetric(value);
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

export function isResearchRow(row: Pick<ComparisonRow, "presentation_tier" | "research_track" | "showcase_eligible" | "not_champion_candidate" | "method" | "benchmark_method">): boolean {
    return row.presentation_tier === "research"
        || row.research_track
        || row.showcase_eligible === false
        || row.not_champion_candidate
        || row.method === "dedicated"
        || row.benchmark_method === "dedicated";
}

export function isChampionCandidateRow(row: Pick<ComparisonRow, "showcase_eligible" | "not_champion_candidate">): boolean {
    return row.showcase_eligible !== false && !row.not_champion_candidate;
}

function isBaselineRow(row: Pick<ComparisonRow, "presentation_tier" | "showcase_eligible" | "research_track" | "not_champion_candidate">): boolean {
    return row.presentation_tier === "baseline"
        || (row.not_champion_candidate && row.showcase_eligible !== false && !row.research_track);
}

export function methodStatusBadges(row: Pick<ComparisonRow, "method_status" | "presentation_tier" | "showcase_eligible" | "research_track" | "not_champion_candidate" | "method" | "benchmark_method">): string[] {
    const badges: string[] = [];
    if ((row.method_status ?? "").toLowerCase() === "experimental" || row.method === "dedicated" || row.benchmark_method === "dedicated") {
        badges.push("Experimental");
    }
    if (isBaselineRow(row)) {
        badges.push("Baseline");
    } else if (isResearchRow(row)) {
        badges.push("Research");
    }
    if (row.showcase_eligible === false) {
        badges.push("Not showcase eligible");
    }
    return [...new Set(badges)];
}

export function researchRunSourceLabel(row: Pick<ComparisonRow, "run_kind" | "status" | "validation_state" | "run_label" | "provenance">): string {
    if (row.status === "partial" || row.validation_state === "partial") {
        return "partial";
    }
    const sourceRoot = row.provenance?.benchmark_source_root;
    if (sourceRoot === "live") {
        if (row.run_kind === "smoke") {
            return "current smoke benchmark";
        }
        if (row.run_kind === "full") {
            return "current full benchmark";
        }
        return "current";
    }
    if (row.run_kind === "smoke") {
        return "archived smoke benchmark";
    }
    if (row.run_kind === "full") {
        return "archived full benchmark";
    }
    if (row.status === "archived" || row.validation_state === "archived") {
        return "archive";
    }
    return row.run_label ?? "research run";
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
