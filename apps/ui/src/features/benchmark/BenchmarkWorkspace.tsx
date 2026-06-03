import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { BarChart3, RefreshCcw, Trophy, Zap } from "lucide-react";
import { BenchmarkComparisonTable } from "../../components/BenchmarkComparisonTable.tsx";
import RequestState from "../../components/RequestState.tsx";
import { INPUT_CLASS_NAME } from "../../shared/ui/inputClasses.ts";
import { MetricTile, StatusPill, WorkspaceHero } from "../../shared/ui/presentation.tsx";
import type { BenchmarkBestMetric, BenchmarkViewMode, BestMethodEntry, ComparisonRow, NamedInfo, ResearchRunGroupInfo } from "../../types";
import BenchmarkEvidencePanel from "./BenchmarkEvidencePanel.tsx";
import {
    bestMetricLabel,
    championValue,
    championTradeoffText,
    formatLatency,
    formatMethodLabel,
    formatMetric,
    formatPairs,
    highlightClassName,
    isChampionCandidateRow,
    isResearchRow,
    sortModeForMetric,
    sortModeLabel,
    viewModeLabel,
} from "./benchmarkPresentation.ts";
import { useBenchmark } from "./hooks/useBenchmark.ts";

const SORT_OPTIONS = [
    { key: "best_accuracy", label: "Best accuracy" },
    { key: "lowest_eer", label: "Lowest EER" },
    { key: "lowest_latency", label: "Lowest latency" },
] as const;

type ComparisonRowsView = {
    rows: ComparisonRow[];
    researchGroupInfoByRowKey: Record<string, ResearchRunGroupInfo>;
    hiddenResearchRowCount: number;
    hasResearchRows: boolean;
};

function researchGroupKey(row: ComparisonRow): string {
    return `${row.dataset}::${row.split}::${row.method || row.benchmark_method}`;
}

function timestampScore(row: ComparisonRow): number {
    const raw = row.provenance?.timestamp_utc;
    if (!raw) {
        return 0;
    }
    const parsed = Date.parse(raw);
    return Number.isFinite(parsed) ? parsed : 0;
}

function compareResearchRepresentative(a: ComparisonRow, b: ComparisonRow): number {
    const aIsCurrent = a.provenance?.benchmark_source_root === "live" ? 0 : 1;
    const bIsCurrent = b.provenance?.benchmark_source_root === "live" ? 0 : 1;
    if (aIsCurrent !== bIsCurrent) {
        return aIsCurrent - bIsCurrent;
    }

    const aIsFull = a.run_kind === "full" ? 0 : 1;
    const bIsFull = b.run_kind === "full" ? 0 : 1;
    if (aIsFull !== bIsFull) {
        return aIsFull - bIsFull;
    }

    const aIsSmoke = a.run_kind === "smoke" ? 1 : 0;
    const bIsSmoke = b.run_kind === "smoke" ? 1 : 0;
    if (aIsSmoke !== bIsSmoke) {
        return aIsSmoke - bIsSmoke;
    }

    const timestampDelta = timestampScore(b) - timestampScore(a);
    if (timestampDelta !== 0) {
        return timestampDelta;
    }

    const artifactDelta = b.artifact_count - a.artifact_count;
    if (artifactDelta !== 0) {
        return artifactDelta;
    }

    return a.run.localeCompare(b.run);
}

function buildComparisonRowsView(
    rows: ComparisonRow[],
    options: {
        showResearchHistory: boolean;
        rowKey: (row: ComparisonRow) => string;
    },
): ComparisonRowsView {
    const { showResearchHistory, rowKey } = options;
    const researchGroups = new Map<string, ComparisonRow[]>();
    for (const row of rows) {
        if (!isResearchRow(row)) {
            continue;
        }
        const key = researchGroupKey(row);
        researchGroups.set(key, [...(researchGroups.get(key) ?? []), row]);
    }

    const representativeByGroup = new Map<string, ComparisonRow>();
    const groupInfoByRepresentativeKey: Record<string, ResearchRunGroupInfo> = {};
    let hiddenResearchRowCount = 0;

    for (const [groupKey, groupRows] of researchGroups) {
        const sorted = [...groupRows].sort(compareResearchRepresentative);
        const representative = sorted[0];
        if (!representative) {
            continue;
        }
        const hiddenRows = sorted.slice(1);
        hiddenResearchRowCount += hiddenRows.length;
        representativeByGroup.set(groupKey, representative);
        const info: ResearchRunGroupInfo = {
            totalCount: groupRows.length,
            hiddenCount: showResearchHistory ? 0 : hiddenRows.length,
            hiddenRows: showResearchHistory ? [] : hiddenRows,
        };

        if (showResearchHistory) {
            for (const row of groupRows) {
                groupInfoByRepresentativeKey[rowKey(row)] = info;
            }
        } else {
            groupInfoByRepresentativeKey[rowKey(representative)] = info;
        }
    }

    if (showResearchHistory) {
        return {
            rows,
            researchGroupInfoByRowKey: groupInfoByRepresentativeKey,
            hiddenResearchRowCount: 0,
            hasResearchRows: researchGroups.size > 0,
        };
    }

    return {
        rows: rows.filter((row) => {
            if (!isResearchRow(row)) {
                return true;
            }
            return representativeByGroup.get(researchGroupKey(row)) === row;
        }),
        researchGroupInfoByRowKey: groupInfoByRepresentativeKey,
        hiddenResearchRowCount,
        hasResearchRows: researchGroups.size > 0,
    };
}

type ChampionCardProps = {
    entry: BestMethodEntry;
    datasetInfo: Record<string, NamedInfo>;
    splitInfo: Record<string, NamedInfo>;
    rows: ComparisonRow[];
    onClick: () => void;
};

function validationStateLabel(validationState: string): string {
    if (validationState === "partial") {
        return "Partial evidence";
    }
    if (validationState === "snapshot") {
        return "Smoke snapshot";
    }
    if (validationState === "archived") {
        return "Archive evidence";
    }
    return "Validated";
}

function validationStateTone(validationState: string): "success" | "warning" | "error" | "neutral" {
    if (validationState === "partial") {
        return "error";
    }
    if (validationState === "snapshot") {
        return "warning";
    }
    if (validationState === "archived") {
        return "neutral";
    }
    return "success";
}

function deriveChampionFallback(rows: ComparisonRow[]): BestMethodEntry[] {
    const championRows = rows.filter(isChampionCandidateRow);
    const bestAuc = championRows.find((row) => row.auc_rank === 1);
    const bestEer = championRows.find((row) => row.eer_rank === 1);
    const bestLatency = championRows.find((row) => row.latency_rank === 1 && row.latency_ms != null);

    const candidates = [
        { metric: "best_auc" as const, row: bestAuc },
        { metric: "best_eer" as const, row: bestEer },
        { metric: "best_latency" as const, row: bestLatency },
    ];

    return candidates.flatMap(({ metric, row }) => {
        if (!row) {
            return [];
        }

        return [{
            dataset: row.dataset,
            split: row.split,
            metric,
            method: row.method,
            benchmark_method: row.benchmark_method,
            method_label: row.method_label ?? null,
            method_status: row.method_status ?? null,
            presentation_tier: row.presentation_tier ?? null,
            showcase_eligible: row.showcase_eligible,
            run: row.run,
            value: championValue(row, metric) ?? 0,
            run_family: row.run_family ?? row.run,
            run_label: row.run_label ?? null,
            view_mode: row.view_mode,
            status: row.status,
            validation_state: row.validation_state,
        }];
    });
}

function mergeChampionEntries(bestEntries: BestMethodEntry[], fallbackEntries: BestMethodEntry[]): BestMethodEntry[] {
    const byMetric = new Map<BenchmarkBestMetric, BestMethodEntry>();
    for (const entry of fallbackEntries) {
        byMetric.set(entry.metric, entry);
    }
    for (const entry of bestEntries) {
        byMetric.set(entry.metric, entry);
    }

    return (["best_auc", "best_eer", "best_latency"] as const)
        .map((metric) => byMetric.get(metric))
        .filter((entry): entry is BestMethodEntry => entry != null);
}

function ChampionCard({ entry, datasetInfo, splitInfo, rows, onClick }: ChampionCardProps) {
    const value =
        entry.metric === "best_latency"
            ? formatLatency(entry.value)
            : formatMetric(entry.value);
    const iconNode = entry.metric === "best_latency"
        ? <Zap className="h-5 w-5" />
        : <Trophy className="h-5 w-5" />;

    return (
        <button
            type="button"
            onClick={onClick}
            className={`surface-card p-5 text-left transition hover:border-[var(--app-brand-border)] ${highlightClassName(sortModeForMetric(entry.metric))}`}
        >
            <div className="flex min-w-0 items-start justify-between gap-4">
                <div className="min-w-0">
                    <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">
                        {bestMetricLabel(entry.metric)}
                    </p>
                    <p className="mt-3 safe-text text-xl font-semibold text-[var(--app-text)]">{formatMethodLabel(entry.method, entry.method_label)}</p>
                </div>
                <div className="rounded-xl border border-[var(--app-brand-border)] bg-[var(--app-brand-surface)] p-3 text-[var(--app-brand-text)]">
                    {iconNode}
                </div>
            </div>
            <div className="mt-5 space-y-2 text-sm text-[var(--app-text-muted)]">
                <div className="font-semibold text-[var(--app-text)]">{value}</div>
                <div className="safe-text font-medium text-[var(--app-text-soft)]">{championTradeoffText(entry, rows)}</div>
                <div className="safe-text">{datasetInfo[entry.dataset]?.label ?? entry.dataset}</div>
                <div className="safe-text">{splitInfo[entry.split]?.label ?? entry.split}</div>
                <div className="safe-text">{entry.run_family ?? entry.run}</div>
            </div>
        </button>
    );
}

function FilterField({
    label,
    caption,
    value,
    onChange,
    disabled,
    children,
}: {
    label: string;
    caption?: string;
    value: string;
    onChange: (value: string) => void;
    disabled?: boolean;
    children: ReactNode;
}) {
    return (
        <label className="space-y-2">
            <span className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">{label}</span>
            <select
                className={INPUT_CLASS_NAME}
                value={value}
                disabled={disabled}
                onChange={(event) => onChange(event.target.value)}
            >
                {children}
            </select>
            {caption ? <p className="text-xs text-[var(--app-text-muted)]">{caption}</p> : null}
        </label>
    );
}

function LoadingSkeleton() {
    return (
        <div className="space-y-6 animate-pulse">
            <div className="surface-card p-6">
                <div className="h-6 w-44 rounded-full bg-[var(--app-surface-muted)]" />
                <div className="mt-4 h-4 w-2/3 rounded-full bg-[var(--app-surface-muted)]" />
                <div className="mt-2 h-4 w-1/2 rounded-full bg-[var(--app-surface-muted)]" />
                <div className="mt-6 grid gap-4 md:grid-cols-4">
                    <div className="h-14 rounded-xl bg-[var(--app-surface-muted)]" />
                    <div className="h-14 rounded-xl bg-[var(--app-surface-muted)]" />
                    <div className="h-14 rounded-xl bg-[var(--app-surface-muted)]" />
                    <div className="h-14 rounded-xl bg-[var(--app-surface-muted)]" />
                </div>
            </div>
            <div className="grid gap-4 xl:grid-cols-3">
                <div className="h-40 rounded-xl bg-[var(--app-surface-muted)]" />
                <div className="h-40 rounded-xl bg-[var(--app-surface-muted)]" />
                <div className="h-40 rounded-xl bg-[var(--app-surface-muted)]" />
            </div>
            <div className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_380px]">
                <div className="h-[28rem] rounded-xl bg-[var(--app-surface-muted)]" />
                <div className="h-[28rem] rounded-xl bg-[var(--app-surface-muted)]" />
            </div>
        </div>
    );
}

function CurrentBenchmarkFindingPanel() {
    return (
        <div className="inline-banner inline-banner--info">
            <div className="inline-banner__body">
                <p className="inline-banner__title">Current benchmark finding</p>
                <p>
                    SourceAFIS remains the strongest validated plain-vs-roll evidence on NIST SD300B/SD300C. SIFT v2
                    is now the strongest custom research baseline with exported latency, and the table includes final
                    classical baselines produced under the same strict pair-audited VAL-to-TEST protocol. Positive-only
                    and negative-only evidence is reported separately in final markdown and metrics artifacts. SourceAFIS
                    evidence still comes from the fingerprint-engine HTTP sidecar path and is not a default interactive
                    runtime method.
                </p>
            </div>
        </div>
    );
}

function sourceLabelForRow(row: ComparisonRow | null | undefined): string {
    if (row?.provenance?.benchmark_source_label) {
        return row.provenance.benchmark_source_label;
    }
    if (row?.provenance?.benchmark_source_root === "reference") {
        return "Reference artifacts";
    }
    if (row?.provenance?.benchmark_source_root === "live") {
        return "Live artifacts";
    }
    return "Benchmark artifacts";
}

function firstRankedRow(rows: ComparisonRow[], rankKey: "auc_rank" | "latency_rank"): ComparisonRow | null {
    const championRows = rows.filter(isChampionCandidateRow);
    return championRows.find((row) => row[rankKey] === 1) ?? championRows[0] ?? null;
}

function viewModeOptions(available: NamedInfo[], selectedViewMode: BenchmarkViewMode): NamedInfo[] {
    if (available.length > 0) {
        return available;
    }
    return [{
        key: selectedViewMode,
        label: viewModeLabel(selectedViewMode),
        summary: "",
    }];
}

export default function BenchmarkWorkspace() {
    const benchmark = useBenchmark();
    const evidencePanelRef = useRef<HTMLDivElement>(null);
    const [showResearchHistory, setShowResearchHistory] = useState(false);
    const summary = benchmark.summary;
    const datasetInfo = benchmark.comparison?.dataset_info ?? {};
    const splitInfo = benchmark.comparison?.split_info ?? {};
    const comparisonRowsView = useMemo(
        () => buildComparisonRowsView(
            benchmark.comparisonRows,
            {
                showResearchHistory,
                rowKey: benchmark.rowKey,
            },
        ),
        [benchmark.comparisonRows, benchmark.rowKey, showResearchHistory],
    );
    const displayRows = comparisonRowsView.rows;
    const displayRowKeys = useMemo(
        () => displayRows.map((row) => benchmark.rowKey(row)),
        [benchmark.rowKey, displayRows],
    );
    const selectedDisplayRow = displayRows.find((row) => benchmark.rowKey(row) === benchmark.selectedRowKey)
        ?? displayRows[0]
        ?? null;
    const selectedResearchGroupInfo = selectedDisplayRow
        ? comparisonRowsView.researchGroupInfoByRowKey[benchmark.rowKey(selectedDisplayRow)] ?? null
        : null;
    const fallbackChampionEntries = deriveChampionFallback(displayRows);
    const championEntries = mergeChampionEntries(benchmark.bestEntries, fallbackChampionEntries);
    const currentRunFamily = selectedDisplayRow?.run_family
        ?? summary?.current_run_families?.[0]
        ?? "Resolving run family";
    const availableViewModes = viewModeOptions(benchmark.availableViewModes, benchmark.selectedViewMode);
    const bestAccuracyRow = firstRankedRow(displayRows, "auc_rank");
    const fastestRow = firstRankedRow(displayRows, "latency_rank");
    const storyPairCount = bestAccuracyRow?.n_pairs ?? fastestRow?.n_pairs ?? selectedDisplayRow?.n_pairs ?? null;
    const storyArtifactCount = selectedDisplayRow?.artifact_count
        ?? Math.max(0, ...displayRows.map((row) => row.artifact_count));
    const storySourceLabel = sourceLabelForRow(selectedDisplayRow ?? bestAccuracyRow);
    const hasResearchRows = comparisonRowsView.hasResearchRows;
    const methodCountLabel = hasResearchRows
        ? `${summary?.method_count ?? 0} methods in comparison`
        : `${summary?.method_count ?? 0} validated benchmark methods`;
    const comparisonDescription = hasResearchRows
        ? `Compare ${viewModeLabel(benchmark.selectedViewMode).toLowerCase()} comparison rows on the active dataset and split. Research methods are grouped and marked separately.`
        : `Compare ${viewModeLabel(benchmark.selectedViewMode).toLowerCase()} validated benchmark methods on the active dataset and split.`;
    const fastestIsMostAccurate = Boolean(
        bestAccuracyRow
        && fastestRow
        && bestAccuracyRow.run === fastestRow.run
        && bestAccuracyRow.method === fastestRow.method
        && bestAccuracyRow.benchmark_method === fastestRow.benchmark_method
        && bestAccuracyRow.split === fastestRow.split,
    );
    const tradeoffDetail = fastestRow
        ? fastestIsMostAccurate
            ? "Best balance on this split."
            : "Fastest is not the top AUC method."
        : "Latency evidence unavailable.";
    const hasArchiveAlternative = benchmark.selectedViewMode === "canonical"
        && benchmark.availableViewModes.some((item) => item.key === "archive");

    useEffect(() => {
        setShowResearchHistory(false);
    }, [benchmark.selectedDataset, benchmark.selectedSplit, benchmark.selectedViewMode]);

    useEffect(() => {
        if (displayRowKeys.length === 0) {
            if (benchmark.selectedRowKey) {
                benchmark.setSelectedRowKey("");
            }
            return;
        }
        if (!displayRowKeys.includes(benchmark.selectedRowKey)) {
            benchmark.setSelectedRowKey(displayRowKeys[0]);
        }
    }, [benchmark.selectedRowKey, benchmark.setSelectedRowKey, displayRowKeys]);

    const handleChampionClick = (entry: BestMethodEntry): void => {
        benchmark.setSelectedSortMode(sortModeForMetric(entry.metric));
        const matchingRow = displayRows.find((row) =>
            row.run === entry.run
            && row.method === entry.method
            && row.benchmark_method === (entry.benchmark_method ?? row.benchmark_method)
            && row.split === entry.split,
        );
        if (matchingRow) {
            benchmark.setSelectedRowKey(benchmark.rowKey(matchingRow));
        }
        window.requestAnimationFrame(() => {
            evidencePanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
            evidencePanelRef.current?.focus({ preventScroll: true });
        });
    };

    return (
        <div className="space-y-6">
            <WorkspaceHero
                eyebrow={viewModeLabel(benchmark.selectedViewMode)}
                title="Validated fingerprint matching benchmark"
                description="A curated benchmark view for comparing tested datasets, method winners, speed trade-offs, and evidence artifacts."
                icon={BarChart3}
            >
                <div className="space-y-4">
                    <div className="flex flex-wrap items-center gap-2">
                        <StatusPill tone="brand">{viewModeLabel(benchmark.selectedViewMode)}</StatusPill>
                        <StatusPill tone={validationStateTone(summary?.validation_state ?? "validated")}>
                            {validationStateLabel(summary?.validation_state ?? "validated")}
                        </StatusPill>
                        <StatusPill>Validated evidence datasets</StatusPill>
                    </div>

                    <div className="grid gap-4 md:grid-cols-5">
                        <FilterField
                            label="View"
                            value={benchmark.selectedViewMode}
                            onChange={(value) => benchmark.setSelectedViewMode(value as BenchmarkViewMode)}
                        >
                            {availableViewModes.map((item) => (
                                <option key={item.key} value={item.key}>
                                    {viewModeLabel(item.key)}
                                </option>
                            ))}
                        </FilterField>

                        <FilterField
                            label="Dataset"
                            caption="Showing datasets with validated benchmark evidence."
                            value={benchmark.selectedDataset}
                            onChange={(value) => benchmark.setSelectedDataset(value)}
                        >
                            {benchmark.availableDatasets.map((item) => (
                                <option key={item.key} value={item.key}>
                                    {item.label}
                                </option>
                            ))}
                        </FilterField>

                        <FilterField
                            label="Split"
                            value={benchmark.selectedSplit}
                            onChange={(value) => benchmark.setSelectedSplit(value)}
                            disabled={benchmark.availableSplits.length === 0}
                        >
                            {benchmark.availableSplits.map((item) => (
                                <option key={item.key} value={item.key}>
                                    {item.label}
                                </option>
                            ))}
                        </FilterField>

                        <FilterField
                            label="Sort"
                            value={benchmark.selectedSortMode}
                            onChange={(value) => benchmark.setSelectedSortMode(value as typeof benchmark.selectedSortMode)}
                        >
                            {SORT_OPTIONS.map((item) => (
                                <option key={item.key} value={item.key}>
                                    {item.label}
                                </option>
                            ))}
                        </FilterField>

                        <div className="space-y-2">
                            <span className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Refresh</span>
                            <button
                                type="button"
                                onClick={() => {
                                    void benchmark.refreshAll();
                                }}
                                disabled={benchmark.isLoading}
                                className="app-button app-button--secondary w-full"
                            >
                                <RefreshCcw className="mr-2 h-4 w-4" />
                                Refresh
                            </button>
                        </div>
                    </div>
                </div>
            </WorkspaceHero>

            {benchmark.summaryState.status === "loading" && !summary ? <LoadingSkeleton /> : null}

            {benchmark.summaryState.status === "error" && benchmark.summaryState.error && !summary ? (
                <RequestState
                    variant="error"
                    title="Failed to load benchmark summary"
                    description={benchmark.summaryState.error}
                    actionLabel="Retry"
                    onAction={() => {
                        void benchmark.refreshSummary();
                    }}
                />
            ) : null}

            {summary ? (
                <>
                    <section className="space-y-4">
                        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                            <div>
                                <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Benchmark Story</p>
                                <h3 className="mt-2 text-2xl font-semibold text-[var(--app-text)]">
                                    Validated fingerprint matching benchmark
                                </h3>
                            </div>
                            <div className="flex flex-wrap gap-3 text-sm text-[var(--app-text-muted)]">
                                <StatusPill>
                                    {summary.result_count} comparison rows
                                </StatusPill>
                            <StatusPill>
                                    {methodCountLabel}
                                </StatusPill>
                                <StatusPill tone="brand">
                                    Sorted by {sortModeLabel(benchmark.selectedSortMode)}
                                </StatusPill>
                            </div>
                        </div>

                        <div className="grid gap-4 xl:grid-cols-3">
                            <MetricTile
                                icon={BarChart3}
                                label="Coverage"
                                value={<span className="safe-truncate">{summary.dataset_info?.label ?? summary.dataset}</span>}
                                detail={`${summary.split_info?.label ?? summary.split} - ${formatPairs(storyPairCount)} pairs`}
                                title={`${summary.dataset_info?.label ?? summary.dataset} - ${summary.split_info?.label ?? summary.split}`}
                                tone="brand"
                            />
                            <MetricTile
                                icon={Trophy}
                                label="Winner"
                                value={<span className="safe-truncate">{bestAccuracyRow ? formatMethodLabel(bestAccuracyRow.method, bestAccuracyRow.method_label) : "Resolving"}</span>}
                                detail={`AUC ${formatMetric(bestAccuracyRow?.auc)}`}
                                title={bestAccuracyRow ? formatMethodLabel(bestAccuracyRow.method, bestAccuracyRow.method_label) : "Resolving winner"}
                                tone="success"
                            />
                            <MetricTile
                                icon={Zap}
                                label="Trade-off"
                                value={<span className="safe-truncate">{fastestRow ? formatMethodLabel(fastestRow.method, fastestRow.method_label) : "Resolving"}</span>}
                                detail={`${formatLatency(fastestRow?.latency_ms)} - ${tradeoffDetail}`}
                                title={fastestRow ? formatMethodLabel(fastestRow.method, fastestRow.method_label) : "Resolving fastest method"}
                                tone="warning"
                            />
                        </div>

                        <div className="flex flex-wrap gap-3">
                            <StatusPill tone={validationStateTone(summary.validation_state)}>
                                {validationStateLabel(summary.validation_state)}
                            </StatusPill>
                            <StatusPill title={currentRunFamily}>
                                {currentRunFamily}
                            </StatusPill>
                            <StatusPill tone="info">
                                {storySourceLabel}
                            </StatusPill>
                            <StatusPill>
                                {storyArtifactCount} artifacts
                            </StatusPill>
                        </div>

                        <CurrentBenchmarkFindingPanel />
                    </section>

                    <section className="space-y-4">
                        <div>
                            <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Champion methods</p>
                            <h3 className="mt-2 text-2xl font-semibold text-[var(--app-text)]">
                                Winners and trade-offs
                            </h3>
                        </div>

                        {championEntries.length > 0 ? (
                            <div className="grid gap-4 xl:grid-cols-3">
                                {championEntries.map((entry) => (
                                    <ChampionCard
                                        key={`${entry.metric}_${entry.run}_${entry.method}_${entry.split}`}
                                        entry={entry}
                                        datasetInfo={datasetInfo}
                                        splitInfo={splitInfo}
                                        rows={benchmark.comparisonRows}
                                        onClick={() => handleChampionClick(entry)}
                                    />
                                ))}
                            </div>
                        ) : benchmark.bestState.status === "loading" || benchmark.comparisonState.status === "loading" ? (
                            <div className="grid gap-4 xl:grid-cols-3">
                                <div className="h-40 rounded-xl bg-[var(--app-surface-muted)] animate-pulse" />
                                <div className="h-40 rounded-xl bg-[var(--app-surface-muted)] animate-pulse" />
                                <div className="h-40 rounded-xl bg-[var(--app-surface-muted)] animate-pulse" />
                            </div>
                        ) : (
                            <RequestState
                                variant="empty"
                                title={`No ${viewModeLabel(benchmark.selectedViewMode).toLowerCase()} winners for this selection`}
                                description="Choose another dataset, split, or view with benchmark rows."
                            />
                        )}
                    </section>

                    <section className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_380px]">
                        <div className="space-y-6">
                            <div className="surface-card p-6">
                                <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                                    <div className="space-y-2">
                                        <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Full comparison</p>
                                        <h3 className="text-2xl font-semibold text-[var(--app-text)]">Method comparison table</h3>
                                        <p className="max-w-2xl text-sm leading-6 text-[var(--app-text-muted)]">
                                            {comparisonDescription}
                                        </p>
                                    </div>
                                    {comparisonRowsView.hiddenResearchRowCount > 0 || showResearchHistory ? (
                                        <div className="flex flex-col items-start gap-2">
                                            {comparisonRowsView.hiddenResearchRowCount > 0 ? (
                                                <StatusPill tone="warning">
                                                    {comparisonRowsView.hiddenResearchRowCount} archived research rows hidden
                                                </StatusPill>
                                            ) : null}
                                            {showResearchHistory ? (
                                                <StatusPill tone="info">Archived research runs visible</StatusPill>
                                            ) : null}
                                            <button
                                                type="button"
                                                className="app-button app-button--secondary"
                                                onClick={() => setShowResearchHistory((current) => !current)}
                                            >
                                                {showResearchHistory ? "Hide archived research runs" : "Show archived research runs"}
                                            </button>
                                        </div>
                                    ) : null}
                                    <StatusPill title={`${summary.dataset_info?.label ?? summary.dataset} - ${summary.split_info?.label ?? summary.split}`}>
                                        {summary.dataset_info?.label ?? summary.dataset} - {summary.split_info?.label ?? summary.split}
                                    </StatusPill>
                                </div>

                                <div className="mt-6">
                                    {benchmark.comparisonState.status === "loading" && displayRows.length === 0 ? (
                                        <RequestState
                                            variant="loading"
                                            title="Loading comparison rows"
                                            description={`Reading ${viewModeLabel(benchmark.selectedViewMode).toLowerCase()} benchmark rows for the active dataset and split.`}
                                        />
                                    ) : null}

                                    {benchmark.comparisonState.status === "error" && benchmark.comparisonState.error ? (
                                        <RequestState
                                            variant="error"
                                            title="Failed to load comparison rows"
                                            description={benchmark.comparisonState.error}
                                            actionLabel="Retry"
                                            onAction={() => {
                                                void benchmark.reloadComparison();
                                            }}
                                        />
                                    ) : null}

                                    {benchmark.comparisonState.status === "success" && displayRows.length === 0 ? (
                                        <RequestState
                                            variant="empty"
                                            title={`No ${viewModeLabel(benchmark.selectedViewMode).toLowerCase()} rows for this selection`}
                                            description={hasArchiveAlternative ? "No showcase rows for this selection. Try Archive." : "Choose another dataset, split, or view mode."}
                                        />
                                    ) : null}

                                    {displayRows.length > 0 ? (
                                        <BenchmarkComparisonTable
                                            rows={displayRows}
                                            selectedRowKey={benchmark.selectedRowKey}
                                            onSelectRow={benchmark.setSelectedRowKey}
                                            rowKey={benchmark.rowKey}
                                            sortMode={benchmark.selectedSortMode}
                                            researchGroupInfoByRowKey={comparisonRowsView.researchGroupInfoByRowKey}
                                        />
                                    ) : null}
                                </div>
                            </div>
                        </div>

                        <div ref={evidencePanelRef} tabIndex={-1} className="focus:outline-none">
                            <BenchmarkEvidencePanel
                                row={selectedDisplayRow}
                                datasetInfo={datasetInfo}
                                splitInfo={splitInfo}
                                researchGroupInfo={selectedResearchGroupInfo}
                                showResearchHistory={showResearchHistory}
                                onShowResearchHistory={() => setShowResearchHistory(true)}
                            />
                        </div>
                    </section>
                </>
            ) : null}
        </div>
    );
}
