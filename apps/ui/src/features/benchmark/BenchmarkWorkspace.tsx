import type { ReactNode } from "react";
import { BarChart3, RefreshCcw, Trophy, Zap } from "lucide-react";
import { BenchmarkComparisonTable } from "../../components/BenchmarkComparisonTable.tsx";
import RequestState from "../../components/RequestState.tsx";
import type { BenchmarkBestMetric, BestMethodEntry, ComparisonRow, NamedInfo } from "../../types";
import BenchmarkEvidencePanel from "./BenchmarkEvidencePanel.tsx";
import {
    bestMetricLabel,
    championValue,
    formatLatency,
    formatMethodLabel,
    formatMetric,
    highlightClassName,
    sortModeForMetric,
    sortModeLabel,
} from "./benchmarkPresentation.ts";
import { useBenchmark } from "./hooks/useBenchmark.ts";

const FILTER_CLASS_NAME = "w-full rounded-2xl border border-slate-800 bg-slate-950/80 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-sky-500/40 focus:ring-4 focus:ring-sky-500/10";

const SORT_OPTIONS = [
    { key: "best_accuracy", label: "Best accuracy" },
    { key: "lowest_eer", label: "Lowest EER" },
    { key: "lowest_latency", label: "Lowest latency" },
] as const;

type ChampionCardProps = {
    entry: BestMethodEntry;
    datasetInfo: Record<string, NamedInfo>;
    splitInfo: Record<string, NamedInfo>;
    onClick: () => void;
};

function validationStateLabel(validationState: string): string {
    if (validationState === "partial") {
        return "Partial evidence";
    }
    return "Validated showcase";
}

function validationStateClassName(validationState: string): string {
    if (validationState === "partial") {
        return "border-rose-500/25 bg-rose-500/10 text-rose-200";
    }
    return "border-emerald-500/25 bg-emerald-500/10 text-emerald-200";
}

function deriveChampionFallback(rows: ComparisonRow[]): BestMethodEntry[] {
    const bestAuc = rows.find((row) => row.auc_rank === 1);
    const bestEer = rows.find((row) => row.eer_rank === 1);
    const bestLatency = rows.find((row) => row.latency_rank === 1 && row.latency_ms != null);

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

function ChampionCard({ entry, datasetInfo, splitInfo, onClick }: ChampionCardProps) {
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
            className={`rounded-[1.6rem] border border-slate-800 bg-slate-950/75 p-5 text-left text-slate-100 shadow-[0_20px_80px_rgba(2,6,23,0.35)] transition hover:border-sky-500/35 hover:bg-slate-950 ${highlightClassName(sortModeForMetric(entry.metric))}`}
        >
            <div className="flex items-start justify-between gap-4">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                        {bestMetricLabel(entry.metric)}
                    </p>
                    <p className="mt-3 text-2xl font-semibold text-slate-50">{formatMethodLabel(entry.method, entry.method_label)}</p>
                </div>
                <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-3 text-sky-300">
                    {iconNode}
                </div>
            </div>
            <div className="mt-5 space-y-2 text-sm text-slate-300">
                <div className="font-semibold text-slate-50">{value}</div>
                <div>{datasetInfo[entry.dataset]?.label ?? entry.dataset}</div>
                <div>{splitInfo[entry.split]?.label ?? entry.split}</div>
                <div className="text-slate-500">{entry.run_family ?? entry.run}</div>
            </div>
        </button>
    );
}

function FilterField({
    label,
    value,
    onChange,
    disabled,
    children,
}: {
    label: string;
    value: string;
    onChange: (value: string) => void;
    disabled?: boolean;
    children: ReactNode;
}) {
    return (
        <label className="space-y-2">
            <span className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">{label}</span>
            <select
                className={FILTER_CLASS_NAME}
                value={value}
                disabled={disabled}
                onChange={(event) => onChange(event.target.value)}
            >
                {children}
            </select>
        </label>
    );
}

function LoadingSkeleton() {
    return (
        <div className="space-y-6 animate-pulse">
            <div className="rounded-[2rem] border border-slate-800 bg-slate-950/75 p-6">
                <div className="h-6 w-44 rounded-full bg-slate-800" />
                <div className="mt-4 h-4 w-2/3 rounded-full bg-slate-900" />
                <div className="mt-2 h-4 w-1/2 rounded-full bg-slate-900" />
                <div className="mt-6 grid gap-4 md:grid-cols-4">
                    <div className="h-14 rounded-2xl bg-slate-900" />
                    <div className="h-14 rounded-2xl bg-slate-900" />
                    <div className="h-14 rounded-2xl bg-slate-900" />
                    <div className="h-14 rounded-2xl bg-slate-900" />
                </div>
            </div>
            <div className="grid gap-4 xl:grid-cols-3">
                <div className="h-40 rounded-[1.6rem] bg-slate-950/75" />
                <div className="h-40 rounded-[1.6rem] bg-slate-950/75" />
                <div className="h-40 rounded-[1.6rem] bg-slate-950/75" />
            </div>
            <div className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_380px]">
                <div className="h-[28rem] rounded-[1.75rem] bg-slate-950/75" />
                <div className="h-[28rem] rounded-[1.75rem] bg-slate-950/75" />
            </div>
        </div>
    );
}

export default function BenchmarkWorkspace() {
    const benchmark = useBenchmark();
    const summary = benchmark.summary;
    const datasetInfo = benchmark.comparison?.dataset_info ?? {};
    const splitInfo = benchmark.comparison?.split_info ?? {};
    const fallbackChampionEntries = deriveChampionFallback(benchmark.comparisonRows);
    const championEntries = mergeChampionEntries(benchmark.bestEntries, fallbackChampionEntries);
    const currentRunFamily = benchmark.selectedRow?.run_family
        ?? summary?.current_run_families?.[0]
        ?? "Resolving run family";

    const handleChampionClick = (entry: BestMethodEntry): void => {
        benchmark.setSelectedSortMode(sortModeForMetric(entry.metric));
        const matchingRow = benchmark.comparisonRows.find((row) =>
            row.run === entry.run
            && row.method === entry.method
            && row.benchmark_method === (entry.benchmark_method ?? row.benchmark_method)
            && row.split === entry.split,
        );
        if (matchingRow) {
            benchmark.setSelectedRowKey(benchmark.rowKey(matchingRow));
        }
    };

    return (
        <div className="space-y-6 text-slate-100">
            <section className="relative overflow-hidden rounded-[2rem] border border-slate-800 bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.18),_transparent_35%),radial-gradient(circle_at_top_right,_rgba(16,185,129,0.12),_transparent_30%),linear-gradient(180deg,rgba(15,23,42,0.98),rgba(2,6,23,1))] p-6 shadow-[0_24px_120px_rgba(2,6,23,0.45)]">
                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-sky-400/50 to-transparent" />
                <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
                    <div className="max-w-3xl space-y-5">
                        <div className="flex flex-wrap items-center gap-2">
                            <span className="inline-flex rounded-full border border-sky-500/25 bg-sky-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-sky-200">
                                Curated full results
                            </span>
                            <span className={`inline-flex rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${validationStateClassName(summary?.validation_state ?? "validated")}`}>
                                {validationStateLabel(summary?.validation_state ?? "validated")}
                            </span>
                        </div>

                        <div>
                            <h2 className="text-3xl font-semibold tracking-tight text-slate-50">Benchmarks</h2>
                            <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-300">
                                Curated full benchmark results for validated showcase runs. Compare accuracy, EER, latency,
                                and method evidence without browsing historical tiers.
                            </p>
                        </div>

                        <div className="grid gap-4 md:grid-cols-4">
                            <FilterField
                                label="Dataset"
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
                                label="Sort by"
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
                                <span className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Refresh results</span>
                                <button
                                    type="button"
                                    onClick={() => {
                                        void benchmark.refreshAll();
                                    }}
                                    disabled={benchmark.isLoading}
                                    className="inline-flex w-full items-center justify-center rounded-2xl border border-slate-700 bg-slate-950/80 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:border-sky-500/35 hover:bg-slate-950 disabled:cursor-not-allowed disabled:opacity-60"
                                >
                                    <RefreshCcw className="mr-2 h-4 w-4" />
                                    Refresh results
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="w-full max-w-md rounded-[1.6rem] border border-slate-800 bg-slate-950/70 p-5">
                        <div className="flex items-start justify-between gap-4">
                            <div>
                                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Current scope</p>
                                <p className="mt-3 text-xl font-semibold text-slate-50">
                                    {summary?.dataset_info?.label ?? "Loading dataset"}
                                </p>
                            </div>
                            <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-3 text-sky-300">
                                <BarChart3 className="h-5 w-5" />
                            </div>
                        </div>
                        <dl className="mt-4 space-y-3 text-sm text-slate-300">
                            <div className="flex items-center justify-between gap-4">
                                <dt className="text-slate-500">Split</dt>
                                <dd>{summary?.split_info?.label ?? "Resolving split"}</dd>
                            </div>
                            <div className="flex items-center justify-between gap-4">
                                <dt className="text-slate-500">Run family</dt>
                                <dd className="text-right">{currentRunFamily}</dd>
                            </div>
                        </dl>
                        <p className="mt-4 text-sm leading-6 text-slate-400">
                            Showing curated full benchmark results.
                        </p>
                    </div>
                </div>
            </section>

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
                                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Executive summary</p>
                                <h3 className="mt-2 text-2xl font-semibold text-slate-50">
                                    Showcase winners and trade-offs
                                </h3>
                            </div>
                            <div className="flex flex-wrap gap-3 text-sm text-slate-400">
                                <div className="rounded-full border border-slate-800 bg-slate-950/80 px-3 py-1.5">
                                    {summary.result_count} comparison rows
                                </div>
                                <div className="rounded-full border border-slate-800 bg-slate-950/80 px-3 py-1.5">
                                    {summary.method_count} methods
                                </div>
                                <div className="rounded-full border border-slate-800 bg-slate-950/80 px-3 py-1.5">
                                    Sorted by {sortModeLabel(benchmark.selectedSortMode)}
                                </div>
                            </div>
                        </div>

                        {championEntries.length > 0 ? (
                            <div className="grid gap-4 xl:grid-cols-3">
                                {championEntries.map((entry) => (
                                    <ChampionCard
                                        key={`${entry.metric}_${entry.run}_${entry.method}_${entry.split}`}
                                        entry={entry}
                                        datasetInfo={datasetInfo}
                                        splitInfo={splitInfo}
                                        onClick={() => handleChampionClick(entry)}
                                    />
                                ))}
                            </div>
                        ) : benchmark.bestState.status === "loading" || benchmark.comparisonState.status === "loading" ? (
                            <div className="grid gap-4 xl:grid-cols-3">
                                <div className="h-40 rounded-[1.6rem] bg-slate-950/75 animate-pulse" />
                                <div className="h-40 rounded-[1.6rem] bg-slate-950/75 animate-pulse" />
                                <div className="h-40 rounded-[1.6rem] bg-slate-950/75 animate-pulse" />
                            </div>
                        ) : (
                            <RequestState
                                variant="empty"
                                title="No showcase winners for this selection"
                                description="Choose another dataset or split with curated full benchmark results."
                            />
                        )}
                    </section>

                    <section className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_380px]">
                        <div className="space-y-6">
                            <div className="rounded-[1.75rem] border border-slate-800 bg-slate-950/80 p-6 shadow-[0_20px_80px_rgba(2,6,23,0.35)]">
                                <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                                    <div className="space-y-2">
                                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Full comparison</p>
                                        <h3 className="text-2xl font-semibold text-slate-50">Method comparison table</h3>
                                        <p className="max-w-2xl text-sm leading-6 text-slate-400">
                                            Compare the validated showcase methods on the active dataset and split. Missing values
                                            remain stable as N/A so the comparison stays readable.
                                        </p>
                                    </div>
                                    <div className="rounded-full border border-slate-800 bg-slate-950/70 px-3 py-2 text-sm text-slate-300">
                                        {summary.dataset_info?.label ?? summary.dataset} - {summary.split_info?.label ?? summary.split}
                                    </div>
                                </div>

                                <div className="mt-6">
                                    {benchmark.comparisonState.status === "loading" && benchmark.comparisonRows.length === 0 ? (
                                        <RequestState
                                            variant="loading"
                                            title="Loading curated comparison rows"
                                            description="Reading validated full benchmark rows for the active dataset and split."
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

                                    {benchmark.comparisonState.status === "success" && benchmark.comparisonRows.length === 0 ? (
                                        <RequestState
                                            variant="empty"
                                            title="No curated full benchmark results for this selection"
                                            description="Choose another dataset or split to continue browsing the showcase."
                                        />
                                    ) : null}

                                    {benchmark.comparisonRows.length > 0 ? (
                                        <BenchmarkComparisonTable
                                            rows={benchmark.comparisonRows}
                                            selectedRowKey={benchmark.selectedRowKey}
                                            onSelectRow={benchmark.setSelectedRowKey}
                                            rowKey={benchmark.rowKey}
                                            sortMode={benchmark.selectedSortMode}
                                        />
                                    ) : null}
                                </div>
                            </div>
                        </div>

                        <BenchmarkEvidencePanel
                            row={benchmark.selectedRow}
                            datasetInfo={datasetInfo}
                            splitInfo={splitInfo}
                        />
                    </section>
                </>
            ) : null}
        </div>
    );
}
