import { useCallback, useEffect, useMemo, useState } from "react";
import {
    fetchBenchmarkBest,
    fetchBenchmarkComparison,
    fetchBenchmarkSummary,
} from "../../../api/matchService.ts";
import {
    createErrorState,
    createIdleState,
    createLoadingState,
    createSuccessState,
    type AsyncState,
} from "../../../shared/request-state";
import type {
    BenchmarkBestMetric,
    BenchmarkSortMode,
    BenchmarkSummaryResponse,
    BestMethodsResponse,
    ComparisonResponse,
    ComparisonRow,
} from "../../../types";
import { toErrorMessage } from "../../../utils/error.ts";

const BENCHMARK_DATASET_QUERY_PARAM = "benchmarkDataset";
const BENCHMARK_SPLIT_QUERY_PARAM = "benchmarkSplit";
const BENCHMARK_VIEW_QUERY_PARAM = "benchmarkView";
const BENCHMARK_SORT_QUERY_PARAM = "benchmarkSort";

type BenchmarkUrlState = {
    dataset: string;
    split: string;
    sortMode: BenchmarkSortMode;
};

type EffectiveSelection = {
    dataset?: string;
    split?: string;
    sortMode: BenchmarkSortMode;
};

const DEFAULT_SORT_MODE: BenchmarkSortMode = "best_accuracy";

function normalizeSortMode(value: string | null | undefined): BenchmarkSortMode {
    if (value === "lowest_eer" || value === "lowest_latency") {
        return value;
    }
    return DEFAULT_SORT_MODE;
}

function readBenchmarkUrlState(): BenchmarkUrlState {
    if (typeof window === "undefined") {
        return {
            dataset: "",
            split: "",
            sortMode: DEFAULT_SORT_MODE,
        };
    }

    const params = new URLSearchParams(window.location.search);
    return {
        dataset: params.get(BENCHMARK_DATASET_QUERY_PARAM) ?? "",
        split: params.get(BENCHMARK_SPLIT_QUERY_PARAM) ?? "",
        sortMode: normalizeSortMode(params.get(BENCHMARK_SORT_QUERY_PARAM)),
    };
}

function syncBenchmarkUrlState(state: BenchmarkUrlState): void {
    const params = new URLSearchParams(window.location.search);

    if (state.dataset) {
        params.set(BENCHMARK_DATASET_QUERY_PARAM, state.dataset);
    } else {
        params.delete(BENCHMARK_DATASET_QUERY_PARAM);
    }

    if (state.split) {
        params.set(BENCHMARK_SPLIT_QUERY_PARAM, state.split);
    } else {
        params.delete(BENCHMARK_SPLIT_QUERY_PARAM);
    }

    params.delete(BENCHMARK_VIEW_QUERY_PARAM);
    params.set(BENCHMARK_SORT_QUERY_PARAM, state.sortMode);

    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
    window.history.replaceState(window.history.state, "", nextUrl);
}

function metricLabel(metric: BenchmarkBestMetric): string {
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

function formatMetricValue(metric: BenchmarkBestMetric, value: number): string {
    if (metric === "best_latency") {
        return `${value.toFixed(2)} ms`;
    }

    return value.toFixed(4);
}

function sortModeLabel(sortMode: BenchmarkSortMode): string {
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

function rowKey(row: ComparisonRow): string {
    return `${row.run}::${row.split}::${row.method}::${row.benchmark_method}`;
}

function normalizeSelectionFromSummary(summary: BenchmarkSummaryResponse): EffectiveSelection {
    return {
        dataset: summary.dataset,
        split: summary.split,
        sortMode: DEFAULT_SORT_MODE,
    };
}

export function useBenchmark() {
    const initialUrlState = useMemo(() => readBenchmarkUrlState(), []);
    const [summaryState, setSummaryState] = useState<AsyncState<BenchmarkSummaryResponse>>(createLoadingState());
    const [comparisonState, setComparisonState] = useState<AsyncState<ComparisonResponse>>(createIdleState());
    const [bestState, setBestState] = useState<AsyncState<BestMethodsResponse>>(createIdleState());
    const [selectedDataset, setSelectedDataset] = useState(initialUrlState.dataset);
    const [selectedSplit, setSelectedSplit] = useState(initialUrlState.split);
    const [selectedSortMode, setSelectedSortMode] = useState<BenchmarkSortMode>(initialUrlState.sortMode);
    const [selectedRowKey, setSelectedRowKey] = useState<string>("");

    const loadSummary = useCallback(async (selection: EffectiveSelection): Promise<BenchmarkSummaryResponse | null> => {
        setSummaryState((current) => createLoadingState(current.data));

        try {
            const payload = await fetchBenchmarkSummary({
                dataset: selection.dataset || undefined,
                split: selection.split || undefined,
            });
            setSummaryState(createSuccessState(payload));
            setSelectedDataset(payload.dataset);
            setSelectedSplit(payload.split);
            return payload;
        } catch (error) {
            setSummaryState((current) => createErrorState(toErrorMessage(error), current.data));
            setComparisonState(createIdleState());
            setBestState(createIdleState());
            return null;
        }
    }, []);

    const loadComparison = useCallback(async (selection: EffectiveSelection): Promise<void> => {
        if (!selection.dataset) {
            setComparisonState(createIdleState());
            return;
        }

        setComparisonState((current) => createLoadingState(current.data));

        try {
            const payload = await fetchBenchmarkComparison({
                dataset: selection.dataset,
                split: selection.split || undefined,
                sort_mode: selection.sortMode,
            });
            setComparisonState(createSuccessState(payload));
        } catch (error) {
            setComparisonState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, []);

    const loadBest = useCallback(async (selection: EffectiveSelection): Promise<void> => {
        if (!selection.dataset) {
            setBestState(createIdleState());
            return;
        }

        setBestState((current) => createLoadingState(current.data));

        try {
            const payload = await fetchBenchmarkBest({
                dataset: selection.dataset,
                split: selection.split || undefined,
            });
            setBestState(createSuccessState(payload));
        } catch (error) {
            setBestState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, []);

    useEffect(() => {
        const handlePopState = (): void => {
            const nextState = readBenchmarkUrlState();
            setSelectedDataset(nextState.dataset);
            setSelectedSplit(nextState.split);
            setSelectedSortMode(nextState.sortMode);
        };

        window.addEventListener("popstate", handlePopState);
        return () => {
            window.removeEventListener("popstate", handlePopState);
        };
    }, []);

    useEffect(() => {
        void loadSummary({
            dataset: selectedDataset || undefined,
            split: selectedSplit || undefined,
            sortMode: DEFAULT_SORT_MODE,
        });
    }, [loadSummary, selectedDataset, selectedSplit]);

    useEffect(() => {
        if (summaryState.status !== "success" || !summaryState.data) {
            return;
        }

        const effectiveSelection: EffectiveSelection = {
            ...normalizeSelectionFromSummary(summaryState.data),
            sortMode: selectedSortMode,
        };

        void Promise.all([
            loadComparison(effectiveSelection),
            loadBest(effectiveSelection),
        ]);
    }, [loadBest, loadComparison, selectedSortMode, summaryState.data, summaryState.status]);

    useEffect(() => {
        syncBenchmarkUrlState({
            dataset: selectedDataset,
            split: selectedSplit,
            sortMode: selectedSortMode,
        });
    }, [selectedDataset, selectedSortMode, selectedSplit]);

    const summary = summaryState.data ?? null;
    const comparison = comparisonState.data ?? null;
    const best = bestState.data ?? null;
    const availableDatasets = summary?.available_datasets ?? [];
    const availableSplits = summary?.available_splits ?? [];
    const comparisonRows = useMemo(
        () => comparison?.rows ?? [],
        [comparison?.rows],
    );

    useEffect(() => {
        if (comparisonRows.length === 0) {
            setSelectedRowKey("");
            return;
        }

        setSelectedRowKey(rowKey(comparisonRows[0]));
    }, [comparisonRows]);

    const selectedRow = useMemo(
        () => comparisonRows.find((row) => rowKey(row) === selectedRowKey) ?? null,
        [comparisonRows, selectedRowKey],
    );

    const refreshSummary = useCallback(async (): Promise<void> => {
        await loadSummary({
            dataset: selectedDataset || undefined,
            split: selectedSplit || undefined,
            sortMode: DEFAULT_SORT_MODE,
        });
    }, [loadSummary, selectedDataset, selectedSplit]);

    const refreshAll = useCallback(async (): Promise<void> => {
        const payload = await loadSummary({
            dataset: selectedDataset || undefined,
            split: selectedSplit || undefined,
            sortMode: selectedSortMode,
        });

        const effectiveSelection: EffectiveSelection = payload
            ? {
                dataset: payload.dataset,
                split: payload.split,
                sortMode: selectedSortMode,
            }
            : {
                dataset: selectedDataset || undefined,
                split: selectedSplit || undefined,
                sortMode: selectedSortMode,
            };

        await Promise.all([
            loadComparison(effectiveSelection),
            loadBest(effectiveSelection),
        ]);
    }, [loadBest, loadComparison, loadSummary, selectedDataset, selectedSortMode, selectedSplit]);

    const isLoading =
        summaryState.status === "loading"
        || comparisonState.status === "loading"
        || bestState.status === "loading";

    return {
        summaryState,
        comparisonState,
        bestState,
        summary,
        comparison,
        best,
        comparisonRows,
        bestEntries: best?.entries ?? [],
        availableDatasets,
        availableSplits,
        selectedDataset,
        setSelectedDataset,
        selectedSplit,
        setSelectedSplit,
        selectedSortMode,
        setSelectedSortMode,
        selectedRow,
        selectedRowKey,
        setSelectedRowKey,
        isLoading,
        metricLabel,
        formatMetricValue,
        sortModeLabel,
        rowKey,
        refreshAll,
        refreshSummary,
        reloadComparison: async () => {
            await loadComparison({
                dataset: summary?.dataset ?? (selectedDataset || undefined),
                split: summary?.split ?? (selectedSplit || undefined),
                sortMode: selectedSortMode,
            });
        },
        reloadBest: async () => {
            await loadBest({
                dataset: summary?.dataset ?? (selectedDataset || undefined),
                split: summary?.split ?? (selectedSplit || undefined),
                sortMode: selectedSortMode,
            });
        },
    };
}
