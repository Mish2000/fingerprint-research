import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchCatalogDatasetBrowser, fetchCatalogDatasets } from "../../../api/matchService.ts";
import {
    createErrorState,
    createLoadingState,
    createSuccessState,
    type AsyncState,
} from "../../../shared/request-state/index.ts";
import type { CatalogBrowserItem, CatalogDatasetSummary, JsonRecord } from "../../../types/index.ts";
import { toErrorMessage } from "../../../utils/error.ts";
import {
    countActiveBrowserFilters,
    createBrowserFilterOptions,
    createBrowserPagination,
    createBrowserPairKey,
    createDefaultBrowserFilters,
    createEmptyBrowserPagination,
    createPairPreviewState,
    inferNextSelectionTarget,
    toCatalogBrowserQuery,
    type BrowserFilters,
    type BrowserPagination,
    type BrowserSelectionSide,
    type PairPreviewState,
} from "../browserModel.ts";
import type { PersistedVerifyBrowserState } from "../persistence.ts";

interface UseCatalogBrowserOptions {
    initialState?: PersistedVerifyBrowserState | null;
}

export function useCatalogBrowser(options: UseCatalogBrowserOptions = {}) {
    const [datasetsState, setDatasetsState] = useState<AsyncState<CatalogDatasetSummary[]>>(createLoadingState());
    const [selectedDataset, setSelectedDatasetState] = useState<CatalogDatasetSummary | null>(null);
    const [browserFilters, setBrowserFilters] = useState<BrowserFilters>(() =>
        options.initialState?.filters ?? createDefaultBrowserFilters(),
    );
    const [browserItems, setBrowserItems] = useState<CatalogBrowserItem[]>([]);
    const [browserLoading, setBrowserLoading] = useState(false);
    const [browserError, setBrowserError] = useState<string | null>(null);
    const [browserPagination, setBrowserPagination] = useState<BrowserPagination>(createEmptyBrowserPagination());
    const [browserSummary, setBrowserSummary] = useState<JsonRecord>({});
    const [selectedAssetA, setSelectedAssetA] = useState<CatalogBrowserItem | null>(
        () => options.initialState?.selectedAssetA ?? null,
    );
    const [selectedAssetB, setSelectedAssetB] = useState<CatalogBrowserItem | null>(
        () => options.initialState?.selectedAssetB ?? null,
    );
    const [replacementTarget, setReplacementTarget] = useState<BrowserSelectionSide | null>(
        () => options.initialState?.replacementTarget ?? null,
    );
    const [browserReloadNonce, setBrowserReloadNonce] = useState(0);
    const preferredDatasetKeyRef = useRef<string | null>(options.initialState?.selectedDatasetKey ?? null);
    const datasetsLoadedRef = useRef(false);
    const browserRequestIdRef = useRef(0);

    const datasets = useMemo(() => datasetsState.data ?? [], [datasetsState.data]);
    const selectedDatasetKey = selectedDataset?.dataset ?? null;
    const browserPageLimit = browserFilters.limit;
    const browserReadyDatasets = useMemo(
        () => datasets.filter((dataset) => dataset.has_browser_assets),
        [datasets],
    );
    const browserQuery = useMemo(
        () => (
            selectedDataset?.has_browser_assets
                ? toCatalogBrowserQuery(selectedDataset.dataset, browserFilters)
                : null
        ),
        [browserFilters, selectedDataset?.dataset, selectedDataset?.has_browser_assets],
    );

    const loadDatasets = useCallback(async (): Promise<void> => {
        setDatasetsState((current) => createLoadingState(current.data));

        try {
            const payload = await fetchCatalogDatasets();
            setDatasetsState(createSuccessState(payload.items));
        } catch (error) {
            setDatasetsState(createErrorState(toErrorMessage(error), []));
        }
    }, []);

    useEffect(() => {
        if (datasetsLoadedRef.current) {
            return;
        }

        datasetsLoadedRef.current = true;
        void loadDatasets();
    }, [loadDatasets]);

    useEffect(() => {
        if (browserReadyDatasets.length === 0) {
            if (selectedDataset !== null) {
                setSelectedDatasetState(null);
            }
            return;
        }

        const preferredDatasetKey = selectedDataset?.dataset ?? preferredDatasetKeyRef.current;
        const matchingDataset = browserReadyDatasets.find((dataset) => dataset.dataset === preferredDatasetKey) ?? null;
        if (matchingDataset) {
            if (matchingDataset !== selectedDataset) {
                setSelectedDatasetState(matchingDataset);
            }
            return;
        }

        setSelectedDatasetState(browserReadyDatasets[0]);
    }, [browserReadyDatasets, selectedDataset]);

    useEffect(() => {
        setSelectedAssetA((current) => {
            if (!current || !selectedDatasetKey || current.dataset === selectedDatasetKey) {
                return current;
            }
            return null;
        });
        setSelectedAssetB((current) => {
            if (!current || !selectedDatasetKey || current.dataset === selectedDatasetKey) {
                return current;
            }
            return null;
        });
        setReplacementTarget(null);
    }, [selectedDatasetKey]);

    useEffect(() => {
        if (!selectedDataset?.has_browser_assets || !browserQuery) {
            setBrowserItems([]);
            setBrowserLoading(false);
            setBrowserError(null);
            setBrowserSummary({});
            setBrowserPagination(createEmptyBrowserPagination(browserPageLimit));
            return;
        }

        const requestId = browserRequestIdRef.current + 1;
        browserRequestIdRef.current = requestId;
        setBrowserLoading(true);
        setBrowserError(null);

        void (async () => {
            try {
                const response = await fetchCatalogDatasetBrowser(browserQuery);

                if (browserRequestIdRef.current !== requestId) {
                    return;
                }

                setBrowserItems(response.items);
                setBrowserSummary(response.summary);
                setBrowserPagination(createBrowserPagination(response));
            } catch (error) {
                if (browserRequestIdRef.current !== requestId) {
                    return;
                }

                setBrowserItems([]);
                setBrowserSummary({});
                setBrowserPagination(createEmptyBrowserPagination(browserPageLimit));
                setBrowserError(toErrorMessage(error));
            } finally {
                if (browserRequestIdRef.current === requestId) {
                    setBrowserLoading(false);
                }
            }
        })();
    }, [
        browserPageLimit,
        browserQuery,
        browserReloadNonce,
        selectedDataset?.has_browser_assets,
    ]);

    const selectDataset = useCallback((dataset: CatalogDatasetSummary): void => {
        preferredDatasetKeyRef.current = dataset.dataset;
        setSelectedDatasetState(dataset);
        setBrowserFilters(createDefaultBrowserFilters());
        setSelectedAssetA(null);
        setSelectedAssetB(null);
        setReplacementTarget(null);
        setBrowserItems([]);
        setBrowserError(null);
        setBrowserSummary({});
        setBrowserPagination(createEmptyBrowserPagination());
    }, []);

    const updateBrowserFilters = useCallback((patch: Partial<BrowserFilters>): void => {
        setBrowserFilters((current) => {
            const next = {
                ...current,
                ...patch,
            };

            if (!Object.prototype.hasOwnProperty.call(patch, "offset")) {
                next.offset = 0;
            }

            return next;
        });
    }, []);

    const resetBrowserFilters = useCallback((): void => {
        setBrowserFilters(createDefaultBrowserFilters());
    }, []);

    const selectBrowserItem = useCallback((item: CatalogBrowserItem): void => {
        const target = inferNextSelectionTarget(selectedAssetA, selectedAssetB, replacementTarget);

        if (target === null) {
            return;
        }

        if (target === "A") {
            if (selectedAssetB?.asset_id === item.asset_id) {
                return;
            }
            setSelectedAssetA(item);
        } else {
            if (selectedAssetA?.asset_id === item.asset_id) {
                return;
            }
            setSelectedAssetB(item);
        }

        setReplacementTarget(null);
    }, [replacementTarget, selectedAssetA, selectedAssetB]);

    const clearSelectedAsset = useCallback((side: BrowserSelectionSide): void => {
        if (side === "A") {
            setSelectedAssetA(null);
        } else {
            setSelectedAssetB(null);
        }
        setReplacementTarget(null);
    }, []);

    const startReplacingAsset = useCallback((side: BrowserSelectionSide): void => {
        setReplacementTarget(side);
    }, []);

    const cancelReplacingAsset = useCallback((): void => {
        setReplacementTarget(null);
    }, []);

    const swapSelectedAssets = useCallback((): void => {
        if (!selectedAssetA || !selectedAssetB) {
            return;
        }

        setSelectedAssetA(selectedAssetB);
        setSelectedAssetB(selectedAssetA);
        setReplacementTarget(null);
    }, [selectedAssetA, selectedAssetB]);

    const resetBrowserContinuity = useCallback((): void => {
        preferredDatasetKeyRef.current = null;
        setSelectedDatasetState(browserReadyDatasets[0] ?? null);
        setBrowserFilters(createDefaultBrowserFilters());
        setSelectedAssetA(null);
        setSelectedAssetB(null);
        setReplacementTarget(null);
        setBrowserItems([]);
        setBrowserError(null);
        setBrowserSummary({});
        setBrowserPagination(createEmptyBrowserPagination());
        setBrowserReloadNonce((current) => current + 1);
    }, [browserReadyDatasets]);

    const nextSelectionTarget = useMemo(
        () => inferNextSelectionTarget(selectedAssetA, selectedAssetB, replacementTarget),
        [replacementTarget, selectedAssetA, selectedAssetB],
    );
    const pairPreviewState: PairPreviewState = useMemo(
        () => createPairPreviewState(selectedAssetA, selectedAssetB, replacementTarget),
        [replacementTarget, selectedAssetA, selectedAssetB],
    );
    const browserFilterOptions = useMemo(
        () => createBrowserFilterOptions(browserSummary, browserItems, selectedAssetA, selectedAssetB),
        [browserItems, browserSummary, selectedAssetA, selectedAssetB],
    );
    const activeFilterCount = useMemo(() => countActiveBrowserFilters(browserFilters), [browserFilters]);
    const browserPairKey = useMemo(() => createBrowserPairKey(selectedAssetA, selectedAssetB), [selectedAssetA, selectedAssetB]);

    return {
        activeFilterCount,
        browserError,
        browserFilterOptions,
        browserFilters,
        browserItems,
        browserLoading,
        browserPagination,
        browserPairKey,
        browserReadyDatasets,
        browserSummary,
        cancelReplacingAsset,
        clearSelectedAsset,
        datasets,
        datasetsState,
        loadDatasets,
        nextSelectionTarget,
        pairPreviewState,
        replacementTarget,
        reloadBrowser: () => setBrowserReloadNonce((current) => current + 1),
        resetBrowserContinuity,
        resetBrowserFilters,
        selectBrowserItem,
        selectDataset,
        selectedAssetA,
        selectedAssetB,
        selectedDataset,
        setBrowserPage: (offset: number) => updateBrowserFilters({ offset }),
        startReplacingAsset,
        swapSelectedAssets,
        updateBrowserFilters,
    };
}
