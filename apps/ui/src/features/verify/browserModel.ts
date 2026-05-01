import type {
    CatalogBrowserItem,
    CatalogDatasetBrowserQuery,
    CatalogDatasetBrowserResponse,
    CatalogBrowserSort,
    JsonRecord,
} from "../../types/index.ts";

export type BrowserSelectionSide = "A" | "B";
export type BrowserUiEligibleFilter = "all" | "eligible" | "ineligible";

export interface BrowserFilters {
    split: string;
    capture: string;
    modality: string;
    subjectId: string;
    finger: string;
    uiEligible: BrowserUiEligibleFilter;
    limit: number;
    offset: number;
    sort: CatalogBrowserSort;
}

export interface BrowserPagination {
    total: number;
    limit: number;
    offset: number;
    hasMore: boolean;
}

export interface BrowserFilterOptions {
    splits: string[];
    captures: string[];
    modalities: string[];
}

export interface PairPreviewState {
    status: "empty" | "needs-a" | "needs-b" | "replacing" | "ready";
    message: string;
}

function sortedValues(values: Iterable<string | null | undefined>): string[] {
    return Array.from(new Set(
        Array.from(values)
            .map((value) => value?.trim() ?? "")
            .filter(Boolean),
    )).sort((left, right) => left.localeCompare(right));
}

function valuesFromSummary(summary: JsonRecord, key: string): string[] {
    const value = summary[key];
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return [];
    }

    return sortedValues(Object.keys(value));
}

export function createDefaultBrowserFilters(): BrowserFilters {
    return {
        split: "",
        capture: "",
        modality: "",
        subjectId: "",
        finger: "",
        uiEligible: "all",
        limit: 48,
        offset: 0,
        sort: "default",
    };
}

export function createEmptyBrowserPagination(limit = 48): BrowserPagination {
    return {
        total: 0,
        limit,
        offset: 0,
        hasMore: false,
    };
}

export function createBrowserPagination(response: CatalogDatasetBrowserResponse): BrowserPagination {
    return {
        total: response.total,
        limit: response.limit,
        offset: response.offset,
        hasMore: response.has_more,
    };
}

export function toCatalogBrowserQuery(dataset: string, filters: BrowserFilters): CatalogDatasetBrowserQuery {
    return {
        dataset,
        split: filters.split || undefined,
        capture: filters.capture || undefined,
        modality: filters.modality || undefined,
        subject_id: filters.subjectId || undefined,
        finger: filters.finger || undefined,
        ui_eligible:
            filters.uiEligible === "eligible"
                ? true
                : filters.uiEligible === "ineligible"
                    ? false
                    : undefined,
        limit: filters.limit,
        offset: filters.offset,
        sort: filters.sort,
    };
}

export function countActiveBrowserFilters(filters: BrowserFilters): number {
    return [
        filters.split,
        filters.capture,
        filters.modality,
        filters.subjectId,
        filters.finger,
        filters.uiEligible !== "all" ? filters.uiEligible : "",
    ].filter(Boolean).length;
}

export function createBrowserFilterOptions(
    summary: JsonRecord,
    items: CatalogBrowserItem[],
    selectedAssetA: CatalogBrowserItem | null,
    selectedAssetB: CatalogBrowserItem | null,
): BrowserFilterOptions {
    const selectedItems = [selectedAssetA, selectedAssetB].filter((item): item is CatalogBrowserItem => item !== null);
    const allItems = [...items, ...selectedItems];

    return {
        splits: valuesFromSummary(summary, "items_by_split").length > 0
            ? valuesFromSummary(summary, "items_by_split")
            : sortedValues(allItems.map((item) => item.split)),
        captures: valuesFromSummary(summary, "items_by_capture").length > 0
            ? valuesFromSummary(summary, "items_by_capture")
            : sortedValues(allItems.map((item) => item.capture)),
        modalities: valuesFromSummary(summary, "items_by_modality").length > 0
            ? valuesFromSummary(summary, "items_by_modality")
            : sortedValues(allItems.map((item) => item.modality)),
    };
}

export function inferNextSelectionTarget(
    selectedAssetA: CatalogBrowserItem | null,
    selectedAssetB: CatalogBrowserItem | null,
    replacementTarget: BrowserSelectionSide | null = null,
): BrowserSelectionSide | null {
    if (replacementTarget) {
        return replacementTarget;
    }

    if (!selectedAssetA) {
        return "A";
    }

    if (!selectedAssetB) {
        return "B";
    }

    return null;
}

export function createPairPreviewState(
    selectedAssetA: CatalogBrowserItem | null,
    selectedAssetB: CatalogBrowserItem | null,
    replacementTarget: BrowserSelectionSide | null = null,
): PairPreviewState {
    if (replacementTarget) {
        return {
            status: "replacing",
            message: `Choose a replacement for side ${replacementTarget}.`,
        };
    }

    if (selectedAssetA && selectedAssetB) {
        return {
            status: "ready",
            message: "Pair ready. Use it as the verify pair when ready.",
        };
    }

    if (selectedAssetA) {
        return {
            status: "needs-b",
            message: "Side A selected. Choose side B next.",
        };
    }

    if (selectedAssetB) {
        return {
            status: "needs-a",
            message: "Side B selected. Choose side A next.",
        };
    }

    return {
        status: "empty",
        message: "Choose side A to begin the pair.",
    };
}

export function createBrowserPairKey(
    selectedAssetA: CatalogBrowserItem | null,
    selectedAssetB: CatalogBrowserItem | null,
): string | null {
    if (!selectedAssetA || !selectedAssetB) {
        return null;
    }

    return `${selectedAssetA.asset_id}::${selectedAssetB.asset_id}`;
}
