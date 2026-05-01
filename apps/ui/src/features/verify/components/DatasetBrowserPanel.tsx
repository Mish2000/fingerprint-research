import type { CatalogBrowserItem, CatalogDatasetSummary, JsonRecord } from "../../../types/index.ts";
import SurfaceCard from "../../../shared/ui/SurfaceCard.tsx";
import type { AsyncState } from "../../../shared/request-state/index.ts";
import type {
    BrowserFilterOptions,
    BrowserFilters as BrowserFiltersState,
    BrowserPagination,
    BrowserSelectionSide,
} from "../browserModel.ts";
import BrowserFilters from "./BrowserFilters.tsx";
import BrowserGrid from "./BrowserGrid.tsx";
import DatasetPicker from "./DatasetPicker.tsx";

interface DatasetBrowserPanelProps {
    datasets: CatalogDatasetSummary[];
    datasetsState: AsyncState<CatalogDatasetSummary[]>;
    selectedDataset: CatalogDatasetSummary | null;
    browserItems: CatalogBrowserItem[];
    browserLoading: boolean;
    browserError: string | null;
    browserFilters: BrowserFiltersState;
    browserFilterOptions: BrowserFilterOptions;
    browserPagination: BrowserPagination;
    browserSummary: JsonRecord;
    activeFilterCount: number;
    nextTarget: BrowserSelectionSide | null;
    replacementTarget: BrowserSelectionSide | null;
    selectedAssetA: CatalogBrowserItem | null;
    selectedAssetB: CatalogBrowserItem | null;
    onSelectDataset: (dataset: CatalogDatasetSummary) => void;
    onRetryDatasets: () => void;
    onUpdateFilters: (patch: Partial<BrowserFiltersState>) => void;
    onResetFilters: () => void;
    onSelectItem: (item: CatalogBrowserItem) => void;
    onRetryBrowser: () => void;
    onPreviousPage: () => void;
    onNextPage: () => void;
}

function formatRange(pagination: BrowserPagination): string {
    if (pagination.total === 0) {
        return "0 items";
    }

    const start = pagination.offset + 1;
    const end = Math.min(pagination.offset + pagination.limit, pagination.total);
    return `${start}-${end} of ${pagination.total}`;
}

export default function DatasetBrowserPanel({
    datasets,
    datasetsState,
    selectedDataset,
    browserItems,
    browserLoading,
    browserError,
    browserFilters,
    browserFilterOptions,
    browserPagination,
    browserSummary,
    activeFilterCount,
    nextTarget,
    replacementTarget,
    selectedAssetA,
    selectedAssetB,
    onSelectDataset,
    onRetryDatasets,
    onUpdateFilters,
    onResetFilters,
    onSelectItem,
    onRetryBrowser,
    onPreviousPage,
    onNextPage,
}: DatasetBrowserPanelProps) {
    return (
        <div className="space-y-6">
            <SurfaceCard
                title="Dataset Picker"
                description="Choose a browser-ready dataset from /api/catalog/datasets. Datasets without browser assets stay visible but disabled."
            >
                <DatasetPicker
                    datasets={datasets}
                    selectedDataset={selectedDataset}
                    loading={datasetsState.status === "loading"}
                    error={datasetsState.error}
                    onSelectDataset={onSelectDataset}
                    onRetry={onRetryDatasets}
                />
            </SurfaceCard>

            <SurfaceCard
                title="Dataset Browser"
                description="Browse paginated real items, preview thumbnails, and assign them to side A or B."
            >
                <div className="space-y-5">
                    <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-700">
                        <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 font-medium text-slate-700">
                            {selectedDataset?.dataset_label ?? "Choose a dataset"}
                        </span>
                        <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 font-medium text-slate-700">
                            Showing {formatRange(browserPagination)}
                        </span>
                        <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 font-medium text-slate-700">
                            Active filters {activeFilterCount}
                        </span>
                        {typeof browserSummary.items_generated === "number" ? (
                            <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 font-medium text-slate-700">
                                Generated {browserSummary.items_generated}
                            </span>
                        ) : null}
                    </div>

                    <BrowserFilters
                        filters={browserFilters}
                        options={browserFilterOptions}
                        activeFilterCount={activeFilterCount}
                        disabled={!selectedDataset}
                        onChange={onUpdateFilters}
                        onReset={onResetFilters}
                    />

                    <BrowserGrid
                        items={browserItems}
                        datasetLabel={selectedDataset?.dataset_label ?? "Dataset"}
                        loading={browserLoading}
                        error={browserError}
                        nextTarget={nextTarget}
                        replacementTarget={replacementTarget}
                        selectedAssetA={selectedAssetA}
                        selectedAssetB={selectedAssetB}
                        onSelectItem={onSelectItem}
                        onRetry={onRetryBrowser}
                    />

                    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-1">
                        <p className="text-sm text-slate-600">
                            Pagination stays server-backed through <code>/api/catalog/dataset-browser</code>.
                        </p>

                        <div className="flex flex-wrap gap-2">
                            <button
                                type="button"
                                onClick={onPreviousPage}
                                disabled={browserLoading || browserPagination.offset === 0}
                                className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                                Previous page
                            </button>

                            <button
                                type="button"
                                onClick={onNextPage}
                                disabled={browserLoading || !browserPagination.hasMore}
                                className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                                Next page
                            </button>
                        </div>
                    </div>
                </div>
            </SurfaceCard>
        </div>
    );
}
