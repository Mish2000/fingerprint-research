import RequestState from "../../../components/RequestState.tsx";
import type { CatalogBrowserItem } from "../../../types/index.ts";
import type { BrowserFilterOptions, BrowserFilters, BrowserPagination } from "../../verify/browserModel.ts";
import BrowserFiltersPanel from "../../verify/components/BrowserFilters.tsx";
import CatalogAssetImage from "../../verify/components/CatalogAssetImage.tsx";

interface IdentificationBrowserProbePanelProps {
    datasetLabel: string;
    items: CatalogBrowserItem[];
    filters: BrowserFilters;
    filterOptions: BrowserFilterOptions;
    activeFilterCount: number;
    loading: boolean;
    error: string | null;
    pagination: BrowserPagination;
    selectedProbeAsset: CatalogBrowserItem | null;
    onChangeFilters: (patch: Partial<BrowserFilters>) => void;
    onResetFilters: () => void;
    onSelectProbeAsset: (item: CatalogBrowserItem) => void;
    onClearProbe: () => void;
}

function metadataLine(label: string, value: string | null | undefined): string {
    return `${label}: ${value || "-"}`;
}

export default function IdentificationBrowserProbePanel({
    datasetLabel,
    items,
    filters,
    filterOptions,
    activeFilterCount,
    loading,
    error,
    pagination,
    selectedProbeAsset,
    onChangeFilters,
    onResetFilters,
    onSelectProbeAsset,
    onClearProbe,
}: IdentificationBrowserProbePanelProps) {
    return (
        <div className="space-y-5">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                        <p className="text-sm font-semibold text-slate-900">Selected probe</p>
                        <p className="mt-1 text-sm text-slate-600">Choose one asset from the dataset browser without opening the system file picker.</p>
                    </div>
                    {selectedProbeAsset ? (
                        <button
                            type="button"
                            onClick={onClearProbe}
                            className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                        >
                            Clear probe
                        </button>
                    ) : null}
                </div>

                {selectedProbeAsset ? (
                    <div className="mt-4 grid gap-4 sm:grid-cols-[180px_1fr]">
                        <CatalogAssetImage
                            src={selectedProbeAsset.preview_url}
                            alt={selectedProbeAsset.asset_id}
                            fallbackLabel={selectedProbeAsset.asset_id}
                            className="h-36"
                        />
                        <div className="space-y-3">
                            <div>
                                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{datasetLabel}</p>
                                <h4 className="mt-1 text-base font-semibold text-slate-900">{selectedProbeAsset.asset_id}</h4>
                            </div>
                            <div className="grid gap-2 text-xs leading-5 text-slate-600 sm:grid-cols-2">
                                <p>{metadataLine("Split", selectedProbeAsset.split)}</p>
                                <p>{metadataLine("Subject", selectedProbeAsset.subject_id)}</p>
                                <p>{metadataLine("Capture", selectedProbeAsset.capture)}</p>
                                <p>{metadataLine("Modality", selectedProbeAsset.modality)}</p>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="mt-4 rounded-2xl border border-dashed border-slate-300 bg-white px-4 py-6 text-sm text-slate-600">
                        No probe selected yet. Pick an item below to use it as the single browser probe.
                    </div>
                )}
            </div>

            <BrowserFiltersPanel
                filters={filters}
                options={filterOptions}
                activeFilterCount={activeFilterCount}
                onChange={onChangeFilters}
                onReset={onResetFilters}
            />

            {loading && items.length === 0 ? (
                <RequestState
                    variant="loading"
                    title="Loading dataset browser"
                    description="Fetching the current page of server-backed browser assets."
                />
            ) : null}

            {!loading && error && items.length === 0 ? (
                <RequestState
                    variant="error"
                    title="Failed to load browser assets"
                    description={error}
                />
            ) : null}

            {!loading && items.length === 0 && !error ? (
                <RequestState
                    variant="empty"
                    title="No browser assets match the current filters"
                    description="Reset one or more filters to see more probe candidates."
                />
            ) : null}

            {items.length > 0 ? (
                <div className="space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-slate-600">
                        <span>
                            Showing {items.length} of {pagination.total} items
                        </span>
                        <span>
                            Offset {pagination.offset}
                        </span>
                    </div>

                    <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
                        {items.map((item) => {
                            const isSelected = selectedProbeAsset?.asset_id === item.asset_id;

                            return (
                                <article
                                    key={item.asset_id}
                                    className={[
                                        "overflow-hidden rounded-2xl border bg-white shadow-sm transition",
                                        isSelected ? "border-brand-300 ring-2 ring-brand-100" : "border-slate-200 hover:border-slate-300",
                                    ].join(" ")}
                                >
                                    <CatalogAssetImage
                                        src={item.thumbnail_url}
                                        alt={item.asset_id}
                                        fallbackLabel={item.asset_id}
                                        className="aspect-[4/3] rounded-none border-0"
                                    />

                                    <div className="space-y-4 p-4">
                                        <div className="flex items-start justify-between gap-3">
                                            <div>
                                                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{datasetLabel}</p>
                                                <h4 className="mt-1 text-sm font-semibold text-slate-900">{item.asset_id}</h4>
                                            </div>
                                            {isSelected ? (
                                                <span className="rounded-full border border-brand-200 bg-brand-50 px-2.5 py-1 text-xs font-semibold text-brand-700">
                                                    Probe
                                                </span>
                                            ) : null}
                                        </div>

                                        <div className="grid gap-2 text-xs leading-5 text-slate-600">
                                            <p>{metadataLine("Split", item.split)}</p>
                                            <p>{metadataLine("Capture", item.capture)}</p>
                                            <p>{metadataLine("Subject", item.subject_id)}</p>
                                            <p>{metadataLine("Modality", item.modality)}</p>
                                        </div>

                                        <button
                                            type="button"
                                            onClick={() => onSelectProbeAsset(item)}
                                            className={[
                                                "w-full rounded-xl px-3 py-2.5 text-sm font-medium transition",
                                                isSelected
                                                    ? "border border-brand-200 bg-brand-50 text-brand-700 hover:bg-brand-100"
                                                    : "bg-slate-900 text-white hover:bg-slate-700",
                                            ].join(" ")}
                                        >
                                            {isSelected ? "Replace probe" : "Use as probe"}
                                        </button>
                                    </div>
                                </article>
                            );
                        })}
                    </div>
                </div>
            ) : null}
        </div>
    );
}
