import RequestState from "../../../components/RequestState.tsx";
import type { CatalogBrowserItem } from "../../../types/index.ts";
import type { BrowserSelectionSide } from "../browserModel.ts";
import BrowserItemCard from "./BrowserItemCard.tsx";

interface BrowserGridProps {
    items: CatalogBrowserItem[];
    datasetLabel: string;
    loading: boolean;
    error: string | null;
    nextTarget: BrowserSelectionSide | null;
    replacementTarget: BrowserSelectionSide | null;
    selectedAssetA: CatalogBrowserItem | null;
    selectedAssetB: CatalogBrowserItem | null;
    onSelectItem: (item: CatalogBrowserItem) => void;
    onRetry: () => void;
}

function sideForItem(
    item: CatalogBrowserItem,
    selectedAssetA: CatalogBrowserItem | null,
    selectedAssetB: CatalogBrowserItem | null,
): BrowserSelectionSide | null {
    if (selectedAssetA?.asset_id === item.asset_id) {
        return "A";
    }
    if (selectedAssetB?.asset_id === item.asset_id) {
        return "B";
    }
    return null;
}

export default function BrowserGrid({
    items,
    datasetLabel,
    loading,
    error,
    nextTarget,
    replacementTarget,
    selectedAssetA,
    selectedAssetB,
    onSelectItem,
    onRetry,
}: BrowserGridProps) {
    if (loading && items.length === 0) {
        return (
            <RequestState
                variant="loading"
                title="Loading browser items"
                description="Fetching the current dataset page from /api/catalog/dataset-browser."
            />
        );
    }

    if (error && items.length === 0) {
        return (
            <RequestState
                variant="error"
                title="Failed to load dataset items"
                description={error}
                actionLabel="Retry"
                onAction={onRetry}
            />
        );
    }

    if (items.length === 0) {
        return (
            <RequestState
                variant="empty"
                title="No items match the current filters"
                description="Reset one or more filters to see more dataset-backed items."
            />
        );
    }

    return (
        <div className="space-y-4">
            {loading ? (
                <div className="rounded-2xl border border-brand-100 bg-brand-50 px-4 py-3 text-sm text-brand-900">
                    Refreshing the current page from the catalog browser...
                </div>
            ) : null}

            {error ? (
                <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                    {error}
                </div>
            ) : null}

            <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
                {items.map((item) => (
                    <BrowserItemCard
                        key={item.asset_id}
                        item={item}
                        datasetLabel={datasetLabel}
                        selectedSide={sideForItem(item, selectedAssetA, selectedAssetB)}
                        nextTarget={nextTarget}
                        replacementTarget={replacementTarget}
                        onSelect={onSelectItem}
                    />
                ))}
            </div>
        </div>
    );
}
