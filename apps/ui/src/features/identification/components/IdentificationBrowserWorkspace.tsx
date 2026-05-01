import SurfaceCard from "../../../shared/ui/SurfaceCard.tsx";
import type { AsyncState } from "../../../shared/request-state";
import type {
    CatalogBrowserItem,
    CatalogDatasetSummary,
    CatalogIdentifyGalleryResponse,
    IdentifyBrowserResetResponse,
    IdentifyBrowserSeedSelectionResponse,
} from "../../../types/index.ts";
import type { BrowserFilterOptions, BrowserFilters, BrowserPagination } from "../../verify/browserModel.ts";
import DatasetPicker from "../../verify/components/DatasetPicker.tsx";
import type { BrowserSearchFormState } from "../hooks/useIdentification.ts";
import IdentificationBrowserGalleryPanel from "./IdentificationBrowserGalleryPanel.tsx";
import IdentificationBrowserProbePanel from "./IdentificationBrowserProbePanel.tsx";
import IdentificationBrowserSummaryPanel from "./IdentificationBrowserSummaryPanel.tsx";

interface IdentificationBrowserWorkspaceProps {
    datasetsState: AsyncState<CatalogDatasetSummary[]>;
    datasets: CatalogDatasetSummary[];
    selectedDataset: CatalogDatasetSummary | null;
    galleryState: AsyncState<CatalogIdentifyGalleryResponse>;
    browserFilters: BrowserFilters;
    browserFilterOptions: BrowserFilterOptions;
    browserActiveFilterCount: number;
    browserItems: CatalogBrowserItem[];
    browserLoading: boolean;
    browserError: string | null;
    browserPagination: BrowserPagination;
    browserSearchForm: BrowserSearchFormState;
    selectedGalleryIdentityIds: string[];
    selectedProbeAsset: CatalogBrowserItem | null;
    browserWarnings: string[];
    browserSeedState: AsyncState<IdentifyBrowserSeedSelectionResponse>;
    browserResetState: AsyncState<IdentifyBrowserResetResponse>;
    busy: boolean;
    onSelectDataset: (dataset: CatalogDatasetSummary) => void;
    onRetryDatasets: () => void | Promise<void>;
    onUpdateBrowserFilters: (patch: Partial<BrowserFilters>) => void;
    onResetBrowserFilters: () => void;
    onUpdateBrowserSearchForm: (patch: Partial<BrowserSearchFormState>) => void;
    onToggleIdentity: (identity: CatalogIdentifyGalleryResponse["items"][number]) => void;
    onSelectProbeAsset: (item: CatalogBrowserItem) => void;
    onClearProbe: () => void;
    onRun: () => void | Promise<void>;
    onResetStore: () => void | Promise<void>;
}

export default function IdentificationBrowserWorkspace({
    datasetsState,
    datasets,
    selectedDataset,
    galleryState,
    browserFilters,
    browserFilterOptions,
    browserActiveFilterCount,
    browserItems,
    browserLoading,
    browserError,
    browserPagination,
    browserSearchForm,
    selectedGalleryIdentityIds,
    selectedProbeAsset,
    browserWarnings,
    browserSeedState,
    busy,
    onSelectDataset,
    onRetryDatasets,
    onUpdateBrowserFilters,
    onResetBrowserFilters,
    onUpdateBrowserSearchForm,
    onToggleIdentity,
    onSelectProbeAsset,
    onClearProbe,
    onRun,
    onResetStore,
}: IdentificationBrowserWorkspaceProps) {
    const identities = galleryState.data?.items ?? [];

    return (
        <div className="space-y-6">
            <SurfaceCard
                title="Browser workspace"
                description="Pick one dataset, choose the gallery identities that should be enrolled into an isolated browser store, then choose a single probe asset from the dataset browser."
                className="min-w-0"
            >
                <DatasetPicker
                    datasets={datasets}
                    selectedDataset={selectedDataset}
                    loading={datasetsState.status === "loading"}
                    error={datasetsState.error}
                    onSelectDataset={onSelectDataset}
                    onRetry={onRetryDatasets}
                    readyDescription="Choose this dataset to browse probe assets and build a catalog-backed 1:N search context for Identification."
                    unavailableDescription="Browser mode requires both browser assets and identify-gallery metadata for this dataset."
                    emptyDescription="The catalog does not currently expose any datasets with both identify-gallery metadata and browser assets."
                />
            </SurfaceCard>

            <div className="grid gap-6">
                <SurfaceCard
                    title="Gallery selection"
                    description="Gallery cards stay identity-aware and come directly from the identify-gallery catalog semantics."
                    className="min-w-0"
                >
                    <IdentificationBrowserGalleryPanel
                        identities={identities}
                        selectedIdentityIds={selectedGalleryIdentityIds}
                        onToggleIdentity={onToggleIdentity}
                    />
                </SurfaceCard>

                <SurfaceCard
                    title="Probe browser"
                    description="Reuse the dataset browser infrastructure from Verify without any A/B pair-builder semantics."
                    className="min-w-0"
                >
                    <IdentificationBrowserProbePanel
                        datasetLabel={selectedDataset?.dataset_label ?? "Dataset"}
                        items={browserItems}
                        filters={browserFilters}
                        filterOptions={browserFilterOptions}
                        activeFilterCount={browserActiveFilterCount}
                        loading={browserLoading}
                        error={browserError}
                        pagination={browserPagination}
                        selectedProbeAsset={selectedProbeAsset}
                        onChangeFilters={onUpdateBrowserFilters}
                        onResetFilters={onResetBrowserFilters}
                        onSelectProbeAsset={onSelectProbeAsset}
                        onClearProbe={onClearProbe}
                    />
                </SurfaceCard>
            </div>

            <SurfaceCard
                title="Run browser identification"
                description="The browser flow seeds the selected gallery into its isolated store and then runs the official identification endpoint against that seeded context."
                className="min-w-0"
            >
                <IdentificationBrowserSummaryPanel
                    dataset={selectedDataset}
                    selectedIdentities={identities.filter((identity) => selectedGalleryIdentityIds.includes(identity.identity_id))}
                    selectedProbeAsset={selectedProbeAsset}
                    browserSearchForm={browserSearchForm}
                    warnings={browserWarnings}
                    busy={busy}
                    browserSeedNotice={browserSeedState.data?.notice ?? null}
                    onUpdate={onUpdateBrowserSearchForm}
                    onRun={onRun}
                    onResetStore={onResetStore}
                />
            </SurfaceCard>
        </div>
    );
}
