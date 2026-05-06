import { Database, Search, Sparkles } from "lucide-react";
import InlineBanner from "../../shared/ui/InlineBanner.tsx";
import { MetricTile, WorkspaceHero } from "../../shared/ui/presentation.tsx";
import IdentificationBrowserResultPanel from "./components/IdentificationBrowserResultPanel.tsx";
import IdentificationBrowserWorkspace from "./components/IdentificationBrowserWorkspace.tsx";
import IdentificationDemoWorkspace from "./components/IdentificationDemoWorkspace.tsx";
import IdentificationModeSwitcher from "./components/IdentificationModeSwitcher.tsx";
import IdentificationResultPanel from "./components/IdentificationResultPanel.tsx";
import { useIdentification } from "./hooks/useIdentification.ts";
import IdentificationOperationalWorkspace from "./IdentificationOperationalWorkspace.tsx";

function includesAnyToken(message: string | null | undefined, tokens: string[]): boolean {
    const normalized = (message ?? "").toLowerCase();
    return tokens.some((token) => normalized.includes(token));
}

function isServiceInitializationError(message: string | null | undefined): boolean {
    return includesAnyToken(message, ["startup", "init", "ctor", "constructor", "not initialized"]);
}

export default function IdentificationWorkspaceProductScreen() {
    const identification = useIdentification();
    const showServiceInitHint = [
        identification.healthState.error,
        identification.adminLayoutState.error,
        identification.statsState.error,
        identification.demoGalleryState.error,
        identification.demoSeedState.error,
        identification.demoResetState.error,
        identification.demoResultState.error,
        identification.enrollState.error,
        identification.searchState.error,
        identification.deleteState.error,
    ].some((message) => isServiceInitializationError(message));
    const showShortlistZeroHint = identification.demoResultState.status === "success"
        && identification.demoResultState.data?.shortlist_size === 0;

    return (
        <div className="space-y-6">
            <WorkspaceHero
                eyebrow="Identification Workspace"
                title="Run guided 1:N demos, browser searches, or operational controls."
                description="Demo Mode stays curated, Browser Mode builds an isolated gallery, and Operational Mode keeps enroll, search, and delete controls available."
                icon={Sparkles}
            >
                <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(18rem,0.75fr)]">
                    <IdentificationModeSwitcher
                        mode={identification.identificationMode}
                        onChange={identification.setIdentificationMode}
                    />

                    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                        <MetricTile label="Demo identities" value={identification.demoIdentities.length} detail="Server-backed gallery" tone="brand" />
                        <MetricTile label="Browser datasets" value={identification.browserReadyDatasets.length} detail="Catalog-ready" />
                        <MetricTile label="Demo probes" value={identification.probeCases.length} detail="Curated 1:N stories" />
                        <MetricTile
                            label="Browser store"
                            value={(identification.statsState.data?.browser_seeded_count ?? 0) > 0 ? "Seeded" : "Empty"}
                            detail={`${identification.statsState.data?.browser_seeded_count ?? 0} identities tracked`}
                            tone={(identification.statsState.data?.browser_seeded_count ?? 0) > 0 ? "success" : "neutral"}
                        />
                    </div>
                </div>
            </WorkspaceHero>

            {identification.notice ? <InlineBanner variant="success">{identification.notice}</InlineBanner> : null}

            {showServiceInitHint ? (
                <InlineBanner variant="warning" title="Backend initialization issue detected">
                    One of the identification endpoints appears to have failed during startup or lazy initialization. Keep the original
                    error visible and treat this as a release-readiness blocker before retrying the flow.
                </InlineBanner>
            ) : null}

            {showShortlistZeroHint ? (
                <InlineBanner variant="warning" title="Shortlist returned zero candidates">
                    The demo request succeeded, but the backend returned an empty shortlist. This is a valid negative path and the UI keeps it readable.
                </InlineBanner>
            ) : null}

            <div className="flex justify-end">
                <button
                    type="button"
                    onClick={identification.clearPersistedWorkspaceState}
                    className="app-button app-button--secondary"
                >
                    Clear saved Identification workspace
                </button>
            </div>

            {identification.identificationMode === "demo" ? (
                <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
                    <IdentificationDemoWorkspace
                        demoGalleryState={identification.demoGalleryState}
                        statsState={identification.statsState}
                        demoSeedState={identification.demoSeedState}
                        demoResetState={identification.demoResetState}
                        demoStoreReady={identification.demoStoreReady}
                        demoSearchForm={identification.demoSearchForm}
                        probeCases={identification.probeCases}
                        recentProbeCases={identification.recentProbeCases}
                        pinnedProbeCases={identification.pinnedProbeCases}
                        selectedProbeCase={identification.selectedProbeCase}
                        selectedProbeCaseId={identification.selectedProbeCaseId}
                        pinnedProbeCaseIds={identification.pinnedProbeCaseIds}
                        onRefreshStats={identification.refreshStats}
                        onRetryGallery={identification.loadDemoGallery}
                        onSeed={identification.seedDemoStore}
                        onReset={identification.resetDemoStore}
                        onSelectProbeCase={(probeCase) => {
                            identification.selectProbeCase(probeCase, false);
                        }}
                        onTogglePinnedProbeCase={identification.togglePinnedProbeCase}
                        onUpdateDemoSearchForm={identification.updateDemoSearchForm}
                        onRun={identification.runDemoIdentification}
                        busy={identification.isDemoBusy}
                    />

                    <IdentificationResultPanel
                        resultState={identification.demoResultState}
                        lastProbeCase={identification.lastDemoRunProbeCase}
                        onRetry={identification.retryDemoRun}
                    />
                </div>
            ) : identification.identificationMode === "browser" ? (
                <div className="grid gap-6 2xl:grid-cols-[1.08fr_0.92fr]">
                    <IdentificationBrowserWorkspace
                        datasetsState={identification.browserDatasetsState}
                        datasets={identification.browserReadyDatasets}
                        selectedDataset={identification.browserSelectedDataset}
                        galleryState={identification.browserGalleryState}
                        browserFilters={identification.browserFilters}
                        browserFilterOptions={identification.browserFilterOptions}
                        browserActiveFilterCount={identification.browserActiveFilterCount}
                        browserItems={identification.browserItems}
                        browserLoading={identification.browserLoading}
                        browserError={identification.browserError}
                        browserPagination={identification.browserPagination}
                        browserSearchForm={identification.browserSearchForm}
                        selectedGalleryIdentityIds={identification.selectedBrowserGalleryIdentityIds}
                        selectedProbeAsset={identification.selectedBrowserProbeAsset}
                        browserWarnings={identification.browserWarnings}
                        browserSeedState={identification.browserSeedState}
                        browserResetState={identification.browserResetState}
                        busy={identification.isBrowserBusy}
                        onSelectDataset={identification.selectBrowserDataset}
                        onRetryDatasets={identification.loadBrowserDatasets}
                        onUpdateBrowserFilters={identification.updateBrowserFilters}
                        onResetBrowserFilters={identification.resetBrowserFilters}
                        onUpdateBrowserSearchForm={identification.updateBrowserSearchForm}
                        onToggleIdentity={identification.toggleBrowserGalleryIdentity}
                        onSelectProbeAsset={identification.selectBrowserProbeAsset}
                        onClearProbe={identification.clearSelectedBrowserProbe}
                        onRun={identification.runBrowserIdentification}
                        onResetStore={identification.resetBrowserStore}
                    />

                    <IdentificationBrowserResultPanel
                        resultState={identification.browserResultState}
                        lastProbeAsset={identification.lastBrowserRunProbeAsset}
                        datasetLabel={identification.lastBrowserRunDatasetLabel}
                        onRetry={identification.retryBrowserRun}
                    />
                </div>
            ) : (
                <>
                    <InlineBanner variant="info" title="Operational controls preserved">
                        Demo and Browser add guided workflows. Stats, enroll, manual search, and delete remain available here.
                    </InlineBanner>
                    <IdentificationOperationalWorkspace identification={identification} />
                </>
            )}

            <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <div className="flex items-center gap-2 text-sm font-semibold text-[var(--app-text)]">
                        <Sparkles className="h-4 w-4 text-[var(--app-brand-text)]" />
                        Guided paths
                    </div>
                    <p className="mt-2 text-sm leading-6 text-[var(--app-text-muted)]">Demo gives a curated walkthrough. Browser builds a catalog-backed search context without file uploads.</p>
                </div>
                <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <div className="flex items-center gap-2 text-sm font-semibold text-[var(--app-text)]">
                        <Search className="h-4 w-4 text-[var(--app-brand-text)]" />
                        Official endpoint
                    </div>
                    <p className="mt-2 text-sm leading-6 text-[var(--app-text-muted)]">Guided flows still reach <code>/api/identify/search</code>; no parallel engine is introduced.</p>
                </div>
                <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <div className="flex items-center gap-2 text-sm font-semibold text-[var(--app-text)]">
                        <Database className="h-4 w-4 text-[var(--app-brand-text)]" />
                        Isolated seeding
                    </div>
                    <p className="mt-2 text-sm leading-6 text-[var(--app-text-muted)]">Browser galleries seed into their own resettable store, keeping operational enrollments untouched.</p>
                </div>
            </div>
        </div>
    );
}
