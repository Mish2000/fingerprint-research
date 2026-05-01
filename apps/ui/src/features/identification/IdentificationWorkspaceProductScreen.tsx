import { Database, Search, Sparkles } from "lucide-react";
import InlineBanner from "../../shared/ui/InlineBanner.tsx";
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
            <section className="overflow-hidden rounded-[28px] border border-slate-200 bg-gradient-to-br from-slate-950 via-emerald-900 to-amber-700 p-6 text-white shadow-sm">
                <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr] lg:items-end">
                    <div className="space-y-4">
                        <div className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-white/80">
                            <Sparkles className="h-4 w-4" />
                            Identification Workspace
                        </div>

                        <div className="space-y-3">
                            <h2 className="text-3xl font-semibold tracking-tight">
                                Demo, browser, and operational 1:N search in one Identification workspace.
                            </h2>
                            <p className="max-w-3xl text-sm leading-7 text-white/80">
                                Demo Mode stays curated, Browser Mode adds identity-aware gallery selection plus a dataset-backed probe browser,
                                and Operational Mode keeps the manual enroll/search/delete controls intact.
                            </p>
                        </div>

                        <IdentificationModeSwitcher
                            mode={identification.identificationMode}
                            onChange={identification.setIdentificationMode}
                        />
                    </div>

                    <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
                        <div className="rounded-2xl border border-white/10 bg-white/10 p-4">
                            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/60">Demo identities</p>
                            <p className="mt-2 text-lg font-semibold">{identification.demoIdentities.length}</p>
                            <p className="mt-1 text-sm text-white/70">Server-backed gallery cards</p>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/10 p-4">
                            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/60">Browser datasets</p>
                            <p className="mt-2 text-lg font-semibold">{identification.browserReadyDatasets.length}</p>
                            <p className="mt-1 text-sm text-white/70">Catalog-ready for guided 1:N</p>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/10 p-4">
                            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/60">Demo probes</p>
                            <p className="mt-2 text-lg font-semibold">{identification.probeCases.length}</p>
                            <p className="mt-1 text-sm text-white/70">Curated 1:N stories</p>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/10 p-4">
                            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/60">Browser store status</p>
                            <p className="mt-2 text-lg font-semibold">
                                {(identification.statsState.data?.browser_seeded_count ?? 0) > 0 ? "Seeded" : "Empty"}
                            </p>
                            <p className="mt-1 text-sm text-white/70">
                                {identification.statsState.data?.browser_seeded_count ?? 0} browser-seeded identities tracked
                            </p>
                        </div>
                    </div>
                </div>
            </section>

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
                    className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
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
                        Demo and Browser add guided product workflows above the existing capabilities. Stats, enroll, manual search, and delete remain available here unchanged.
                    </InlineBanner>
                    <IdentificationOperationalWorkspace identification={identification} />
                </>
            )}

            <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
                        <Sparkles className="h-4 w-4 text-brand-600" />
                        Guided paths
                    </div>
                    <p className="mt-2 text-sm leading-6 text-slate-600">Demo gives a curated walkthrough first, while Browser lets you build a catalog-backed search context without falling back to file uploads.</p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
                        <Search className="h-4 w-4 text-brand-600" />
                        Official endpoint
                    </div>
                    <p className="mt-2 text-sm leading-6 text-slate-600">The guided flow still reaches <code>/api/identify/search</code>; there is no parallel identification engine.</p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
                        <Database className="h-4 w-4 text-brand-600" />
                        Isolated seeding
                    </div>
                    <p className="mt-2 text-sm leading-6 text-slate-600">Browser-selected galleries seed into their own resettable store so operational enrollments stay untouched.</p>
                </div>
            </div>
        </div>
    );
}
