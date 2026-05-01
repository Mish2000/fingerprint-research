import { ChevronRight, Database, LoaderCircle, Play, SlidersHorizontal, Sparkles, Thermometer, Upload } from "lucide-react";
import FileDropBox from "../../components/FileDropBox.tsx";
import { MatchCanvas } from "../../components/MatchCanvas.tsx";
import RequestState from "../../components/RequestState.tsx";
import { ResultSummary } from "../../components/ResultSummary.tsx";
import VerifyDemoCasesPanel from "../../components/VerifyDemoCasesPanel.tsx";
import InlineBanner from "../../shared/ui/InlineBanner.tsx";
import SurfaceCard from "../../shared/ui/SurfaceCard.tsx";
import FormField from "../../shared/ui/FormField.tsx";
import { CHECKBOX_CLASS_NAME, INPUT_CLASS_NAME } from "../../shared/ui/inputClasses.ts";
import { formatMethodLabel } from "../../shared/storytelling.ts";
import { CAPTURE_OPTIONS, METHOD_PROFILES } from "./config.ts";
import DatasetBrowserPanel from "./components/DatasetBrowserPanel.tsx";
import PairBuilderPanel from "./components/PairBuilderPanel.tsx";
import VerifyOutcomeStoryPanel from "./components/VerifyOutcomeStoryPanel.tsx";
import { useVerifyWorkspace } from "./hooks/useVerifyWorkspace.ts";
import { VERIFY_DEMO_FILTERS, formatGroundTruthLabel } from "./model.ts";

function isServiceInitializationError(message: string | null | undefined): boolean {
    const normalized = (message ?? "").toLowerCase();

    return (
        normalized.includes("startup")
        || normalized.includes("init")
        || normalized.includes("ctor")
        || normalized.includes("constructor")
        || normalized.includes("not initialized")
    );
}

function isMissingDemoAssetError(message: string | null | undefined): boolean {
    const normalized = (message ?? "").toLowerCase();
    return normalized.includes("demo asset") || normalized.includes("404") || normalized.includes("not found");
}

function stageLabel(
    stage: ReturnType<typeof useVerifyWorkspace>["stage"],
    activeMode: ReturnType<typeof useVerifyWorkspace>["activeMode"],
): string {
    if (stage === "loading-demo") {
        return "Loading demo...";
    }
    if (stage === "warming") {
        return "Warming matcher...";
    }
    if (stage === "matching") {
        return activeMode === "demo" ? "Running demo..." : "Running verify...";
    }
    return activeMode === "demo" ? "Run Selected Case" : "Run Verification";
}

function demoCatalogHealthTitle(status: "healthy" | "degraded" | "incomplete"): string {
    return status === "incomplete"
        ? "Curated demo catalog is incomplete"
        : "Curated demo evidence is degraded";
}

export default function VerifyWorkspaceProductScreen() {
    const verify = useVerifyWorkspace();
    const browser = verify.browser;
    const overlayMatches = verify.currentResult?.overlay?.matches ?? [];
    const showCanvas = Boolean(verify.manualFiles.probeFile && verify.manualFiles.referenceFile && overlayMatches.length > 0);
    const selectedDemoCase = verify.selectedDemoCase;

    const verifyError = verify.resultState.error;
    const demoCasesError = verify.demoCasesState.error;
    const browserDatasetsError = browser.datasetsState.error;
    const browserItemsError = browser.browserError;
    const showServiceInitHint = isServiceInitializationError(verifyError)
        || isServiceInitializationError(demoCasesError)
        || isServiceInitializationError(browserDatasetsError)
        || isServiceInitializationError(browserItemsError);
    const showDemoAssetHint = verify.resultState.status === "error" && isMissingDemoAssetError(verifyError);
    const usingMethodOverride = Boolean(
        verify.lastRunContext?.mode === "demo"
        && verify.lastRunContext.recommendedMethod
        && verify.lastRunContext.method !== verify.lastRunContext.recommendedMethod,
    );
    const isSelectedDemoCasePinned = Boolean(
        selectedDemoCase && verify.pinnedDemoCaseIds.includes(selectedDemoCase.case_id),
    );
    const showManualReuploadHint = verify.activeMode === "manual"
        && Boolean(verify.manualPairReminder)
        && !verify.form.probeFile
        && !verify.form.referenceFile;
    const demoCatalogBuildHealth = verify.demoCatalogBuildHealth;
    const showDemoCatalogHealthBanner = Boolean(
        demoCatalogBuildHealth
        && demoCatalogBuildHealth.catalog_build_status !== "healthy",
    );

    const canRunPrimaryAction = verify.activeMode === "demo"
        ? Boolean(selectedDemoCase) && !verify.isBusy
        : verify.activeMode === "browser"
            ? Boolean(
                browser.selectedAssetA
                && browser.selectedAssetB
                && verify.isCurrentBrowserPairApplied
                && verify.form.probeFile
                && verify.form.referenceFile,
            ) && !verify.isBusy
            : Boolean(verify.form.probeFile && verify.form.referenceFile) && !verify.isBusy;

    const runPrimaryAction = (): void => {
        if (verify.activeMode === "demo") {
            void verify.runSelectedDemoCase();
            return;
        }

        if (verify.activeMode === "browser") {
            void verify.runBrowserPair();
            return;
        }

        void verify.runMatch();
    };

    const loadingDescription = verify.lastRunContext?.mode === "demo"
        ? `Loading server-backed assets for "${verify.lastRunContext.title}" and waiting for the match result.`
        : verify.lastRunContext?.mode === "browser"
            ? "Submitting the applied dataset-browser pair through /api/match."
            : "Uploading the two selected files and waiting for the backend MatchResponse.";

    const emptyResultDescription = verify.activeMode === "demo"
        ? "Choose a curated case and run it to see the decision, score, threshold, and latency in one place."
        : verify.activeMode === "browser"
            ? "Build a pair from the dataset browser, use it as the verify pair, then run verify from the same workspace."
            : "Upload two files, choose a method, and run verify to see the structured result here.";

    return (
        <div className="space-y-6">
            <section className="overflow-hidden rounded-[28px] border border-slate-200 bg-gradient-to-br from-slate-950 via-slate-900 to-brand-700 p-6 text-white shadow-sm">
                <div className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr] lg:items-end">
                    <div className="space-y-4">
                        <div className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-white/80">
                            <Sparkles className="h-4 w-4" />
                            Verify Workspace
                        </div>

                        <div className="space-y-3">
                            <h2 className="text-3xl font-semibold tracking-tight">
                                Run curated demos, browse real datasets, or upload your own pair.
                            </h2>
                            <p className="max-w-3xl text-sm leading-7 text-white/80">
                                Demo Mode stays the default, Manual Upload stays separate, and Dataset Browser / Pair Builder adds a
                                third path that turns real catalog-backed assets into a verify-ready pair without touching File Explorer.
                            </p>
                        </div>

                        <div className="flex flex-wrap gap-3">
                            {[
                                { value: "demo" as const, label: "Demo Mode", description: "Curated one-click verify", icon: Sparkles },
                                { value: "browser" as const, label: "Dataset Browser", description: "Build a pair from real data", icon: Database },
                                { value: "manual" as const, label: "Manual Upload", description: "Bring your own two files", icon: Upload },
                            ].map((mode) => {
                                const Icon = mode.icon;
                                const isActive = verify.activeMode === mode.value;

                                return (
                                    <button
                                        key={mode.value}
                                        type="button"
                                        onClick={() => verify.setActiveMode(mode.value)}
                                        className={[
                                            "min-w-52 rounded-2xl border px-4 py-3 text-left transition",
                                            isActive
                                                ? "border-white/30 bg-white/15 shadow-sm"
                                                : "border-white/10 bg-black/10 hover:border-white/20 hover:bg-white/10",
                                        ].join(" ")}
                                        aria-pressed={isActive}
                                    >
                                        <div className="flex items-start gap-3">
                                            <div className="rounded-xl bg-white/10 p-2 text-white">
                                                <Icon className="h-4 w-4" />
                                            </div>
                                            <div>
                                                <div className="font-semibold">{mode.label}</div>
                                                <div className="mt-1 text-sm text-white/70">{mode.description}</div>
                                            </div>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
                        <div className="rounded-2xl border border-white/10 bg-white/10 p-4">
                            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/60">Default Path</p>
                            <p className="mt-2 text-lg font-semibold">Demo Mode</p>
                            <p className="mt-1 text-sm text-white/70">Curated first-run experience</p>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/10 p-4">
                            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/60">Browser-Ready Datasets</p>
                            <p className="mt-2 text-lg font-semibold">{browser.browserReadyDatasets.length}</p>
                            <p className="mt-1 text-sm text-white/70">Loaded from catalog datasets</p>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/10 p-4">
                            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/60">Latest Context</p>
                            <p className="mt-2 text-lg font-semibold">{verify.lastRunContext?.title ?? "No run yet"}</p>
                            <p className="mt-1 text-sm text-white/70">
                                {verify.lastRunContext ? formatMethodLabel(verify.lastRunContext.method) : "Waiting for first execution"}
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {verify.notice ? <InlineBanner variant="success">{verify.notice}</InlineBanner> : null}

            {showServiceInitHint ? (
                <InlineBanner variant="warning" title="Backend initialization issue detected">
                    The backend appears to have failed during service startup or lazy initialization. Keep the original error visible,
                    then verify the server before retrying the flow.
                </InlineBanner>
            ) : null}

            {showDemoAssetHint ? (
                <InlineBanner variant="warning" title="Curated demo asset is unavailable">
                    One of the files for the selected case could not be downloaded from the server. The workspace keeps the rest of the
                    UI usable and lets you retry the same case explicitly.
                </InlineBanner>
            ) : null}

            <div className="flex justify-end">
                <button
                    type="button"
                    onClick={verify.clearPersistedWorkspaceState}
                    className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                >
                    Clear saved Verify workspace
                </button>
            </div>

            <div className="grid gap-6 xl:grid-cols-[1.12fr_0.88fr]">
                <div className="space-y-6">
                    {verify.activeMode === "demo" ? (
                        <SurfaceCard
                            title="Demo Mode"
                            description="Start with a ready-made verify case. Metadata comes from the catalog layer and the files are pulled from the server when you run."
                        >
                            <div className="space-y-5">
                                {showDemoCatalogHealthBanner && demoCatalogBuildHealth ? (
                                    <InlineBanner
                                        variant="warning"
                                        title={demoCatalogHealthTitle(demoCatalogBuildHealth.catalog_build_status)}
                                    >
                                        {demoCatalogBuildHealth.summary_message}
                                    </InlineBanner>
                                ) : null}

                                <div className="rounded-2xl border border-brand-100 bg-brand-50 px-4 py-4 text-sm leading-6 text-brand-900">
                                    Pick a case to inspect its metadata, keep the recommended method or override it, then run the same
                                    workspace flow you would use for manual verification.
                                </div>

                                {selectedDemoCase ? (
                                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                                        <div className="flex flex-wrap items-start justify-between gap-4">
                                            <div>
                                                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Selected Case</p>
                                                <h3 className="mt-2 text-xl font-semibold text-slate-900">{selectedDemoCase.title}</h3>
                                                <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{selectedDemoCase.description}</p>
                                            </div>

                                            <div className="flex flex-wrap gap-2">
                                                <button
                                                    type="button"
                                                    onClick={() => {
                                                        verify.togglePinnedDemoCase(selectedDemoCase);
                                                    }}
                                                    className={[
                                                        "rounded-xl border px-4 py-2 text-sm font-medium transition",
                                                        isSelectedDemoCasePinned
                                                            ? "border-amber-200 bg-amber-50 text-amber-800"
                                                            : "border-slate-200 bg-white text-slate-700 hover:bg-slate-100",
                                                    ].join(" ")}
                                                >
                                                    {isSelectedDemoCasePinned ? "Unpin case" : "Pin case"}
                                                </button>

                                                <button
                                                    type="button"
                                                    onClick={() => {
                                                        void verify.runSelectedDemoCase();
                                                    }}
                                                    disabled={verify.isBusy}
                                                    className="inline-flex items-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-brand-300"
                                                >
                                                    {verify.isBusy && verify.runningDemoCaseId === selectedDemoCase.case_id ? (
                                                        <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                                                    ) : (
                                                        <Play className="mr-2 h-4 w-4" />
                                                    )}
                                                    Run Selected Case
                                                </button>
                                            </div>
                                        </div>

                                        <div className="mt-4 flex flex-wrap gap-2 text-xs font-medium text-slate-600">
                                            <span className="rounded-full border border-slate-200 bg-white px-3 py-1">{selectedDemoCase.dataset_label}</span>
                                            <span className="rounded-full border border-slate-200 bg-white px-3 py-1">{selectedDemoCase.split}</span>
                                            <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                                {formatGroundTruthLabel(selectedDemoCase.ground_truth)}
                                            </span>
                                            <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                                Recommended {formatMethodLabel(selectedDemoCase.recommended_method)}
                                            </span>
                                            {selectedDemoCase.evidence_quality ? (
                                                <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                                    {selectedDemoCase.evidence_quality.evidence_status === "strong"
                                                        ? "Strong evidence"
                                                        : selectedDemoCase.evidence_quality.evidence_status === "fallback"
                                                            ? "Fallback evidence"
                                                            : "Degraded evidence"}
                                                </span>
                                            ) : null}
                                        </div>

                                        {selectedDemoCase.evidence_quality ? (
                                            <p className="mt-3 text-sm leading-6 text-slate-600">
                                                {selectedDemoCase.evidence_quality.evidence_note}
                                            </p>
                                        ) : null}
                                    </div>
                                ) : null}

                                {verify.pinnedDemoCases.length > 0 ? (
                                    <div className="rounded-2xl border border-amber-200 bg-amber-50/60 p-4">
                                        <div className="flex flex-wrap items-center justify-between gap-3">
                                            <div>
                                                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-amber-700">Pinned demo cases</p>
                                                <p className="mt-1 text-sm text-amber-900">
                                                    Keep a small verify playlist ready across reloads without restoring old results.
                                                </p>
                                            </div>
                                            <span className="rounded-full border border-amber-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-amber-700">
                                                {verify.pinnedDemoCases.length} pinned
                                            </span>
                                        </div>

                                        <div className="mt-4 grid gap-3 md:grid-cols-2">
                                            {verify.pinnedDemoCases.map((demoCase) => (
                                                <div key={demoCase.case_id} className="rounded-2xl border border-amber-200 bg-white p-4">
                                                    <p className="text-sm font-semibold text-slate-900">{demoCase.title}</p>
                                                    <p className="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">
                                                        {demoCase.dataset_label} / {demoCase.split}
                                                    </p>

                                                    <div className="mt-3 flex flex-wrap gap-2">
                                                        <button
                                                            type="button"
                                                            onClick={() => verify.selectDemoCase(demoCase, true)}
                                                            className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                                                        >
                                                            Select
                                                        </button>
                                                        <button
                                                            type="button"
                                                            onClick={() => {
                                                                verify.togglePinnedDemoCase(demoCase);
                                                            }}
                                                            className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm font-medium text-amber-800 transition hover:bg-amber-100"
                                                        >
                                                            Unpin
                                                        </button>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ) : null}

                                <div className="flex flex-wrap gap-2">
                                    {VERIFY_DEMO_FILTERS.map((filter) => {
                                        const isActive = verify.demoFilter === filter.value;

                                        return (
                                            <button
                                                key={filter.value}
                                                type="button"
                                                onClick={() => verify.setDemoFilter(filter.value)}
                                                className={[
                                                    "rounded-full border px-3 py-1.5 text-sm font-medium transition",
                                                    isActive
                                                        ? "border-brand-200 bg-brand-50 text-brand-700"
                                                        : "border-slate-200 bg-white text-slate-600 hover:border-slate-300",
                                                ].join(" ")}
                                                aria-pressed={isActive}
                                            >
                                                {filter.label}
                                            </button>
                                        );
                                    })}
                                </div>

                                {verify.demoCasesState.status === "error" && verify.demoCasesState.error ? (
                                    <RequestState
                                        variant="error"
                                        title="Failed to load curated verify cases"
                                        description={verify.demoCasesState.error}
                                        actionLabel="Retry"
                                        onAction={() => {
                                            void verify.retryLoadDemoCases();
                                        }}
                                    />
                                ) : (
                                    <VerifyDemoCasesPanel
                                        cases={verify.filteredDemoCases}
                                        loading={verify.demoCasesState.status === "loading"}
                                        busy={verify.isBusy}
                                        selectedCaseId={verify.selectedDemoCaseId}
                                        runningCaseId={verify.runningDemoCaseId}
                                        onSelectDemo={verify.selectDemoCase}
                                        onRunDemo={(demoCase) => {
                                            void verify.runDemoCase(demoCase, verify.selectedDemoCaseId === demoCase.case_id);
                                        }}
                                    />
                                )}
                            </div>
                        </SurfaceCard>
                    ) : null}

                    {verify.activeMode === "browser" ? (
                        <>
                            <DatasetBrowserPanel
                                datasets={browser.datasets}
                                datasetsState={browser.datasetsState}
                                selectedDataset={browser.selectedDataset}
                                browserItems={browser.browserItems}
                                browserLoading={browser.browserLoading}
                                browserError={browser.browserError}
                                browserFilters={browser.browserFilters}
                                browserFilterOptions={browser.browserFilterOptions}
                                browserPagination={browser.browserPagination}
                                browserSummary={browser.browserSummary}
                                activeFilterCount={browser.activeFilterCount}
                                nextTarget={browser.nextSelectionTarget}
                                replacementTarget={browser.replacementTarget}
                                selectedAssetA={browser.selectedAssetA}
                                selectedAssetB={browser.selectedAssetB}
                                onSelectDataset={browser.selectDataset}
                                onRetryDatasets={() => {
                                    void browser.loadDatasets();
                                }}
                                onUpdateFilters={browser.updateBrowserFilters}
                                onResetFilters={browser.resetBrowserFilters}
                                onSelectItem={browser.selectBrowserItem}
                                onRetryBrowser={browser.reloadBrowser}
                                onPreviousPage={() => {
                                    browser.setBrowserPage(Math.max(0, browser.browserPagination.offset - browser.browserPagination.limit));
                                }}
                                onNextPage={() => {
                                    browser.setBrowserPage(browser.browserPagination.offset + browser.browserPagination.limit);
                                }}
                            />

                            <SurfaceCard
                                title="Pair Builder"
                                description="Choose two real dataset items, preview both sides, and load the pair into Verify without manual upload."
                            >
                                <PairBuilderPanel
                                    datasetLabel={browser.selectedDataset?.dataset_label ?? null}
                                    selectedAssetA={browser.selectedAssetA}
                                    selectedAssetB={browser.selectedAssetB}
                                    pairPreviewState={browser.pairPreviewState}
                                    replacementTarget={browser.replacementTarget}
                                    applyPairState={verify.applyPairState}
                                    isCurrentPairApplied={verify.isCurrentBrowserPairApplied}
                                    onClearAsset={browser.clearSelectedAsset}
                                    onStartReplacing={browser.startReplacingAsset}
                                    onCancelReplacing={browser.cancelReplacingAsset}
                                    onSwap={browser.swapSelectedAssets}
                                    onApply={() => {
                                        void verify.applyBrowserPairToVerify();
                                    }}
                                />
                            </SurfaceCard>
                        </>
                    ) : null}

                    {verify.activeMode === "manual" ? (
                        <SurfaceCard
                            title="Manual Upload"
                            description="Bring your own two files, keep control over capture metadata and method selection, and run the same verify endpoint."
                        >
                            <div className="space-y-5">
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 text-sm leading-6 text-slate-700">
                                    Manual Upload stays intentionally separate from Demo Mode and Dataset Browser. Nothing here depends on
                                    local file paths or catalog manifests: you pick two files, choose method and captures, and run verify
                                    directly.
                                </div>

                                {showManualReuploadHint ? (
                                    <InlineBanner variant="warning" title="Previous manual pair needs re-upload">
                                        {verify.manualPairReminder?.probeFileName ?? "Probe file"} and {verify.manualPairReminder?.referenceFileName ?? "reference file"}
                                        {` `}were remembered only as lightweight labels. Upload the files again before running verify.
                                    </InlineBanner>
                                ) : null}

                                <div className="grid gap-5 md:grid-cols-2">
                                    <SurfaceCard title="Probe Image" description="Upload the probe image." className="h-full">
                                        <FileDropBox
                                            file={verify.form.probeFile}
                                            onChange={(file) => {
                                                verify.updateForm({ probeFile: file });
                                            }}
                                            disabled={verify.isBusy}
                                            title="Probe fingerprint"
                                            description="Drag and drop or browse for the probe image."
                                        />
                                    </SurfaceCard>

                                    <SurfaceCard title="Reference Image" description="Upload the reference image." className="h-full">
                                        <FileDropBox
                                            file={verify.form.referenceFile}
                                            onChange={(file) => {
                                                verify.updateForm({ referenceFile: file });
                                            }}
                                            disabled={verify.isBusy}
                                            title="Reference fingerprint"
                                            description="Drag and drop or browse for the reference image."
                                        />
                                    </SurfaceCard>
                                </div>
                            </div>
                        </SurfaceCard>
                    ) : null}
                </div>

                <div className="space-y-6">
                    <SurfaceCard
                        title={verify.activeMode === "demo" ? "Run Controls" : verify.activeMode === "browser" ? "Browser Pair Controls" : "Manual Controls"}
                        description={
                            verify.activeMode === "demo"
                                ? "The selected case defaults to its recommended method, but you can still override method or capture metadata before running."
                                : verify.activeMode === "browser"
                                    ? "Dataset Browser only fills the pair. Method choice stays yours, and Run Verification stays explicit."
                                    : "Choose the method and request options for the two files you uploaded."
                        }
                    >
                        <div className="space-y-5">
                            {verify.activeMode === "demo" && selectedDemoCase ? (
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                        <div>
                                            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Ready To Run</p>
                                            <p className="mt-1 text-base font-semibold text-slate-900">{selectedDemoCase.title}</p>
                                            <p className="mt-1 text-sm text-slate-600">
                                                {selectedDemoCase.dataset_label} / {selectedDemoCase.split} / {formatGroundTruthLabel(selectedDemoCase.ground_truth)}
                                            </p>
                                        </div>
                                        <div className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-slate-600">
                                            Recommended {formatMethodLabel(selectedDemoCase.recommended_method)}
                                        </div>
                                    </div>
                                </div>
                            ) : null}

                            {verify.activeMode === "browser" ? (
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                        <div>
                                            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Ready To Run</p>
                                            <p className="mt-1 text-base font-semibold text-slate-900">
                                                {browser.selectedAssetA && browser.selectedAssetB
                                                    ? `${browser.selectedAssetA.asset_id} vs ${browser.selectedAssetB.asset_id}`
                                                    : "Select both sides in Pair Builder"}
                                            </p>
                                            <p className="mt-1 text-sm text-slate-600">
                                                {browser.selectedDataset
                                                    ? `${browser.selectedDataset.dataset_label} / ${browser.selectedAssetA?.split ?? "-"} to ${browser.selectedAssetB?.split ?? "-"}`
                                                    : "Choose a browser-ready dataset first."}
                                            </p>
                                        </div>
                                        <div className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-slate-600">
                                            {verify.isCurrentBrowserPairApplied ? "Pair Applied" : "Apply Pair First"}
                                        </div>
                                    </div>
                                </div>
                            ) : null}

                            <div className="grid gap-4 md:grid-cols-2">
                                <FormField
                                    label="Method"
                                    hint={
                                        verify.activeMode === "demo" && selectedDemoCase
                                            ? `Catalog default: ${formatMethodLabel(selectedDemoCase.recommended_method)}. ${verify.selectedMethod.hint}`
                                            : verify.selectedMethod.hint
                                    }
                                >
                                    <select
                                        className={INPUT_CLASS_NAME}
                                        value={verify.form.method}
                                        disabled={verify.isBusy}
                                        onChange={(event) => {
                                            verify.updateForm({ method: event.target.value as keyof typeof METHOD_PROFILES });
                                        }}
                                    >
                                        {Object.values(METHOD_PROFILES).map((profile) => (
                                            <option key={profile.value} value={profile.value}>
                                                {profile.label}
                                            </option>
                                        ))}
                                    </select>
                                </FormField>

                                <FormField label="Max Visualized Matches" hint="A positive integer used only by the client-side canvas.">
                                    <input
                                        className={INPUT_CLASS_NAME}
                                        value={verify.form.maxMatchesText}
                                        disabled={verify.isBusy}
                                        onChange={(event) => {
                                            verify.updateForm({ maxMatchesText: event.target.value });
                                        }}
                                    />
                                </FormField>

                                <FormField label="Capture A" hint={verify.selectedMethod.captureHelp}>
                                    <select
                                        className={INPUT_CLASS_NAME}
                                        value={verify.form.captureA}
                                        disabled={verify.isBusy}
                                        onChange={(event) => {
                                            verify.updateForm({ captureA: event.target.value as typeof verify.form.captureA });
                                        }}
                                    >
                                        {CAPTURE_OPTIONS.map((option) => (
                                            <option key={option.value} value={option.value}>
                                                {option.label}
                                            </option>
                                        ))}
                                    </select>
                                </FormField>

                                <FormField label="Capture B" hint={verify.selectedMethod.captureHelp}>
                                    <select
                                        className={INPUT_CLASS_NAME}
                                        value={verify.form.captureB}
                                        disabled={verify.isBusy}
                                        onChange={(event) => {
                                            verify.updateForm({ captureB: event.target.value as typeof verify.form.captureB });
                                        }}
                                    >
                                        {CAPTURE_OPTIONS.map((option) => (
                                            <option key={option.value} value={option.value}>
                                                {option.label}
                                            </option>
                                        ))}
                                    </select>
                                </FormField>

                                <FormField label="Threshold Mode" hint={verify.selectedMethod.thresholdHelp}>
                                    <select
                                        className={INPUT_CLASS_NAME}
                                        value={verify.form.thresholdMode}
                                        disabled={verify.isBusy}
                                        onChange={(event) => {
                                            verify.updateForm({ thresholdMode: event.target.value as typeof verify.form.thresholdMode });
                                        }}
                                    >
                                        <option value="default">Use method default</option>
                                        <option value="custom">Custom threshold</option>
                                    </select>
                                </FormField>

                                <FormField label="Threshold Value" hint="Ignored when threshold mode is set to default.">
                                    <input
                                        className={INPUT_CLASS_NAME}
                                        value={verify.form.thresholdText}
                                        disabled={verify.isBusy || verify.form.thresholdMode === "default"}
                                        onChange={(event) => {
                                            verify.updateForm({ thresholdText: event.target.value });
                                        }}
                                    />
                                </FormField>
                            </div>

                            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                <p className="text-sm font-semibold text-slate-800">Execution Toggles</p>
                                <div className="mt-3 grid gap-3 md:grid-cols-2">
                                    <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                        <input
                                            type="checkbox"
                                            className={CHECKBOX_CLASS_NAME}
                                            checked={verify.form.returnOverlay}
                                            disabled={verify.isBusy || !verify.selectedMethod.supportsOverlay}
                                            onChange={(event) => {
                                                verify.updateForm({ returnOverlay: event.target.checked });
                                            }}
                                        />
                                        Return overlay
                                    </label>

                                    <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                        <input
                                            type="checkbox"
                                            className={CHECKBOX_CLASS_NAME}
                                            checked={verify.form.warmUpEnabled}
                                            disabled={verify.isBusy}
                                            onChange={(event) => {
                                                verify.updateForm({ warmUpEnabled: event.target.checked });
                                            }}
                                        />
                                        Warm up matcher
                                    </label>

                                    <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                        <input
                                            type="checkbox"
                                            className={CHECKBOX_CLASS_NAME}
                                            checked={verify.form.showOutliers}
                                            disabled={verify.isBusy}
                                            onChange={(event) => {
                                                verify.updateForm({ showOutliers: event.target.checked });
                                            }}
                                        />
                                        Show outliers on canvas
                                    </label>

                                    <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                        <input
                                            type="checkbox"
                                            className={CHECKBOX_CLASS_NAME}
                                            checked={verify.form.showTentative}
                                            disabled={verify.isBusy}
                                            onChange={(event) => {
                                                verify.updateForm({ showTentative: event.target.checked });
                                            }}
                                        />
                                        Show tentative on canvas
                                    </label>
                                </div>
                            </div>

                            <div className="flex flex-wrap items-center gap-3">
                                <button
                                    type="button"
                                    onClick={runPrimaryAction}
                                    disabled={!canRunPrimaryAction}
                                    className="inline-flex items-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-brand-300"
                                >
                                    {verify.isBusy ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                                    {stageLabel(verify.stage, verify.activeMode)}
                                </button>

                                <span className="inline-flex items-center rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                                    <SlidersHorizontal className="mr-2 h-4 w-4" />
                                    {verify.selectedMethod.label}
                                </span>

                                <span className="inline-flex items-center rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                                    <Thermometer className="mr-2 h-4 w-4" />
                                    Threshold {verify.form.thresholdMode === "default" ? "default" : verify.form.thresholdText || "custom"}
                                </span>

                                <span className="inline-flex items-center rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                                    <ChevronRight className="mr-2 h-4 w-4" />
                                    Max {verify.maxMatches} canvas matches
                                </span>
                            </div>
                        </div>
                    </SurfaceCard>

                    <SurfaceCard
                        title="Latest Result"
                        description="The decision stays attached to the exact case or file pair that produced it."
                    >
                        <div className="space-y-5">
                            {verify.lastRunContext ? (
                                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                        <div>
                                            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                                                {verify.lastRunContext.mode === "demo"
                                                    ? "Demo Result Context"
                                                    : verify.lastRunContext.mode === "browser"
                                                        ? "Browser Result Context"
                                                        : "Manual Result Context"}
                                            </p>
                                            <p className="mt-1 text-base font-semibold text-slate-900">{verify.lastRunContext.title}</p>
                                            <p className="mt-1 text-sm text-slate-600">{verify.lastRunContext.subtitle}</p>
                                        </div>
                                        <div className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold tracking-[0.16em] text-slate-600">
                                            Method {formatMethodLabel(verify.lastRunContext.method)}
                                        </div>
                                    </div>

                                    <div className="mt-4 grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
                                        {verify.lastRunContext.mode === "demo" ? (
                                            <>
                                                <div>
                                                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Dataset / Split</p>
                                                    <p className="mt-1 font-medium text-slate-900">
                                                        {verify.lastRunContext.datasetLabel} / {verify.lastRunContext.split}
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Method Behavior</p>
                                                    <p className="mt-1 font-medium text-slate-900">
                                                        {usingMethodOverride && verify.lastRunContext.recommendedMethod
                                                            ? `Override from ${formatMethodLabel(verify.lastRunContext.recommendedMethod)}`
                                                            : "Using recommended method"}
                                                    </p>
                                                </div>
                                            </>
                                        ) : verify.lastRunContext.mode === "browser" ? (
                                            <>
                                                <div>
                                                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Dataset / Split</p>
                                                    <p className="mt-1 font-medium text-slate-900">
                                                        {verify.lastRunContext.datasetLabel} / {verify.lastRunContext.split}
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Asset Pair</p>
                                                    <p className="mt-1 font-medium text-slate-900">
                                                        {verify.lastRunContext.assetAId ?? "-"} vs {verify.lastRunContext.assetBId ?? "-"}
                                                    </p>
                                                </div>
                                            </>
                                        ) : (
                                            <>
                                                <div>
                                                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Probe File</p>
                                                    <p className="mt-1 font-medium text-slate-900">{verify.lastRunContext.probeFileName ?? "-"}</p>
                                                </div>
                                                <div>
                                                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Reference File</p>
                                                    <p className="mt-1 font-medium text-slate-900">{verify.lastRunContext.referenceFileName ?? "-"}</p>
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>
                            ) : null}

                            {verify.resultState.status === "loading" ? (
                                <RequestState
                                    variant="loading"
                                    title="Verification request in progress"
                                    description={loadingDescription}
                                />
                            ) : null}

                            {verify.resultState.status === "error" && verify.resultState.error ? (
                                <RequestState
                                    variant="error"
                                    title="Verification failed"
                                    description={verify.resultState.error}
                                    actionLabel="Try again"
                                    onAction={() => {
                                        void verify.retryLastRun();
                                    }}
                                />
                            ) : null}

                            {verify.resultState.status === "success" && verify.currentResult ? (
                                <>
                                    <VerifyOutcomeStoryPanel story={verify.verifyStoryState} />
                                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                        <div className="flex flex-wrap items-center justify-between gap-3">
                                            <div>
                                                <p className="text-sm font-semibold text-slate-900">Raw metrics</p>
                                                <p className="mt-1 text-sm text-slate-600">
                                                    The storytelling layer stays on top, while the original metric block remains available underneath.
                                                </p>
                                            </div>
                                            <button
                                                type="button"
                                                onClick={() => {
                                                    verify.setStoryVisibilityState({
                                                        rawDetailsExpanded: !verify.storyVisibilityState.rawDetailsExpanded,
                                                    });
                                                }}
                                                className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
                                            >
                                                {verify.storyVisibilityState.rawDetailsExpanded ? "Hide raw metrics" : "Show raw metrics"}
                                            </button>
                                        </div>
                                    </div>
                                    {verify.storyVisibilityState.rawDetailsExpanded ? (
                                        <ResultSummary resp={verify.currentResult} />
                                    ) : null}
                                    {showCanvas ? (
                                        <MatchCanvas
                                            fileA={verify.manualFiles.probeFile as File}
                                            fileB={verify.manualFiles.referenceFile as File}
                                            matches={overlayMatches}
                                            showOutliers={verify.form.showOutliers}
                                            showTentative={verify.form.showTentative}
                                            maxMatches={verify.maxMatches}
                                        />
                                    ) : (
                                        <RequestState
                                            variant="empty"
                                            title="No drawable overlay available"
                                            description={verify.notice ?? "The current response does not contain overlay matches for canvas visualization."}
                                        />
                                    )}
                                </>
                            ) : null}

                            {verify.resultState.status === "idle" ? (
                                <RequestState
                                    variant="empty"
                                    title="No result yet"
                                    description={emptyResultDescription}
                                />
                            ) : null}
                        </div>
                    </SurfaceCard>

                    <InlineBanner variant="info" title="Server-backed execution">
                        Dataset Browser uses <code>/api/catalog/datasets</code>, <code>/api/catalog/dataset-browser</code>, and the
                        server-returned asset URLs from <code>/api/catalog/assets/...</code> before the final request reaches
                        <code> /api/match</code>. No client-side filesystem assumptions are used.
                    </InlineBanner>
                </div>
            </div>
        </div>
    );
}
