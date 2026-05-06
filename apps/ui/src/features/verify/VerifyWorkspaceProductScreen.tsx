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
import { MetricTile, StatusPill, WorkspaceHero } from "../../shared/ui/presentation.tsx";
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
            <WorkspaceHero
                eyebrow="Verify Workspace"
                title="Run curated cases, browse datasets, or upload a pair."
                description="Demo Mode stays first. Dataset Browser builds a server-backed pair, and Manual Upload keeps the direct 1:1 flow available."
                icon={Sparkles}
            >
                <div className="grid gap-3 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.65fr)]">
                    <div className="grid gap-3 md:grid-cols-3">
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
                                    className={`mode-card ${isActive ? "mode-card--active" : ""}`.trim()}
                                    aria-pressed={isActive}
                                >
                                    <div className="mode-card__content">
                                        <div className="mode-card__icon">
                                            <Icon className="h-4 w-4" />
                                        </div>
                                        <div className="min-w-0">
                                            <div className="mode-card__label">{mode.label}</div>
                                            <div className="mode-card__description text-clamp-2">{mode.description}</div>
                                        </div>
                                    </div>
                                </button>
                            );
                        })}
                    </div>

                    <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
                        <MetricTile label="Default Path" value="Demo Mode" detail="Curated first run" tone="brand" />
                        <MetricTile label="Browser datasets" value={browser.browserReadyDatasets.length} detail="Catalog-ready" />
                        <MetricTile
                            label="Latest Context"
                            value={<span className="safe-truncate">{verify.lastRunContext?.title ?? "No run yet"}</span>}
                            detail={verify.lastRunContext ? formatMethodLabel(verify.lastRunContext.method) : "Waiting for first execution"}
                            title={verify.lastRunContext?.title ?? "No run yet"}
                        />
                    </div>
                </div>
            </WorkspaceHero>

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
                    className="app-button app-button--secondary"
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

                                <div className="rounded-xl border border-[var(--app-brand-border)] bg-[var(--app-brand-surface)] px-4 py-3 text-sm leading-6 text-[var(--app-brand-text)]">
                                    Pick a case, keep the recommended method or override it, then run the same verify flow.
                                </div>

                                {selectedDemoCase ? (
                                    <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-5">
                                        <div className="flex flex-wrap items-start justify-between gap-4">
                                            <div className="min-w-0">
                                                <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Selected Case</p>
                                                <h3 className="mt-2 safe-text text-xl font-semibold text-[var(--app-text)]">{selectedDemoCase.title}</h3>
                                                <p className="mt-2 max-w-3xl text-clamp-2 text-sm leading-6 text-[var(--app-text-muted)]">{selectedDemoCase.description}</p>
                                            </div>

                                            <div className="flex flex-wrap gap-2">
                                                <button
                                                    type="button"
                                                    onClick={() => {
                                                        verify.togglePinnedDemoCase(selectedDemoCase);
                                                    }}
                                                    className={[
                                                        "app-button",
                                                        isSelectedDemoCasePinned
                                                            ? "app-button--secondary border-[var(--app-warning-border)] bg-[var(--app-warning-surface)] text-[var(--app-warning-text)]"
                                                            : "app-button--secondary",
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
                                                    className="app-button app-button--primary"
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

                                        <div className="mt-4 flex min-w-0 flex-wrap gap-2">
                                            <StatusPill title={selectedDemoCase.dataset_label}>{selectedDemoCase.dataset_label}</StatusPill>
                                            <StatusPill title={selectedDemoCase.split}>{selectedDemoCase.split}</StatusPill>
                                            <StatusPill>{formatGroundTruthLabel(selectedDemoCase.ground_truth)}</StatusPill>
                                            <StatusPill tone="brand">
                                                Recommended {formatMethodLabel(selectedDemoCase.recommended_method)}
                                            </StatusPill>
                                            {selectedDemoCase.evidence_quality ? (
                                                <StatusPill tone={selectedDemoCase.evidence_quality.evidence_status === "strong" ? "success" : "warning"}>
                                                    {selectedDemoCase.evidence_quality.evidence_status === "strong"
                                                        ? "Strong evidence"
                                                        : selectedDemoCase.evidence_quality.evidence_status === "fallback"
                                                            ? "Fallback evidence"
                                                            : "Degraded evidence"}
                                                </StatusPill>
                                            ) : null}
                                        </div>

                                        {selectedDemoCase.evidence_quality ? (
                                            <p className="mt-3 text-clamp-2 text-sm leading-6 text-[var(--app-text-muted)]">
                                                {selectedDemoCase.evidence_quality.evidence_note}
                                            </p>
                                        ) : null}
                                    </div>
                                ) : null}

                                {verify.pinnedDemoCases.length > 0 ? (
                                    <div className="rounded-xl border border-[var(--app-warning-border)] bg-[var(--app-warning-surface)] p-4">
                                        <div className="flex flex-wrap items-center justify-between gap-3">
                                            <div className="min-w-0">
                                                <p className="text-xs font-semibold uppercase text-[var(--app-warning-text)]">Pinned demo cases</p>
                                                <p className="mt-1 text-sm text-[var(--app-warning-text)]">
                                                    Keep a small verify playlist ready across reloads.
                                                </p>
                                            </div>
                                            <StatusPill tone="warning">
                                                {verify.pinnedDemoCases.length} pinned
                                            </StatusPill>
                                        </div>

                                        <div className="mt-4 grid gap-3 md:grid-cols-2">
                                            {verify.pinnedDemoCases.map((demoCase) => (
                                                <div key={demoCase.case_id} className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                                                    <p className="safe-text text-sm font-semibold text-[var(--app-text)]">{demoCase.title}</p>
                                                    <p className="mt-1 safe-text text-xs uppercase text-[var(--app-text-muted)]">
                                                        {demoCase.dataset_label} / {demoCase.split}
                                                    </p>

                                                    <div className="mt-3 flex flex-wrap gap-2">
                                                        <button
                                                            type="button"
                                                            onClick={() => verify.selectDemoCase(demoCase, true)}
                                                            className="app-button app-button--secondary"
                                                        >
                                                            Select
                                                        </button>
                                                        <button
                                                            type="button"
                                                            onClick={() => {
                                                                verify.togglePinnedDemoCase(demoCase);
                                                            }}
                                                            className="app-button app-button--secondary border-[var(--app-warning-border)] bg-[var(--app-warning-surface)] text-[var(--app-warning-text)]"
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
                                                    "status-pill cursor-pointer transition",
                                                    isActive
                                                        ? "status-pill--brand"
                                                        : "status-pill--neutral hover:border-[var(--app-brand-border)]",
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
                                <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] px-4 py-3 text-sm leading-6 text-[var(--app-text-soft)]">
                                    Manual Upload stays intentionally separate from Demo Mode and Dataset Browser. Pick two files, choose method and captures, and run verify directly.
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
                                <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                        <div className="min-w-0">
                                            <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Ready To Run</p>
                                            <p className="mt-1 safe-text text-base font-semibold text-[var(--app-text)]">{selectedDemoCase.title}</p>
                                            <p className="mt-1 safe-text text-sm text-[var(--app-text-muted)]">
                                                {selectedDemoCase.dataset_label} / {selectedDemoCase.split} / {formatGroundTruthLabel(selectedDemoCase.ground_truth)}
                                            </p>
                                        </div>
                                        <StatusPill tone="brand">
                                            Recommended {formatMethodLabel(selectedDemoCase.recommended_method)}
                                        </StatusPill>
                                    </div>
                                </div>
                            ) : null}

                            {verify.activeMode === "browser" ? (
                                <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                        <div className="min-w-0">
                                            <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Ready To Run</p>
                                            <p className="mt-1 safe-text text-base font-semibold text-[var(--app-text)]">
                                                {browser.selectedAssetA && browser.selectedAssetB
                                                    ? `${browser.selectedAssetA.asset_id} vs ${browser.selectedAssetB.asset_id}`
                                                    : "Select both sides in Pair Builder"}
                                            </p>
                                            <p className="mt-1 safe-text text-sm text-[var(--app-text-muted)]">
                                                {browser.selectedDataset
                                                    ? `${browser.selectedDataset.dataset_label} / ${browser.selectedAssetA?.split ?? "-"} to ${browser.selectedAssetB?.split ?? "-"}`
                                                    : "Choose a browser-ready dataset first."}
                                            </p>
                                        </div>
                                        <StatusPill tone={verify.isCurrentBrowserPairApplied ? "success" : "warning"}>
                                            {verify.isCurrentBrowserPairApplied ? "Pair Applied" : "Apply Pair First"}
                                        </StatusPill>
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

                            <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                                <p className="text-sm font-semibold text-[var(--app-text)]">Execution Toggles</p>
                                <div className="mt-3 grid gap-3 md:grid-cols-2">
                                    <label className="inline-flex min-w-0 items-center gap-2 text-sm text-[var(--app-text-soft)]">
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

                                    <label className="inline-flex min-w-0 items-center gap-2 text-sm text-[var(--app-text-soft)]">
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

                                    <label className="inline-flex min-w-0 items-center gap-2 text-sm text-[var(--app-text-soft)]">
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

                                    <label className="inline-flex min-w-0 items-center gap-2 text-sm text-[var(--app-text-soft)]">
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
                                    className="app-button app-button--primary"
                                >
                                    {verify.isBusy ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                                    {stageLabel(verify.stage, verify.activeMode)}
                                </button>

                                <StatusPill icon={SlidersHorizontal} title={verify.selectedMethod.label}>
                                    {verify.selectedMethod.label}
                                </StatusPill>

                                <StatusPill icon={Thermometer}>
                                    Threshold {verify.form.thresholdMode === "default" ? "default" : verify.form.thresholdText || "custom"}
                                </StatusPill>

                                <StatusPill icon={ChevronRight}>
                                    Max {verify.maxMatches} canvas matches
                                </StatusPill>
                            </div>
                        </div>
                    </SurfaceCard>

                    <SurfaceCard
                        title="Latest Result"
                        description="The decision stays attached to the exact case or file pair that produced it."
                    >
                        <div className="space-y-5">
                            {verify.lastRunContext ? (
                                <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                        <div className="min-w-0">
                                            <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">
                                                {verify.lastRunContext.mode === "demo"
                                                    ? "Demo Result Context"
                                                    : verify.lastRunContext.mode === "browser"
                                                        ? "Browser Result Context"
                                                        : "Manual Result Context"}
                                            </p>
                                            <p className="mt-1 safe-text text-base font-semibold text-[var(--app-text)]">{verify.lastRunContext.title}</p>
                                            <p className="mt-1 safe-text text-sm text-[var(--app-text-muted)]">{verify.lastRunContext.subtitle}</p>
                                        </div>
                                        <StatusPill tone="brand">
                                            Method {formatMethodLabel(verify.lastRunContext.method)}
                                        </StatusPill>
                                    </div>

                                    <div className="mt-4 grid gap-3 text-sm text-[var(--app-text-soft)] sm:grid-cols-2">
                                        {verify.lastRunContext.mode === "demo" ? (
                                            <>
                                                <div className="min-w-0">
                                                    <p className="text-xs font-medium uppercase text-[var(--app-text-muted)]">Dataset / Split</p>
                                                    <p className="mt-1 safe-text font-medium text-[var(--app-text)]">
                                                        {verify.lastRunContext.datasetLabel} / {verify.lastRunContext.split}
                                                    </p>
                                                </div>
                                                <div className="min-w-0">
                                                    <p className="text-xs font-medium uppercase text-[var(--app-text-muted)]">Method Behavior</p>
                                                    <p className="mt-1 safe-text font-medium text-[var(--app-text)]">
                                                        {usingMethodOverride && verify.lastRunContext.recommendedMethod
                                                            ? `Override from ${formatMethodLabel(verify.lastRunContext.recommendedMethod)}`
                                                            : "Using recommended method"}
                                                    </p>
                                                </div>
                                            </>
                                        ) : verify.lastRunContext.mode === "browser" ? (
                                            <>
                                                <div className="min-w-0">
                                                    <p className="text-xs font-medium uppercase text-[var(--app-text-muted)]">Dataset / Split</p>
                                                    <p className="mt-1 safe-text font-medium text-[var(--app-text)]">
                                                        {verify.lastRunContext.datasetLabel} / {verify.lastRunContext.split}
                                                    </p>
                                                </div>
                                                <div className="min-w-0">
                                                    <p className="text-xs font-medium uppercase text-[var(--app-text-muted)]">Asset Pair</p>
                                                    <p className="mt-1 safe-text font-medium text-[var(--app-text)]">
                                                        {verify.lastRunContext.assetAId ?? "-"} vs {verify.lastRunContext.assetBId ?? "-"}
                                                    </p>
                                                </div>
                                            </>
                                        ) : (
                                            <>
                                                <div className="min-w-0">
                                                    <p className="text-xs font-medium uppercase text-[var(--app-text-muted)]">Probe File</p>
                                                    <p className="mt-1 safe-text font-medium text-[var(--app-text)]">{verify.lastRunContext.probeFileName ?? "-"}</p>
                                                </div>
                                                <div className="min-w-0">
                                                    <p className="text-xs font-medium uppercase text-[var(--app-text-muted)]">Reference File</p>
                                                    <p className="mt-1 safe-text font-medium text-[var(--app-text)]">{verify.lastRunContext.referenceFileName ?? "-"}</p>
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
                                    <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                                        <div className="flex flex-wrap items-center justify-between gap-3">
                                            <div className="min-w-0">
                                                <p className="text-sm font-semibold text-[var(--app-text)]">Raw metrics</p>
                                                <p className="mt-1 text-sm text-[var(--app-text-muted)]">
                                                    Original score, threshold, latency, and overlay metrics remain available.
                                                </p>
                                            </div>
                                            <button
                                                type="button"
                                                onClick={() => {
                                                    verify.setStoryVisibilityState({
                                                        rawDetailsExpanded: !verify.storyVisibilityState.rawDetailsExpanded,
                                                    });
                                                }}
                                                className="app-button app-button--secondary"
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
