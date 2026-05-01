import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
    fetchCatalogVerifyCases,
    loadCatalogBrowserPairFiles,
    loadCatalogVerifyCaseFiles,
    matchFingerprints,
    warmUpMatcher,
} from "../../../api/matchService.ts";
import {
    CAPTURE_VALUES,
    type Capture,
    type CatalogBuildHealth,
    type CatalogVerifyCase,
    type MatchResponse,
    type Method,
} from "../../../types/index.ts";
import {
    createErrorState,
    createIdleState,
    createLoadingState,
    createSuccessState,
    type AsyncState,
} from "../../../shared/request-state/index.ts";
import { formatMethodLabel } from "../../../shared/storytelling.ts";
import { toErrorMessage } from "../../../utils/error.ts";
import { METHOD_PROFILES, formatThresholdValue, type MethodProfile } from "../config.ts";
import {
    buildDemoRunConfiguration,
    createBrowserRunContext,
    createDemoRunContext,
    createManualRunContext,
    filterDemoCases,
    readVerifyModeFromSessionStorage,
    type ThresholdMode,
    type VerifyDemoFilter,
    type VerifyMode,
    type VerifyRunContext,
} from "../model.ts";
import {
    clearAllPersistedVerifyState,
    readPersistedVerifySessionState,
    readPersistedVerifyWorkspaceState,
    writePersistedVerifySessionState,
    writePersistedVerifyWorkspaceState,
    type PersistedVerifyBrowserState,
    type PersistedVerifyManualPair,
    type PersistedVerifyPreferences,
} from "../persistence.ts";
import { createVerifyStoryState } from "../storyModel.ts";
import { useCatalogBrowser } from "./useCatalogBrowser.ts";

export type VerifyStage = "idle" | "loading-demo" | "warming" | "matching";

export interface VerifyFormState {
    probeFile: File | null;
    referenceFile: File | null;
    method: Method;
    captureA: Capture;
    captureB: Capture;
    thresholdText: string;
    thresholdMode: ThresholdMode;
    returnOverlay: boolean;
    warmUpEnabled: boolean;
    showOutliers: boolean;
    showTentative: boolean;
    maxMatchesText: string;
}

interface RunMatchOverrides {
    probeFile?: File | null;
    referenceFile?: File | null;
    method?: Method;
    captureA?: Capture;
    captureB?: Capture;
    thresholdMode?: ThresholdMode;
    thresholdText?: string;
    returnOverlay?: boolean;
    warmUpEnabled?: boolean;
    context?: VerifyRunContext;
}

function parseThreshold(value: string): number | undefined | null {
    const trimmed = value.trim();
    if (!trimmed) {
        return undefined;
    }

    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
}

function parsePositiveInteger(value: string): number | null {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed <= 0) {
        return null;
    }
    return parsed;
}

function resolveCaptureOverride(value: string | null | undefined, fallback: Capture): Capture {
    return CAPTURE_VALUES.includes(value as Capture) ? (value as Capture) : fallback;
}

function createOverlayFallbackDescription(
    methodProfile: MethodProfile,
    overlayRequested: boolean,
    response: MatchResponse,
): string {
    if (!methodProfile.supportsOverlay) {
        return "This matcher does not return overlay data on the backend, so overlay = null is expected.";
    }

    if (!overlayRequested) {
        return "Overlay was disabled for this request, so the UI only renders the structured MatchResponse summary.";
    }

    if (response.overlay === null) {
        return "The server returned a valid MatchResponse without an overlay object for this request.";
    }

    if (response.overlay.matches.length === 0) {
        return "The server returned an overlay object, but it did not contain drawable matches for the current pair.";
    }

    return "Overlay data is available and ready for visualization.";
}

function createDefaultVerifyFormState(preferences?: PersistedVerifyPreferences | null): VerifyFormState {
    const method = preferences?.method ?? "vit";

    return {
        probeFile: null,
        referenceFile: null,
        method,
        captureA: preferences?.captureA ?? "plain",
        captureB: preferences?.captureB ?? "plain",
        thresholdText: preferences?.thresholdMode === "custom"
            ? preferences.thresholdText
            : formatThresholdValue(METHOD_PROFILES[method].defaultThreshold),
        thresholdMode: preferences?.thresholdMode ?? "default",
        returnOverlay: preferences?.returnOverlay ?? false,
        warmUpEnabled: preferences?.warmUpEnabled ?? true,
        showOutliers: preferences?.showOutliers ?? true,
        showTentative: preferences?.showTentative ?? true,
        maxMatchesText: preferences?.maxMatchesText ?? "100",
    };
}

function createManualPairPersistence(
    probeFile: File | null,
    referenceFile: File | null,
    fallback: PersistedVerifyManualPair | null,
): PersistedVerifyManualPair | null {
    const probeFileName = probeFile?.name ?? fallback?.probeFileName ?? null;
    const referenceFileName = referenceFile?.name ?? fallback?.referenceFileName ?? null;

    if (!probeFileName && !referenceFileName) {
        return null;
    }

    return {
        probeFileName,
        referenceFileName,
        requiresReupload: true,
    };
}

export function useVerifyWorkspace() {
    const persistedWorkspaceRef = useRef(readPersistedVerifyWorkspaceState());
    const persistedSessionRef = useRef(readPersistedVerifySessionState());
    const persistedWorkspace = persistedWorkspaceRef.current;
    const persistedSession = persistedSessionRef.current;

    const [form, setForm] = useState<VerifyFormState>(() => createDefaultVerifyFormState(persistedWorkspace?.preferences));
    const [activeMode, setActiveModeState] = useState<VerifyMode>(() => persistedWorkspace?.mode ?? readVerifyModeFromSessionStorage());
    const [demoFilter, setDemoFilter] = useState<VerifyDemoFilter>(() => persistedWorkspace?.demoFilter ?? "all");
    const [stage, setStage] = useState<VerifyStage>("idle");
    const [resultState, setResultState] = useState<AsyncState<MatchResponse>>(createIdleState());
    const [demoCasesState, setDemoCasesState] = useState<AsyncState<CatalogVerifyCase[]>>(createLoadingState());
    const [demoCatalogBuildHealth, setDemoCatalogBuildHealth] = useState<CatalogBuildHealth | null>(null);
    const [notice, setNotice] = useState<string | null>(null);
    const [selectedDemoCaseId, setSelectedDemoCaseId] = useState<string | null>(() => persistedWorkspace?.selectedDemoCaseId ?? null);
    const [pinnedDemoCaseIds, setPinnedDemoCaseIds] = useState<string[]>(() => persistedWorkspace?.pinnedDemoCaseIds ?? []);
    const [runningDemoCaseId, setRunningDemoCaseId] = useState<string | null>(null);
    const [lastRunContext, setLastRunContext] = useState<VerifyRunContext | null>(null);
    const [applyPairState, setApplyPairState] = useState<AsyncState<{ pairKey: string }>>(createIdleState());
    const [storyVisibilityState, setStoryVisibilityState] = useState(() => persistedSession?.storyVisibility ?? { rawDetailsExpanded: false });
    const [manualPairReminder, setManualPairReminder] = useState<PersistedVerifyManualPair | null>(() => persistedWorkspace?.manualPair ?? null);
    const demoCasesLoadedRef = useRef(false);
    const browser = useCatalogBrowser({ initialState: persistedWorkspace?.browser as PersistedVerifyBrowserState | null });

    const demoCases = useMemo(
        () => demoCasesState.data ?? [],
        [demoCasesState.data],
    );
    const filteredDemoCases = useMemo(
        () => filterDemoCases(demoCases, demoFilter),
        [demoCases, demoFilter],
    );
    const selectedDemoCase = useMemo(
        () => demoCases.find((demoCase) => demoCase.case_id === selectedDemoCaseId) ?? null,
        [demoCases, selectedDemoCaseId],
    );
    const pinnedDemoCases = useMemo(
        () => pinnedDemoCaseIds
            .map((caseId) => demoCases.find((demoCase) => demoCase.case_id === caseId) ?? null)
            .filter((demoCase): demoCase is CatalogVerifyCase => demoCase !== null),
        [demoCases, pinnedDemoCaseIds],
    );
    const selectedMethod = useMemo(() => METHOD_PROFILES[form.method], [form.method]);
    const isBusy = stage !== "idle";
    const maxMatches = parsePositiveInteger(form.maxMatchesText) ?? 100;
    const currentResult = resultState.data;
    const verifyStoryState = useMemo(
        () => createVerifyStoryState({ resultState, context: lastRunContext }),
        [lastRunContext, resultState],
    );
    const errorState = resultState.error ?? demoCasesState.error ?? browser.browserError ?? browser.datasetsState.error;
    const manualFiles = useMemo(
        () => ({
            probeFile: form.probeFile,
            referenceFile: form.referenceFile,
        }),
        [form.probeFile, form.referenceFile],
    );
    const isCurrentBrowserPairApplied = Boolean(
        browser.browserPairKey
        && browser.browserPairKey === applyPairState.data?.pairKey,
    );

    useEffect(() => {
        const nextWorkspaceState = {
            mode: activeMode,
            demoFilter,
            selectedDemoCaseId,
            pinnedDemoCaseIds,
            browser: {
                selectedDatasetKey: browser.selectedDataset?.dataset ?? null,
                filters: browser.browserFilters,
                selectedAssetA: browser.selectedAssetA,
                selectedAssetB: browser.selectedAssetB,
                replacementTarget: browser.replacementTarget,
            },
            manualPair: activeMode === "manual"
                ? createManualPairPersistence(form.probeFile, form.referenceFile, manualPairReminder)
                : manualPairReminder,
            preferences: {
                method: form.method,
                captureA: form.captureA,
                captureB: form.captureB,
                thresholdMode: form.thresholdMode,
                thresholdText: form.thresholdText,
                returnOverlay: form.returnOverlay,
                warmUpEnabled: form.warmUpEnabled,
                showOutliers: form.showOutliers,
                showTentative: form.showTentative,
                maxMatchesText: form.maxMatchesText,
            },
        };

        const defaultForm = createDefaultVerifyFormState(null);
        const hasDefaultFirstDemoSelection = selectedDemoCaseId === null || selectedDemoCaseId === demoCases[0]?.case_id;
        const firstDemoCase = demoCases[0] ?? null;
        const defaultDemoForm = firstDemoCase
            ? (() => {
                const demoConfig = buildDemoRunConfiguration(
                    {
                        method: defaultForm.method,
                        captureA: defaultForm.captureA,
                        captureB: defaultForm.captureB,
                        thresholdMode: defaultForm.thresholdMode,
                        thresholdText: defaultForm.thresholdText,
                    },
                    firstDemoCase,
                    false,
                );

                return {
                    ...defaultForm,
                    method: demoConfig.method,
                    captureA: demoConfig.captureA,
                    captureB: demoConfig.captureB,
                    thresholdText: demoConfig.thresholdText,
                    returnOverlay: demoConfig.returnOverlay,
                };
            })()
            : defaultForm;
        const baselineForm = activeMode === "demo" && hasDefaultFirstDemoSelection ? defaultDemoForm : defaultForm;
        const hasBrowserSelection = Boolean(browser.selectedAssetA || browser.selectedAssetB);
        const hasBrowserFilters =
            browser.browserFilters.split !== ""
            || browser.browserFilters.capture !== ""
            || browser.browserFilters.modality !== ""
            || browser.browserFilters.subjectId !== ""
            || browser.browserFilters.finger !== ""
            || browser.browserFilters.uiEligible !== "all"
            || browser.browserFilters.limit !== 48
            || browser.browserFilters.offset !== 0
            || browser.browserFilters.sort !== "default";
        const hasNonDefaultBrowserDataset = Boolean(
            browser.selectedDataset?.dataset
            && browser.selectedDataset.dataset !== browser.browserReadyDatasets[0]?.dataset,
        );
        const hasNonDefaultPreferences =
            form.method !== baselineForm.method
            || form.captureA !== baselineForm.captureA
            || form.captureB !== baselineForm.captureB
            || form.thresholdMode !== baselineForm.thresholdMode
            || form.thresholdText !== baselineForm.thresholdText
            || form.returnOverlay !== baselineForm.returnOverlay
            || form.warmUpEnabled !== baselineForm.warmUpEnabled
            || form.showOutliers !== baselineForm.showOutliers
            || form.showTentative !== baselineForm.showTentative
            || form.maxMatchesText !== baselineForm.maxMatchesText;
        const shouldPersist =
            activeMode !== "demo"
            || demoFilter !== "all"
            || !hasDefaultFirstDemoSelection
            || pinnedDemoCaseIds.length > 0
            || Boolean(nextWorkspaceState.manualPair)
            || hasBrowserSelection
            || hasBrowserFilters
            || hasNonDefaultBrowserDataset
            || hasNonDefaultPreferences;

        if (!shouldPersist) {
            clearAllPersistedVerifyState();
            return;
        }

        writePersistedVerifyWorkspaceState(nextWorkspaceState);
    }, [
        browser.browserReadyDatasets,
        activeMode,
        browser.browserFilters,
        browser.replacementTarget,
        browser.selectedAssetA,
        browser.selectedAssetB,
        browser.selectedDataset?.dataset,
        demoFilter,
        form.captureA,
        form.captureB,
        form.maxMatchesText,
        form.method,
        form.probeFile,
        form.referenceFile,
        form.returnOverlay,
        form.showOutliers,
        form.showTentative,
        form.thresholdMode,
        form.thresholdText,
        form.warmUpEnabled,
        manualPairReminder,
        pinnedDemoCaseIds,
        selectedDemoCaseId,
    ]);

    useEffect(() => {
        writePersistedVerifySessionState({
            storyVisibility: {
                rawDetailsExpanded: storyVisibilityState.rawDetailsExpanded,
            },
        });
    }, [storyVisibilityState.rawDetailsExpanded]);

    useEffect(() => {
        if (activeMode !== "manual") {
            return;
        }

        const manualPair = createManualPairPersistence(form.probeFile, form.referenceFile, null);
        if (!manualPair) {
            return;
        }

        setManualPairReminder(manualPair);
    }, [activeMode, form.probeFile, form.referenceFile]);

    useEffect(() => {
        if (form.thresholdMode !== "default") {
            return;
        }

        setForm((current) => ({
            ...current,
            thresholdText: formatThresholdValue(METHOD_PROFILES[current.method].defaultThreshold),
        }));
    }, [form.thresholdMode, form.method]);

    useEffect(() => {
        if (selectedMethod.supportsOverlay) {
            return;
        }

        setForm((current) => (current.returnOverlay ? { ...current, returnOverlay: false } : current));
    }, [selectedMethod.supportsOverlay]);

    useEffect(() => {
        setStoryVisibilityState({ rawDetailsExpanded: false });
    }, [lastRunContext?.title, resultState.status]);

    const loadCuratedDemoCases = useCallback(async (): Promise<void> => {
        setDemoCasesState(createLoadingState());

        try {
            const payload = await fetchCatalogVerifyCases();
            setDemoCasesState(createSuccessState(payload.items));
            setDemoCatalogBuildHealth(payload.catalog_build_health ?? null);
        } catch (error) {
            setDemoCasesState(createErrorState(toErrorMessage(error), []));
            setDemoCatalogBuildHealth(null);
        }
    }, []);

    useEffect(() => {
        if (demoCasesLoadedRef.current) {
            return;
        }

        demoCasesLoadedRef.current = true;
        void loadCuratedDemoCases();
    }, [loadCuratedDemoCases]);

    const updateForm = useCallback((patch: Partial<VerifyFormState>): void => {
        setForm((current) => ({ ...current, ...patch }));
    }, []);

    const selectDemoCase = useCallback((demoCase: CatalogVerifyCase, preserveUserChoices = false): void => {
        setSelectedDemoCaseId(demoCase.case_id);
        setForm((current) => {
            const demoConfig = buildDemoRunConfiguration(
                {
                    method: current.method,
                    captureA: current.captureA,
                    captureB: current.captureB,
                    thresholdMode: current.thresholdMode,
                    thresholdText: current.thresholdText,
                },
                demoCase,
                preserveUserChoices,
            );

            return {
                ...current,
                method: demoConfig.method,
                captureA: demoConfig.captureA,
                captureB: demoConfig.captureB,
                thresholdText: demoConfig.thresholdText,
                returnOverlay: demoConfig.returnOverlay,
            };
        });
    }, []);

    useEffect(() => {
        if (demoCases.length === 0) {
            if (demoCasesState.status === "loading") {
                return;
            }

            if (selectedDemoCaseId !== null) {
                setSelectedDemoCaseId(null);
            }
            if (pinnedDemoCaseIds.length > 0) {
                setPinnedDemoCaseIds([]);
            }
            return;
        }

        setPinnedDemoCaseIds((current) => {
            const next = current.filter((caseId) => demoCases.some((demoCase) => demoCase.case_id === caseId));
            return next.length === current.length && next.every((caseId, index) => caseId === current[index]) ? current : next;
        });

        if (selectedDemoCaseId && demoCases.some((demoCase) => demoCase.case_id === selectedDemoCaseId)) {
            return;
        }

        const firstCase = demoCases[0];
        setSelectedDemoCaseId(firstCase.case_id);

        if (activeMode === "demo") {
            setForm((current) => {
                const demoConfig = buildDemoRunConfiguration(
                    {
                        method: current.method,
                        captureA: current.captureA,
                        captureB: current.captureB,
                        thresholdMode: current.thresholdMode,
                        thresholdText: current.thresholdText,
                    },
                    firstCase,
                    false,
                );

                return {
                    ...current,
                    method: demoConfig.method,
                    captureA: demoConfig.captureA,
                    captureB: demoConfig.captureB,
                    thresholdText: demoConfig.thresholdText,
                    returnOverlay: demoConfig.returnOverlay,
                };
            });
        }
    }, [activeMode, demoCases, demoCasesState.status, pinnedDemoCaseIds, selectedDemoCaseId]);

    const setActiveMode = useCallback(
        (mode: VerifyMode): void => {
            setActiveModeState(mode);

            if (mode !== "demo") {
                return;
            }

            const targetCase = selectedDemoCase ?? demoCases[0] ?? null;
            if (targetCase) {
                selectDemoCase(targetCase, false);
            }
        },
        [demoCases, selectDemoCase, selectedDemoCase],
    );

    const togglePinnedDemoCase = useCallback((demoCase: CatalogVerifyCase): void => {
        setPinnedDemoCaseIds((current) => {
            if (current.includes(demoCase.case_id)) {
                return current.filter((caseId) => caseId !== demoCase.case_id);
            }

            return [demoCase.case_id, ...current].slice(0, 8);
        });
    }, []);

    const clearPersistedWorkspaceState = useCallback((): void => {
        clearAllPersistedVerifyState();
        browser.resetBrowserContinuity();
        setForm(createDefaultVerifyFormState(null));
        setActiveModeState("demo");
        setDemoFilter("all");
        setSelectedDemoCaseId(null);
        setPinnedDemoCaseIds([]);
        setRunningDemoCaseId(null);
        setLastRunContext(null);
        setApplyPairState(createIdleState());
        setStoryVisibilityState({ rawDetailsExpanded: false });
        setManualPairReminder(null);
        setStage("idle");
        setResultState(createIdleState());
        setNotice("Cleared saved Verify workspace continuity.");
    }, [browser.resetBrowserContinuity]);

    const applyBrowserPairToVerify = useCallback(async (): Promise<boolean> => {
        if (!browser.selectedAssetA || !browser.selectedAssetB || !browser.browserPairKey) {
            setApplyPairState(createErrorState(
                "Choose two dataset items before sending them to Verify.",
                applyPairState.data,
            ));
            return false;
        }

        setApplyPairState(createLoadingState(applyPairState.data));
        setNotice(null);

        try {
            const files = await loadCatalogBrowserPairFiles(browser.selectedAssetA, browser.selectedAssetB);
            setForm((current) => ({
                ...current,
                probeFile: files.fileA,
                referenceFile: files.fileB,
                captureA: resolveCaptureOverride(browser.selectedAssetA?.capture, current.captureA),
                captureB: resolveCaptureOverride(browser.selectedAssetB?.capture, current.captureB),
            }));
            setApplyPairState(createSuccessState({ pairKey: browser.browserPairKey }));
            setNotice(
                `Loaded the selected browser pair from ${browser.selectedDataset?.dataset_label ?? browser.selectedAssetA.dataset} into Verify. Review the controls and run when ready.`,
            );
            return true;
        } catch (error) {
            setApplyPairState(createErrorState(toErrorMessage(error), applyPairState.data));
            return false;
        }
    }, [
        applyPairState.data,
        browser.browserPairKey,
        browser.selectedAssetA,
        browser.selectedAssetB,
        browser.selectedDataset?.dataset_label,
    ]);

    const runMatch = useCallback(
        async (overrides: RunMatchOverrides = {}): Promise<boolean> => {
            const effectiveMethod = overrides.method ?? form.method;
            const methodProfile = METHOD_PROFILES[effectiveMethod];
            const effectiveProbeFile = overrides.probeFile ?? form.probeFile;
            const effectiveReferenceFile = overrides.referenceFile ?? form.referenceFile;
            const effectiveCaptureA = overrides.captureA ?? form.captureA;
            const effectiveCaptureB = overrides.captureB ?? form.captureB;
            const effectiveThresholdMode = overrides.thresholdMode ?? form.thresholdMode;
            const effectiveThresholdText = overrides.thresholdText ?? form.thresholdText;
            const effectiveReturnOverlay = methodProfile.supportsOverlay && (overrides.returnOverlay ?? form.returnOverlay);
            const effectiveWarmUpEnabled = overrides.warmUpEnabled ?? form.warmUpEnabled;
            const effectiveContext = overrides.context ?? createManualRunContext(
                effectiveProbeFile?.name,
                effectiveReferenceFile?.name,
                effectiveMethod,
                effectiveCaptureA,
                effectiveCaptureB,
            );

            setLastRunContext(effectiveContext);
            setNotice(null);

            if (!effectiveProbeFile || !effectiveReferenceFile) {
                setResultState(createErrorState("Please provide both a probe image and a reference image before running verify."));
                return false;
            }

            const threshold = effectiveThresholdMode === "custom"
                ? parseThreshold(effectiveThresholdText)
                : methodProfile.defaultThreshold;

            if (threshold === null) {
                setResultState(createErrorState("Threshold must be empty or a valid numeric value."));
                return false;
            }

            if (parsePositiveInteger(form.maxMatchesText) === null) {
                setResultState(createErrorState("Max matches must be a positive integer."));
                return false;
            }

            try {
                if (effectiveWarmUpEnabled && methodProfile.recommendedWarmUp) {
                    setStage("warming");
                    setResultState(createLoadingState(resultState.data));
                    await warmUpMatcher(effectiveMethod);
                }

                setStage("matching");
                setResultState(createLoadingState(resultState.data));
                const response = await matchFingerprints({
                    method: effectiveMethod,
                    fileA: effectiveProbeFile,
                    fileB: effectiveReferenceFile,
                    captureA: effectiveCaptureA,
                    captureB: effectiveCaptureB,
                    returnOverlay: effectiveReturnOverlay,
                    threshold,
                });
                setResultState(createSuccessState(response));
                setNotice(createOverlayFallbackDescription(methodProfile, effectiveReturnOverlay, response));
                return true;
            } catch (error) {
                setResultState(createErrorState(toErrorMessage(error), resultState.data));
                return false;
            } finally {
                setStage("idle");
            }
        },
        [form, resultState.data],
    );

    const runBrowserPair = useCallback(async (): Promise<void> => {
        if (
            !browser.selectedAssetA
            || !browser.selectedAssetB
            || !browser.browserPairKey
            || !isCurrentBrowserPairApplied
            || !form.probeFile
            || !form.referenceFile
        ) {
            setResultState(createErrorState(
                "Use the selected dataset pair as the verify pair before running verification.",
                resultState.data,
            ));
            return;
        }

        await runMatch({
            context: createBrowserRunContext(
                browser.selectedAssetA,
                browser.selectedAssetB,
                form.method,
                browser.selectedDataset?.dataset_label ?? browser.selectedAssetA.dataset,
            ),
        });
    }, [
        browser.browserPairKey,
        browser.selectedAssetA,
        browser.selectedAssetB,
        browser.selectedDataset?.dataset_label,
        form.method,
        form.probeFile,
        form.referenceFile,
        isCurrentBrowserPairApplied,
        resultState.data,
        runMatch,
    ]);

    const runDemoCase = useCallback(
        async (demoCase: CatalogVerifyCase, preserveUserChoices = selectedDemoCaseId === demoCase.case_id): Promise<void> => {
            const demoConfig = buildDemoRunConfiguration(
                {
                    method: form.method,
                    captureA: form.captureA,
                    captureB: form.captureB,
                    thresholdMode: form.thresholdMode,
                    thresholdText: form.thresholdText,
                },
                demoCase,
                preserveUserChoices,
            );

            const context = createDemoRunContext(demoCase, demoConfig.method);
            setSelectedDemoCaseId(demoCase.case_id);
            setRunningDemoCaseId(demoCase.case_id);
            setLastRunContext(context);
            setNotice(null);

            try {
                setStage("loading-demo");
                const files = await loadCatalogVerifyCaseFiles(demoCase);
                setForm((current) => ({
                    ...current,
                    probeFile: files.fileA,
                    referenceFile: files.fileB,
                    method: demoConfig.method,
                    captureA: demoConfig.captureA,
                    captureB: demoConfig.captureB,
                    thresholdText: demoConfig.thresholdText,
                    returnOverlay: demoConfig.returnOverlay,
                }));
                const didSucceed = await runMatch({
                    probeFile: files.fileA,
                    referenceFile: files.fileB,
                    method: demoConfig.method,
                    captureA: demoConfig.captureA,
                    captureB: demoConfig.captureB,
                    thresholdMode: form.thresholdMode,
                    thresholdText: demoConfig.thresholdText,
                    returnOverlay: demoConfig.returnOverlay,
                    context,
                });
                if (didSucceed) {
                    const methodLabel = formatMethodLabel(demoConfig.method);
                    const recommendedSuffix = demoConfig.method === demoCase.recommended_method
                        ? "using the catalog-recommended method."
                        : `with a manual method override from ${formatMethodLabel(demoCase.recommended_method)} to ${methodLabel}.`;
                    setNotice(`Loaded "${demoCase.title}" from ${demoCase.dataset_label} (${demoCase.split}) ${recommendedSuffix}`);
                }
            } catch (error) {
                setResultState(createErrorState(toErrorMessage(error), resultState.data));
            } finally {
                setStage("idle");
                setRunningDemoCaseId(null);
            }
        },
        [form.captureA, form.captureB, form.method, form.thresholdMode, form.thresholdText, resultState.data, runMatch, selectedDemoCaseId],
    );

    const runSelectedDemoCase = useCallback(async (): Promise<void> => {
        if (!selectedDemoCase) {
            setResultState(createErrorState("Select a curated demo case before running verification.", resultState.data));
            return;
        }

        await runDemoCase(selectedDemoCase, true);
    }, [resultState.data, runDemoCase, selectedDemoCase]);

    const retryLastRun = useCallback(async (): Promise<void> => {
        if (lastRunContext?.mode === "demo") {
            const demoCase = demoCases.find((item) => item.case_id === lastRunContext.demoCaseId) ?? selectedDemoCase;
            if (demoCase) {
                await runDemoCase(demoCase, true);
                return;
            }
        }

        await runMatch({ context: lastRunContext ?? undefined });
    }, [demoCases, lastRunContext, runDemoCase, runMatch, selectedDemoCase]);

    return {
        activeMode,
        applyBrowserPairToVerify,
        applyPairState,
        browser,
        currentResult,
        demoCases,
        demoCatalogBuildHealth,
        demoCasesState,
        demoFilter,
        errorState,
        filteredDemoCases,
        form,
        isCurrentBrowserPairApplied,
        isBusy,
        lastRunContext,
        manualPairReminder,
        manualFiles,
        maxMatches,
        notice,
        pinnedDemoCaseIds,
        pinnedDemoCases,
        resultState,
        runningDemoCaseId,
        selectedDemoCase,
        selectedDemoCaseId,
        selectedMethod,
        storyDerivedConfidenceState: verifyStoryState.confidenceBand,
        storyErrorState: verifyStoryState.storyErrorState,
        storyVisibilityState,
        stage,
        verifyStoryState,
        clearPersistedWorkspaceState,
        setActiveMode,
        setDemoFilter,
        setNotice,
        setStoryVisibilityState,
        selectDemoCase,
        togglePinnedDemoCase,
        updateForm,
        runBrowserPair,
        runDemoCase,
        runMatch,
        runSelectedDemoCase,
        retryLastRun,
        retryLoadDemoCases: loadCuratedDemoCases,
    };
}
