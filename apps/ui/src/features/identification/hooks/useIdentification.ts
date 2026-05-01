import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchCatalogDatasetBrowser, fetchCatalogDatasets, loadCatalogBrowserItemFile } from "../../../api/matchService.ts";
import {
    deleteIdentity,
    enrollFingerprint,
    fetchIdentificationAdminLayout,
    fetchIdentificationAdminReconciliationReport,
    fetchIdentificationCatalogGallery,
    fetchIdentificationDemoGallery,
    fetchIdentificationHealth,
    fetchIdentificationStats,
    identifyFingerprint,
    loadIdentificationProbeCaseFile,
    resetIdentificationBrowserStore,
    resetIdentificationDemoStore,
    seedIdentificationBrowserSelection,
    seedIdentificationDemoStore,
} from "../../../api/identificationService.ts";
import {
    createErrorState,
    createIdleState,
    createLoadingState,
    createSuccessState,
    type AsyncState,
} from "../../../shared/request-state";
import { CAPTURE_VALUES } from "../../../types";
import type {
    Capture,
    CatalogBrowserItem,
    CatalogDatasetSummary,
    CatalogIdentifyGalleryResponse,
    CatalogIdentifyProbeCase,
    CatalogIdentityItem,
    DeleteIdentityResponse,
    EnrollFingerprintResponse,
    IdentificationAdminInspectionResponse,
    IdentificationAdminReconciliationResponse,
    IdentificationHealthResponse,
    IdentifyBrowserResetResponse,
    IdentifyBrowserSeedSelectionResponse,
    IdentificationRetrievalMethod,
    IdentificationStatsResponse,
    IdentifyDemoResetResponse,
    IdentifyDemoSeedResponse,
    IdentifyResponse,
    JsonRecord,
    Method,
} from "../../../types";
import { toErrorMessage } from "../../../utils/error.ts";
import {
    countActiveBrowserFilters,
    createBrowserFilterOptions,
    createBrowserPagination,
    createDefaultBrowserFilters,
    createEmptyBrowserPagination,
    toCatalogBrowserQuery,
    type BrowserFilters,
    type BrowserPagination,
} from "../../verify/browserModel.ts";
import {
    createDemoExpectationSummary,
    persistIdentificationModeInSessionStorage,
    readIdentificationModeFromSessionStorage,
    type DemoExpectationSummary,
    type IdentificationMode,
} from "../model.ts";
import {
    clearPersistedIdentificationWorkspaceState,
    readPersistedIdentificationWorkspaceState,
    writePersistedIdentificationWorkspaceState,
    type PersistedIdentificationWorkspaceState,
    type PersistedProbeEntry,
} from "../persistence.ts";
import { createIdentificationStoryState } from "../storyModel.ts";

export interface EnrollFormState {
    file: File | null;
    fullName: string;
    nationalId: string;
    capture: Capture;
    includeDl: boolean;
    includeVit: boolean;
    replaceExisting: boolean;
}

export interface SearchFormState {
    file: File | null;
    capture: Capture;
    retrievalMethod: IdentificationRetrievalMethod;
    rerankMethod: Method;
    shortlistSizeText: string;
    thresholdText: string;
    namePattern: string;
    nationalIdPattern: string;
    createdFrom: string;
    createdTo: string;
}

export interface DemoSearchFormState {
    retrievalMethod: IdentificationRetrievalMethod;
    rerankMethod: Method;
    shortlistSizeText: string;
    thresholdText: string;
    advancedVisible: boolean;
    namePattern: string;
    nationalIdPattern: string;
    createdFrom: string;
    createdTo: string;
}

export type BrowserSearchFormState = DemoSearchFormState;

export interface DeleteFormState {
    randomId: string;
    confirmChecked: boolean;
}

function parseShortlistSize(value: string): number | null {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed <= 0) {
        return null;
    }
    return parsed;
}

function parseThreshold(value: string): number | undefined | null {
    const trimmed = value.trim();
    if (!trimmed) {
        return undefined;
    }

    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
}

function resolveCapture(value: string | null | undefined): Capture {
    return (CAPTURE_VALUES as readonly string[]).includes(String(value)) ? (value as Capture) : "plain";
}

function applyProbeDefaults(
    current: DemoSearchFormState,
    probeCase: CatalogIdentifyProbeCase,
    preserveUserChoices: boolean,
): DemoSearchFormState {
    if (preserveUserChoices) {
        return current;
    }

    return {
        ...current,
        retrievalMethod: probeCase.recommended_retrieval_method ?? current.retrievalMethod,
        rerankMethod: probeCase.recommended_rerank_method ?? current.rerankMethod,
        shortlistSizeText: probeCase.recommended_shortlist_size != null
            ? String(probeCase.recommended_shortlist_size)
            : current.shortlistSizeText,
    };
}

function createDefaultDemoSearchFormState(
    persisted: PersistedIdentificationWorkspaceState["demoSearchPreferences"] | null,
): DemoSearchFormState {
    return {
        retrievalMethod: persisted?.retrievalMethod ?? "dl",
        rerankMethod: persisted?.rerankMethod ?? "sift",
        shortlistSizeText: persisted?.shortlistSizeText ?? "10",
        thresholdText: persisted?.thresholdText ?? "",
        advancedVisible: persisted?.advancedVisible ?? false,
        namePattern: persisted?.namePattern ?? "",
        nationalIdPattern: persisted?.nationalIdPattern ?? "",
        createdFrom: persisted?.createdFrom ?? "",
        createdTo: persisted?.createdTo ?? "",
    };
}

function createDefaultBrowserSearchFormState(
    persisted: PersistedIdentificationWorkspaceState["browserSearchPreferences"] | null,
): BrowserSearchFormState {
    return {
        retrievalMethod: persisted?.retrievalMethod ?? "dl",
        rerankMethod: persisted?.rerankMethod ?? "sift",
        shortlistSizeText: persisted?.shortlistSizeText ?? "10",
        thresholdText: persisted?.thresholdText ?? "",
        advancedVisible: persisted?.advancedVisible ?? false,
        namePattern: persisted?.namePattern ?? "",
        nationalIdPattern: persisted?.nationalIdPattern ?? "",
        createdFrom: persisted?.createdFrom ?? "",
        createdTo: persisted?.createdTo ?? "",
    };
}

function createDefaultSearchFormState(
    persisted: PersistedIdentificationWorkspaceState["operationalSearchPreferences"] | null,
): SearchFormState {
    return {
        file: null,
        capture: persisted?.capture ?? "plain",
        retrievalMethod: persisted?.retrievalMethod ?? "dl",
        rerankMethod: persisted?.rerankMethod ?? "sift",
        shortlistSizeText: persisted?.shortlistSizeText ?? "25",
        thresholdText: persisted?.thresholdText ?? "",
        namePattern: persisted?.namePattern ?? "",
        nationalIdPattern: persisted?.nationalIdPattern ?? "",
        createdFrom: persisted?.createdFrom ?? "",
        createdTo: persisted?.createdTo ?? "",
    };
}

function toPersistedProbeEntry(probeCase: CatalogIdentifyProbeCase): PersistedProbeEntry {
    return {
        id: probeCase.id,
        title: probeCase.title,
        datasetLabel: probeCase.dataset_label,
        expectedOutcome: probeCase.expected_outcome ?? null,
    };
}

function prependRecentProbeEntry(current: PersistedProbeEntry[], probeCase: CatalogIdentifyProbeCase): PersistedProbeEntry[] {
    const nextEntry = toPersistedProbeEntry(probeCase);

    return [
        nextEntry,
        ...current.filter((entry) => entry.id !== nextEntry.id),
    ].slice(0, 6);
}

function createEmptyIdentifyGalleryResponse(): CatalogIdentifyGalleryResponse {
    return {
        items: [],
        demo_identities: [],
        probe_cases: [],
        total: 0,
        limit: 20,
        offset: 0,
        has_more: false,
        total_probe_cases: 0,
    };
}

function sameStringArray(left: string[], right: string[]): boolean {
    return left.length === right.length && left.every((value, index) => value === right[index]);
}

function hasNonDefaultSearchPreferences(
    current: DemoSearchFormState | BrowserSearchFormState,
    defaults: DemoSearchFormState | BrowserSearchFormState,
): boolean {
    return current.retrievalMethod !== defaults.retrievalMethod
        || current.rerankMethod !== defaults.rerankMethod
        || current.shortlistSizeText !== defaults.shortlistSizeText
        || current.thresholdText !== defaults.thresholdText
        || current.advancedVisible !== defaults.advancedVisible
        || current.namePattern !== defaults.namePattern
        || current.nationalIdPattern !== defaults.nationalIdPattern
        || current.createdFrom !== defaults.createdFrom
        || current.createdTo !== defaults.createdTo;
}

export function useIdentification() {
    const persistedWorkspaceRef = useRef(readPersistedIdentificationWorkspaceState());
    const persistedWorkspace = persistedWorkspaceRef.current;
    const didInitialLoadRef = useRef(false);
    const browserRequestIdRef = useRef(0);
    const browserGalleryRequestIdRef = useRef(0);
    const browserPreferredDatasetKeyRef = useRef<string | null>(persistedWorkspace?.browser.selectedDatasetKey ?? null);

    const [identificationMode, setIdentificationModeState] = useState<IdentificationMode>(
        () => persistedWorkspace?.mode ?? readIdentificationModeFromSessionStorage(),
    );
    const [statsState, setStatsState] = useState<AsyncState<IdentificationStatsResponse>>(createLoadingState());
    const [demoGalleryState, setDemoGalleryState] = useState<AsyncState<CatalogIdentifyGalleryResponse>>(createLoadingState());
    const [browserDatasetsState, setBrowserDatasetsState] = useState<AsyncState<CatalogDatasetSummary[]>>(createLoadingState());
    const [browserGalleryState, setBrowserGalleryState] = useState<AsyncState<CatalogIdentifyGalleryResponse>>(createIdleState());
    const [demoSeedState, setDemoSeedState] = useState<AsyncState<IdentifyDemoSeedResponse>>(createIdleState());
    const [demoResetState, setDemoResetState] = useState<AsyncState<IdentifyDemoResetResponse>>(createIdleState());
    const [demoRunState, setDemoRunState] = useState<AsyncState<{ probeCaseId: string }>>(createIdleState());
    const [demoResultState, setDemoResultState] = useState<AsyncState<IdentifyResponse>>(createIdleState());
    const [browserSeedState, setBrowserSeedState] = useState<AsyncState<IdentifyBrowserSeedSelectionResponse>>(createIdleState());
    const [browserResetState, setBrowserResetState] = useState<AsyncState<IdentifyBrowserResetResponse>>(createIdleState());
    const [browserRunState, setBrowserRunState] = useState<AsyncState<{ dataset: string }>>(createIdleState());
    const [browserResultState, setBrowserResultState] = useState<AsyncState<IdentifyResponse>>(createIdleState());
    const [enrollState, setEnrollState] = useState<AsyncState<EnrollFingerprintResponse>>(createIdleState());
    const [searchState, setSearchState] = useState<AsyncState<IdentifyResponse>>(createIdleState());
    const [deleteState, setDeleteState] = useState<AsyncState<DeleteIdentityResponse>>(createIdleState());
    const [healthState, setHealthState] = useState<AsyncState<IdentificationHealthResponse>>(createLoadingState());
    const [adminLayoutState, setAdminLayoutState] = useState<AsyncState<IdentificationAdminInspectionResponse>>(createLoadingState());
    const [adminReconciliationState, setAdminReconciliationState] = useState<AsyncState<IdentificationAdminReconciliationResponse>>(createIdleState());
    const [browserSelectedDataset, setBrowserSelectedDataset] = useState<CatalogDatasetSummary | null>(null);
    const [browserFilters, setBrowserFilters] = useState<BrowserFilters>(() => persistedWorkspace?.browser.filters ?? createDefaultBrowserFilters());
    const [browserItems, setBrowserItems] = useState<CatalogBrowserItem[]>([]);
    const [browserLoading, setBrowserLoading] = useState(false);
    const [browserError, setBrowserError] = useState<string | null>(null);
    const [browserPagination, setBrowserPagination] = useState<BrowserPagination>(createEmptyBrowserPagination());
    const [browserSummary, setBrowserSummary] = useState<JsonRecord>({});
    const [selectedBrowserGalleryIdentityIds, setSelectedBrowserGalleryIdentityIds] = useState<string[]>(
        () => persistedWorkspace?.browser.selectedGalleryIdentityIds ?? [],
    );
    const [selectedBrowserProbeAsset, setSelectedBrowserProbeAsset] = useState<CatalogBrowserItem | null>(
        () => persistedWorkspace?.browser.selectedProbeAsset ?? null,
    );
    const [lastBrowserRunProbeAsset, setLastBrowserRunProbeAsset] = useState<CatalogBrowserItem | null>(null);
    const [lastBrowserRunDatasetLabel, setLastBrowserRunDatasetLabel] = useState<string | null>(null);
    const [enrollForm, setEnrollForm] = useState<EnrollFormState>({
        file: null,
        fullName: "",
        nationalId: "",
        capture: "plain",
        includeDl: true,
        includeVit: true,
        replaceExisting: false,
    });
    const [searchForm, setSearchForm] = useState<SearchFormState>(() => createDefaultSearchFormState(persistedWorkspace?.operationalSearchPreferences ?? null));
    const [demoSearchForm, setDemoSearchForm] = useState<DemoSearchFormState>(() => createDefaultDemoSearchFormState(persistedWorkspace?.demoSearchPreferences ?? null));
    const [browserSearchForm, setBrowserSearchForm] = useState<BrowserSearchFormState>(() => createDefaultBrowserSearchFormState(persistedWorkspace?.browserSearchPreferences ?? null));
    const [deleteForm, setDeleteForm] = useState<DeleteFormState>({ randomId: "", confirmChecked: false });
    const [selectedProbeCaseId, setSelectedProbeCaseId] = useState<string | null>(() => persistedWorkspace?.selectedProbeCaseId ?? null);
    const [recentProbes, setRecentProbes] = useState<PersistedProbeEntry[]>(() => persistedWorkspace?.recentProbes ?? []);
    const [pinnedProbeCaseIds, setPinnedProbeCaseIds] = useState<string[]>(() => persistedWorkspace?.pinnedProbeCaseIds ?? []);
    const [lastDemoRunProbeCaseId, setLastDemoRunProbeCaseId] = useState<string | null>(null);
    const [notice, setNotice] = useState<string | null>(null);
    const [storyVisibilityState, setStoryVisibilityState] = useState({ rawDetailsExpanded: false });

    const refreshStats = useCallback(async (): Promise<void> => {
        setStatsState((current) => createLoadingState(current.data));
        try {
            const payload = await fetchIdentificationStats();
            setStatsState(createSuccessState(payload));
        } catch (error) {
            setStatsState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, []);

    const refreshHealth = useCallback(async (): Promise<void> => {
        setHealthState((current) => createLoadingState(current.data));
        try {
            const payload = await fetchIdentificationHealth();
            setHealthState(createSuccessState(payload));
        } catch (error) {
            setHealthState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, []);

    const refreshAdminLayout = useCallback(async (): Promise<void> => {
        setAdminLayoutState((current) => createLoadingState(current.data));
        try {
            const payload = await fetchIdentificationAdminLayout();
            setAdminLayoutState(createSuccessState(payload));
        } catch (error) {
            setAdminLayoutState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, []);

    const refreshAdminReconciliationReport = useCallback(async (): Promise<void> => {
        setAdminReconciliationState((current) => createLoadingState(current.data));
        try {
            const payload = await fetchIdentificationAdminReconciliationReport();
            setAdminReconciliationState(createSuccessState(payload));
        } catch (error) {
            setAdminReconciliationState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, []);

    const refreshRuntimeReadiness = useCallback(async (): Promise<void> => {
        await Promise.all([refreshHealth(), refreshAdminLayout()]);
    }, [refreshAdminLayout, refreshHealth]);

    const loadDemoGallery = useCallback(async (): Promise<void> => {
        setDemoGalleryState((current) => createLoadingState(current.data));
        try {
            const payload = await fetchIdentificationDemoGallery();
            setDemoGalleryState(createSuccessState(payload));
        } catch (error) {
            setDemoGalleryState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, []);

    const loadBrowserDatasets = useCallback(async (): Promise<void> => {
        setBrowserDatasetsState((current) => createLoadingState(current.data));
        try {
            const payload = await fetchCatalogDatasets();
            setBrowserDatasetsState(createSuccessState(payload.items));
        } catch (error) {
            setBrowserDatasetsState((current) => createErrorState(toErrorMessage(error), current.data ?? []));
        }
    }, []);

    useEffect(() => {
        if (didInitialLoadRef.current) {
            return;
        }

        didInitialLoadRef.current = true;
        void refreshStats();
        void refreshRuntimeReadiness();
        void loadDemoGallery();
        void loadBrowserDatasets();
    }, [loadBrowserDatasets, loadDemoGallery, refreshRuntimeReadiness, refreshStats]);

    const demoGallery = demoGalleryState.data;
    const demoIdentities = useMemo(() => demoGallery?.demo_identities ?? [], [demoGallery]);
    const probeCases = useMemo(() => demoGallery?.probe_cases ?? [], [demoGallery]);
    const selectedProbeCase = useMemo(
        () => probeCases.find((probeCase) => probeCase.id === selectedProbeCaseId) ?? null,
        [probeCases, selectedProbeCaseId],
    );
    const pinnedProbeCases = useMemo(
        () => pinnedProbeCaseIds
            .map((probeCaseId) => probeCases.find((probeCase) => probeCase.id === probeCaseId) ?? null)
            .filter((probeCase): probeCase is CatalogIdentifyProbeCase => probeCase !== null),
        [pinnedProbeCaseIds, probeCases],
    );
    const recentProbeCases = useMemo(
        () => recentProbes
            .map((entry) => probeCases.find((probeCase) => probeCase.id === entry.id) ?? null)
            .filter((probeCase): probeCase is CatalogIdentifyProbeCase => probeCase !== null),
        [probeCases, recentProbes],
    );
    const lastDemoRunProbeCase = useMemo(
        () => probeCases.find((probeCase) => probeCase.id === lastDemoRunProbeCaseId) ?? null,
        [lastDemoRunProbeCaseId, probeCases],
    );
    const demoExpectationState: DemoExpectationSummary = useMemo(
        () => createDemoExpectationSummary(lastDemoRunProbeCase, demoResultState.data ?? null),
        [demoResultState.data, lastDemoRunProbeCase],
    );

    const browserDatasets = useMemo(() => browserDatasetsState.data ?? [], [browserDatasetsState.data]);
    const browserReadyDatasets = useMemo(
        () => browserDatasets.filter((dataset) => dataset.has_browser_assets && dataset.has_identify_gallery),
        [browserDatasets],
    );
    const browserGallery = browserGalleryState.data ?? createEmptyIdentifyGalleryResponse();
    const browserIdentities = useMemo(() => browserGallery.items ?? [], [browserGallery.items]);
    const selectedBrowserGalleryIdentities = useMemo(
        () => browserIdentities.filter((identity) => selectedBrowserGalleryIdentityIds.includes(identity.identity_id)),
        [browserIdentities, selectedBrowserGalleryIdentityIds],
    );
    const browserSelectedDatasetKey = browserSelectedDataset?.dataset ?? null;
    const browserQuery = useMemo(
        () => (
            browserSelectedDataset?.has_browser_assets
                ? toCatalogBrowserQuery(browserSelectedDataset.dataset, browserFilters)
                : null
        ),
        [browserFilters, browserSelectedDataset?.dataset, browserSelectedDataset?.has_browser_assets],
    );
    const browserActiveFilterCount = useMemo(
        () => countActiveBrowserFilters(browserFilters),
        [browserFilters],
    );
    const browserFilterOptions = useMemo(
        () => createBrowserFilterOptions(browserSummary, browserItems, selectedBrowserProbeAsset, null),
        [browserItems, browserSummary, selectedBrowserProbeAsset],
    );
    const probeDatasetMismatch = Boolean(
        selectedBrowserProbeAsset
        && browserSelectedDataset
        && selectedBrowserProbeAsset.dataset !== browserSelectedDataset.dataset,
    );
    const galleryTooLarge = selectedBrowserGalleryIdentityIds.length >= 12;
    const browserWarnings = useMemo(() => {
        const warnings: string[] = [];
        if (!selectedBrowserGalleryIdentityIds.length) {
            warnings.push("Select at least one gallery identity before running browser identification.");
        }
        if (!selectedBrowserProbeAsset) {
            warnings.push("Select a probe asset from the dataset browser before running browser identification.");
        }
        if (probeDatasetMismatch) {
            warnings.push("Probe and gallery dataset must match in Browser mode.");
        }
        if (galleryTooLarge) {
            warnings.push("Large gallery selections can slow down the browser-seeded walkthrough.");
        }
        return warnings;
    }, [galleryTooLarge, probeDatasetMismatch, selectedBrowserGalleryIdentityIds.length, selectedBrowserProbeAsset]);

    const demoStoryState = useMemo(
        () => createIdentificationStoryState({ resultState: demoResultState, probeCase: lastDemoRunProbeCase }),
        [demoResultState, lastDemoRunProbeCase],
    );
    const browserStoryState = useMemo(
        () => createIdentificationStoryState({ resultState: browserResultState, probeCase: null }),
        [browserResultState],
    );
    const searchStoryState = useMemo(
        () => createIdentificationStoryState({ resultState: searchState, probeCase: null }),
        [searchState],
    );
    const identifyStoryState = identificationMode === "demo"
        ? demoStoryState
        : identificationMode === "browser"
            ? browserStoryState
            : searchStoryState;
    const demoStoreReady = (statsState.data?.demo_seeded_count ?? 0) > 0;
    const isDemoBusy = demoSeedState.status === "loading"
        || demoResetState.status === "loading"
        || demoRunState.status === "loading";
    const isBrowserBusy = browserSeedState.status === "loading"
        || browserResetState.status === "loading"
        || browserRunState.status === "loading"
        || browserResultState.status === "loading";

    useEffect(() => {
        if (browserReadyDatasets.length === 0) {
            if (browserSelectedDataset !== null) {
                setBrowserSelectedDataset(null);
            }
            browserPreferredDatasetKeyRef.current = null;
            return;
        }

        const requestedDatasetKey = browserSelectedDataset?.dataset ?? browserPreferredDatasetKeyRef.current;
        const matchingDataset = browserReadyDatasets.find((dataset) => dataset.dataset === requestedDatasetKey) ?? null;
        if (matchingDataset) {
            if (matchingDataset !== browserSelectedDataset) {
                setBrowserSelectedDataset(matchingDataset);
            }
            browserPreferredDatasetKeyRef.current = matchingDataset.dataset;
            return;
        }

        browserPreferredDatasetKeyRef.current = browserReadyDatasets[0].dataset;
        setBrowserSelectedDataset(browserReadyDatasets[0]);
    }, [browserReadyDatasets, browserSelectedDataset]);

    useEffect(() => {
        if (!browserSelectedDataset) {
            setBrowserGalleryState(createIdleState());
            return;
        }

        const requestId = browserGalleryRequestIdRef.current + 1;
        browserGalleryRequestIdRef.current = requestId;
        setBrowserGalleryState((current) => createLoadingState(current.data));

        void (async () => {
            try {
                const payload = await fetchIdentificationCatalogGallery(browserSelectedDataset.dataset);
                if (browserGalleryRequestIdRef.current !== requestId) {
                    return;
                }
                setBrowserGalleryState(createSuccessState(payload));
            } catch (error) {
                if (browserGalleryRequestIdRef.current !== requestId) {
                    return;
                }
                setBrowserGalleryState((current) => createErrorState(toErrorMessage(error), current.data));
            }
        })();
    }, [browserSelectedDataset]);

    useEffect(() => {
        if (!browserSelectedDataset?.has_browser_assets || !browserQuery) {
            setBrowserItems([]);
            setBrowserLoading(false);
            setBrowserError(null);
            setBrowserSummary({});
            setBrowserPagination(createEmptyBrowserPagination(browserFilters.limit));
            return;
        }

        const requestId = browserRequestIdRef.current + 1;
        browserRequestIdRef.current = requestId;
        setBrowserLoading(true);
        setBrowserError(null);

        void (async () => {
            try {
                const response = await fetchCatalogDatasetBrowser(browserQuery);
                if (browserRequestIdRef.current !== requestId) {
                    return;
                }
                setBrowserItems(response.items);
                setBrowserSummary(response.summary);
                setBrowserPagination(createBrowserPagination(response));
            } catch (error) {
                if (browserRequestIdRef.current !== requestId) {
                    return;
                }
                setBrowserItems([]);
                setBrowserSummary({});
                setBrowserPagination(createEmptyBrowserPagination(browserFilters.limit));
                setBrowserError(toErrorMessage(error));
            } finally {
                if (browserRequestIdRef.current === requestId) {
                    setBrowserLoading(false);
                }
            }
        })();
    }, [browserFilters.limit, browserQuery, browserSelectedDataset?.has_browser_assets]);

    useEffect(() => {
        if (!selectedBrowserProbeAsset || !browserSelectedDatasetKey || selectedBrowserProbeAsset.dataset === browserSelectedDatasetKey) {
            return;
        }
        setSelectedBrowserProbeAsset(null);
    }, [browserSelectedDatasetKey, selectedBrowserProbeAsset]);

    useEffect(() => {
        if (browserItems.length === 0) {
            return;
        }
        setSelectedBrowserProbeAsset((current) => browserItems.find((item) => item.asset_id === current?.asset_id) ?? current);
    }, [browserItems]);

    useEffect(() => {
        if (browserIdentities.length === 0) {
            if (browserGalleryState.status === "loading") {
                return;
            }
            if (selectedBrowserGalleryIdentityIds.length > 0) {
                setSelectedBrowserGalleryIdentityIds([]);
            }
            return;
        }

        const validIds = selectedBrowserGalleryIdentityIds.filter((identityId) => browserIdentities.some((identity) => identity.identity_id === identityId));
        if (!sameStringArray(validIds, selectedBrowserGalleryIdentityIds)) {
            setSelectedBrowserGalleryIdentityIds(validIds);
            return;
        }

        if (validIds.length === 0) {
            setSelectedBrowserGalleryIdentityIds([browserIdentities[0].identity_id]);
        }
    }, [browserGalleryState.status, browserIdentities, selectedBrowserGalleryIdentityIds]);

    useEffect(() => {
        if (probeCases.length === 0) {
            if (demoGalleryState.status === "loading") {
                return;
            }
            if (selectedProbeCaseId !== null) {
                setSelectedProbeCaseId(null);
            }
            if (recentProbes.length > 0) {
                setRecentProbes([]);
            }
            if (pinnedProbeCaseIds.length > 0) {
                setPinnedProbeCaseIds([]);
            }
            return;
        }

        setRecentProbes((current) => {
            const next = current.filter((entry) => probeCases.some((probeCase) => probeCase.id === entry.id));
            return next.length === current.length && next.every((entry, index) => entry.id === current[index]?.id) ? current : next;
        });
        setPinnedProbeCaseIds((current) => {
            const next = current.filter((probeCaseId) => probeCases.some((probeCase) => probeCase.id === probeCaseId));
            return next.length === current.length && next.every((probeCaseId, index) => probeCaseId === current[index]) ? current : next;
        });

        if (selectedProbeCaseId && probeCases.some((probeCase) => probeCase.id === selectedProbeCaseId)) {
            return;
        }

        const firstProbeCase = probeCases[0];
        setSelectedProbeCaseId(firstProbeCase.id);
        setDemoSearchForm((current) => applyProbeDefaults(current, firstProbeCase, false));
    }, [demoGalleryState.status, pinnedProbeCaseIds, probeCases, recentProbes, selectedProbeCaseId]);

    useEffect(() => {
        persistIdentificationModeInSessionStorage(identificationMode);
    }, [identificationMode]);

    useEffect(() => {
        setStoryVisibilityState({ rawDetailsExpanded: false });
    }, [browserResultState.status, demoResultState.status, identificationMode, lastDemoRunProbeCaseId, searchState.status]);

    useEffect(() => {
        const nextWorkspaceState = {
            mode: identificationMode,
            selectedProbeCaseId,
            recentProbes,
            pinnedProbeCaseIds,
            demoSearchPreferences: {
                retrievalMethod: demoSearchForm.retrievalMethod,
                rerankMethod: demoSearchForm.rerankMethod,
                shortlistSizeText: demoSearchForm.shortlistSizeText,
                thresholdText: demoSearchForm.thresholdText,
                advancedVisible: demoSearchForm.advancedVisible,
                namePattern: demoSearchForm.namePattern,
                nationalIdPattern: demoSearchForm.nationalIdPattern,
                createdFrom: demoSearchForm.createdFrom,
                createdTo: demoSearchForm.createdTo,
            },
            browserSearchPreferences: {
                retrievalMethod: browserSearchForm.retrievalMethod,
                rerankMethod: browserSearchForm.rerankMethod,
                shortlistSizeText: browserSearchForm.shortlistSizeText,
                thresholdText: browserSearchForm.thresholdText,
                advancedVisible: browserSearchForm.advancedVisible,
                namePattern: browserSearchForm.namePattern,
                nationalIdPattern: browserSearchForm.nationalIdPattern,
                createdFrom: browserSearchForm.createdFrom,
                createdTo: browserSearchForm.createdTo,
            },
            browser: {
                selectedDatasetKey: browserSelectedDatasetKey,
                filters: browserFilters,
                selectedGalleryIdentityIds: selectedBrowserGalleryIdentityIds,
                selectedProbeAsset: selectedBrowserProbeAsset,
            },
            operationalSearchPreferences: {
                capture: searchForm.capture,
                retrievalMethod: searchForm.retrievalMethod,
                rerankMethod: searchForm.rerankMethod,
                shortlistSizeText: searchForm.shortlistSizeText,
                thresholdText: searchForm.thresholdText,
                namePattern: searchForm.namePattern,
                nationalIdPattern: searchForm.nationalIdPattern,
                createdFrom: searchForm.createdFrom,
                createdTo: searchForm.createdTo,
            },
        };

        const defaultSearchForm = createDefaultSearchFormState(null);
        const defaultDemoSearchForm = selectedProbeCaseId === probeCases[0]?.id && probeCases[0]
            ? applyProbeDefaults(createDefaultDemoSearchFormState(null), probeCases[0], false)
            : createDefaultDemoSearchFormState(null);
        const defaultBrowserSearchForm = createDefaultBrowserSearchFormState(null);
        const defaultBrowserFilters = createDefaultBrowserFilters();
        const isDefaultProbeSelection = selectedProbeCaseId === null || selectedProbeCaseId === probeCases[0]?.id;
        const hasNonDefaultDemoSearchPreferences = hasNonDefaultSearchPreferences(demoSearchForm, defaultDemoSearchForm);
        const hasNonDefaultBrowserSearchPreferences = hasNonDefaultSearchPreferences(browserSearchForm, defaultBrowserSearchForm);
        const hasNonDefaultOperationalSearchPreferences =
            searchForm.capture !== defaultSearchForm.capture
            || searchForm.retrievalMethod !== defaultSearchForm.retrievalMethod
            || searchForm.rerankMethod !== defaultSearchForm.rerankMethod
            || searchForm.shortlistSizeText !== defaultSearchForm.shortlistSizeText
            || searchForm.thresholdText !== defaultSearchForm.thresholdText
            || searchForm.namePattern !== defaultSearchForm.namePattern
            || searchForm.nationalIdPattern !== defaultSearchForm.nationalIdPattern
            || searchForm.createdFrom !== defaultSearchForm.createdFrom
            || searchForm.createdTo !== defaultSearchForm.createdTo;
        const hasNonDefaultBrowserState =
            browserSelectedDatasetKey !== null
            || browserFilters.split !== defaultBrowserFilters.split
            || browserFilters.capture !== defaultBrowserFilters.capture
            || browserFilters.modality !== defaultBrowserFilters.modality
            || browserFilters.subjectId !== defaultBrowserFilters.subjectId
            || browserFilters.finger !== defaultBrowserFilters.finger
            || browserFilters.uiEligible !== defaultBrowserFilters.uiEligible
            || browserFilters.limit !== defaultBrowserFilters.limit
            || browserFilters.offset !== defaultBrowserFilters.offset
            || browserFilters.sort !== defaultBrowserFilters.sort
            || selectedBrowserGalleryIdentityIds.length > 0
            || selectedBrowserProbeAsset !== null;
        const shouldPersist =
            identificationMode !== "demo"
            || !isDefaultProbeSelection
            || recentProbes.length > 0
            || pinnedProbeCaseIds.length > 0
            || hasNonDefaultDemoSearchPreferences
            || hasNonDefaultBrowserSearchPreferences
            || hasNonDefaultBrowserState
            || hasNonDefaultOperationalSearchPreferences;

        if (!shouldPersist) {
            clearPersistedIdentificationWorkspaceState();
            return;
        }

        writePersistedIdentificationWorkspaceState(nextWorkspaceState);
    }, [
        browserFilters,
        browserSearchForm.advancedVisible,
        browserSearchForm.createdFrom,
        browserSearchForm.createdTo,
        browserSearchForm.namePattern,
        browserSearchForm.nationalIdPattern,
        browserSearchForm.rerankMethod,
        browserSearchForm.retrievalMethod,
        browserSearchForm.shortlistSizeText,
        browserSearchForm.thresholdText,
        browserSelectedDatasetKey,
        demoSearchForm.advancedVisible,
        demoSearchForm.createdFrom,
        demoSearchForm.createdTo,
        demoSearchForm.namePattern,
        demoSearchForm.nationalIdPattern,
        demoSearchForm.rerankMethod,
        demoSearchForm.retrievalMethod,
        demoSearchForm.shortlistSizeText,
        demoSearchForm.thresholdText,
        identificationMode,
        pinnedProbeCaseIds,
        probeCases,
        recentProbes,
        searchForm.capture,
        searchForm.createdFrom,
        searchForm.createdTo,
        searchForm.namePattern,
        searchForm.nationalIdPattern,
        searchForm.rerankMethod,
        searchForm.retrievalMethod,
        searchForm.shortlistSizeText,
        searchForm.thresholdText,
        selectedBrowserGalleryIdentityIds,
        selectedBrowserProbeAsset,
        selectedProbeCaseId,
    ]);

    const setIdentificationMode = useCallback((mode: IdentificationMode): void => {
        setIdentificationModeState(mode);
    }, []);

    const updateEnrollForm = useCallback((patch: Partial<EnrollFormState>): void => {
        setEnrollForm((current) => ({ ...current, ...patch }));
    }, []);

    const updateSearchForm = useCallback((patch: Partial<SearchFormState>): void => {
        setSearchForm((current) => ({ ...current, ...patch }));
    }, []);

    const updateDemoSearchForm = useCallback((patch: Partial<DemoSearchFormState>): void => {
        setDemoSearchForm((current) => ({ ...current, ...patch }));
    }, []);

    const updateBrowserSearchForm = useCallback((patch: Partial<BrowserSearchFormState>): void => {
        setBrowserSearchForm((current) => ({ ...current, ...patch }));
    }, []);

    const updateDeleteForm = useCallback((patch: Partial<DeleteFormState>): void => {
        setDeleteForm((current) => ({ ...current, ...patch }));
    }, []);

    const selectProbeCase = useCallback((probeCase: CatalogIdentifyProbeCase, preserveUserChoices = false): void => {
        setSelectedProbeCaseId(probeCase.id);
        setRecentProbes((current) => prependRecentProbeEntry(current, probeCase));
        setDemoSearchForm((current) => applyProbeDefaults(current, probeCase, preserveUserChoices));
    }, []);

    const togglePinnedProbeCase = useCallback((probeCase: CatalogIdentifyProbeCase): void => {
        setPinnedProbeCaseIds((current) => {
            if (current.includes(probeCase.id)) {
                return current.filter((probeCaseId) => probeCaseId !== probeCase.id);
            }
            return [probeCase.id, ...current].slice(0, 8);
        });
    }, []);

    const selectBrowserDataset = useCallback((dataset: CatalogDatasetSummary): void => {
        if (browserSelectedDataset?.dataset === dataset.dataset) {
            return;
        }
        browserPreferredDatasetKeyRef.current = dataset.dataset;
        setBrowserSelectedDataset(dataset);
        setBrowserFilters(createDefaultBrowserFilters());
        setSelectedBrowserGalleryIdentityIds([]);
        setSelectedBrowserProbeAsset(null);
        setBrowserItems([]);
        setBrowserError(null);
        setBrowserSummary({});
        setBrowserPagination(createEmptyBrowserPagination());
        setBrowserSeedState(createIdleState());
        setBrowserResetState(createIdleState());
        setBrowserRunState(createIdleState());
        setBrowserResultState(createIdleState());
        setLastBrowserRunProbeAsset(null);
        setLastBrowserRunDatasetLabel(null);
    }, [browserSelectedDataset?.dataset]);

    const updateBrowserFilters = useCallback((patch: Partial<BrowserFilters>): void => {
        setBrowserFilters((current) => {
            const next = { ...current, ...patch };
            if (!Object.prototype.hasOwnProperty.call(patch, "offset")) {
                next.offset = 0;
            }
            return next;
        });
    }, []);

    const resetBrowserFilters = useCallback((): void => {
        setBrowserFilters(createDefaultBrowserFilters());
    }, []);

    const toggleBrowserGalleryIdentity = useCallback((identity: CatalogIdentityItem): void => {
        setSelectedBrowserGalleryIdentityIds((current) => (
            current.includes(identity.identity_id)
                ? current.filter((identityId) => identityId !== identity.identity_id)
                : [...current, identity.identity_id]
        ));
    }, []);

    const selectBrowserProbeAsset = useCallback((item: CatalogBrowserItem): void => {
        setSelectedBrowserProbeAsset(item);
    }, []);

    const clearSelectedBrowserProbe = useCallback((): void => {
        setSelectedBrowserProbeAsset(null);
    }, []);

    const clearPersistedWorkspaceState = useCallback((): void => {
        clearPersistedIdentificationWorkspaceState();
        setIdentificationModeState("demo");
        setEnrollForm({
            file: null,
            fullName: "",
            nationalId: "",
            capture: "plain",
            includeDl: true,
            includeVit: true,
            replaceExisting: false,
        });
        setDemoSearchForm(createDefaultDemoSearchFormState(null));
        setBrowserSearchForm(createDefaultBrowserSearchFormState(null));
        setSearchForm(createDefaultSearchFormState(null));
        setDeleteForm({ randomId: "", confirmChecked: false });
        setSelectedProbeCaseId(null);
        setRecentProbes([]);
        setPinnedProbeCaseIds([]);
        setLastDemoRunProbeCaseId(null);
        setBrowserFilters(createDefaultBrowserFilters());
        setSelectedBrowserGalleryIdentityIds([]);
        setSelectedBrowserProbeAsset(null);
        setLastBrowserRunProbeAsset(null);
        setLastBrowserRunDatasetLabel(null);
        setNotice("Cleared saved Identification workspace continuity.");
        setDemoSeedState(createIdleState());
        setDemoResetState(createIdleState());
        setDemoRunState(createIdleState());
        setDemoResultState(createIdleState());
        setBrowserSeedState(createIdleState());
        setBrowserResetState(createIdleState());
        setBrowserRunState(createIdleState());
        setBrowserResultState(createIdleState());
        setEnrollState(createIdleState());
        setSearchState(createIdleState());
        setDeleteState(createIdleState());
        setStoryVisibilityState({ rawDetailsExpanded: false });
    }, []);

    const submitEnroll = useCallback(async (): Promise<void> => {
        setNotice(null);
        if (!enrollForm.file) {
            setEnrollState(createErrorState("Please upload a fingerprint image before enrolling an identity."));
            return;
        }
        if (!enrollForm.fullName.trim() || !enrollForm.nationalId.trim()) {
            setEnrollState(createErrorState("Full name and national ID are required for enrollment."));
            return;
        }

        const vectorMethods = [enrollForm.includeDl ? "dl" : null, enrollForm.includeVit ? "vit" : null]
            .filter((item): item is string => item !== null);
        if (vectorMethods.length === 0) {
            setEnrollState(createErrorState("Choose at least one vector method for enrollment."));
            return;
        }

        setEnrollState((current) => createLoadingState(current.data));
        try {
            const payload = await enrollFingerprint({
                file: enrollForm.file,
                fullName: enrollForm.fullName.trim(),
                nationalId: enrollForm.nationalId.trim(),
                capture: enrollForm.capture,
                vectorMethods,
                replaceExisting: enrollForm.replaceExisting,
            });
            setEnrollState(createSuccessState(payload));
            setNotice(`Enrolled ${enrollForm.fullName.trim()} as ${payload.random_id}.`);
            await Promise.all([refreshStats(), refreshAdminLayout()]);
        } catch (error) {
            setEnrollState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, [enrollForm, refreshAdminLayout, refreshStats]);

    const submitSearch = useCallback(async (): Promise<void> => {
        setNotice(null);
        if (!searchForm.file) {
            setSearchState(createErrorState("Please upload a probe fingerprint before running identification."));
            return;
        }
        const shortlistSize = parseShortlistSize(searchForm.shortlistSizeText);
        if (shortlistSize === null) {
            setSearchState(createErrorState("Shortlist size must be a positive integer."));
            return;
        }
        const threshold = parseThreshold(searchForm.thresholdText);
        if (threshold === null) {
            setSearchState(createErrorState("Threshold must be empty or a valid number."));
            return;
        }
        if (searchForm.createdFrom && searchForm.createdTo && searchForm.createdFrom > searchForm.createdTo) {
            setSearchState(createErrorState("Created from must be earlier than or equal to created to."));
            return;
        }

        setSearchState((current) => createLoadingState(current.data));
        try {
            const payload = await identifyFingerprint({
                file: searchForm.file,
                capture: searchForm.capture,
                retrievalMethod: searchForm.retrievalMethod,
                rerankMethod: searchForm.rerankMethod,
                shortlistSize,
                threshold,
                namePattern: searchForm.namePattern.trim() || undefined,
                nationalIdPattern: searchForm.nationalIdPattern.trim() || undefined,
                createdFrom: searchForm.createdFrom || undefined,
                createdTo: searchForm.createdTo || undefined,
            });
            setSearchState(createSuccessState(payload));
            setNotice(
                payload.top_candidate
                    ? `Top candidate: ${payload.top_candidate.full_name} (${payload.top_candidate.random_id}).`
                    : "Identification completed without a top candidate.",
            );
        } catch (error) {
            setSearchState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, [searchForm]);

    const submitDelete = useCallback(async (): Promise<void> => {
        setNotice(null);
        if (!deleteForm.randomId.trim()) {
            setDeleteState(createErrorState("Provide a random ID before requesting deletion."));
            return;
        }
        if (!deleteForm.confirmChecked) {
            setDeleteState(createErrorState("You must confirm deletion before removing an identity."));
            return;
        }

        setDeleteState((current) => createLoadingState(current.data));
        try {
            const payload = await deleteIdentity(deleteForm.randomId.trim());
            setDeleteState(createSuccessState(payload));
            setDeleteForm({ randomId: payload.random_id, confirmChecked: false });
            setNotice(payload.removed ? `Deleted ${payload.random_id}.` : `${payload.random_id} was not found.`);
            await Promise.all([refreshStats(), refreshAdminLayout()]);
        } catch (error) {
            setDeleteState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, [deleteForm, refreshAdminLayout, refreshStats]);

    const seedDemoStore = useCallback(async (): Promise<void> => {
        setNotice(null);
        setDemoSeedState((current) => createLoadingState(current.data));
        try {
            const payload = await seedIdentificationDemoStore();
            setDemoSeedState(createSuccessState(payload));
            setDemoResetState(createIdleState());
            setNotice(payload.notice ?? "Demo identities seeded successfully.");
            await Promise.all([refreshStats(), refreshAdminLayout()]);
        } catch (error) {
            setDemoSeedState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, [refreshAdminLayout, refreshStats]);

    const resetDemoStore = useCallback(async (): Promise<void> => {
        setNotice(null);
        setDemoResetState((current) => createLoadingState(current.data));
        try {
            const payload = await resetIdentificationDemoStore();
            setDemoResetState(createSuccessState(payload));
            setDemoSeedState(createIdleState());
            setNotice(payload.notice ?? "Demo store reset completed.");
            await Promise.all([refreshStats(), refreshAdminLayout()]);
        } catch (error) {
            setDemoResetState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, [refreshAdminLayout, refreshStats]);

    const resetBrowserStore = useCallback(async (): Promise<void> => {
        setNotice(null);
        setBrowserResetState((current) => createLoadingState(current.data));
        try {
            const payload = await resetIdentificationBrowserStore();
            setBrowserResetState(createSuccessState(payload));
            setBrowserSeedState(createIdleState());
            setNotice(payload.notice ?? "Browser store reset completed.");
            await Promise.all([refreshStats(), refreshAdminLayout()]);
        } catch (error) {
            setBrowserResetState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }, [refreshAdminLayout, refreshStats]);

    const runDemoIdentification = useCallback(async (probeCaseOverride?: CatalogIdentifyProbeCase | null): Promise<void> => {
        setNotice(null);
        const activeProbeCase = probeCaseOverride ?? selectedProbeCase;
        if (!activeProbeCase) {
            setDemoRunState(createErrorState("Select a probe case before running identification."));
            return;
        }
        if (!demoStoreReady) {
            setDemoRunState(createErrorState("Seed demo identities before running guided identification."));
            return;
        }

        const shortlistSize = parseShortlistSize(demoSearchForm.shortlistSizeText);
        if (shortlistSize === null) {
            setDemoRunState(createErrorState("Shortlist size must be a positive integer."));
            return;
        }
        const threshold = parseThreshold(demoSearchForm.thresholdText);
        if (threshold === null) {
            setDemoRunState(createErrorState("Threshold must be empty or a valid number."));
            return;
        }
        if (demoSearchForm.createdFrom && demoSearchForm.createdTo && demoSearchForm.createdFrom > demoSearchForm.createdTo) {
            setDemoRunState(createErrorState("Created from must be earlier than or equal to created to."));
            return;
        }

        setLastDemoRunProbeCaseId(activeProbeCase.id);
        setRecentProbes((current) => prependRecentProbeEntry(current, activeProbeCase));
        setDemoRunState((current) => createLoadingState(current.data));
        setDemoResultState((current) => createLoadingState(current.data));

        try {
            const file = await loadIdentificationProbeCaseFile(activeProbeCase);
            const payload = await identifyFingerprint({
                file,
                capture: resolveCapture(activeProbeCase.capture),
                retrievalMethod: demoSearchForm.retrievalMethod,
                rerankMethod: demoSearchForm.rerankMethod,
                shortlistSize,
                threshold,
                namePattern: demoSearchForm.namePattern.trim() || undefined,
                nationalIdPattern: demoSearchForm.nationalIdPattern.trim() || undefined,
                createdFrom: demoSearchForm.createdFrom || undefined,
                createdTo: demoSearchForm.createdTo || undefined,
            });
            setDemoRunState(createSuccessState({ probeCaseId: activeProbeCase.id }));
            setDemoResultState(createSuccessState(payload));
            setNotice(
                payload.top_candidate
                    ? `Demo top candidate: ${payload.top_candidate.full_name} (${payload.top_candidate.random_id}).`
                    : "Demo identification completed without a top candidate.",
            );
        } catch (error) {
            const message = toErrorMessage(error);
            setDemoRunState((current) => createErrorState(message, current.data));
            setDemoResultState((current) => createErrorState(message, current.data));
        }
    }, [demoSearchForm, demoStoreReady, selectedProbeCase]);

    const runBrowserIdentification = useCallback(async (): Promise<void> => {
        setNotice(null);
        if (!browserSelectedDataset) {
            setBrowserRunState(createErrorState("Select a browser-ready dataset before running identification."));
            return;
        }
        if (selectedBrowserGalleryIdentityIds.length === 0) {
            setBrowserRunState(createErrorState("Select at least one gallery identity before running browser identification."));
            return;
        }
        if (!selectedBrowserProbeAsset) {
            setBrowserRunState(createErrorState("Select a probe asset from the dataset browser before running browser identification."));
            return;
        }
        if (probeDatasetMismatch) {
            setBrowserRunState(createErrorState("Probe and gallery dataset must match in Browser mode."));
            return;
        }

        const shortlistSize = parseShortlistSize(browserSearchForm.shortlistSizeText);
        if (shortlistSize === null) {
            setBrowserRunState(createErrorState("Shortlist size must be a positive integer."));
            return;
        }
        const threshold = parseThreshold(browserSearchForm.thresholdText);
        if (threshold === null) {
            setBrowserRunState(createErrorState("Threshold must be empty or a valid number."));
            return;
        }
        if (browserSearchForm.createdFrom && browserSearchForm.createdTo && browserSearchForm.createdFrom > browserSearchForm.createdTo) {
            setBrowserRunState(createErrorState("Created from must be earlier than or equal to created to."));
            return;
        }

        setBrowserRunState((current) => createLoadingState(current.data));
        setBrowserSeedState((current) => createLoadingState(current.data));
        setBrowserResultState((current) => createLoadingState(current.data));

        try {
            const seedPayload = await seedIdentificationBrowserSelection({
                dataset: browserSelectedDataset.dataset,
                selected_identity_ids: selectedBrowserGalleryIdentityIds,
                overwrite: true,
                metadata: {
                    source: "identification_browser_mode",
                    selected_probe_asset_id: selectedBrowserProbeAsset.asset_id,
                },
            });
            setBrowserSeedState(createSuccessState(seedPayload));
            if (seedPayload.errors.length > 0 || !seedPayload.store_ready) {
                const message = seedPayload.errors[0] ?? "Browser-selected gallery could not be seeded.";
                setBrowserRunState(createErrorState(message, { dataset: browserSelectedDataset.dataset }));
                setBrowserResultState(createErrorState(message));
                return;
            }

            // Browser catalog assets currently expose only "thumbnail" and "preview";
            // preview is the highest-fidelity source available for identification runs.
            const probeFile = await loadCatalogBrowserItemFile(selectedBrowserProbeAsset, "preview");
            const payload = await identifyFingerprint({
                file: probeFile,
                capture: resolveCapture(selectedBrowserProbeAsset.capture),
                retrievalMethod: browserSearchForm.retrievalMethod,
                rerankMethod: browserSearchForm.rerankMethod,
                shortlistSize,
                threshold,
                namePattern: browserSearchForm.namePattern.trim() || undefined,
                nationalIdPattern: browserSearchForm.nationalIdPattern.trim() || undefined,
                createdFrom: browserSearchForm.createdFrom || undefined,
                createdTo: browserSearchForm.createdTo || undefined,
                storeScope: "browser",
            });
            setBrowserRunState(createSuccessState({ dataset: browserSelectedDataset.dataset }));
            setBrowserResultState(createSuccessState(payload));
            setLastBrowserRunProbeAsset(selectedBrowserProbeAsset);
            setLastBrowserRunDatasetLabel(browserSelectedDataset.dataset_label);
            setNotice(
                payload.top_candidate
                    ? `Browser top candidate: ${payload.top_candidate.full_name} (${payload.top_candidate.random_id}).`
                    : "Browser identification completed without a top candidate.",
            );
        } catch (error) {
            const message = toErrorMessage(error);
            setBrowserRunState((current) => createErrorState(message, current.data));
            setBrowserResultState((current) => createErrorState(message, current.data));
            setBrowserSeedState((current) => current.status === "loading" ? createErrorState(message, current.data) : current);
        }
    }, [
        browserSearchForm,
        browserSelectedDataset,
        probeDatasetMismatch,
        selectedBrowserGalleryIdentityIds,
        selectedBrowserProbeAsset,
    ]);

    const retryDemoRun = useCallback(async (): Promise<void> => {
        if (!lastDemoRunProbeCase) {
            await runDemoIdentification();
            return;
        }
        selectProbeCase(lastDemoRunProbeCase, true);
        await runDemoIdentification(lastDemoRunProbeCase);
    }, [lastDemoRunProbeCase, runDemoIdentification, selectProbeCase]);

    const retryBrowserRun = useCallback(async (): Promise<void> => {
        await runBrowserIdentification();
    }, [runBrowserIdentification]);

    return {
        identificationMode,
        statsState,
        healthState,
        adminLayoutState,
        adminReconciliationState,
        demoGalleryState,
        demoIdentities,
        probeCases,
        demoSeedState,
        demoResetState,
        demoRunState,
        demoResultState,
        demoExpectationState,
        demoStoryState,
        demoSearchForm,
        demoStoreReady,
        browserDatasetsState,
        browserReadyDatasets,
        browserGalleryState,
        browserIdentities,
        browserFilters,
        browserItems,
        browserLoading,
        browserError,
        browserPagination,
        browserSummary,
        browserFilterOptions,
        browserActiveFilterCount,
        browserSelectedDataset,
        browserSelectedDatasetKey,
        browserSearchForm,
        browserSeedState,
        browserResetState,
        browserRunState,
        browserResultState,
        browserStoryState,
        selectedBrowserGalleryIdentityIds,
        selectedBrowserGalleryIdentities,
        selectedBrowserProbeAsset,
        lastBrowserRunProbeAsset,
        lastBrowserRunDatasetLabel,
        browserWarnings,
        probeDatasetMismatch,
        galleryTooLarge,
        isBrowserBusy,
        enrollState,
        identifyStoryState,
        pinnedProbeCaseIds,
        pinnedProbeCases,
        recentProbeCases,
        recentProbes,
        searchState,
        searchStoryState,
        deleteState,
        enrollForm,
        searchForm,
        deleteForm,
        selectedProbeCase,
        selectedProbeCaseId,
        lastDemoRunProbeCase,
        isDemoBusy,
        notice,
        selectedNarrativeCaseMetadata: identificationMode === "demo" ? lastDemoRunProbeCase : null,
        storyDerivedConfidenceState: identifyStoryState.confidenceBand,
        storyErrorState: identifyStoryState.storyErrorState,
        storyVisibilityState,
        clearPersistedWorkspaceState,
        setIdentificationMode,
        updateEnrollForm,
        updateSearchForm,
        updateDemoSearchForm,
        updateBrowserSearchForm,
        updateDeleteForm,
        selectProbeCase,
        togglePinnedProbeCase,
        selectBrowserDataset,
        updateBrowserFilters,
        resetBrowserFilters,
        toggleBrowserGalleryIdentity,
        selectBrowserProbeAsset,
        clearSelectedBrowserProbe,
        submitEnroll,
        submitSearch,
        submitDelete,
        refreshStats,
        refreshHealth,
        refreshAdminLayout,
        refreshAdminReconciliationReport,
        refreshRuntimeReadiness,
        loadDemoGallery,
        loadBrowserDatasets,
        seedDemoStore,
        resetDemoStore,
        resetBrowserStore,
        runDemoIdentification,
        runBrowserIdentification,
        retryDemoRun,
        retryBrowserRun,
        setNotice,
        setStoryVisibilityState,
        setBrowserPage: (offset: number) => updateBrowserFilters({ offset }),
    };
}
