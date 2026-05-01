import {
    CATALOG_BROWSER_SORT_VALUES,
    CAPTURE_VALUES,
    normalizeMethodValue,
    type CatalogBrowserItem,
    type Capture,
    type Method,
} from "../../types/index.ts";
import type { BrowserFilters, BrowserSelectionSide } from "./browserModel.ts";
import { createDefaultBrowserFilters } from "./browserModel.ts";
import {
    clearVerifyModeInSessionStorage,
    sanitizeVerifyMode,
    type ThresholdMode,
    type VerifyDemoFilter,
    type VerifyMode,
} from "./model.ts";
import {
    clearWorkspaceState,
    createWorkspaceStorageKey,
    readWorkspaceState,
    sanitizeBoolean,
    sanitizeFileName,
    sanitizeNumber,
    sanitizeOptionalString,
    sanitizeRelativeStorageUrl,
    sanitizeStringArray,
    writeWorkspaceState,
} from "../../shared/persistence/workspacePersistence.ts";

export interface PersistedVerifyManualPair {
    probeFileName: string | null;
    referenceFileName: string | null;
    requiresReupload: true;
}

export interface PersistedVerifyPreferences {
    method: Method;
    captureA: Capture;
    captureB: Capture;
    thresholdMode: ThresholdMode;
    thresholdText: string;
    returnOverlay: boolean;
    warmUpEnabled: boolean;
    showOutliers: boolean;
    showTentative: boolean;
    maxMatchesText: string;
}

export interface PersistedVerifyBrowserState {
    selectedDatasetKey: string | null;
    filters: BrowserFilters;
    selectedAssetA: CatalogBrowserItem | null;
    selectedAssetB: CatalogBrowserItem | null;
    replacementTarget: BrowserSelectionSide | null;
}

export interface PersistedVerifyWorkspaceState {
    mode: VerifyMode;
    demoFilter: VerifyDemoFilter;
    selectedDemoCaseId: string | null;
    pinnedDemoCaseIds: string[];
    browser: PersistedVerifyBrowserState;
    manualPair: PersistedVerifyManualPair | null;
    preferences: PersistedVerifyPreferences;
}

export interface PersistedVerifySessionState {
    storyVisibility: {
        rawDetailsExpanded: boolean;
    };
}

const VERIFY_LOCAL_STORAGE_KEY = createWorkspaceStorageKey("verify", "local");
const VERIFY_SESSION_STORAGE_KEY = createWorkspaceStorageKey("verify", "session");
const VERIFY_DEMO_FILTER_VALUES: VerifyDemoFilter[] = ["all", "easy", "hard", "genuine", "impostor"];

function sanitizeMethod(value: unknown, fallback: Method): Method {
    return normalizeMethodValue(value) ?? fallback;
}

function sanitizeCapture(value: unknown, fallback: Capture): Capture {
    return CAPTURE_VALUES.includes(value as Capture) ? (value as Capture) : fallback;
}

function sanitizeThresholdMode(value: unknown): ThresholdMode {
    return value === "custom" ? "custom" : "default";
}

function sanitizeVerifyDemoFilter(value: unknown): VerifyDemoFilter {
    return VERIFY_DEMO_FILTER_VALUES.includes(value as VerifyDemoFilter) ? (value as VerifyDemoFilter) : "all";
}

function sanitizeBrowserSelectionSide(value: unknown): BrowserSelectionSide | null {
    return value === "A" || value === "B" ? value : null;
}

function sanitizeBrowserFilters(value: unknown): BrowserFilters {
    const defaults = createDefaultBrowserFilters();

    if (typeof value !== "object" || value === null) {
        return defaults;
    }

    const record = value as Record<string, unknown>;
    const nextLimit = sanitizeNumber(record.limit, defaults.limit, 1, 96);

    return {
        split: sanitizeOptionalString(record.split, 80) ?? "",
        capture: sanitizeOptionalString(record.capture, 80) ?? "",
        modality: sanitizeOptionalString(record.modality, 80) ?? "",
        subjectId: sanitizeOptionalString(record.subjectId, 80) ?? "",
        finger: sanitizeOptionalString(record.finger, 40) ?? "",
        uiEligible: record.uiEligible === "eligible" || record.uiEligible === "ineligible" ? record.uiEligible : "all",
        limit: nextLimit,
        offset: sanitizeNumber(record.offset, 0, 0),
        sort: CATALOG_BROWSER_SORT_VALUES.includes(record.sort as BrowserFilters["sort"]) ? (record.sort as BrowserFilters["sort"]) : defaults.sort,
    };
}

function sanitizeDimensions(value: unknown): CatalogBrowserItem["original_dimensions"] | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;
    const width = sanitizeNumber(record.width, 0, 0);
    const height = sanitizeNumber(record.height, 0, 0);

    if (width === 0 && height === 0) {
        return null;
    }

    return { width, height };
}

function sanitizeCatalogBrowserItem(value: unknown): CatalogBrowserItem | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;
    const assetId = sanitizeOptionalString(record.asset_id, 160);
    const dataset = sanitizeOptionalString(record.dataset, 120);
    const split = sanitizeOptionalString(record.split, 80);
    const selectionReason = sanitizeOptionalString(record.selection_reason, 240);
    const selectionPolicy = sanitizeOptionalString(record.selection_policy, 120);
    const thumbnailUrl = sanitizeRelativeStorageUrl(record.thumbnail_url);
    const previewUrl = sanitizeRelativeStorageUrl(record.preview_url);
    const availabilityStatus = sanitizeOptionalString(record.availability_status, 80);
    const originalDimensions = sanitizeDimensions(record.original_dimensions);
    const thumbnailDimensions = sanitizeDimensions(record.thumbnail_dimensions);
    const previewDimensions = sanitizeDimensions(record.preview_dimensions);

    if (
        !assetId
        || !dataset
        || !split
        || !selectionReason
        || !selectionPolicy
        || !thumbnailUrl
        || !previewUrl
        || !availabilityStatus
        || !originalDimensions
        || !thumbnailDimensions
        || !previewDimensions
    ) {
        return null;
    }

    return {
        asset_id: assetId,
        dataset,
        split,
        subject_id: sanitizeOptionalString(record.subject_id, 80),
        finger: sanitizeOptionalString(record.finger, 40),
        capture: sanitizeOptionalString(record.capture, 80),
        modality: sanitizeOptionalString(record.modality, 80),
        ui_eligible: sanitizeBoolean(record.ui_eligible, false),
        selection_reason: selectionReason,
        selection_policy: selectionPolicy,
        thumbnail_url: thumbnailUrl,
        preview_url: previewUrl,
        availability_status: availabilityStatus,
        original_dimensions: originalDimensions,
        thumbnail_dimensions: thumbnailDimensions,
        preview_dimensions: previewDimensions,
    };
}

function sanitizeManualPair(value: unknown): PersistedVerifyManualPair | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;
    const probeFileName = sanitizeFileName(record.probeFileName);
    const referenceFileName = sanitizeFileName(record.referenceFileName);

    if (!probeFileName && !referenceFileName) {
        return null;
    }

    return {
        probeFileName,
        referenceFileName,
        requiresReupload: true,
    };
}

function sanitizeVerifyPreferences(value: unknown): PersistedVerifyPreferences | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;

    return {
        method: sanitizeMethod(record.method, "vit"),
        captureA: sanitizeCapture(record.captureA, "plain"),
        captureB: sanitizeCapture(record.captureB, "plain"),
        thresholdMode: sanitizeThresholdMode(record.thresholdMode),
        thresholdText: sanitizeOptionalString(record.thresholdText, 40) ?? "",
        returnOverlay: sanitizeBoolean(record.returnOverlay, false),
        warmUpEnabled: sanitizeBoolean(record.warmUpEnabled, true),
        showOutliers: sanitizeBoolean(record.showOutliers, true),
        showTentative: sanitizeBoolean(record.showTentative, true),
        maxMatchesText: sanitizeOptionalString(record.maxMatchesText, 12) ?? "100",
    };
}

function sanitizeVerifyBrowserState(value: unknown): PersistedVerifyBrowserState | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;

    return {
        selectedDatasetKey: sanitizeOptionalString(record.selectedDatasetKey, 120),
        filters: sanitizeBrowserFilters(record.filters),
        selectedAssetA: sanitizeCatalogBrowserItem(record.selectedAssetA),
        selectedAssetB: sanitizeCatalogBrowserItem(record.selectedAssetB),
        replacementTarget: sanitizeBrowserSelectionSide(record.replacementTarget),
    };
}

function sanitizeVerifyWorkspaceState(value: unknown): PersistedVerifyWorkspaceState | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;
    const browser = sanitizeVerifyBrowserState(record.browser);
    const preferences = sanitizeVerifyPreferences(record.preferences);

    if (!browser || !preferences) {
        return null;
    }

    return {
        mode: sanitizeVerifyMode(record.mode as string | null | undefined),
        demoFilter: sanitizeVerifyDemoFilter(record.demoFilter),
        selectedDemoCaseId: sanitizeOptionalString(record.selectedDemoCaseId, 160),
        pinnedDemoCaseIds: sanitizeStringArray(record.pinnedDemoCaseIds, 8, 160),
        browser,
        manualPair: sanitizeManualPair(record.manualPair),
        preferences,
    };
}

function sanitizeVerifySessionState(value: unknown): PersistedVerifySessionState | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;
    const storyVisibility = typeof record.storyVisibility === "object" && record.storyVisibility !== null
        ? record.storyVisibility as Record<string, unknown>
        : null;

    return {
        storyVisibility: {
            rawDetailsExpanded: sanitizeBoolean(storyVisibility?.rawDetailsExpanded, false),
        },
    };
}

export function readPersistedVerifyWorkspaceState(): PersistedVerifyWorkspaceState | null {
    return readWorkspaceState("local", VERIFY_LOCAL_STORAGE_KEY, sanitizeVerifyWorkspaceState);
}

export function writePersistedVerifyWorkspaceState(state: PersistedVerifyWorkspaceState): boolean {
    const sanitized = sanitizeVerifyWorkspaceState(state);
    if (!sanitized) {
        clearPersistedVerifyWorkspaceState();
        return false;
    }

    return writeWorkspaceState("local", VERIFY_LOCAL_STORAGE_KEY, sanitized);
}

export function clearPersistedVerifyWorkspaceState(): void {
    clearWorkspaceState("local", VERIFY_LOCAL_STORAGE_KEY);
}

export function readPersistedVerifySessionState(): PersistedVerifySessionState | null {
    return readWorkspaceState("session", VERIFY_SESSION_STORAGE_KEY, sanitizeVerifySessionState);
}

export function writePersistedVerifySessionState(state: PersistedVerifySessionState): boolean {
    const sanitized = sanitizeVerifySessionState(state);
    if (!sanitized) {
        clearPersistedVerifySessionState();
        return false;
    }

    return writeWorkspaceState("session", VERIFY_SESSION_STORAGE_KEY, sanitized);
}

export function clearPersistedVerifySessionState(): void {
    clearWorkspaceState("session", VERIFY_SESSION_STORAGE_KEY);
}

export function clearAllPersistedVerifyState(): void {
    clearPersistedVerifyWorkspaceState();
    clearPersistedVerifySessionState();
    clearVerifyModeInSessionStorage();
}
