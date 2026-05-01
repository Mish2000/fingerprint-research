import {
    CATALOG_BROWSER_SORT_VALUES,
    CAPTURE_VALUES,
    IDENTIFICATION_RETRIEVAL_METHOD_VALUES,
    normalizeMethodValue,
    type CatalogBrowserItem,
    type Capture,
    type IdentificationRetrievalMethod,
    type Method,
} from "../../types/index.ts";
import { clearIdentificationModeInSessionStorage, type IdentificationMode } from "./model.ts";
import type { BrowserFilters } from "../verify/browserModel.ts";
import { createDefaultBrowserFilters } from "../verify/browserModel.ts";
import {
    clearWorkspaceState,
    createWorkspaceStorageKey,
    readWorkspaceState,
    sanitizeBoolean,
    sanitizeNumber,
    sanitizeOptionalString,
    sanitizeRelativeStorageUrl,
    sanitizeStringArray,
    writeWorkspaceState,
} from "../../shared/persistence/workspacePersistence.ts";

export interface PersistedProbeEntry {
    id: string;
    title: string;
    datasetLabel: string;
    expectedOutcome: string | null;
}

export interface PersistedIdentificationDemoSearchPreferences {
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

export interface PersistedIdentificationBrowserSearchPreferences {
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

export interface PersistedIdentificationOperationalSearchPreferences {
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

export interface PersistedIdentificationBrowserState {
    selectedDatasetKey: string | null;
    filters: BrowserFilters;
    selectedGalleryIdentityIds: string[];
    selectedProbeAsset: CatalogBrowserItem | null;
}

export interface PersistedIdentificationWorkspaceState {
    mode: IdentificationMode;
    selectedProbeCaseId: string | null;
    recentProbes: PersistedProbeEntry[];
    pinnedProbeCaseIds: string[];
    demoSearchPreferences: PersistedIdentificationDemoSearchPreferences;
    browserSearchPreferences: PersistedIdentificationBrowserSearchPreferences;
    browser: PersistedIdentificationBrowserState;
    operationalSearchPreferences: PersistedIdentificationOperationalSearchPreferences;
}

const IDENTIFICATION_LOCAL_STORAGE_KEY = createWorkspaceStorageKey("identification", "local");

function sanitizeIdentificationMode(value: unknown): IdentificationMode {
    return value === "operational" || value === "browser" ? value : "demo";
}

function sanitizeRetrievalMethod(
    value: unknown,
    fallback: IdentificationRetrievalMethod,
): IdentificationRetrievalMethod {
    return IDENTIFICATION_RETRIEVAL_METHOD_VALUES.includes(value as IdentificationRetrievalMethod)
        ? (value as IdentificationRetrievalMethod)
        : fallback;
}

function sanitizeMethod(value: unknown, fallback: Method): Method {
    return normalizeMethodValue(value) ?? fallback;
}

function sanitizeCapture(value: unknown, fallback: Capture): Capture {
    return CAPTURE_VALUES.includes(value as Capture) ? (value as Capture) : fallback;
}

function sanitizeProbeEntry(value: unknown): PersistedProbeEntry | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;
    const id = sanitizeOptionalString(record.id, 160);
    const title = sanitizeOptionalString(record.title, 160);
    const datasetLabel = sanitizeOptionalString(record.datasetLabel, 120);

    if (!id || !title || !datasetLabel) {
        return null;
    }

    return {
        id,
        title,
        datasetLabel,
        expectedOutcome: sanitizeOptionalString(record.expectedOutcome, 40),
    };
}

function sanitizeRecentProbeEntries(value: unknown): PersistedProbeEntry[] {
    if (!Array.isArray(value)) {
        return [];
    }

    const entries: PersistedProbeEntry[] = [];
    const seen = new Set<string>();

    for (const item of value) {
        const probe = sanitizeProbeEntry(item);
        if (!probe || seen.has(probe.id)) {
            continue;
        }

        seen.add(probe.id);
        entries.push(probe);

        if (entries.length >= 6) {
            break;
        }
    }

    return entries;
}

function sanitizeSearchPreferences(
    value: unknown,
    defaultShortlist: string,
): PersistedIdentificationDemoSearchPreferences | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;

    return {
        retrievalMethod: sanitizeRetrievalMethod(record.retrievalMethod, "dl"),
        rerankMethod: sanitizeMethod(record.rerankMethod, "sift"),
        shortlistSizeText: sanitizeOptionalString(record.shortlistSizeText, 12) ?? defaultShortlist,
        thresholdText: sanitizeOptionalString(record.thresholdText, 32) ?? "",
        advancedVisible: sanitizeBoolean(record.advancedVisible, false),
        namePattern: sanitizeOptionalString(record.namePattern, 80) ?? "",
        nationalIdPattern: sanitizeOptionalString(record.nationalIdPattern, 80) ?? "",
        createdFrom: sanitizeOptionalString(record.createdFrom, 20) ?? "",
        createdTo: sanitizeOptionalString(record.createdTo, 20) ?? "",
    };
}

function sanitizeDemoSearchPreferences(value: unknown): PersistedIdentificationDemoSearchPreferences | null {
    return sanitizeSearchPreferences(value, "10");
}

function sanitizeBrowserSearchPreferences(value: unknown): PersistedIdentificationBrowserSearchPreferences | null {
    return sanitizeSearchPreferences(value, "10");
}

function sanitizeOperationalSearchPreferences(value: unknown): PersistedIdentificationOperationalSearchPreferences | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;

    return {
        capture: sanitizeCapture(record.capture, "plain"),
        retrievalMethod: sanitizeRetrievalMethod(record.retrievalMethod, "dl"),
        rerankMethod: sanitizeMethod(record.rerankMethod, "sift"),
        shortlistSizeText: sanitizeOptionalString(record.shortlistSizeText, 12) ?? "25",
        thresholdText: sanitizeOptionalString(record.thresholdText, 32) ?? "",
        namePattern: sanitizeOptionalString(record.namePattern, 80) ?? "",
        nationalIdPattern: sanitizeOptionalString(record.nationalIdPattern, 80) ?? "",
        createdFrom: sanitizeOptionalString(record.createdFrom, 20) ?? "",
        createdTo: sanitizeOptionalString(record.createdTo, 20) ?? "",
    };
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

function sanitizeBrowserState(value: unknown): PersistedIdentificationBrowserState | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;

    return {
        selectedDatasetKey: sanitizeOptionalString(record.selectedDatasetKey, 120),
        filters: sanitizeBrowserFilters(record.filters),
        selectedGalleryIdentityIds: sanitizeStringArray(record.selectedGalleryIdentityIds, 32, 160),
        selectedProbeAsset: sanitizeCatalogBrowserItem(record.selectedProbeAsset),
    };
}

function sanitizeIdentificationWorkspaceState(value: unknown): PersistedIdentificationWorkspaceState | null {
    if (typeof value !== "object" || value === null) {
        return null;
    }

    const record = value as Record<string, unknown>;
    const demoSearchPreferences = sanitizeDemoSearchPreferences(record.demoSearchPreferences);
    const browserSearchPreferences = sanitizeBrowserSearchPreferences(record.browserSearchPreferences);
    const browser = sanitizeBrowserState(record.browser);
    const operationalSearchPreferences = sanitizeOperationalSearchPreferences(record.operationalSearchPreferences);

    if (!demoSearchPreferences || !browserSearchPreferences || !browser || !operationalSearchPreferences) {
        return null;
    }

    return {
        mode: sanitizeIdentificationMode(record.mode),
        selectedProbeCaseId: sanitizeOptionalString(record.selectedProbeCaseId, 160),
        recentProbes: sanitizeRecentProbeEntries(record.recentProbes),
        pinnedProbeCaseIds: sanitizeStringArray(record.pinnedProbeCaseIds, 8, 160),
        demoSearchPreferences,
        browserSearchPreferences,
        browser,
        operationalSearchPreferences,
    };
}

export function readPersistedIdentificationWorkspaceState(): PersistedIdentificationWorkspaceState | null {
    return readWorkspaceState("local", IDENTIFICATION_LOCAL_STORAGE_KEY, sanitizeIdentificationWorkspaceState);
}

export function writePersistedIdentificationWorkspaceState(state: PersistedIdentificationWorkspaceState): boolean {
    const sanitized = sanitizeIdentificationWorkspaceState(state);
    if (!sanitized) {
        clearPersistedIdentificationWorkspaceState();
        return false;
    }

    return writeWorkspaceState("local", IDENTIFICATION_LOCAL_STORAGE_KEY, sanitized);
}

export function clearPersistedIdentificationWorkspaceState(): void {
    clearWorkspaceState("local", IDENTIFICATION_LOCAL_STORAGE_KEY);
}

export function clearAllPersistedIdentificationState(): void {
    clearPersistedIdentificationWorkspaceState();
    clearIdentificationModeInSessionStorage();
}
