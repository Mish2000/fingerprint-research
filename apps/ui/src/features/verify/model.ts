import type { CatalogBrowserItem, CatalogVerifyCase, Capture, Method } from "../../types/index.ts";
import { METHOD_PROFILES, formatThresholdValue } from "./config.ts";

export type VerifyMode = "demo" | "manual" | "browser";
export type VerifyDemoFilter = "all" | "easy" | "hard" | "genuine" | "impostor";
export type ThresholdMode = "default" | "custom";

export interface DemoFormSelection {
    method: Method;
    captureA: Capture;
    captureB: Capture;
    thresholdMode: ThresholdMode;
    thresholdText: string;
}

export interface DemoRunConfiguration {
    method: Method;
    captureA: Capture;
    captureB: Capture;
    thresholdText: string;
    returnOverlay: boolean;
}

export interface VerifyRunContext {
    mode: VerifyMode;
    method: Method;
    title: string;
    subtitle: string;
    description?: string | null;
    demoCaseId?: string | null;
    datasetLabel?: string | null;
    split?: string | null;
    recommendedMethod?: Method | null;
    probeFileName?: string | null;
    referenceFileName?: string | null;
    assetAId?: string | null;
    assetBId?: string | null;
    captureA?: Capture | null;
    captureB?: Capture | null;
    difficulty?: string | null;
    groundTruth?: string | null;
    caseType?: string | null;
    selectionReason?: string | null;
    modalityRelation?: string | null;
    tags?: string[];
}

export const VERIFY_MODE_SESSION_KEY = "fp-research.verify.mode";
export const VERIFY_DEMO_FILTERS: Array<{ value: VerifyDemoFilter; label: string }> = [
    { value: "all", label: "All cases" },
    { value: "easy", label: "Easy" },
    { value: "hard", label: "Hard" },
    { value: "genuine", label: "Genuine" },
    { value: "impostor", label: "Impostor" },
];

function formatWords(value: string): string {
    return value
        .split(/[_-]/g)
        .filter(Boolean)
        .map((part) => part[0].toUpperCase() + part.slice(1))
        .join(" ");
}

export function sanitizeVerifyMode(value: string | null | undefined): VerifyMode {
    if (value === "manual" || value === "browser") {
        return value;
    }
    return "demo";
}

export function readVerifyModeFromSessionStorage(): VerifyMode {
    if (typeof window === "undefined") {
        return "demo";
    }

    try {
        return sanitizeVerifyMode(window.sessionStorage.getItem(VERIFY_MODE_SESSION_KEY));
    } catch {
        return "demo";
    }
}

export function persistVerifyModeInSessionStorage(mode: VerifyMode): void {
    if (typeof window === "undefined") {
        return;
    }

    try {
        window.sessionStorage.setItem(VERIFY_MODE_SESSION_KEY, mode);
    } catch {
        // Ignore session-storage failures and keep the UI usable.
    }
}

export function clearVerifyModeInSessionStorage(): void {
    if (typeof window === "undefined") {
        return;
    }

    try {
        window.sessionStorage.removeItem(VERIFY_MODE_SESSION_KEY);
    } catch {
        // Ignore session-storage failures and keep the UI usable.
    }
}

export function filterDemoCases(cases: CatalogVerifyCase[], filter: VerifyDemoFilter): CatalogVerifyCase[] {
    switch (filter) {
        case "easy":
            return cases.filter((item) => item.difficulty.toLowerCase() === "easy");
        case "hard":
            return cases.filter((item) => item.difficulty.toLowerCase() === "hard");
        case "genuine":
            return cases.filter((item) => item.ground_truth.toLowerCase() === "match");
        case "impostor":
            return cases.filter((item) => item.ground_truth.toLowerCase() === "non_match");
        case "all":
        default:
            return cases;
    }
}

export function formatGroundTruthLabel(value: string): string {
    const normalized = value.toLowerCase();
    if (normalized === "match") {
        return "Genuine";
    }
    if (normalized === "non_match") {
        return "Impostor";
    }
    return formatWords(value);
}

export function formatCaseMetadataLabel(value: string | null | undefined): string {
    if (!value) {
        return "-";
    }
    return formatWords(value);
}

function toOptionalCapture(value: string | null | undefined): Capture | null {
    return value === "plain" || value === "roll" || value === "contactless" || value === "contact_based"
        ? value
        : null;
}

export function buildDemoRunConfiguration(
    currentSelection: DemoFormSelection,
    demoCase: CatalogVerifyCase,
    preserveUserChoices: boolean,
): DemoRunConfiguration {
    const method = preserveUserChoices ? currentSelection.method : demoCase.recommended_method;
    const captureA = preserveUserChoices ? currentSelection.captureA : demoCase.capture_a;
    const captureB = preserveUserChoices ? currentSelection.captureB : demoCase.capture_b;

    return {
        method,
        captureA,
        captureB,
        thresholdText:
            currentSelection.thresholdMode === "default"
                ? formatThresholdValue(METHOD_PROFILES[method].defaultThreshold)
                : currentSelection.thresholdText,
        returnOverlay: METHOD_PROFILES[method].supportsOverlay,
    };
}

export function createDemoRunContext(demoCase: CatalogVerifyCase, method: Method): VerifyRunContext {
    return {
        mode: "demo",
        method,
        title: demoCase.title,
        subtitle: `${demoCase.dataset_label} / ${demoCase.split} / ${formatGroundTruthLabel(demoCase.ground_truth)}`,
        description: demoCase.description,
        demoCaseId: demoCase.case_id,
        datasetLabel: demoCase.dataset_label,
        split: demoCase.split,
        recommendedMethod: demoCase.recommended_method,
        captureA: demoCase.capture_a,
        captureB: demoCase.capture_b,
        difficulty: demoCase.difficulty,
        groundTruth: demoCase.ground_truth,
        caseType: demoCase.case_type,
        selectionReason: demoCase.selection_reason,
        modalityRelation: demoCase.modality_relation ?? null,
        tags: [...demoCase.tags],
    };
}

export function createManualRunContext(
    probeFileName: string | null | undefined,
    referenceFileName: string | null | undefined,
    method: Method,
    captureA?: Capture | null,
    captureB?: Capture | null,
): VerifyRunContext {
    return {
        mode: "manual",
        method,
        title: "Manual upload",
        subtitle: "User-provided probe/reference pair",
        probeFileName: probeFileName ?? null,
        referenceFileName: referenceFileName ?? null,
        captureA: captureA ?? null,
        captureB: captureB ?? null,
        tags: [],
    };
}

export function createBrowserRunContext(
    assetA: CatalogBrowserItem,
    assetB: CatalogBrowserItem,
    method: Method,
    datasetLabel: string,
): VerifyRunContext {
    return {
        mode: "browser",
        method,
        title: `${datasetLabel} browser pair`,
        subtitle: `${assetA.asset_id} vs ${assetB.asset_id}`,
        datasetLabel,
        split: `${assetA.split} -> ${assetB.split}`,
        assetAId: assetA.asset_id,
        assetBId: assetB.asset_id,
        captureA: toOptionalCapture(assetA.capture),
        captureB: toOptionalCapture(assetB.capture),
        probeFileName: `${assetA.asset_id}_preview.png`,
        referenceFileName: `${assetB.asset_id}_preview.png`,
        tags: [],
    };
}

