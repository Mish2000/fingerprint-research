import { describe, expect, it } from "vitest";
import {
    MAX_PERSISTED_STATE_BYTES,
    createWorkspaceStorageKey,
    readWorkspaceState,
    writeWorkspaceState,
} from "../src/shared/persistence/workspacePersistence.ts";
import { IDENTIFICATION_MODE_KEY } from "../src/features/identification/model.ts";
import { clearAllPersistedIdentificationState } from "../src/features/identification/persistence.ts";
import { VERIFY_MODE_SESSION_KEY } from "../src/features/verify/model.ts";
import {
    clearAllPersistedVerifyState,
    readPersistedVerifyWorkspaceState,
    writePersistedVerifyWorkspaceState,
    type PersistedVerifyWorkspaceState,
} from "../src/features/verify/persistence.ts";
import {
    APP_LAST_TAB_STORAGE_KEY,
    APP_PREFERENCES_STORAGE_KEY,
    APP_PREFERENCES_SCHEMA_VERSION,
    clearPersistedLastActiveTab,
    readPersistedAppPreferences,
    readPersistedLastActiveTab,
    writePersistedAppPreferences,
    writePersistedLastActiveTab,
} from "../src/shared/preferences/persistence.ts";
import { createDefaultAppPreferences } from "../src/shared/preferences/defaults.ts";
import type { AppPreferences } from "../src/shared/preferences/types.ts";

function createVerifyPersistenceState(): PersistedVerifyWorkspaceState {
    return {
        mode: "manual",
        demoFilter: "all",
        selectedDemoCaseId: "case_easy_match",
        pinnedDemoCaseIds: ["case_easy_match"],
        browser: {
            selectedDatasetKey: "nist_sd300b",
            filters: {
                split: "",
                capture: "",
                modality: "",
                subjectId: "",
                finger: "",
                uiEligible: "all",
                limit: 48,
                offset: 0,
                sort: "default",
            },
            selectedAssetA: null,
            selectedAssetB: null,
            replacementTarget: null,
        },
        manualPair: {
            probeFileName: "probe.png",
            referenceFileName: "reference.png",
            requiresReupload: true,
        },
        preferences: {
            method: "vit",
            captureA: "plain",
            captureB: "plain",
            thresholdMode: "default",
            thresholdText: "0.5",
            returnOverlay: false,
            warmUpEnabled: true,
            showOutliers: true,
            showTentative: true,
            maxMatchesText: "100",
        },
    };
}

describe("workspace persistence infrastructure", () => {
    it("rejects schema mismatches and corrupted JSON safely", () => {
        const verifyKey = createWorkspaceStorageKey("verify", "local");

        localStorage.setItem(verifyKey, JSON.stringify({
            version: 999,
            savedAt: "2026-04-02T00:00:00Z",
            data: createVerifyPersistenceState(),
        }));

        expect(readPersistedVerifyWorkspaceState()).toBeNull();
        expect(localStorage.getItem(verifyKey)).toBeNull();

        localStorage.setItem(verifyKey, "{bad json");
        expect(readPersistedVerifyWorkspaceState()).toBeNull();
        expect(localStorage.getItem(verifyKey)).toBeNull();
    });

    it("sanitizes manual file names so local paths are not persisted", () => {
        const verifyKey = createWorkspaceStorageKey("verify", "local");
        const payload = createVerifyPersistenceState();

        payload.manualPair = {
            probeFileName: "C:\\Users\\demo\\probe.png",
            referenceFileName: "D:\\Temp\\reference.png",
            requiresReupload: true,
        };

        expect(writePersistedVerifyWorkspaceState(payload)).toBe(true);

        const raw = localStorage.getItem(verifyKey);
        expect(raw).toContain("probe.png");
        expect(raw).toContain("reference.png");
        expect(raw).not.toContain("C:\\\\Users");
        expect(raw).not.toContain("D:\\\\Temp");
    });

    it("drops oversized payloads instead of persisting unnecessary state", () => {
        const oversizedKey = "workspace-persistence-oversized";
        const didPersist = writeWorkspaceState("local", oversizedKey, {
            text: "x".repeat(MAX_PERSISTED_STATE_BYTES * 2),
        });

        expect(didPersist).toBe(false);
        expect(localStorage.getItem(oversizedKey)).toBeNull();
    });

    it("lets callers safely sanitize arbitrary raw payloads", () => {
        const genericKey = "workspace-persistence-generic";

        writeWorkspaceState("local", genericKey, {
            keep: "value",
            drop: { nested: true },
        });

        const parsed = readWorkspaceState("local", genericKey, (value) => {
            if (typeof value !== "object" || value === null) {
                return null;
            }

            const record = value as Record<string, unknown>;
            return typeof record.keep === "string" ? { keep: record.keep } : null;
        });

        expect(parsed).toEqual({ keep: "value" });
    });

    it("clears saved workspace state without touching global preferences", () => {
        const verifyLocalKey = createWorkspaceStorageKey("verify", "local");
        const verifySessionKey = createWorkspaceStorageKey("verify", "session");
        const identificationLocalKey = createWorkspaceStorageKey("identification", "local");
        const preferences: AppPreferences = {
            ...createDefaultAppPreferences(),
            theme: "dark",
            defaultTab: "identify",
        };

        localStorage.setItem(verifyLocalKey, "verify local");
        sessionStorage.setItem(verifySessionKey, "verify session");
        sessionStorage.setItem(VERIFY_MODE_SESSION_KEY, "browser");
        localStorage.setItem(identificationLocalKey, "identification local");
        sessionStorage.setItem(IDENTIFICATION_MODE_KEY, "operational");
        writePersistedLastActiveTab("identify");
        writePersistedAppPreferences(preferences);

        clearAllPersistedVerifyState();
        clearAllPersistedIdentificationState();
        clearPersistedLastActiveTab();

        expect(localStorage.getItem(verifyLocalKey)).toBeNull();
        expect(sessionStorage.getItem(verifySessionKey)).toBeNull();
        expect(sessionStorage.getItem(VERIFY_MODE_SESSION_KEY)).toBeNull();
        expect(localStorage.getItem(identificationLocalKey)).toBeNull();
        expect(sessionStorage.getItem(IDENTIFICATION_MODE_KEY)).toBeNull();
        expect(readPersistedLastActiveTab()).toBeNull();
        expect(readPersistedAppPreferences()).toEqual(preferences);
    });
});

describe("global preferences persistence", () => {
    it("round-trips a versioned preferences payload", () => {
        const preferences: AppPreferences = {
            theme: "dark",
            language: "he",
            density: "compact",
            defaultTab: "verify",
            rememberLastTab: false,
            reducedMotion: true,
        };

        expect(writePersistedAppPreferences(preferences)).toBe(true);

        const raw = localStorage.getItem(APP_PREFERENCES_STORAGE_KEY);
        expect(raw).toContain(`"version":${APP_PREFERENCES_SCHEMA_VERSION}`);
        expect(readPersistedAppPreferences()).toEqual(preferences);
    });

    it("sanitizes invalid preference fields back to defaults", () => {
        localStorage.setItem(APP_PREFERENCES_STORAGE_KEY, JSON.stringify({
            version: APP_PREFERENCES_SCHEMA_VERSION,
            savedAt: "2026-04-07T00:00:00Z",
            data: {
                theme: "midnight",
                language: "fr",
                density: "dense",
                defaultTab: "contracts",
                rememberLastTab: "yes",
                reducedMotion: "no",
            },
        }));

        expect(readPersistedAppPreferences()).toEqual(createDefaultAppPreferences());
        expect(localStorage.getItem(APP_PREFERENCES_STORAGE_KEY)).not.toBeNull();
    });

    it("removes corrupt preference and remembered-tab payloads safely", () => {
        localStorage.setItem(APP_PREFERENCES_STORAGE_KEY, "{bad json");
        expect(readPersistedAppPreferences()).toBeNull();
        expect(localStorage.getItem(APP_PREFERENCES_STORAGE_KEY)).toBeNull();

        localStorage.setItem(APP_LAST_TAB_STORAGE_KEY, JSON.stringify({
            version: APP_PREFERENCES_SCHEMA_VERSION,
            savedAt: "2026-04-07T00:00:00Z",
            data: "contracts",
        }));
        expect(readPersistedLastActiveTab()).toBeNull();
        expect(localStorage.getItem(APP_LAST_TAB_STORAGE_KEY)).toBeNull();
    });
});
