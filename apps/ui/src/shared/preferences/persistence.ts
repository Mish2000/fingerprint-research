import { isObject } from "../../utils/error.ts";
import { sanitizeBoolean } from "../persistence/workspacePersistence.ts";
import { createDefaultAppPreferences } from "./defaults.ts";
import {
    APP_DENSITY_VALUES,
    APP_LANGUAGE_VALUES,
    APP_TAB_VALUES,
    APP_THEME_VALUES,
    type AppDensityPreference,
    type AppLanguage,
    type AppPreferences,
    type AppTabPreference,
    type AppThemePreference,
} from "./types.ts";

export const APP_PREFERENCES_STORAGE_KEY = "fp-research.preferences.local";
export const APP_LAST_TAB_STORAGE_KEY = "fp-research.preferences.last-tab";
export const APP_PREFERENCES_SCHEMA_VERSION = 1;
const MAX_PREFERENCES_PAYLOAD_BYTES = 4_000;

interface PersistedEnvelope<T> {
    version: number;
    savedAt: string;
    data: T;
}

function getLocalStorage(): Storage | null {
    if (typeof window === "undefined") {
        return null;
    }

    try {
        return window.localStorage;
    } catch {
        return null;
    }
}

function measureStringSize(value: string): number {
    if (typeof TextEncoder !== "undefined") {
        return new TextEncoder().encode(value).length;
    }

    return value.length;
}

function sanitizeEnum<T extends string>(value: unknown, values: readonly T[], fallback: T): T {
    return values.includes(value as T) ? (value as T) : fallback;
}

function sanitizeTheme(value: unknown, fallback: AppThemePreference): AppThemePreference {
    return sanitizeEnum(value, APP_THEME_VALUES, fallback);
}

function sanitizeLanguage(value: unknown, fallback: AppLanguage): AppLanguage {
    return sanitizeEnum(value, APP_LANGUAGE_VALUES, fallback);
}

function sanitizeDensity(value: unknown, fallback: AppDensityPreference): AppDensityPreference {
    return sanitizeEnum(value, APP_DENSITY_VALUES, fallback);
}

export function sanitizeAppTab(value: unknown, fallback: AppTabPreference): AppTabPreference {
    return sanitizeEnum(value, APP_TAB_VALUES, fallback);
}

function sanitizeOptionalAppTab(value: unknown): AppTabPreference | null {
    return APP_TAB_VALUES.includes(value as AppTabPreference) ? (value as AppTabPreference) : null;
}

export function sanitizeAppPreferences(value: unknown): AppPreferences | null {
    if (!isObject(value)) {
        return null;
    }

    const defaults = createDefaultAppPreferences();

    return {
        theme: sanitizeTheme(value.theme, defaults.theme),
        language: sanitizeLanguage(value.language, defaults.language),
        density: sanitizeDensity(value.density, defaults.density),
        defaultTab: sanitizeAppTab(value.defaultTab, defaults.defaultTab),
        rememberLastTab: sanitizeBoolean(value.rememberLastTab, defaults.rememberLastTab),
        reducedMotion: sanitizeBoolean(value.reducedMotion, defaults.reducedMotion),
    };
}

function readEnvelope<T>(
    key: string,
    sanitize: (value: unknown) => T | null,
): T | null {
    const storage = getLocalStorage();
    if (!storage) {
        return null;
    }

    try {
        const raw = storage.getItem(key);
        if (!raw) {
            return null;
        }

        const parsed = JSON.parse(raw) as unknown;
        if (!isObject(parsed) || parsed.version !== APP_PREFERENCES_SCHEMA_VERSION || !("data" in parsed)) {
            storage.removeItem(key);
            return null;
        }

        const sanitized = sanitize(parsed.data);
        if (sanitized === null) {
            storage.removeItem(key);
            return null;
        }

        return sanitized;
    } catch {
        try {
            storage.removeItem(key);
        } catch {
            // Ignore cleanup failures after a bad payload.
        }
        return null;
    }
}

function writeEnvelope<T>(key: string, data: T): boolean {
    const storage = getLocalStorage();
    if (!storage) {
        return false;
    }

    const payload: PersistedEnvelope<T> = {
        version: APP_PREFERENCES_SCHEMA_VERSION,
        savedAt: new Date().toISOString(),
        data,
    };

    try {
        const serialized = JSON.stringify(payload);
        if (!serialized || measureStringSize(serialized) > MAX_PREFERENCES_PAYLOAD_BYTES) {
            storage.removeItem(key);
            return false;
        }

        storage.setItem(key, serialized);
        return true;
    } catch {
        return false;
    }
}

function clearStorageKey(key: string): void {
    const storage = getLocalStorage();
    if (!storage) {
        return;
    }

    try {
        storage.removeItem(key);
    } catch {
        // Ignore storage cleanup failures.
    }
}

export function readPersistedAppPreferences(): AppPreferences | null {
    return readEnvelope(APP_PREFERENCES_STORAGE_KEY, sanitizeAppPreferences);
}

export function writePersistedAppPreferences(preferences: AppPreferences): boolean {
    const sanitized = sanitizeAppPreferences(preferences);
    if (!sanitized) {
        clearPersistedAppPreferences();
        return false;
    }

    return writeEnvelope(APP_PREFERENCES_STORAGE_KEY, sanitized);
}

export function clearPersistedAppPreferences(): void {
    clearStorageKey(APP_PREFERENCES_STORAGE_KEY);
}

export function readPersistedLastActiveTab(): AppTabPreference | null {
    return readEnvelope(APP_LAST_TAB_STORAGE_KEY, sanitizeOptionalAppTab);
}

export function writePersistedLastActiveTab(tab: AppTabPreference): boolean {
    const sanitized = sanitizeAppTab(tab, "benchmark");
    return writeEnvelope(APP_LAST_TAB_STORAGE_KEY, sanitized);
}

export function clearPersistedLastActiveTab(): void {
    clearStorageKey(APP_LAST_TAB_STORAGE_KEY);
}
