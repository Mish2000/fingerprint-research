export const WORKSPACE_STATE_SCHEMA_VERSION = 1;
export const MAX_PERSISTED_STATE_BYTES = 80_000;

export type WorkspaceStorageScope = "local" | "session";

interface PersistedWorkspaceEnvelope<T> {
    version: number;
    savedAt: string;
    data: T;
}

function getStorage(scope: WorkspaceStorageScope): Storage | null {
    if (typeof window === "undefined") {
        return null;
    }

    try {
        return scope === "local" ? window.localStorage : window.sessionStorage;
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

export function createWorkspaceStorageKey(feature: string, scope: WorkspaceStorageScope): string {
    return `fp-research.workspace.${feature}.${scope}`;
}

export function readWorkspaceState<T>(
    scope: WorkspaceStorageScope,
    key: string,
    sanitize: (value: unknown) => T | null,
): T | null {
    const storage = getStorage(scope);
    if (!storage) {
        return null;
    }

    try {
        const raw = storage.getItem(key);
        if (!raw) {
            return null;
        }

        const parsed = JSON.parse(raw) as unknown;
        if (typeof parsed !== "object" || parsed === null) {
            storage.removeItem(key);
            return null;
        }

        const envelope = parsed as Partial<PersistedWorkspaceEnvelope<unknown>>;
        if (envelope.version !== WORKSPACE_STATE_SCHEMA_VERSION || !("data" in envelope)) {
            storage.removeItem(key);
            return null;
        }

        const sanitized = sanitize(envelope.data);
        if (sanitized === null) {
            storage.removeItem(key);
            return null;
        }

        return sanitized;
    } catch {
        try {
            storage.removeItem(key);
        } catch {
            // Ignore storage failures after a bad payload.
        }
        return null;
    }
}

export function writeWorkspaceState<T>(
    scope: WorkspaceStorageScope,
    key: string,
    data: T,
): boolean {
    const storage = getStorage(scope);
    if (!storage) {
        return false;
    }

    const envelope: PersistedWorkspaceEnvelope<T> = {
        version: WORKSPACE_STATE_SCHEMA_VERSION,
        savedAt: new Date().toISOString(),
        data,
    };

    try {
        const serialized = JSON.stringify(envelope);
        if (!serialized || measureStringSize(serialized) > MAX_PERSISTED_STATE_BYTES) {
            storage.removeItem(key);
            return false;
        }

        storage.setItem(key, serialized);
        return true;
    } catch {
        return false;
    }
}

export function clearWorkspaceState(scope: WorkspaceStorageScope, key: string): void {
    const storage = getStorage(scope);
    if (!storage) {
        return;
    }

    try {
        storage.removeItem(key);
    } catch {
        // Ignore storage cleanup failures.
    }
}

export function sanitizeBoolean(value: unknown, fallback: boolean): boolean {
    return typeof value === "boolean" ? value : fallback;
}

export function sanitizeNumber(
    value: unknown,
    fallback: number,
    min = Number.NEGATIVE_INFINITY,
    max = Number.POSITIVE_INFINITY,
): number {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return fallback;
    }

    return Math.min(Math.max(value, min), max);
}

export function sanitizeOptionalString(value: unknown, maxLength = 500): string | null {
    if (typeof value !== "string") {
        return null;
    }

    const trimmed = value.trim();
    if (!trimmed) {
        return null;
    }

    return trimmed.slice(0, maxLength);
}

export function sanitizeFileName(value: unknown, maxLength = 180): string | null {
    const text = sanitizeOptionalString(value, maxLength);
    if (!text) {
        return null;
    }

    const normalized = text.replace(/\\/g, "/");
    const fileName = normalized.split("/").filter(Boolean).pop() ?? "";
    return fileName ? fileName.slice(0, maxLength) : null;
}

export function sanitizeRelativeStorageUrl(value: unknown, maxLength = 2_000): string | null {
    const text = sanitizeOptionalString(value, maxLength);
    if (!text) {
        return null;
    }

    if (/^[a-z][a-z0-9+.-]*:/i.test(text)) {
        return null;
    }

    if (text.includes("..")) {
        return null;
    }

    return text;
}

export function sanitizeStringArray(value: unknown, maxItems: number, maxLength = 160): string[] {
    if (!Array.isArray(value)) {
        return [];
    }

    const items: string[] = [];
    const seen = new Set<string>();
    for (const item of value) {
        const sanitized = sanitizeOptionalString(item, maxLength);
        if (!sanitized || seen.has(sanitized)) {
            continue;
        }

        seen.add(sanitized);
        items.push(sanitized);
        if (items.length >= maxItems) {
            break;
        }
    }

    return items;
}
