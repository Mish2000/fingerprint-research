export function isObject(v: unknown): v is Record<string, unknown> {
    return typeof v === "object" && v !== null;
}

export function toErrorMessage(v: unknown): string {
    if (v instanceof Error) return v.message;
    if (typeof v === "string") return v;
    try {
        return JSON.stringify(v);
    } catch {
        return String(v);
    }
}

export function extractApiErrorMessage(payload: unknown): string {
    if (isObject(payload) && "detail" in payload) {
        const d = payload.detail as unknown;
        if (typeof d === "string") return d;
        return toErrorMessage(d);
    }
    return toErrorMessage(payload);
}