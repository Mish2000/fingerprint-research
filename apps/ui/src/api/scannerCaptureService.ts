import { extractApiErrorMessage, isObject } from "../utils/error.ts";
import { readJsonOrThrow, readResponsePayload } from "./http.ts";

type UnknownRecord = Record<string, unknown>;

export interface ScannerImportResponse {
    capture_id: string;
    original_filename: string;
    normalized_filename: string;
    normalized_url: string;
    mime_type: string;
    size_bytes: number;
    modified_at: string;
    age_seconds: number;
}

export type ScannerCaptureMode = "auto" | "twain" | "saved_file_bridge";

export interface ScannerStatusResponse {
    active_mode: string;
    available_modes: string[];
    direct_capture_available: boolean;
    saved_file_bridge_available: boolean;
    device_name: string | null;
    driver_detected: boolean | null;
    twain_source_detected: boolean | null;
    last_error: string | null;
    configured: boolean;
    enabled: boolean;
    direct_capture_enabled: boolean;
    saved_file_bridge_enabled: boolean;
    capture_dir_display: string | null;
    normalized_dir_display: string | null;
}

export interface ScannerCaptureRequest {
    mode: ScannerCaptureMode;
    timeout_ms?: number;
    fallback_allowed?: boolean;
    normalize?: boolean;
    show_ui?: boolean;
    settle_after_enable_ms?: number;
}

export interface ScannerCaptureDevice {
    name: string | null;
    provider: string | null;
}

export interface ScannerCaptureSuccessResponse {
    ok: true;
    mode_used: Exclude<ScannerCaptureMode, "auto">;
    direct_capture: boolean;
    normalized_url: string;
    capture_id: string;
    raw_file: UnknownRecord | null;
    normalized_file: UnknownRecord | null;
    duration_ms: number;
    device: ScannerCaptureDevice;
    warning: string | null;
    metadata: UnknownRecord | null;
}

export interface ScannerCaptureFailureResponse {
    ok: false;
    error_code: string;
    message: string;
    mode_requested: string;
    fallback_available: boolean;
    diagnostics: UnknownRecord;
}

export type ScannerCaptureResponse = ScannerCaptureSuccessResponse | ScannerCaptureFailureResponse;

export interface ScannerCaptureAssetReference {
    normalized_url: string;
    normalized_filename?: string | null;
    capture_id?: string | null;
}

function expectObject(payload: unknown, label: string): UnknownRecord {
    if (!isObject(payload)) {
        throw new Error(`${label} must be an object.`);
    }
    return payload;
}

function expectString(record: UnknownRecord, key: string, label: string): string {
    const value = record[key];
    if (typeof value !== "string") {
        throw new Error(`${label}.${key} must be a string.`);
    }
    return value;
}

function expectBoolean(record: UnknownRecord, key: string, label: string): boolean {
    const value = record[key];
    if (typeof value !== "boolean") {
        throw new Error(`${label}.${key} must be a boolean.`);
    }
    return value;
}

function maybeString(record: UnknownRecord, key: string): string | null {
    const value = record[key];
    return typeof value === "string" ? value : null;
}

function maybeBoolean(record: UnknownRecord, key: string): boolean | null {
    const value = record[key];
    return typeof value === "boolean" ? value : null;
}

function maybeObject(record: UnknownRecord, key: string, label: string): UnknownRecord | null {
    const value = record[key];
    if (value == null) {
        return null;
    }
    return expectObject(value, `${label}.${key}`);
}

function expectStringArray(payload: unknown, label: string): string[] {
    if (!Array.isArray(payload)) {
        throw new Error(`${label} must be an array.`);
    }
    return payload.map((item, index) => {
        if (typeof item !== "string") {
            throw new Error(`${label}[${index}] must be a string.`);
        }
        return item;
    });
}

function expectNumber(record: UnknownRecord, key: string, label: string): number {
    const value = record[key];
    if (typeof value !== "number" || Number.isNaN(value)) {
        throw new Error(`${label}.${key} must be a number.`);
    }
    return value;
}

function normalizeScannerCaptureMode(value: unknown, label: string): Exclude<ScannerCaptureMode, "auto"> {
    if (value === "twain" || value === "saved_file_bridge") {
        return value;
    }
    throw new Error(`${label} must be twain or saved_file_bridge.`);
}

function normalizeScannerImportResponse(payload: unknown): ScannerImportResponse {
    const record = expectObject(payload, "ScannerImportResponse");
    return {
        capture_id: expectString(record, "capture_id", "ScannerImportResponse"),
        original_filename: expectString(record, "original_filename", "ScannerImportResponse"),
        normalized_filename: expectString(record, "normalized_filename", "ScannerImportResponse"),
        normalized_url: expectString(record, "normalized_url", "ScannerImportResponse"),
        mime_type: expectString(record, "mime_type", "ScannerImportResponse"),
        size_bytes: expectNumber(record, "size_bytes", "ScannerImportResponse"),
        modified_at: expectString(record, "modified_at", "ScannerImportResponse"),
        age_seconds: expectNumber(record, "age_seconds", "ScannerImportResponse"),
    };
}

function normalizeScannerStatusResponse(payload: unknown): ScannerStatusResponse {
    const record = expectObject(payload, "ScannerStatusResponse");
    return {
        active_mode: expectString(record, "active_mode", "ScannerStatusResponse"),
        available_modes: expectStringArray(record.available_modes ?? [], "ScannerStatusResponse.available_modes"),
        direct_capture_available: expectBoolean(record, "direct_capture_available", "ScannerStatusResponse"),
        saved_file_bridge_available: expectBoolean(record, "saved_file_bridge_available", "ScannerStatusResponse"),
        device_name: maybeString(record, "device_name"),
        driver_detected: maybeBoolean(record, "driver_detected"),
        twain_source_detected: maybeBoolean(record, "twain_source_detected"),
        last_error: maybeString(record, "last_error"),
        configured: expectBoolean(record, "configured", "ScannerStatusResponse"),
        enabled: expectBoolean(record, "enabled", "ScannerStatusResponse"),
        direct_capture_enabled: expectBoolean(record, "direct_capture_enabled", "ScannerStatusResponse"),
        saved_file_bridge_enabled: expectBoolean(record, "saved_file_bridge_enabled", "ScannerStatusResponse"),
        capture_dir_display: maybeString(record, "capture_dir_display"),
        normalized_dir_display: maybeString(record, "normalized_dir_display"),
    };
}

function normalizeScannerCaptureDevice(payload: unknown): ScannerCaptureDevice {
    if (payload == null) {
        return { name: null, provider: null };
    }

    const record = expectObject(payload, "ScannerCaptureResponse.device");
    return {
        name: maybeString(record, "name"),
        provider: maybeString(record, "provider"),
    };
}

function normalizeScannerCaptureResponse(payload: unknown): ScannerCaptureResponse {
    const record = expectObject(payload, "ScannerCaptureResponse");
    const ok = expectBoolean(record, "ok", "ScannerCaptureResponse");

    if (!ok) {
        return {
            ok: false,
            error_code: expectString(record, "error_code", "ScannerCaptureResponse"),
            message: expectString(record, "message", "ScannerCaptureResponse"),
            mode_requested: expectString(record, "mode_requested", "ScannerCaptureResponse"),
            fallback_available: expectBoolean(record, "fallback_available", "ScannerCaptureResponse"),
            diagnostics: maybeObject(record, "diagnostics", "ScannerCaptureResponse") ?? {},
        };
    }

    return {
        ok: true,
        mode_used: normalizeScannerCaptureMode(record.mode_used, "ScannerCaptureResponse.mode_used"),
        direct_capture: expectBoolean(record, "direct_capture", "ScannerCaptureResponse"),
        normalized_url: expectString(record, "normalized_url", "ScannerCaptureResponse"),
        capture_id: expectString(record, "capture_id", "ScannerCaptureResponse"),
        raw_file: maybeObject(record, "raw_file", "ScannerCaptureResponse"),
        normalized_file: maybeObject(record, "normalized_file", "ScannerCaptureResponse"),
        duration_ms: expectNumber(record, "duration_ms", "ScannerCaptureResponse"),
        device: normalizeScannerCaptureDevice(record.device),
        warning: maybeString(record, "warning"),
        metadata: maybeObject(record, "metadata", "ScannerCaptureResponse"),
    };
}

function formatCaptureAssetError(response: Response, payload: unknown): string {
    const extractedMessage = extractApiErrorMessage(payload);
    const statusLabel = response.status ? `${response.status}` : "request";

    if (extractedMessage) {
        return `Failed to load normalized scanner capture (${statusLabel}): ${extractedMessage}`;
    }

    return `Failed to load normalized scanner capture (${statusLabel}).`;
}

function scannerCaptureFilename(capture: ScannerCaptureAssetReference): string {
    if (capture.normalized_filename) {
        return capture.normalized_filename;
    }

    if (capture.capture_id) {
        return capture.capture_id.endsWith(".png") ? capture.capture_id : `${capture.capture_id}.png`;
    }

    const urlSegment = capture.normalized_url.split("/").filter(Boolean).at(-1);
    return urlSegment ? `${urlSegment.replace(/\.png$/i, "")}.png` : "scanner_capture.png";
}

export async function getScannerStatus(): Promise<ScannerStatusResponse> {
    const response = await fetch("/api/scanner/status");
    return readJsonOrThrow(response, normalizeScannerStatusResponse);
}

export async function captureScanner(request: ScannerCaptureRequest): Promise<ScannerCaptureResponse> {
    const response = await fetch("/api/scanner/capture", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
    });
    return readJsonOrThrow(response, normalizeScannerCaptureResponse);
}

export async function importLatestSavedScannerCapture(): Promise<ScannerImportResponse> {
    const response = await fetch("/api/scanner/import-latest", { method: "POST" });
    return readJsonOrThrow(response, normalizeScannerImportResponse);
}

export async function loadScannerCaptureFile(capture: ScannerCaptureAssetReference): Promise<File> {
    const response = await fetch(capture.normalized_url);

    if (!response.ok) {
        const payload = await readResponsePayload(response);
        throw new Error(formatCaptureAssetError(response, payload));
    }

    const blob = await response.blob();
    return new File([blob], scannerCaptureFilename(capture), { type: "image/png" });
}
