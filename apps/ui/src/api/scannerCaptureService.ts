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

function expectNumber(record: UnknownRecord, key: string, label: string): number {
    const value = record[key];
    if (typeof value !== "number" || Number.isNaN(value)) {
        throw new Error(`${label}.${key} must be a number.`);
    }
    return value;
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

function formatCaptureAssetError(response: Response, payload: unknown): string {
    const extractedMessage = extractApiErrorMessage(payload);
    const statusLabel = response.status ? `${response.status}` : "request";

    if (extractedMessage) {
        return `Failed to load normalized scanner capture (${statusLabel}): ${extractedMessage}`;
    }

    return `Failed to load normalized scanner capture (${statusLabel}).`;
}

export async function importLatestSavedScannerCapture(): Promise<ScannerImportResponse> {
    const response = await fetch("/api/scanner/import-latest", { method: "POST" });
    return readJsonOrThrow(response, normalizeScannerImportResponse);
}

export async function loadScannerCaptureFile(importedCapture: ScannerImportResponse): Promise<File> {
    const response = await fetch(importedCapture.normalized_url);

    if (!response.ok) {
        const payload = await readResponsePayload(response);
        throw new Error(formatCaptureAssetError(response, payload));
    }

    const blob = await response.blob();
    return new File([blob], importedCapture.normalized_filename, { type: "image/png" });
}
