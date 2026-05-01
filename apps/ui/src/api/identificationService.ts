import { readJsonOrThrow } from "./http.ts";
import { loadFileFromUrl } from "./assetLoader.ts";
import {
    normalizeCatalogIdentifyGalleryResponse,
    normalizeDeleteIdentityResponse,
    normalizeEnrollFingerprintResponse,
    normalizeIdentificationAdminLayoutResponse,
    normalizeIdentificationAdminReconciliationResponse,
    normalizeIdentificationHealthResponse,
    normalizeIdentifyBrowserResetResponse,
    normalizeIdentifyBrowserSeedSelectionResponse,
    normalizeIdentifyDemoResetResponse,
    normalizeIdentifyDemoSeedResponse,
    normalizeIdentificationStatsResponse,
    normalizeIdentifyResponse,
} from "./contracts.ts";
import type {
    CatalogIdentifyGalleryResponse,
    CatalogIdentifyProbeCase,
    DeleteIdentityResponse,
    EnrollFingerprintRequest,
    EnrollFingerprintResponse,
    IdentificationAdminInspectionResponse,
    IdentificationAdminReconciliationResponse,
    IdentificationHealthResponse,
    IdentifyBrowserResetResponse,
    IdentifyBrowserSeedSelectionRequest,
    IdentifyBrowserSeedSelectionResponse,
    IdentifyDemoResetResponse,
    IdentifyDemoSeedResponse,
    IdentificationStatsResponse,
    IdentifyFingerprintRequest,
    IdentifyResponse,
} from "../types/index.ts";


export async function fetchIdentificationStats(): Promise<IdentificationStatsResponse> {
    const response = await fetch("/api/identify/stats");
    return readJsonOrThrow(response, normalizeIdentificationStatsResponse);
}

interface IdentificationAdminQueryOptions {
    storeScope?: "operational" | "browser";
    tablePrefix?: string;
}

function buildIdentificationAdminQuery(options?: IdentificationAdminQueryOptions): string {
    const params = new URLSearchParams();
    if (options?.storeScope && options.storeScope !== "operational") {
        params.set("store_scope", options.storeScope);
    }
    if (options?.tablePrefix) {
        params.set("table_prefix", options.tablePrefix);
    }
    const query = params.toString();
    return query ? `?${query}` : "";
}

export async function fetchIdentificationHealth(): Promise<IdentificationHealthResponse> {
    const response = await fetch("/api/health");
    return readJsonOrThrow(response, normalizeIdentificationHealthResponse);
}

export async function fetchIdentificationAdminLayout(
    options?: IdentificationAdminQueryOptions,
): Promise<IdentificationAdminInspectionResponse> {
    const response = await fetch(`/api/identify/admin/layout${buildIdentificationAdminQuery(options)}`);
    return readJsonOrThrow(response, normalizeIdentificationAdminLayoutResponse);
}

export async function fetchIdentificationAdminReconciliationReport(
    options?: IdentificationAdminQueryOptions,
): Promise<IdentificationAdminReconciliationResponse> {
    const response = await fetch(`/api/identify/admin/reconcile${buildIdentificationAdminQuery(options)}`);
    return readJsonOrThrow(response, normalizeIdentificationAdminReconciliationResponse);
}

export async function fetchIdentificationCatalogGallery(dataset?: string): Promise<CatalogIdentifyGalleryResponse> {
    const params = new URLSearchParams();
    if (dataset) {
        params.set("dataset", dataset);
    }

    const suffix = params.toString() ? `?${params.toString()}` : "";
    const response = await fetch(`/api/catalog/identify-gallery${suffix}`);
    return readJsonOrThrow(response, normalizeCatalogIdentifyGalleryResponse);
}

export async function fetchIdentificationDemoGallery(): Promise<CatalogIdentifyGalleryResponse> {
    return fetchIdentificationCatalogGallery();
}

export async function enrollFingerprint(
    request: EnrollFingerprintRequest,
): Promise<EnrollFingerprintResponse> {
    const form = new FormData();
    form.append("img", request.file);
    form.append("full_name", request.fullName);
    form.append("national_id", request.nationalId);
    form.append("capture", request.capture);
    form.append("vector_methods", (request.vectorMethods ?? ["dl", "vit"]).join(","));
    form.append("replace_existing", String(Boolean(request.replaceExisting)));

    const response = await fetch("/api/identify/enroll", { method: "POST", body: form });
    return readJsonOrThrow(response, normalizeEnrollFingerprintResponse);
}

export async function identifyFingerprint(
    request: IdentifyFingerprintRequest,
): Promise<IdentifyResponse> {
    const form = new FormData();
    form.append("img", request.file);
    form.append("capture", request.capture);
    form.append("retrieval_method", request.retrievalMethod ?? "dl");
    form.append("rerank_method", request.rerankMethod ?? "sift");
    form.append("shortlist_size", String(request.shortlistSize ?? 25));

    if (request.threshold !== undefined) {
        form.append("threshold", String(request.threshold));
    }
    if (request.namePattern) {
        form.append("name_pattern", request.namePattern);
    }
    if (request.nationalIdPattern) {
        form.append("national_id_pattern", request.nationalIdPattern);
    }
    if (request.createdFrom) {
        form.append("created_from", request.createdFrom);
    }
    if (request.createdTo) {
        form.append("created_to", request.createdTo);
    }
    if (request.storeScope && request.storeScope !== "operational") {
        form.append("store_scope", request.storeScope);
    }

    const response = await fetch("/api/identify/search", { method: "POST", body: form });
    return readJsonOrThrow(response, normalizeIdentifyResponse);
}

export async function deleteIdentity(randomId: string): Promise<DeleteIdentityResponse> {
    const response = await fetch(`/api/identify/person/${encodeURIComponent(randomId)}`, {
        method: "DELETE",
    });
    return readJsonOrThrow(response, normalizeDeleteIdentityResponse);
}

export async function seedIdentificationDemoStore(): Promise<IdentifyDemoSeedResponse> {
    const response = await fetch("/api/identify/demo/seed", {
        method: "POST",
    });
    return readJsonOrThrow(response, normalizeIdentifyDemoSeedResponse);
}

export async function resetIdentificationDemoStore(): Promise<IdentifyDemoResetResponse> {
    const response = await fetch("/api/identify/demo/reset", {
        method: "POST",
    });
    return readJsonOrThrow(response, normalizeIdentifyDemoResetResponse);
}

export async function seedIdentificationBrowserSelection(
    request: IdentifyBrowserSeedSelectionRequest,
): Promise<IdentifyBrowserSeedSelectionResponse> {
    const response = await fetch("/api/identify/browser/seed-selection", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
    });
    return readJsonOrThrow(response, normalizeIdentifyBrowserSeedSelectionResponse);
}

export async function resetIdentificationBrowserStore(): Promise<IdentifyBrowserResetResponse> {
    const response = await fetch("/api/identify/browser/reset", {
        method: "POST",
    });
    return readJsonOrThrow(response, normalizeIdentifyBrowserResetResponse);
}

export async function loadIdentificationProbeCaseFile(probeCase: CatalogIdentifyProbeCase): Promise<File> {
    const extension = probeCase.probe_asset_url.toLowerCase().endsWith(".jpg") || probeCase.probe_asset_url.toLowerCase().endsWith(".jpeg")
        ? "jpg"
        : "png";
    return loadFileFromUrl(
        probeCase.probe_asset_url,
        `${probeCase.id}_probe.${extension}`,
        "identification probe asset",
    );
}
