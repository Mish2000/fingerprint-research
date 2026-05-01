import { readJsonOrThrow, readResponsePayload } from "./http.ts";
import {
    normalizeBenchmarkRunsResponse,
    normalizeBenchmarkSummaryResponse,
    normalizeBestMethodsResponse,
    normalizeCatalogDatasetBrowserResponse,
    normalizeCatalogDatasetsResponse,
    normalizeCatalogVerifyCasesResponse,
    normalizeComparisonResponse,
    normalizeDemoCasesResponse,
    normalizeMatchResponse,
} from "./contracts.ts";
import type {
    BenchmarkComparisonQuery,
    BenchmarkBestQuery,
    BenchmarkRunsResponse,
    BenchmarkSummaryQuery,
    BenchmarkSummaryResponse,
    BestMethodsResponse,
    CatalogBrowserItem,
    CatalogDatasetBrowserQuery,
    CatalogDatasetBrowserResponse,
    CatalogDatasetsResponse,
    CatalogVerifyCase,
    CatalogVerifyCasesResponse,
    ComparisonResponse,
    DemoCase,
    LoadedDemoCaseFiles,
    DemoCasesResponse,
    MatchRequest,
    MatchResponse,
    Method,
} from "../types/index.ts";
import { extractApiErrorMessage, toErrorMessage } from "../utils/error.ts";

const METHODS_WITH_CAPTURE = new Set<Method>(["dedicated", "dl", "vit"]);
const WARM_UP_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

const warmUpPromises = new Map<Method, Promise<void>>();
let warmUpTemplatePromise: Promise<File> | null = null;

type AssetBackedCase = {
    id: string;
    image_a_url: string;
    image_b_url: string;
    asset_a_id?: string | null;
    asset_b_id?: string | null;
};

type CatalogAssetVariant = "thumbnail" | "preview";

function formatAssetError(response: Response, payload: unknown, assetLabel: string): string {
    const extractedMessage = extractApiErrorMessage(payload);
    const statusLabel = response.status ? `${response.status}` : "request";

    if (extractedMessage) {
        return `Failed to load ${assetLabel} (${statusLabel}): ${extractedMessage}`;
    }

    return `Failed to load ${assetLabel} (${statusLabel}).`;
}

export async function matchFingerprints(request: MatchRequest): Promise<MatchResponse> {
    const form = new FormData();
    form.append("method", request.method);
    form.append("img_a", request.fileA);
    form.append("img_b", request.fileB);

    if (METHODS_WITH_CAPTURE.has(request.method)) {
        form.append("capture_a", request.captureA);
        form.append("capture_b", request.captureB);
    }

    form.append("return_overlay", String(request.returnOverlay));

    if (request.threshold !== undefined && request.threshold !== "") {
        form.append("threshold", String(request.threshold));
    }

    try {
        const response = await fetch("/api/match", { method: "POST", body: form });
        return await readJsonOrThrow(response, normalizeMatchResponse);
    } catch (error) {
        throw new Error(toErrorMessage(error));
    }
}

async function getWarmUpTemplateFile(): Promise<File> {
    if (warmUpTemplatePromise === null) {
        warmUpTemplatePromise = (async () => {
            const response = await fetch(`data:image/png;base64,${WARM_UP_IMAGE_BASE64}`);
            const blob = await response.blob();
            return new File([blob], "verify_warmup.png", { type: "image/png" });
        })();
    }

    const template = await warmUpTemplatePromise;
    return new File([template], template.name, { type: template.type });
}

export function warmUpMatcher(method: Method): Promise<void> {
    const existingPromise = warmUpPromises.get(method);
    if (existingPromise) {
        return existingPromise;
    }

    const promise = (async () => {
        const fileA = await getWarmUpTemplateFile();
        const fileB = await getWarmUpTemplateFile();

        await matchFingerprints({
            method,
            fileA,
            fileB,
            captureA: "plain",
            captureB: "plain",
            returnOverlay: false,
        });
    })().finally(() => {
        warmUpPromises.delete(method);
    });

    warmUpPromises.set(method, promise);
    return promise;
}

function buildBenchmarkQuery(query: BenchmarkSummaryQuery | BenchmarkComparisonQuery | BenchmarkBestQuery = {}): string {
    const params = new URLSearchParams();

    if (query.dataset) {
        params.set("dataset", query.dataset);
    }
    if (query.split) {
        params.set("split", query.split);
    }
    if ("view_mode" in query && query.view_mode) {
        params.set("view_mode", query.view_mode);
    }
    if ("sort_mode" in query && query.sort_mode) {
        params.set("sort_mode", query.sort_mode);
    }

    const serialized = params.toString();
    return serialized ? `?${serialized}` : "";
}

export async function fetchBenchmarkSummary(
    query: BenchmarkSummaryQuery = {},
): Promise<BenchmarkSummaryResponse> {
    const response = await fetch(`/api/benchmark/summary${buildBenchmarkQuery(query)}`);
    return readJsonOrThrow(response, normalizeBenchmarkSummaryResponse);
}

export async function fetchBenchmarkRuns(): Promise<BenchmarkRunsResponse> {
    const response = await fetch("/api/benchmark/runs");
    return readJsonOrThrow(response, normalizeBenchmarkRunsResponse);
}

export async function fetchBenchmarkComparison(
    query: BenchmarkComparisonQuery = {},
): Promise<ComparisonResponse> {
    const response = await fetch(`/api/benchmark/comparison${buildBenchmarkQuery(query)}`);
    return readJsonOrThrow(response, normalizeComparisonResponse);
}

export async function fetchBenchmarkBest(
    query: BenchmarkBestQuery = {},
): Promise<BestMethodsResponse> {
    const response = await fetch(`/api/benchmark/best${buildBenchmarkQuery(query)}`);
    return readJsonOrThrow(response, normalizeBestMethodsResponse);
}

export async function fetchDemoCases(): Promise<DemoCasesResponse> {
    const response = await fetch("/api/demo/cases");
    return readJsonOrThrow(response, normalizeDemoCasesResponse);
}

export async function fetchCatalogVerifyCases(): Promise<CatalogVerifyCasesResponse> {
    const response = await fetch("/api/catalog/verify-cases");
    return readJsonOrThrow(response, normalizeCatalogVerifyCasesResponse);
}

export async function fetchCatalogDatasets(): Promise<CatalogDatasetsResponse> {
    const response = await fetch("/api/catalog/datasets");
    return readJsonOrThrow(response, normalizeCatalogDatasetsResponse);
}

export async function fetchCatalogDatasetBrowser(
    query: CatalogDatasetBrowserQuery,
): Promise<CatalogDatasetBrowserResponse> {
    const params = new URLSearchParams();
    params.set("dataset", query.dataset);

    if (query.split) {
        params.set("split", query.split);
    }
    if (query.capture) {
        params.set("capture", query.capture);
    }
    if (query.modality) {
        params.set("modality", query.modality);
    }
    if (query.subject_id) {
        params.set("subject_id", query.subject_id);
    }
    if (query.finger) {
        params.set("finger", query.finger);
    }
    if (typeof query.ui_eligible === "boolean") {
        params.set("ui_eligible", String(query.ui_eligible));
    }
    if (typeof query.limit === "number") {
        params.set("limit", String(query.limit));
    }
    if (typeof query.offset === "number") {
        params.set("offset", String(query.offset));
    }
    if (query.sort) {
        params.set("sort", query.sort);
    }

    const response = await fetch(`/api/catalog/dataset-browser?${params.toString()}`);
    return readJsonOrThrow(response, normalizeCatalogDatasetBrowserResponse);
}

async function fetchFileFromUrl(
    url: string,
    fallbackName: string,
    assetLabel: string,
): Promise<File> {
    const response = await fetch(url);

    if (!response.ok) {
        const payload = await readResponsePayload(response);
        throw new Error(formatAssetError(response, payload, assetLabel));
    }

    const blob = await response.blob();
    const urlTail = url.split("/").pop() || "";
    const fileName = urlTail.includes(".") ? urlTail : fallbackName;
    const contentType = response.headers.get("content-type") || blob.type || "";

    if (contentType.includes("json") || fileName.toLowerCase().endsWith(".json")) {
        throw new Error(
            "Catalog asset is a descriptor rather than a runnable binary file. " +
            "Regenerate the demo asset bundle or provide binary assets under data/samples/assets/.",
        );
    }

    return new File([blob], fileName, { type: blob.type || "image/png" });
}

async function loadCaseFiles(item: AssetBackedCase): Promise<LoadedDemoCaseFiles> {
    const [fileA, fileB] = await Promise.all([
        fetchFileFromUrl(item.image_a_url, `${item.asset_a_id ?? item.id}_a.png`, "demo asset"),
        fetchFileFromUrl(item.image_b_url, `${item.asset_b_id ?? item.id}_b.png`, "demo asset"),
    ]);

    return { fileA, fileB };
}

export async function loadDemoCaseFiles(item: DemoCase): Promise<LoadedDemoCaseFiles> {
    return loadCaseFiles({
        id: item.id,
        image_a_url: item.image_a_url,
        image_b_url: item.image_b_url,
        asset_a_id: item.asset_a_id,
        asset_b_id: item.asset_b_id,
    });
}

export async function loadCatalogVerifyCaseFiles(item: CatalogVerifyCase): Promise<LoadedDemoCaseFiles> {
    return loadCaseFiles({
        id: item.case_id,
        image_a_url: item.image_a_url,
        image_b_url: item.image_b_url,
        asset_a_id: item.asset_a_id,
        asset_b_id: item.asset_b_id,
    });
}

export async function loadCatalogBrowserItemFile(
    item: CatalogBrowserItem,
    variant: CatalogAssetVariant = "preview",
): Promise<File> {
    const url = variant === "thumbnail" ? item.thumbnail_url : item.preview_url;
    const fileExtension = url.toLowerCase().endsWith(".jpg") || url.toLowerCase().endsWith(".jpeg") ? "jpg" : "png";
    return fetchFileFromUrl(url, `${item.asset_id}_${variant}.${fileExtension}`, "browser asset");
}

export async function loadCatalogBrowserPairFiles(
    itemA: CatalogBrowserItem,
    itemB: CatalogBrowserItem,
): Promise<LoadedDemoCaseFiles> {
    const [fileA, fileB] = await Promise.all([
        loadCatalogBrowserItemFile(itemA, "preview"),
        loadCatalogBrowserItemFile(itemB, "preview"),
    ]);

    return { fileA, fileB };
}
