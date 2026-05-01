import fs from "node:fs";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import IdentificationWorkspace from "../src/features/identification/IdentificationWorkspace.tsx";
import type {
    CatalogDatasetBrowserResponse,
    CatalogDatasetsResponse,
    CatalogIdentifyGalleryResponse,
    IdentifyBrowserResetResponse,
    IdentifyBrowserSeedSelectionResponse,
    IdentifyDemoResetResponse,
    IdentifyDemoSeedResponse,
    IdentificationStatsResponse,
    IdentifyResponse,
} from "../src/types/index.ts";

type RenderedWorkspace = {
    container: HTMLDivElement;
    root: Root;
};

function createJsonResponse(payload: unknown): Response {
    return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "content-type": "application/json" },
    });
}

function createImageResponse(label: string): Response {
    return new Response(new Blob([label], { type: "image/png" }), {
        status: 200,
        headers: { "content-type": "image/png" },
    });
}

function normalizeText(value: string | null | undefined): string {
    return (value ?? "").replace(/\s+/g, " ").trim();
}

function getButtons(container: HTMLElement): HTMLButtonElement[] {
    return Array.from(container.querySelectorAll("button"));
}

function getButtonByText(container: HTMLElement, text: string): HTMLButtonElement {
    const match = getButtons(container).find((button) => normalizeText(button.textContent).includes(text));
    if (!match) {
        throw new Error(`Unable to find button with text: ${text}`);
    }
    return match;
}

function getLabelField<T extends HTMLInputElement | HTMLSelectElement>(container: HTMLElement, label: string): T {
    const labels = Array.from(container.querySelectorAll("label"));
    const normalizedLabel = normalizeText(label);
    const match = labels.find((field) => normalizeText(field.textContent) === normalizedLabel)
        ?? labels.find((field) => normalizeText(field.textContent).startsWith(normalizedLabel))
        ?? labels.find((field) => normalizeText(field.textContent).includes(normalizedLabel));
    if (!match) {
        throw new Error(`Unable to find field with label: ${label}`);
    }

    const control = match.querySelector("input, select");
    if (!control) {
        throw new Error(`Unable to find control for label: ${label}`);
    }

    return control as T;
}

async function click(button: HTMLButtonElement): Promise<void> {
    await act(async () => {
        button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
}

async function changeFieldValue(field: HTMLInputElement | HTMLSelectElement, value: string): Promise<void> {
    await act(async () => {
        const prototype = field instanceof HTMLInputElement ? HTMLInputElement.prototype : HTMLSelectElement.prototype;
        const valueSetter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
        valueSetter?.call(field, value);
        field.dispatchEvent(new Event("input", { bubbles: true }));
        field.dispatchEvent(new Event("change", { bubbles: true }));
    });
}

async function uploadFile(input: HTMLInputElement, file: File): Promise<void> {
    await act(async () => {
        Object.defineProperty(input, "files", {
            configurable: true,
            value: {
                0: file,
                length: 1,
                item: (index: number) => (index === 0 ? file : null),
            },
        });
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
}

async function flush(): Promise<void> {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
}

async function waitFor(assertion: () => void, timeoutMs = 2000): Promise<void> {
    const start = Date.now();
    let lastError: unknown;

    while (Date.now() - start < timeoutMs) {
        try {
            assertion();
            return;
        } catch (error) {
            lastError = error;
            await act(async () => {
                await new Promise((resolve) => setTimeout(resolve, 20));
            });
        }
    }

    throw lastError instanceof Error ? lastError : new Error("Timed out while waiting for UI state.");
}

async function renderWorkspace(): Promise<RenderedWorkspace> {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    await act(async () => {
        root.render(<IdentificationWorkspace />);
    });

    await flush();
    return { container, root };
}

async function unmountWorkspace(root: Root): Promise<void> {
    await act(async () => {
        root.unmount();
    });
}

function createStats(totalEnrolled: number, demoSeededCount: number, browserSeededCount = 0): IdentificationStatsResponse {
    return {
        total_enrolled: totalEnrolled,
        demo_seeded_count: demoSeededCount,
        browser_seeded_count: browserSeededCount,
        storage_layout: {
            backend: "postgresql",
            person_table: "biometric.person_directory",
            raw_fingerprints_table: "biometric.raw_fingerprints",
            feature_vectors_table: "biometric.feature_vectors",
            identity_map_table: "identity.identity_map",
        },
    };
}

const identifyGalleryItems = [
    {
        identity_id: "identity_subject_001",
        dataset: "identify_demo",
        dataset_label: "Identify Demo",
        display_name: "Subject 001",
        subject_id: "1",
        gallery_role: "standard",
        tags: ["identify_demo", "gallery"],
        is_demo_safe: true,
        enrollment_candidates: [
            {
                asset_id: "asset_subject_001_enroll",
                capture: "plain",
                finger: "1",
                recommended_usage: "recommended_enrollment",
                asset_reference: "/api/catalog/assets/identify_demo/asset_subject_001_enroll/preview",
                has_servable_asset: true,
                availability_status: "available",
            },
        ],
        probe_candidates: [
            {
                asset_id: "asset_subject_001_probe",
                capture: "roll",
                finger: "1",
                recommended_usage: "recommended_probe",
                asset_reference: "/api/catalog/assets/identify_demo/asset_subject_001_probe/preview",
                has_servable_asset: true,
                availability_status: "available",
            },
        ],
        preview_url: "/api/catalog/assets/identify_demo/uiasset_subject_001/preview",
        thumbnail_url: "/api/catalog/assets/identify_demo/uiasset_subject_001/thumbnail",
        recommended_enrollment_asset_id: "asset_subject_001_enroll",
        recommended_probe_asset_id: "asset_subject_001_probe",
        recommended_enrollment_capture: "plain",
        recommended_probe_capture: "roll",
    },
    {
        identity_id: "identity_subject_002",
        dataset: "identify_demo",
        dataset_label: "Identify Demo",
        display_name: "Subject 002",
        subject_id: "2",
        gallery_role: "standard",
        tags: ["identify_demo", "gallery"],
        is_demo_safe: true,
        enrollment_candidates: [
            {
                asset_id: "asset_subject_002_enroll",
                capture: "plain",
                finger: "2",
                recommended_usage: "recommended_enrollment",
                asset_reference: "/api/catalog/assets/identify_demo/asset_subject_002_enroll/preview",
                has_servable_asset: true,
                availability_status: "available",
            },
        ],
        probe_candidates: [
            {
                asset_id: "asset_subject_002_probe",
                capture: "roll",
                finger: "2",
                recommended_usage: "recommended_probe",
                asset_reference: "/api/catalog/assets/identify_demo/asset_subject_002_probe/preview",
                has_servable_asset: true,
                availability_status: "available",
            },
        ],
        preview_url: "/api/catalog/assets/identify_demo/uiasset_subject_002/preview",
        thumbnail_url: "/api/catalog/assets/identify_demo/uiasset_subject_002/thumbnail",
        recommended_enrollment_asset_id: "asset_subject_002_enroll",
        recommended_probe_asset_id: "asset_subject_002_probe",
        recommended_enrollment_capture: "plain",
        recommended_probe_capture: "roll",
    },
] as const;

const identifyGallery: CatalogIdentifyGalleryResponse = {
    items: [...identifyGalleryItems],
    demo_identities: [
        {
            id: "identity_subject_001",
            dataset: "identify_demo",
            dataset_label: "Identify Demo",
            display_label: "Subject 001",
            capture: "plain",
            thumbnail_url: "/api/catalog/assets/identify_demo/uiasset_subject_001/thumbnail",
            preview_url: "/api/catalog/assets/identify_demo/uiasset_subject_001/preview",
            subject_id: "1",
            gallery_role: "standard",
            tags: ["identify_demo", "gallery"],
            recommended_enrollment_asset_id: "asset_subject_001_enroll",
            recommended_probe_asset_id: "asset_subject_001_probe",
        },
        {
            id: "identity_subject_002",
            dataset: "identify_demo",
            dataset_label: "Identify Demo",
            display_label: "Subject 002",
            capture: "plain",
            thumbnail_url: "/api/catalog/assets/identify_demo/uiasset_subject_002/thumbnail",
            preview_url: "/api/catalog/assets/identify_demo/uiasset_subject_002/preview",
            subject_id: "2",
            gallery_role: "standard",
            tags: ["identify_demo", "gallery"],
            recommended_enrollment_asset_id: "asset_subject_002_enroll",
            recommended_probe_asset_id: "asset_subject_002_probe",
        },
    ],
    probe_cases: [
        {
            id: "probe_positive",
            title: "Positive identify probe",
            description: "Probe the seeded gallery with subject 001.",
            dataset: "identify_demo",
            dataset_label: "Identify Demo",
            capture: "plain",
            difficulty: "easy",
            probe_thumbnail_url: "/api/catalog/assets/identify_demo/probe_positive/thumbnail",
            probe_preview_url: "/api/catalog/assets/identify_demo/probe_positive/preview",
            probe_asset_url: "/api/catalog/assets/identify_demo/probe_positive/preview",
            expected_outcome: "match",
            expected_top_identity_id: "identity_subject_001",
            expected_top_identity_label: "Subject 001",
            recommended_retrieval_method: "dl",
            recommended_rerank_method: "sift",
            recommended_shortlist_size: 7,
            scenario_type: "positive_identification",
            tags: ["positive", "demo"],
        },
        {
            id: "probe_no_match",
            title: "No match walkthrough",
            description: "Exercise the no-match decision path.",
            dataset: "identify_demo",
            dataset_label: "Identify Demo",
            capture: "plain",
            difficulty: "hard",
            probe_thumbnail_url: "/api/catalog/assets/identify_demo/probe_no_match/thumbnail",
            probe_preview_url: "/api/catalog/assets/identify_demo/probe_no_match/preview",
            probe_asset_url: "/api/catalog/assets/identify_demo/probe_no_match/preview",
            expected_outcome: "no_match",
            expected_top_identity_id: null,
            expected_top_identity_label: null,
            recommended_retrieval_method: "dl",
            recommended_rerank_method: "sift",
            recommended_shortlist_size: 5,
            scenario_type: "no_match",
            tags: ["negative", "demo"],
        },
        {
            id: "probe_shortlist_zero",
            title: "Shortlist zero walkthrough",
            description: "Exercise the zero-shortlist path.",
            dataset: "identify_demo",
            dataset_label: "Identify Demo",
            capture: "plain",
            difficulty: "challenging",
            probe_thumbnail_url: "/api/catalog/assets/identify_demo/probe_shortlist_zero/thumbnail",
            probe_preview_url: "/api/catalog/assets/identify_demo/probe_shortlist_zero/preview",
            probe_asset_url: "/api/catalog/assets/identify_demo/probe_shortlist_zero/preview",
            expected_outcome: "no_match",
            expected_top_identity_id: null,
            expected_top_identity_label: null,
            recommended_retrieval_method: "dl",
            recommended_rerank_method: "sift",
            recommended_shortlist_size: 3,
            scenario_type: "difficult_identification",
            tags: ["negative", "demo"],
        },
    ],
    total: 2,
    limit: 20,
    offset: 0,
    has_more: false,
    total_probe_cases: 3,
};

const identifyGalleryAlt: CatalogIdentifyGalleryResponse = {
    items: [
        {
            identity_id: "identity_alt_subject_101",
            dataset: "identify_demo_alt",
            dataset_label: "Identify Demo Alt",
            display_name: "Subject 101",
            subject_id: "101",
            gallery_role: "challenging",
            tags: ["identify_demo_alt", "gallery"],
            is_demo_safe: true,
            enrollment_candidates: [
                {
                    asset_id: "asset_alt_subject_101_enroll",
                    capture: "plain",
                    finger: "3",
                    recommended_usage: "recommended_enrollment",
                    asset_reference: "/api/catalog/assets/identify_demo_alt/asset_alt_subject_101_enroll/preview",
                    has_servable_asset: true,
                    availability_status: "available",
                },
            ],
            probe_candidates: [
                {
                    asset_id: "asset_alt_subject_101_probe",
                    capture: "latent",
                    finger: "3",
                    recommended_usage: "recommended_probe",
                    asset_reference: "/api/catalog/assets/identify_demo_alt/asset_alt_subject_101_probe/preview",
                    has_servable_asset: true,
                    availability_status: "available",
                },
            ],
            preview_url: "/api/catalog/assets/identify_demo_alt/uiasset_subject_101/preview",
            thumbnail_url: "/api/catalog/assets/identify_demo_alt/uiasset_subject_101/thumbnail",
            recommended_enrollment_asset_id: "asset_alt_subject_101_enroll",
            recommended_probe_asset_id: "asset_alt_subject_101_probe",
            recommended_enrollment_capture: "plain",
            recommended_probe_capture: "latent",
        },
    ],
    demo_identities: [],
    probe_cases: [],
    total: 1,
    limit: 20,
    offset: 0,
    has_more: false,
    total_probe_cases: 0,
};

const catalogDatasets: CatalogDatasetsResponse = {
    items: [
        {
            dataset: "identify_demo",
            dataset_label: "Identify Demo",
            has_verify_cases: false,
            has_identify_gallery: true,
            has_browser_assets: true,
            verify_case_count: 0,
            identify_identity_count: 2,
            browser_item_count: 2,
            browser_validation_status: "pass",
            selection_policy: "deterministic_round_robin",
            available_features: ["identify_gallery", "dataset_browser"],
        },
        {
            dataset: "identify_demo_alt",
            dataset_label: "Identify Demo Alt",
            has_verify_cases: false,
            has_identify_gallery: true,
            has_browser_assets: true,
            verify_case_count: 0,
            identify_identity_count: 1,
            browser_item_count: 1,
            browser_validation_status: "pass",
            selection_policy: "deterministic_round_robin",
            available_features: ["identify_gallery", "dataset_browser"],
        },
    ],
};

const browserDatasetDemo: CatalogDatasetBrowserResponse = {
    dataset: "identify_demo",
    dataset_label: "Identify Demo",
    selection_policy: "deterministic_round_robin",
    validation_status: "pass",
    total: 2,
    limit: 48,
    offset: 0,
    has_more: false,
    generated_at: "2026-04-02T00:00:00Z",
    generator_version: "1.0.0",
    warning_count: 0,
    summary: { items_generated: 2 },
    items: [
        {
            asset_id: "uiasset_subject_001",
            dataset: "identify_demo",
            split: "val",
            subject_id: "1",
            finger: "1",
            capture: "plain",
            modality: "optical_2d",
            ui_eligible: true,
            selection_reason: "Deterministic identify demo preview.",
            selection_policy: "deterministic_round_robin",
            thumbnail_url: "/api/catalog/assets/identify_demo/uiasset_subject_001/thumbnail",
            preview_url: "/api/catalog/assets/identify_demo/uiasset_subject_001/preview",
            availability_status: "available",
            original_dimensions: { width: 640, height: 480 },
            thumbnail_dimensions: { width: 160, height: 120 },
            preview_dimensions: { width: 512, height: 384 },
        },
        {
            asset_id: "uiasset_subject_002",
            dataset: "identify_demo",
            split: "val",
            subject_id: "2",
            finger: "2",
            capture: "roll",
            modality: "optical_2d",
            ui_eligible: true,
            selection_reason: "Deterministic identify demo preview.",
            selection_policy: "deterministic_round_robin",
            thumbnail_url: "/api/catalog/assets/identify_demo/uiasset_subject_002/thumbnail",
            preview_url: "/api/catalog/assets/identify_demo/uiasset_subject_002/preview",
            availability_status: "available",
            original_dimensions: { width: 640, height: 480 },
            thumbnail_dimensions: { width: 160, height: 120 },
            preview_dimensions: { width: 512, height: 384 },
        },
    ],
};

const browserDatasetAlt: CatalogDatasetBrowserResponse = {
    dataset: "identify_demo_alt",
    dataset_label: "Identify Demo Alt",
    selection_policy: "deterministic_round_robin",
    validation_status: "pass",
    total: 1,
    limit: 48,
    offset: 0,
    has_more: false,
    generated_at: "2026-04-02T00:00:00Z",
    generator_version: "1.0.0",
    warning_count: 0,
    summary: { items_generated: 1 },
    items: [
        {
            asset_id: "uiasset_subject_101",
            dataset: "identify_demo_alt",
            split: "eval",
            subject_id: "101",
            finger: "3",
            capture: "latent",
            modality: "optical_2d",
            ui_eligible: true,
            selection_reason: "Deterministic alternate browser preview.",
            selection_policy: "deterministic_round_robin",
            thumbnail_url: "/api/catalog/assets/identify_demo_alt/uiasset_subject_101/thumbnail",
            preview_url: "/api/catalog/assets/identify_demo_alt/uiasset_subject_101/preview",
            availability_status: "available",
            original_dimensions: { width: 640, height: 480 },
            thumbnail_dimensions: { width: 160, height: 120 },
            preview_dimensions: { width: 512, height: 384 },
        },
    ],
};

const successResponse: IdentifyResponse = {
    retrieval_method: "dl",
    rerank_method: "sift",
    threshold: 0.5,
    decision: true,
    total_enrolled: 2,
    candidate_pool_size: 2,
    shortlist_size: 2,
    hints_applied: {},
    top_candidate: {
        rank: 1,
        random_id: "demo_identify_subject_001",
        full_name: "Subject 001",
        national_id_masked: "*****0001",
        created_at: "2026-04-02T00:00:00Z",
        capture: "plain",
        retrieval_score: 0.97,
        rerank_score: 0.95,
        decision: true,
    },
    candidates: [
        {
            rank: 1,
            random_id: "demo_identify_subject_001",
            full_name: "Subject 001",
            national_id_masked: "*****0001",
            created_at: "2026-04-02T00:00:00Z",
            capture: "plain",
            retrieval_score: 0.97,
            rerank_score: 0.95,
            decision: true,
        },
        {
            rank: 2,
            random_id: "demo_identify_subject_002",
            full_name: "Subject 002",
            national_id_masked: "*****0002",
            created_at: "2026-04-02T00:01:00Z",
            capture: "plain",
            retrieval_score: 0.65,
            rerank_score: 0.35,
            decision: false,
        },
    ],
    latency_ms: {
        probe_embed_ms: 12,
        shortlist_scan_ms: 4,
        rerank_ms: 21,
        total_ms: 39,
    },
    storage_layout: {},
};

const noMatchResponse: IdentifyResponse = {
    ...successResponse,
    decision: false,
    top_candidate: {
        ...successResponse.top_candidate!,
        full_name: "Subject 002",
        random_id: "demo_identify_subject_002",
        national_id_masked: "*****0002",
        retrieval_score: 0.52,
        rerank_score: 0.11,
        decision: false,
    },
    candidates: [
        {
            rank: 1,
            random_id: "demo_identify_subject_002",
            full_name: "Subject 002",
            national_id_masked: "*****0002",
            created_at: "2026-04-02T00:01:00Z",
            capture: "plain",
            retrieval_score: 0.52,
            rerank_score: 0.11,
            decision: false,
        },
    ],
};

const shortlistZeroResponse: IdentifyResponse = {
    ...successResponse,
    decision: false,
    candidate_pool_size: 0,
    shortlist_size: 0,
    top_candidate: null,
    candidates: [],
};

const partialLatencyResponse: IdentifyResponse = {
    ...successResponse,
    latency_ms: {
        probe_embed_ms: 12,
        total_ms: 39,
    },
};

const browserSuccessResponse: IdentifyResponse = {
    ...successResponse,
    total_enrolled: 2,
    top_candidate: {
        ...successResponse.top_candidate!,
        random_id: "browser_identify_identify_demo_identity_subject_001",
        full_name: "Subject 001",
    },
    candidates: successResponse.candidates.map((candidate, index) => ({
        ...candidate,
        random_id: index === 0
            ? "browser_identify_identify_demo_identity_subject_001"
            : "browser_identify_identify_demo_identity_subject_002",
    })),
};

type FetchMockOptions = {
    datasets?: CatalogDatasetsResponse;
    galleryByDataset?: Record<string, CatalogIdentifyGalleryResponse>;
    browserByDataset?: Record<string, CatalogDatasetBrowserResponse>;
};

function installFetchMock(options: FetchMockOptions = {}) {
    const requests: string[] = [];
    const datasetsPayload = options.datasets ?? catalogDatasets;
    const galleryByDataset = options.galleryByDataset ?? {
        identify_demo: identifyGallery,
        identify_demo_alt: identifyGalleryAlt,
    };
    const browserByDataset = options.browserByDataset ?? {
        identify_demo: browserDatasetDemo,
        identify_demo_alt: browserDatasetAlt,
    };
    let currentStats = createStats(0, 0, 0);
    let submittedSearchFormData: FormData | null = null;
    let submittedBrowserSeedPayload: Record<string, unknown> | null = null;
    let demoSearchResponsePayload: IdentifyResponse = successResponse;
    let browserSearchResponsePayload: IdentifyResponse = browserSuccessResponse;

    const seedResponse = (): IdentifyDemoSeedResponse => ({
        seeded_count: 2,
        updated_count: currentStats.demo_seeded_count > 0 ? 2 : 0,
        skipped_count: 0,
        total_enrolled: currentStats.total_enrolled,
        demo_seeded_count: 2,
        storage_layout: currentStats.storage_layout,
        notice: "Demo store now contains 2 seeded identities.",
    });

    const resetResponse = (): IdentifyDemoResetResponse => ({
        removed_count: 2,
        total_enrolled: currentStats.total_enrolled,
        demo_seeded_count: 0,
        storage_layout: currentStats.storage_layout,
        notice: "Demo store reset completed.",
    });

    const browserSeedResponse = (): IdentifyBrowserSeedSelectionResponse => {
        const dataset = String(submittedBrowserSeedPayload?.dataset ?? "identify_demo");
        const selectedIds = Array.isArray(submittedBrowserSeedPayload?.selected_identity_ids)
            ? (submittedBrowserSeedPayload?.selected_identity_ids as string[])
            : [];
        return {
            dataset,
            selected_count: selectedIds.length,
            seeded_count: selectedIds.length,
            updated_count: 0,
            skipped_count: 0,
            total_enrolled: currentStats.total_enrolled,
            browser_seeded_count: selectedIds.length,
            store_ready: selectedIds.length > 0,
            seeded_identity_ids: selectedIds,
            storage_layout: currentStats.storage_layout,
            warnings: [],
            errors: [],
            notice: `Browser store prepared with ${selectedIds.length} selected identities.`,
        };
    };

    const browserResetResponse = (): IdentifyBrowserResetResponse => ({
        removed_count: currentStats.browser_seeded_count,
        total_enrolled: currentStats.total_enrolled,
        browser_seeded_count: 0,
        storage_layout: currentStats.storage_layout,
        notice: "Browser-seeded store reset completed. Operational enrollments were not touched.",
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        requests.push(url);
        const parsedUrl = new URL(url, "http://localhost");

        if (parsedUrl.pathname === "/api/catalog/identify-gallery") {
            const dataset = parsedUrl.searchParams.get("dataset");
            return createJsonResponse(dataset ? galleryByDataset[dataset] : identifyGallery);
        }

        if (parsedUrl.pathname === "/api/catalog/datasets") {
            return createJsonResponse(datasetsPayload);
        }

        if (parsedUrl.pathname === "/api/catalog/dataset-browser") {
            const dataset = parsedUrl.searchParams.get("dataset");
            if (!dataset || !browserByDataset[dataset]) {
                throw new Error(`Unexpected dataset browser request: ${url}`);
            }
            return createJsonResponse(browserByDataset[dataset]);
        }

        if (url === "/api/identify/stats") {
            return createJsonResponse(currentStats);
        }

        if (url === "/api/identify/demo/seed") {
            currentStats = createStats(
                currentStats.total_enrolled - currentStats.demo_seeded_count + 2,
                2,
                currentStats.browser_seeded_count,
            );
            return createJsonResponse(seedResponse());
        }

        if (url === "/api/identify/demo/reset") {
            currentStats = createStats(
                Math.max(currentStats.total_enrolled - currentStats.demo_seeded_count, 0),
                0,
                currentStats.browser_seeded_count,
            );
            return createJsonResponse(resetResponse());
        }

        if (url === identifyGallery.probe_cases[0].probe_asset_url || url === identifyGallery.probe_cases[1].probe_asset_url || url === identifyGallery.probe_cases[2].probe_asset_url) {
            return createImageResponse(url);
        }

        if (url === "/api/identify/browser/seed-selection") {
            submittedBrowserSeedPayload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
            const selectedIds = Array.isArray(submittedBrowserSeedPayload.selected_identity_ids)
                ? (submittedBrowserSeedPayload.selected_identity_ids as string[])
                : [];
            currentStats = createStats(currentStats.total_enrolled, currentStats.demo_seeded_count, selectedIds.length);
            return createJsonResponse(browserSeedResponse());
        }

        if (url === "/api/identify/browser/reset") {
            currentStats = createStats(currentStats.total_enrolled, currentStats.demo_seeded_count, 0);
            return createJsonResponse(browserResetResponse());
        }

        if (url.startsWith("/api/catalog/assets/identify_demo/") || url.startsWith("/api/catalog/assets/identify_demo_alt/")) {
            return createImageResponse(url);
        }

        if (url === "/api/identify/search") {
            submittedSearchFormData = init?.body as FormData;
            const storeScope = String(submittedSearchFormData?.get("store_scope") ?? "operational");
            return createJsonResponse(storeScope === "browser" ? browserSearchResponsePayload : demoSearchResponsePayload);
        }

        if (url === "/api/identify/enroll") {
            currentStats = createStats(currentStats.total_enrolled + 1, currentStats.demo_seeded_count, currentStats.browser_seeded_count);
            return createJsonResponse({
                random_id: "manual_identity_001",
                created_at: "2026-04-02T00:00:00Z",
                vector_methods: ["dl", "vit"],
                image_sha256: "hash",
                storage_layout: currentStats.storage_layout,
            });
        }

        if (url === "/api/identify/person/manual_identity_001" && init?.method === "DELETE") {
            currentStats = createStats(
                Math.max(currentStats.total_enrolled - 1, 0),
                currentStats.demo_seeded_count,
                currentStats.browser_seeded_count,
            );
            return createJsonResponse({
                random_id: "manual_identity_001",
                removed: true,
                storage_layout: currentStats.storage_layout,
            });
        }

        throw new Error(`Unexpected fetch call: ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);
    return {
        requests,
        setSearchResponsePayload: (payload: IdentifyResponse) => {
            demoSearchResponsePayload = payload;
        },
        setBrowserSearchResponsePayload: (payload: IdentifyResponse) => {
            browserSearchResponsePayload = payload;
        },
        getSubmittedSearchFormData: () => submittedSearchFormData,
        getSubmittedBrowserSeedPayload: () => submittedBrowserSeedPayload,
    };
}

afterEach(() => {
    localStorage.clear();
    sessionStorage.clear();
});

describe("Identification demo gallery workspace", () => {
    it("loads the identify gallery by default and keeps the selected probe stable across mode switches", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Positive identify probe");
            expect(normalizeText(container.textContent)).toContain("Run guided identification");
            expect(normalizeText(container.textContent)).toContain("Expected Subject 001");
        });

        expect(Array.from(container.querySelectorAll('input[type="file"]')).length).toBe(0);

        await click(getButtonByText(container, "No match walkthrough"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("No match walkthrough");
        });

        await click(getButtonByText(container, "Operational Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Enroll identity");
            expect(normalizeText(container.textContent)).toContain("Search identity");
        });

        await click(getButtonByText(container, "Demo Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("No match walkthrough");
        });

        const sourcePaths = [
            "../src/features/identification/IdentificationWorkspaceProductScreen.tsx",
            "../src/features/identification/hooks/useIdentification.ts",
            "../src/api/identificationService.ts",
        ];
        for (const sourcePath of sourcePaths) {
            const source = fs.readFileSync(new URL(sourcePath, import.meta.url), "utf-8");
            expect(source.includes("data/samples")).toBe(false);
            expect(source.includes("data/processed")).toBe(false);
            expect(source.includes("postgresql://")).toBe(false);
        }

        await unmountWorkspace(root);
    });

    it("shows both split classic methods in demo and operational rerank selectors", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Positive identify probe");
        });

        const demoRerankField = getLabelField<HTMLSelectElement>(container, "Re-rank method");
        const demoLabels = Array.from(demoRerankField.options).map((option) => normalizeText(option.textContent));
        expect(demoLabels).toContain("Classic (ORB)");
        expect(demoLabels).toContain("Classic (ROI GFTT+ORB)");

        await click(getButtonByText(container, "Operational Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Search identity");
        });

        const operationalRerankField = getLabelField<HTMLSelectElement>(container, "Re-rank method");
        const operationalLabels = Array.from(operationalRerankField.options).map((option) => normalizeText(option.textContent));
        expect(operationalLabels).toContain("Classic (ORB)");
        expect(operationalLabels).toContain("Classic (ROI GFTT+ORB)");

        await unmountWorkspace(root);
    });

    it("seeds and resets the demo store through dedicated endpoints and refreshes stats without breaking probe state", async () => {
        const { requests } = installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Demo Store");
        });

        await click(getButtonByText(container, "Seed demo identities"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Demo seeding completed");
            expect(normalizeText(container.textContent)).toContain("Demo seeded identities2");
        });

        expect(requests).toContain("/api/identify/demo/seed");
        expect(requests.filter((request) => request === "/api/identify/stats").length).toBeGreaterThanOrEqual(2);

        await click(getButtonByText(container, "Reset demo store"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Demo reset completed");
            expect(normalizeText(container.textContent)).toContain("Positive identify probe");
            expect(normalizeText(container.textContent)).toContain("Seed demo store first");
        });

        expect(requests).toContain("/api/identify/demo/reset");

        await unmountWorkspace(root);
    });

    it("renders Browser mode, avoids file pickers, and restores the selected browser context across reload", async () => {
        installFetchMock();
        let rendered = await renderWorkspace();

        await click(getButtonByText(rendered.container, "Browser Mode"));
        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Browser workspace");
            expect(getButtonByText(rendered.container, "Identify Demo").getAttribute("aria-pressed")).toBe("true");
        });

        expect(Array.from(rendered.container.querySelectorAll('input[type="file"]')).length).toBe(0);

        await click(getButtonByText(rendered.container, "Use as probe"));
        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("uiasset_subject_001");
        });

        await changeFieldValue(getLabelField<HTMLSelectElement>(rendered.container, "Re-rank method"), "dl");
        await changeFieldValue(getLabelField<HTMLInputElement>(rendered.container, "Shortlist size"), "11");

        await unmountWorkspace(rendered.root);
        rendered = await renderWorkspace();

        await waitFor(() => {
            expect(getButtonByText(rendered.container, "Browser Mode").getAttribute("aria-pressed")).toBe("true");
            expect(normalizeText(rendered.container.textContent)).toContain("uiasset_subject_001");
        });

        expect(getButtonByText(rendered.container, "Identify Demo").getAttribute("aria-pressed")).toBe("true");
        expect(getLabelField<HTMLSelectElement>(rendered.container, "Re-rank method").value).toBe("dl");
        expect(getLabelField<HTMLInputElement>(rendered.container, "Shortlist size").value).toBe("11");

        await unmountWorkspace(rendered.root);
    });

    it("shows Browser guardrails when no probe is selected and does not expose Verify pair-builder controls", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Browser Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Browser workspace");
            expect(normalizeText(container.textContent)).toContain("Select a probe asset from the dataset browser before running browser identification.");
        });

        expect(getButtonByText(container, "Seed gallery and run").disabled).toBe(true);
        expect(normalizeText(container.textContent)).not.toContain("Use as verify pair");
        expect(normalizeText(container.textContent)).not.toContain("Pair Status");
        expect(normalizeText(container.textContent)).toContain("Use as probe");

        await unmountWorkspace(root);
    });

    it("cannot run Browser mode when the selected dataset exposes no gallery identities", async () => {
        installFetchMock({
            datasets: {
                items: [
                    {
                        dataset: "identify_empty_browser",
                        dataset_label: "Identify Empty Browser",
                        has_verify_cases: false,
                        has_identify_gallery: true,
                        has_browser_assets: true,
                        verify_case_count: 0,
                        identify_identity_count: 0,
                        browser_item_count: 1,
                        browser_validation_status: "pass",
                        selection_policy: "deterministic_round_robin",
                        available_features: ["identify_gallery", "dataset_browser"],
                    },
                ],
            },
            galleryByDataset: {
                identify_empty_browser: {
                    items: [],
                    demo_identities: [],
                    probe_cases: [],
                    total: 0,
                    limit: 20,
                    offset: 0,
                    has_more: false,
                    total_probe_cases: 0,
                },
            },
            browserByDataset: {
                identify_empty_browser: {
                    dataset: "identify_empty_browser",
                    dataset_label: "Identify Empty Browser",
                    selection_policy: "deterministic_round_robin",
                    validation_status: "pass",
                    total: 1,
                    limit: 48,
                    offset: 0,
                    has_more: false,
                    generated_at: "2026-04-02T00:00:00Z",
                    generator_version: "1.0.0",
                    warning_count: 0,
                    summary: { items_generated: 1 },
                    items: [
                        {
                            asset_id: "uiasset_empty_subject",
                            dataset: "identify_empty_browser",
                            split: "eval",
                            subject_id: "404",
                            finger: "4",
                            capture: "plain",
                            modality: "optical_2d",
                            ui_eligible: true,
                            selection_reason: "Single browser asset for empty gallery coverage.",
                            selection_policy: "deterministic_round_robin",
                            thumbnail_url: "/api/catalog/assets/identify_empty_browser/uiasset_empty_subject/thumbnail",
                            preview_url: "/api/catalog/assets/identify_empty_browser/uiasset_empty_subject/preview",
                            availability_status: "available",
                            original_dimensions: { width: 640, height: 480 },
                            thumbnail_dimensions: { width: 160, height: 120 },
                            preview_dimensions: { width: 512, height: 384 },
                        },
                    ],
                },
            },
        });
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Browser Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("This dataset does not currently expose any identify-gallery identities for Browser mode.");
            expect(normalizeText(container.textContent)).toContain("Select at least one gallery identity before running browser identification.");
        });

        expect(getButtonByText(container, "Seed gallery and run").disabled).toBe(true);

        await unmountWorkspace(root);
    });

    it("runs Browser mode through browser seeding plus store_scope=browser, then resets dependent state on dataset switch", async () => {
        const controls = installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Browser Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Probe browser");
        });

        await click(getButtonByText(container, "Use as probe"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("uiasset_subject_001");
        });

        await changeFieldValue(getLabelField<HTMLInputElement>(container, "Subject ID"), "1");
        expect(getLabelField<HTMLInputElement>(container, "Subject ID").value).toBe("1");

        await click(getButtonByText(container, "Seed gallery and run"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Latest browser result");
            expect(normalizeText(container.textContent)).toContain("Top candidate");
        });

        const browserSeedPayload = controls.getSubmittedBrowserSeedPayload();
        expect(browserSeedPayload).not.toBeNull();
        expect(browserSeedPayload?.dataset).toBe("identify_demo");
        expect(browserSeedPayload?.selected_identity_ids).toEqual(["identity_subject_001"]);

        const submittedFormData = controls.getSubmittedSearchFormData();
        expect(submittedFormData?.get("store_scope")).toBe("browser");
        expect(submittedFormData?.get("retrieval_method")).toBe("dl");
        expect((submittedFormData?.get("img") as File).name).toBe("uiasset_subject_001_preview.png");

        await click(getButtonByText(container, "Identify Demo Alt"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("No browser identification result yet");
            expect(normalizeText(container.textContent)).toContain("No probe selected yet");
        });

        expect(getLabelField<HTMLInputElement>(container, "Subject ID").value).toBe("");

        await unmountWorkspace(root);
    });

    it("keeps operational controls intact after Browser mode seeding and reset", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Browser Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Browser workspace");
        });

        await click(getButtonByText(container, "Use as probe"));
        await click(getButtonByText(container, "Seed gallery and run"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Latest browser result");
        });

        await click(getButtonByText(container, "Reset browser store"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Browser-seeded store reset completed");
        });

        await click(getButtonByText(container, "Operational Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Enroll identity");
            expect(normalizeText(container.textContent)).toContain("Search identity");
            expect(normalizeText(container.textContent)).toContain("Delete identity");
        });

        expect(normalizeText(container.textContent)).not.toContain("Use as verify pair");

        await unmountWorkspace(root);
    });

    it("shows a clear empty state when no browser-ready datasets are available", async () => {
        installFetchMock({ datasets: { items: [] } });
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Browser Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("No browser-ready datasets");
            expect(normalizeText(container.textContent)).toContain("The catalog does not currently expose any datasets with both identify-gallery metadata and browser assets.");
        });

        await unmountWorkspace(root);
    });

    it("runs guided identification through the server probe asset URL and /api/identify/search, then shows top candidate and shortlist", async () => {
        const { requests, getSubmittedSearchFormData } = installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Seed demo identities"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Demo seeding completed");
        });

        await click(getButtonByText(container, "Run identification"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Top candidate");
            expect(normalizeText(container.textContent)).toContain("Subject 001");
            expect(normalizeText(container.textContent)).toContain("Candidate ranking");
            expect(normalizeText(container.textContent)).toContain("Expected vs actual");
            expect(normalizeText(container.textContent)).toContain("Method story");
            expect(normalizeText(container.textContent)).toContain("Confidence band");
        });

        expect(requests).toContain("/api/catalog/assets/identify_demo/probe_positive/preview");
        expect(requests).toContain("/api/identify/search");

        const submittedFormData = getSubmittedSearchFormData();
        expect(submittedFormData).not.toBeNull();
        expect((submittedFormData?.get("img") as File).name).toBe("probe_positive_probe.png");
        expect(submittedFormData?.get("retrieval_method")).toBe("dl");
        expect(submittedFormData?.get("rerank_method")).toBe("sift");

        await unmountWorkspace(root);
    });

    it("renders no-match and shortlist-zero negative paths clearly", async () => {
        const controls = installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Seed demo identities"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Demo seeding completed");
        });

        controls.setSearchResponsePayload(noMatchResponse);
        await click(getButtonByText(container, "No match walkthrough"));
        await click(getButtonByText(container, "Run identification"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("NO MATCH");
            expect(normalizeText(container.textContent)).toContain("Expected vs actual");
        });

        controls.setSearchResponsePayload(shortlistZeroResponse);
        await click(getButtonByText(container, "Shortlist zero walkthrough"));
        await click(getButtonByText(container, "Run identification"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("No shortlist candidates");
        });

        await unmountWorkspace(root);
    });

    it("keeps the latency story readable when only partial latency metadata is returned", async () => {
        const controls = installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Seed demo identities"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Demo seeding completed");
        });

        controls.setSearchResponsePayload(partialLatencyResponse);
        await click(getButtonByText(container, "Run identification"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Latency");
            expect(normalizeText(container.textContent)).toContain("39.0 ms");
            expect(normalizeText(container.textContent)).toContain("Probe embed: 12.0 ms");
        });

        await unmountWorkspace(root);
    });

    it("preserves operational enroll, manual search, and delete workflows", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Operational Mode"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Enroll identity");
            expect(normalizeText(container.textContent)).toContain("Delete identity");
            expect(normalizeText(container.textContent)).toContain("Search identity");
        });

        const fileInputs = Array.from(container.querySelectorAll('input[type="file"]')) as HTMLInputElement[];
        await uploadFile(fileInputs[0], new File([new Blob(["manual-enroll"], { type: "image/png" })], "manual-enroll.png", { type: "image/png" }));
        await changeFieldValue(getLabelField<HTMLInputElement>(container, "Full name"), "Manual Person");
        await changeFieldValue(getLabelField<HTMLInputElement>(container, "National ID"), "123456789");
        await click(getButtonByText(container, "Enroll identity"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Enrollment completed");
            expect(normalizeText(container.textContent)).toContain("manual_identity_001");
        });

        await uploadFile(fileInputs[1], new File([new Blob(["manual-search"], { type: "image/png" })], "manual-search.png", { type: "image/png" }));
        await click(getButtonByText(container, "Search identities"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Candidate ranking");
        });

        await changeFieldValue(getLabelField<HTMLInputElement>(container, "Random ID"), "manual_identity_001");
        const deleteConfirmLabel = Array.from(container.querySelectorAll("label")).find((label) =>
            normalizeText(label.textContent).includes("I understand this will permanently purge the selected identity."),
        );
        const deleteCheckbox = deleteConfirmLabel?.querySelector('input[type="checkbox"]') as HTMLInputElement;
        await act(async () => {
            deleteCheckbox.dispatchEvent(new MouseEvent("click", { bubbles: true }));
        });
        await click(getButtonByText(container, "Delete identity"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Identity removed");
        });

        await unmountWorkspace(root);
    });

    it("rehydrates the selected probe, recent probes, and pinned probes across reload", async () => {
        installFetchMock();
        let rendered = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Positive identify probe");
        });

        await click(getButtonByText(rendered.container, "No match walkthrough"));
        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Selected probe");
            expect(normalizeText(rendered.container.textContent)).toContain("No match walkthrough");
        });

        await click(getButtonByText(rendered.container, "Pin probe"));
        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Pinned probes");
            expect(normalizeText(rendered.container.textContent)).toContain("Recent probes");
        });

        const persistedState = JSON.parse(localStorage.getItem("fp-research.workspace.identification.local") ?? "{}");
        expect(persistedState.data.recentProbes).toHaveLength(1);
        expect(persistedState.data.recentProbes[0].id).toBe("probe_no_match");

        await unmountWorkspace(rendered.root);
        rendered = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Selected probe");
            expect(normalizeText(rendered.container.textContent)).toContain("Pinned probes");
            expect(normalizeText(rendered.container.textContent)).toContain("Recent probes");
            expect(normalizeText(rendered.container.textContent)).toContain("No match walkthrough");
        });

        await unmountWorkspace(rendered.root);
    });

    it("restores operational mode without auto-running identification on reload", async () => {
        const { requests } = installFetchMock();
        let rendered = await renderWorkspace();

        await click(getButtonByText(rendered.container, "Operational Mode"));
        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Enroll identity");
            expect(normalizeText(rendered.container.textContent)).toContain("Search identity");
        });

        const searchCallsBeforeReload = requests.filter((request) => request === "/api/identify/search").length;
        await unmountWorkspace(rendered.root);

        rendered = await renderWorkspace();
        await waitFor(() => {
            expect(getButtonByText(rendered.container, "Operational Mode").getAttribute("aria-pressed")).toBe("true");
            expect(normalizeText(rendered.container.textContent)).toContain("Search identity");
        });

        const searchCallsAfterReload = requests.filter((request) => request === "/api/identify/search").length;
        expect(searchCallsAfterReload).toBe(searchCallsBeforeReload);

        await unmountWorkspace(rendered.root);
    });

    it("clears identification continuity and removes stale persisted probe references safely", async () => {
        localStorage.setItem("fp-research.workspace.identification.local", JSON.stringify({
            version: 1,
            savedAt: "2026-04-02T00:00:00Z",
            data: {
                mode: "demo",
                selectedProbeCaseId: "missing_probe",
                recentProbes: [
                    {
                        id: "missing_probe",
                        title: "Missing probe",
                        datasetLabel: "Missing dataset",
                        expectedOutcome: "no_match",
                    },
                ],
                pinnedProbeCaseIds: ["missing_probe"],
                demoSearchPreferences: {
                    retrievalMethod: "dl",
                    rerankMethod: "sift",
                    shortlistSizeText: "10",
                    thresholdText: "",
                    advancedVisible: false,
                    namePattern: "",
                    nationalIdPattern: "",
                    createdFrom: "",
                    createdTo: "",
                },
                operationalSearchPreferences: {
                    capture: "plain",
                    retrievalMethod: "dl",
                    rerankMethod: "sift",
                    shortlistSizeText: "25",
                    thresholdText: "",
                    namePattern: "",
                    nationalIdPattern: "",
                    createdFrom: "",
                    createdTo: "",
                },
            },
        }));

        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Positive identify probe");
            expect(normalizeText(container.textContent)).not.toContain("Missing probe");
        });

        await click(getButtonByText(container, "Clear saved Identification workspace"));
        await waitFor(() => {
            expect(getButtonByText(container, "Demo Mode").getAttribute("aria-pressed")).toBe("true");
            expect(normalizeText(container.textContent)).not.toContain("Pinned probes");
            expect(normalizeText(container.textContent)).not.toContain("Recent probes");
        });

        const storedState = JSON.parse(localStorage.getItem("fp-research.workspace.identification.local") ?? "{}");
        expect(storedState.data.mode).toBe("demo");
        expect(storedState.data.pinnedProbeCaseIds).toEqual([]);
        expect(storedState.data.recentProbes).toEqual([]);
        expect(storedState.data.browser.selectedProbeAsset).toBeNull();

        await unmountWorkspace(root);
    });
});
