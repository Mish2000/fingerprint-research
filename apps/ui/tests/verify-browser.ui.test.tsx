import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import VerifyWorkspace from "../src/features/verify/VerifyWorkspace.tsx";
import type { CatalogBrowserItem, CatalogDatasetSummary, CatalogVerifyCase, MatchResponse } from "../src/types/index.ts";

type RenderedWorkspace = {
    container: HTMLDivElement;
    root: Root;
};

const demoCases: CatalogVerifyCase[] = [
    {
        case_id: "case_easy_match",
        title: "Easy genuine",
        description: "A clean genuine pair for a first-run demo.",
        dataset: "nist_sd300b",
        dataset_label: "NIST SD300B",
        split: "val",
        difficulty: "easy",
        case_type: "easy_genuine",
        ground_truth: "match",
        recommended_method: "sift",
        capture_a: "plain",
        capture_b: "roll",
        modality_relation: null,
        tags: ["easy", "genuine"],
        selection_policy: "benchmark_top",
        selection_reason: "High-confidence reference case for the default walkthrough.",
        image_a_url: "/api/demo/cases/case_easy_match/a/probe.png",
        image_b_url: "/api/demo/cases/case_easy_match/b/reference.png",
        availability_status: "available",
        asset_a_id: "asset_easy_a",
        asset_b_id: "asset_easy_b",
    },
];

const datasets: CatalogDatasetSummary[] = [
    {
        dataset: "nist_sd300b",
        dataset_label: "NIST SD300B",
        has_verify_cases: true,
        has_identify_gallery: false,
        has_browser_assets: true,
        verify_case_count: 1,
        identify_identity_count: 0,
        browser_item_count: 3,
        browser_validation_status: "pass",
        selection_policy: "deterministic_round_robin",
        available_features: ["verify_cases", "dataset_browser"],
        demo_health: {
            planned_verify_cases: 1,
            built_verify_cases: 1,
            benchmark_backed_cases: 0,
            heuristic_fallback_cases: 1,
            missing_benchmark_evidence: true,
            status: "degraded",
            note: "1 of 1 curated verify case(s) use heuristic fallback because benchmark evidence is incomplete.",
        },
    },
    {
        dataset: "legacy_only",
        dataset_label: "Legacy Only",
        has_verify_cases: false,
        has_identify_gallery: false,
        has_browser_assets: false,
        verify_case_count: 0,
        identify_identity_count: 0,
        browser_item_count: 0,
        browser_validation_status: null,
        selection_policy: null,
        available_features: [],
        demo_health: null,
    },
];

const browserItems: CatalogBrowserItem[] = [
    {
        asset_id: "asset_browser_a",
        dataset: "nist_sd300b",
        split: "test",
        subject_id: "100001",
        finger: "1",
        capture: "plain",
        modality: "optical_2d",
        ui_eligible: true,
        selection_reason: "Representative test item A.",
        selection_policy: "deterministic_round_robin",
        thumbnail_url: "/api/catalog/assets/nist_sd300b/asset_browser_a/thumbnail",
        preview_url: "/api/catalog/assets/nist_sd300b/asset_browser_a/preview",
        availability_status: "available",
        original_dimensions: { width: 640, height: 480 },
        thumbnail_dimensions: { width: 160, height: 120 },
        preview_dimensions: { width: 512, height: 384 },
    },
    {
        asset_id: "asset_browser_b",
        dataset: "nist_sd300b",
        split: "val",
        subject_id: "100002",
        finger: "2",
        capture: "roll",
        modality: "optical_2d",
        ui_eligible: true,
        selection_reason: "Representative validation item B.",
        selection_policy: "deterministic_round_robin",
        thumbnail_url: "/api/catalog/assets/nist_sd300b/asset_browser_b/thumbnail",
        preview_url: "/api/catalog/assets/nist_sd300b/asset_browser_b/preview",
        availability_status: "available",
        original_dimensions: { width: 640, height: 480 },
        thumbnail_dimensions: { width: 160, height: 120 },
        preview_dimensions: { width: 512, height: 384 },
    },
    {
        asset_id: "asset_browser_c",
        dataset: "nist_sd300b",
        split: "train",
        subject_id: "100003",
        finger: "3",
        capture: "contactless",
        modality: "latent",
        ui_eligible: true,
        selection_reason: "Representative train item C.",
        selection_policy: "deterministic_round_robin",
        thumbnail_url: "/api/catalog/assets/nist_sd300b/asset_browser_c/thumbnail",
        preview_url: "/api/catalog/assets/nist_sd300b/asset_browser_c/preview",
        availability_status: "available",
        original_dimensions: { width: 640, height: 480 },
        thumbnail_dimensions: { width: 160, height: 120 },
        preview_dimensions: { width: 512, height: 384 },
    },
];

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

function createMatchResponse(): MatchResponse {
    return {
        method: "sift",
        score: 0.92,
        decision: true,
        threshold: 0.01,
        latency_ms: 18,
        meta: { matches: 34, inliers: 21 },
        overlay: null,
    };
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
    const match = Array.from(container.querySelectorAll("label")).find((field) =>
        normalizeText(field.textContent).includes(label),
    );
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
        field.value = value;
        field.dispatchEvent(new Event("input", { bubbles: true }));
        field.dispatchEvent(new Event("change", { bubbles: true }));
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
        root.render(<VerifyWorkspace />);
    });

    await flush();
    return { container, root };
}

async function unmountWorkspace(root: Root): Promise<void> {
    await act(async () => {
        root.unmount();
    });
}

function browserPageFor(url: string): CatalogBrowserItem[] {
    const params = new URL(url, "http://localhost").searchParams;
    const split = params.get("split");

    if (split) {
        return browserItems.filter((item) => item.split === split);
    }

    return browserItems;
}

function installFetchMock(
    overrides: Partial<Record<string, (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>>> = {},
) {
    const requests: string[] = [];
    let submittedMatchFormData: FormData | null = null;

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        requests.push(url);

        if (overrides[url]) {
            return overrides[url]!(input, init);
        }

        if (url === "/api/catalog/verify-cases") {
            return createJsonResponse({
                items: demoCases,
                total: demoCases.length,
                limit: demoCases.length,
                offset: 0,
                has_more: false,
            });
        }

        if (url === "/api/catalog/datasets") {
            return createJsonResponse({
                items: datasets,
                catalog_build_health: {
                    catalog_build_status: "degraded",
                    total_verify_cases: 1,
                    benchmark_backed_case_count: 0,
                    heuristic_fallback_case_count: 1,
                    datasets_with_missing_benchmark_evidence: ["nist_sd300b"],
                    summary_message: "1 of 1 curated verify case(s) use heuristic fallback across 1 dataset(s).",
                },
            });
        }

        if (url.startsWith("/api/catalog/dataset-browser?")) {
            const items = browserPageFor(url);
            return createJsonResponse({
                dataset: "nist_sd300b",
                dataset_label: "NIST SD300B",
                selection_policy: "deterministic_round_robin",
                validation_status: "pass",
                total: items.length,
                limit: 48,
                offset: 0,
                has_more: false,
                generated_at: "2026-03-31T00:00:00Z",
                generator_version: "1.0.0",
                warning_count: 0,
                summary: {
                    items_by_split: { test: 1, val: 1, train: 1 },
                    items_by_capture: { plain: 1, roll: 1, contactless: 1 },
                    items_by_modality: { optical_2d: 2, latent: 1 },
                },
                items,
            });
        }

        const matchedBrowserItem = browserItems.find((item) => item.preview_url === url || item.thumbnail_url === url);
        if (matchedBrowserItem) {
            return createImageResponse(matchedBrowserItem.asset_id);
        }

        if (url === "/api/match") {
            submittedMatchFormData = init?.body as FormData;
            return createJsonResponse(createMatchResponse());
        }

        throw new Error(`Unexpected fetch call: ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);
    return {
        fetchMock,
        requests,
        getSubmittedMatchFormData: () => submittedMatchFormData,
    };
}

afterEach(() => {
    localStorage.clear();
    sessionStorage.clear();
});

describe("Verify Workspace dataset browser flow", () => {
    it("loads browser-ready datasets and dataset-browser items from catalog endpoints", async () => {
        const { requests } = installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Dataset Browser"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("NIST SD300B");
            expect(normalizeText(container.textContent)).toContain("asset_browser_a");
            expect(normalizeText(container.textContent)).toContain("asset_browser_b");
        });

        expect(requests).toContain("/api/catalog/datasets");
        expect(requests).toContain("/api/catalog/dataset-browser?dataset=nist_sd300b&limit=48&offset=0&sort=default");

        await unmountWorkspace(root);
    });

    it("shows dataset-level demo health warnings without hiding browser-ready datasets", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Dataset Browser"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Degraded demo evidence");
            expect(normalizeText(container.textContent)).toContain("1 of 1 curated verify case(s) use heuristic fallback");
            expect(normalizeText(container.textContent)).toContain("NIST SD300B");
        });

        await unmountWorkspace(root);
    });

    it("updates the dataset-browser query and results when filters change", async () => {
        const { requests } = installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Dataset Browser"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("asset_browser_a");
        });

        const splitField = getLabelField<HTMLSelectElement>(container, "Split");
        await changeFieldValue(splitField, "train");

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("asset_browser_c");
            expect(normalizeText(container.textContent)).not.toContain("asset_browser_a");
        });

        expect(requests.some((request) => request.includes("/api/catalog/dataset-browser?dataset=nist_sd300b&split=train"))).toBe(true);

        await unmountWorkspace(root);
    });

    it("supports selecting A and B, clearing A, and preserving browser state across mode switches", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Dataset Browser"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("asset_browser_a");
        });

        await click(getButtonByText(container, "Choose A"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Side A");
            expect(normalizeText(container.textContent)).toContain("asset_browser_a");
        });

        await click(getButtonByText(container, "Choose B"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("asset_browser_b");
            expect(normalizeText(container.textContent)).toContain("Use as verify pair");
        });

        await click(getButtonByText(container, "Clear A"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Waiting for selection");
            expect(normalizeText(container.textContent)).toContain("asset_browser_b");
        });

        await click(getButtonByText(container, "Manual Upload"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Manual Upload stays intentionally separate");
        });

        await click(getButtonByText(container, "Dataset Browser"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("asset_browser_b");
            expect(normalizeText(container.textContent)).toContain("Waiting for selection");
        });

        await unmountWorkspace(root);
    });

    it("loads the selected browser pair through server asset URLs and runs verify after apply", async () => {
        const { getSubmittedMatchFormData } = installFetchMock();
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Dataset Browser"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("asset_browser_a");
        });

        await click(getButtonByText(container, "Choose A"));
        await click(getButtonByText(container, "Choose B"));
        await click(getButtonByText(container, "Swap A / B"));
        await click(getButtonByText(container, "Use as verify pair"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Pair loaded into Verify");
        });

        const captureAField = getLabelField<HTMLSelectElement>(container, "Capture A");
        const captureBField = getLabelField<HTMLSelectElement>(container, "Capture B");
        expect(captureAField.value).toBe("roll");
        expect(captureBField.value).toBe("plain");

        await click(getButtonByText(container, "Run Verification"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Same finger");
            expect(normalizeText(container.textContent)).toContain("Rolled vs Plain");
            expect(normalizeText(container.textContent)).toContain("Confidence band");
        });

        const submittedFormData = getSubmittedMatchFormData();
        expect(submittedFormData).not.toBeNull();
        expect((submittedFormData?.get("img_a") as File).name).toBe("asset_browser_b_preview.png");
        expect((submittedFormData?.get("img_b") as File).name).toBe("asset_browser_a_preview.png");

        await unmountWorkspace(root);
    });

    it("keeps the selected pair visible when browser asset loading fails", async () => {
        installFetchMock({
            "/api/catalog/assets/nist_sd300b/asset_browser_b/preview": () =>
                new Response(JSON.stringify({ detail: "Asset missing" }), {
                    status: 404,
                    headers: { "content-type": "application/json" },
                }),
        });

        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Dataset Browser"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("asset_browser_a");
        });

        await click(getButtonByText(container, "Choose A"));
        await click(getButtonByText(container, "Choose B"));
        await click(getButtonByText(container, "Use as verify pair"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Failed to load browser asset (404): Asset missing");
            expect(normalizeText(container.textContent)).toContain("asset_browser_a");
            expect(normalizeText(container.textContent)).toContain("asset_browser_b");
        });

        await unmountWorkspace(root);
    });

    it("rehydrates dataset-browser filters and pair context across reload without auto-applying or auto-running", async () => {
        const { fetchMock } = installFetchMock();
        let rendered = await renderWorkspace();

        await click(getButtonByText(rendered.container, "Dataset Browser"));
        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("asset_browser_a");
        });

        await click(getButtonByText(rendered.container, "Choose A"));
        await click(getButtonByText(rendered.container, "Choose B"));

        const splitField = getLabelField<HTMLSelectElement>(rendered.container, "Split");
        await changeFieldValue(splitField, "train");
        await waitFor(() => {
            expect(splitField.value).toBe("train");
            expect(normalizeText(rendered.container.textContent)).toContain("asset_browser_c");
        });

        const matchCallsBeforeReload = fetchMock.mock.calls.filter(([input]) => String(input) === "/api/match").length;
        await unmountWorkspace(rendered.root);

        rendered = await renderWorkspace();
        await waitFor(() => {
            expect(getButtonByText(rendered.container, "Dataset Browser").getAttribute("aria-pressed")).toBe("true");
            expect(getLabelField<HTMLSelectElement>(rendered.container, "Split").value).toBe("train");
            expect(normalizeText(rendered.container.textContent)).toContain("asset_browser_a");
            expect(normalizeText(rendered.container.textContent)).toContain("asset_browser_b");
            expect(normalizeText(rendered.container.textContent)).toContain("Use as verify pair");
        });

        const matchCallsAfterReload = fetchMock.mock.calls.filter(([input]) => String(input) === "/api/match").length;
        expect(matchCallsAfterReload).toBe(matchCallsBeforeReload);

        await unmountWorkspace(rendered.root);
    });

    it("falls back safely when persisted browser dataset context is stale", async () => {
        localStorage.setItem("fp-research.workspace.verify.local", JSON.stringify({
            version: 1,
            savedAt: "2026-04-02T00:00:00Z",
            data: {
                mode: "browser",
                demoFilter: "all",
                selectedDemoCaseId: null,
                pinnedDemoCaseIds: [],
                browser: {
                    selectedDatasetKey: "missing_dataset",
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
                manualPair: null,
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
            },
        }));

        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(getButtonByText(container, "Dataset Browser").getAttribute("aria-pressed")).toBe("true");
            expect(normalizeText(container.textContent)).toContain("NIST SD300B");
            expect(normalizeText(container.textContent)).toContain("asset_browser_a");
        });

        expect(normalizeText(container.textContent)).not.toContain("missing_dataset");

        await unmountWorkspace(root);
    });
});
