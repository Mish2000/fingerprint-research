import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import VerifyWorkspaceScreen from "../src/features/verify/VerifyWorkspace.tsx";
import type { CatalogVerifyCase, MatchResponse } from "../src/types/index.ts";

type Deferred<T> = {
    promise: Promise<T>;
    resolve: (value: T) => void;
    reject: (reason?: unknown) => void;
};

type RenderedWorkspace = {
    container: HTMLDivElement;
    root: Root;
};

const catalogCases: CatalogVerifyCase[] = [
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
    evidence_quality: {
        selection_driver: "benchmark_driven",
        benchmark_backed_selection: true,
        heuristic_fallback_used: false,
        benchmark_discovery_outcome: "benchmark_best_resolved",
        evidence_status: "strong",
        evidence_note: "Selected directly from benchmark evidence.",
    },
},
{
        case_id: "case_hard_non_match",
        title: "Hard impostor",
        description: "A tougher impostor pair that stays deterministic.",
        dataset: "fvc2004",
        dataset_label: "FVC 2004",
        split: "test",
        difficulty: "hard",
        case_type: "hard_impostor",
        ground_truth: "non_match",
        recommended_method: "sift",
        capture_a: "contactless",
        capture_b: "contact_based",
        modality_relation: "cross_sensor",
        tags: ["hard", "impostor"],
        selection_policy: "hard_negative",
        selection_reason: "Representative hard negative from the curated catalog.",
        image_a_url: "/api/demo/cases/case_hard_non_match/a/probe.png",
        image_b_url: "/api/demo/cases/case_hard_non_match/b/reference.png",
        availability_status: "available",
        asset_a_id: "asset_hard_a",
        asset_b_id: "asset_hard_b",
        evidence_quality: {
            selection_driver: "heuristic_fallback",
            benchmark_backed_selection: false,
            heuristic_fallback_used: true,
            benchmark_discovery_outcome: "dataset_fallback_no_benchmark_evidence",
            evidence_status: "degraded",
            evidence_note: "Benchmark evidence was unavailable, so this case used a heuristic fallback.",
        },
    },
];

function createDeferred<T>(): Deferred<T> {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
}

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

function createMatchResponse(method: MatchResponse["method"] = "sift"): MatchResponse {
    return {
        method,
        score: 0.92,
        decision: true,
        threshold: 0.01,
        latency_ms: 18,
        meta: {
            matches: 34,
            inliers: 21,
            k1: 56,
            k2: 61,
        },
        overlay: null,
    };
}

function createCatalogResponse(items: CatalogVerifyCase[] = catalogCases): Response {
    return createJsonResponse({
        items,
        total: items.length,
        limit: items.length,
        offset: 0,
        has_more: false,
        catalog_build_health: {
            catalog_build_status: "degraded",
            total_verify_cases: items.length,
            benchmark_backed_case_count: 1,
            heuristic_fallback_case_count: 1,
            datasets_with_missing_benchmark_evidence: ["fvc2004"],
            summary_message: "1 of 2 curated verify case(s) use heuristic fallback across 1 dataset(s).",
        },
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

function getCaseButton(container: HTMLElement, title: string): HTMLButtonElement {
    const match = Array.from(container.querySelectorAll<HTMLButtonElement>("button[aria-pressed]")).find((button) =>
        normalizeText(button.textContent).includes(title),
    );
    if (!match) {
        throw new Error(`Unable to find case button for: ${title}`);
    }
    return match;
}

function getSelectByLabel(container: HTMLElement, label: string): HTMLSelectElement {
    const normalizedLabel = normalizeText(label);
    const labels = Array.from(container.querySelectorAll("label"));
    const match = labels.find((field) => normalizeText(field.textContent) === normalizedLabel)
        ?? labels.find((field) => normalizeText(field.textContent).startsWith(normalizedLabel))
        ?? labels.find((field) => normalizeText(field.textContent).includes(normalizedLabel));
    if (!match) {
        throw new Error(`Unable to find select with label: ${label}`);
    }

    const control = match.querySelector("select");
    if (!control) {
        throw new Error(`Unable to find select control for label: ${label}`);
    }

    return control;
}

async function click(button: HTMLButtonElement): Promise<void> {
    await act(async () => {
        button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
}

async function flush(): Promise<void> {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
}

async function waitFor(assertion: () => void, timeoutMs = 1500): Promise<void> {
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
        root.render(<VerifyWorkspaceScreen />);
    });

    await flush();
    return { container, root };
}

async function unmountWorkspace(root: Root): Promise<void> {
    await act(async () => {
        root.unmount();
    });
}

function installFetchMock(
    overrides: Partial<Record<string, (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>>> = {},
): ReturnType<typeof vi.fn> {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);

        if (overrides[url]) {
            return overrides[url]!(input, init);
        }

        if (url === "/api/catalog/verify-cases") {
            return createCatalogResponse();
        }

        const matchedCase = catalogCases.find((item) => item.image_a_url === url || item.image_b_url === url);
        if (matchedCase) {
            const assetLabel = url === matchedCase.image_a_url ? "probe" : "reference";
            return createImageResponse(assetLabel);
        }

        if (url === "/api/match") {
            return createJsonResponse(createMatchResponse());
        }

        throw new Error(`Unexpected fetch call: ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
}

function createFileList(file: File): FileList {
    return {
        0: file,
        length: 1,
        item: (index: number) => (index === 0 ? file : null),
        [Symbol.iterator]: function* iterator() {
            yield file;
        },
    } as unknown as FileList;
}

async function uploadFile(input: HTMLInputElement, file: File): Promise<void> {
    Object.defineProperty(input, "files", {
        configurable: true,
        value: createFileList(file),
    });

    await act(async () => {
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
}

afterEach(() => {
    localStorage.clear();
    sessionStorage.clear();
});

describe("Verify Workspace dual-mode UX", () => {
    it("opens in Demo Mode by default and loads catalog-backed cases", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Easy genuine");
            expect(normalizeText(container.textContent)).toContain("Hard impostor");
        });

        expect(getButtonByText(container, "Demo Mode").getAttribute("aria-pressed")).toBe("true");
        expect(getButtonByText(container, "Manual Upload").getAttribute("aria-pressed")).toBe("false");
        expect(normalizeText(container.textContent)).toContain("Selected Case");
        expect(getCaseButton(container, "Easy genuine").getAttribute("aria-pressed")).toBe("true");

        await unmountWorkspace(root);
    });

    it("updates selection when the user picks a different catalog case", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Hard impostor");
        });

        await click(getCaseButton(container, "Hard impostor"));

        await waitFor(() => {
            expect(getCaseButton(container, "Hard impostor").getAttribute("aria-pressed")).toBe("true");
            expect(getCaseButton(container, "Easy genuine").getAttribute("aria-pressed")).toBe("false");
        });

        await unmountWorkspace(root);
    });

    it("shows both split classic methods distinctly in the verify selector", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Easy genuine");
        });

        const methodField = getSelectByLabel(container, "Method");
        const optionLabels = Array.from(methodField.options).map((option) => normalizeText(option.textContent));

        expect(optionLabels).toContain("Classic (ORB)");
        expect(optionLabels).toContain("Classic (ROI GFTT+ORB)");

        await unmountWorkspace(root);
    });

    it("shows compact evidence badges and a degraded catalog banner when demo evidence falls back to heuristics", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Curated demo evidence is degraded");
            expect(normalizeText(container.textContent)).toContain("Strong evidence");
            expect(normalizeText(container.textContent)).toContain("Degraded evidence");
            expect(normalizeText(container.textContent)).toContain("Heuristic fallback");
        });

        await unmountWorkspace(root);
    });

    it("shows a running state while the selected demo case is in flight", async () => {
        const matchResponse = createDeferred<Response>();
        installFetchMock({
            "/api/match": () => matchResponse.promise,
        });
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Easy genuine");
        });

        await click(getButtonByText(container, "Run Selected Case"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Verification request in progress");
            expect(normalizeText(container.textContent)).toContain("Running");
        });

        await act(async () => {
            matchResponse.resolve(createJsonResponse(createMatchResponse()));
            await matchResponse.promise;
        });

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Same finger");
            expect(normalizeText(container.textContent)).toContain("Confidence band");
            expect(normalizeText(container.textContent)).toContain("Expected vs actual");
        });

        await unmountWorkspace(root);
    });

    it("switches between Demo Mode and Manual Upload without losing the demo selection", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Hard impostor");
        });

        await click(getCaseButton(container, "Hard impostor"));
        await waitFor(() => {
            expect(getCaseButton(container, "Hard impostor").getAttribute("aria-pressed")).toBe("true");
        });

        await click(getButtonByText(container, "Manual Upload"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Manual Upload stays intentionally separate");
        });

        expect(getButtonByText(container, "Manual Upload").getAttribute("aria-pressed")).toBe("true");

        await click(getButtonByText(container, "Demo Mode"));
        await waitFor(() => {
            expect(getButtonByText(container, "Demo Mode").getAttribute("aria-pressed")).toBe("true");
            expect(getCaseButton(container, "Hard impostor").getAttribute("aria-pressed")).toBe("true");
        });

        await unmountWorkspace(root);
    });

    it("runs the selected demo case through catalog metadata, server assets, and the match endpoint", async () => {
        const requests: string[] = [];
        let submittedFormData: FormData | null = null;

        installFetchMock({
            "/api/catalog/verify-cases": () => {
                requests.push("/api/catalog/verify-cases");
                return createCatalogResponse();
            },
            [catalogCases[1].image_a_url]: () => {
                requests.push(catalogCases[1].image_a_url);
                return createImageResponse("hard-probe");
            },
            [catalogCases[1].image_b_url]: () => {
                requests.push(catalogCases[1].image_b_url);
                return createImageResponse("hard-reference");
            },
            "/api/match": (_input, init) => {
                requests.push("/api/match");
                submittedFormData = init?.body as FormData;
                return createJsonResponse(createMatchResponse("sift"));
            },
        });

        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Hard impostor");
        });

        await click(getCaseButton(container, "Hard impostor"));
        await click(getButtonByText(container, "Run Selected Case"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Same finger");
            expect(normalizeText(container.textContent)).toContain("Hard impostor");
            expect(normalizeText(container.textContent)).toContain("Actual decision did not match the expected outcome.");
            expect(normalizeText(container.textContent)).toContain("Cross-capture setup.");
        });

        expect(requests).toEqual([
            "/api/catalog/verify-cases",
            catalogCases[1].image_a_url,
            catalogCases[1].image_b_url,
            "/api/match",
        ]);
        expect(submittedFormData).not.toBeNull();
        expect(submittedFormData?.get("method")).toBe(catalogCases[1].recommended_method);
        expect((submittedFormData?.get("img_a") as File).name).toBe("probe.png");
        expect((submittedFormData?.get("img_b") as File).name).toBe("reference.png");
        expect(normalizeText(container.textContent)).toContain("Method Classic (SIFT)");

        await unmountWorkspace(root);
    });

    it("preserves the manual upload flow and submits the user-selected files", async () => {
        const matchCalls: FormData[] = [];
        installFetchMock({
            "/api/match": (_input, init) => {
                matchCalls.push(init?.body as FormData);
                return createJsonResponse(createMatchResponse("sift"));
            },
        });

        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Easy genuine");
        });

        await click(getButtonByText(container, "Manual Upload"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Manual Upload stays intentionally separate");
        });

        const [probeInput, referenceInput] = Array.from(container.querySelectorAll<HTMLInputElement>("input[type=\"file\"]"));
        expect(probeInput).toBeTruthy();
        expect(referenceInput).toBeTruthy();

        await uploadFile(probeInput, new File(["probe-manual"], "manual-probe.png", { type: "image/png" }));
        await uploadFile(referenceInput, new File(["reference-manual"], "manual-reference.png", { type: "image/png" }));
        await click(getButtonByText(container, "Run Verification"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Same finger");
            expect(normalizeText(container.textContent)).toContain("Manual upload");
            expect(normalizeText(container.textContent)).toContain("Difficulty metadata is not available for this pair.");
        });

        expect(matchCalls).toHaveLength(1);
        expect((matchCalls[0].get("img_a") as File).name).toBe("manual-probe.png");
        expect((matchCalls[0].get("img_b") as File).name).toBe("manual-reference.png");

        await unmountWorkspace(root);
    });

    it("rehydrates the last selected verify demo case and pinned cases across reload without auto-running verify", async () => {
        const fetchMock = installFetchMock();
        let rendered = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Hard impostor");
        });

        await click(getCaseButton(rendered.container, "Hard impostor"));
        await click(getButtonByText(rendered.container, "Pin case"));

        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Pinned demo cases");
            expect(getCaseButton(rendered.container, "Hard impostor").getAttribute("aria-pressed")).toBe("true");
        });

        const matchCallsBeforeReload = fetchMock.mock.calls.filter(([input]) => String(input) === "/api/match").length;
        await unmountWorkspace(rendered.root);

        rendered = await renderWorkspace();

        await waitFor(() => {
            expect(getCaseButton(rendered.container, "Hard impostor").getAttribute("aria-pressed")).toBe("true");
            expect(normalizeText(rendered.container.textContent)).toContain("Pinned demo cases");
            expect(normalizeText(rendered.container.textContent)).toContain("Hard impostor");
        });

        const matchCallsAfterReload = fetchMock.mock.calls.filter(([input]) => String(input) === "/api/match").length;
        expect(matchCallsAfterReload).toBe(matchCallsBeforeReload);

        await unmountWorkspace(rendered.root);
    });

    it("restores manual mode as a re-upload reminder instead of pretending local files still exist", async () => {
        installFetchMock();
        let rendered = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Easy genuine");
        });

        await click(getButtonByText(rendered.container, "Manual Upload"));
        await waitFor(() => {
            expect(normalizeText(rendered.container.textContent)).toContain("Manual Upload stays intentionally separate");
        });

        const [probeInput, referenceInput] = Array.from(rendered.container.querySelectorAll<HTMLInputElement>("input[type=\"file\"]"));
        await uploadFile(probeInput, new File(["probe-manual"], "manual-probe.png", { type: "image/png" }));
        await uploadFile(referenceInput, new File(["reference-manual"], "manual-reference.png", { type: "image/png" }));

        await unmountWorkspace(rendered.root);
        rendered = await renderWorkspace();

        await waitFor(() => {
            expect(getButtonByText(rendered.container, "Manual Upload").getAttribute("aria-pressed")).toBe("true");
            expect(normalizeText(rendered.container.textContent)).toContain("Previous manual pair needs re-upload");
            expect(normalizeText(rendered.container.textContent)).toContain("manual-probe.png");
            expect(normalizeText(rendered.container.textContent)).toContain("manual-reference.png");
        });

        expect(normalizeText(rendered.container.textContent)).not.toContain("Same finger");

        await unmountWorkspace(rendered.root);
    });

    it("clears saved verify workspace continuity and falls back to the default state", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Hard impostor");
        });

        await click(getCaseButton(container, "Hard impostor"));
        await click(getButtonByText(container, "Pin case"));
        await click(getButtonByText(container, "Clear saved Verify workspace"));

        await waitFor(() => {
            expect(getButtonByText(container, "Demo Mode").getAttribute("aria-pressed")).toBe("true");
            expect(getCaseButton(container, "Easy genuine").getAttribute("aria-pressed")).toBe("true");
            expect(normalizeText(container.textContent)).not.toContain("Pinned demo cases");
        });

        expect(localStorage.getItem("fp-research.workspace.verify.local")).toBeNull();

        await unmountWorkspace(root);
    });
});
