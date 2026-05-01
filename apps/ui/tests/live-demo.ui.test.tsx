import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import FingerprintLiveDemoWorkspace from "../src/features/live-demo/FingerprintLiveDemoWorkspace.tsx";
import type { IdentifyCandidate, IdentifyResponse } from "../src/types/index.ts";

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

function normalizeText(value: string | null | undefined): string {
    return (value ?? "").replace(/\s+/g, " ").trim();
}

function getButtonByText(container: HTMLElement, text: string): HTMLButtonElement {
    const button = Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find((item) =>
        normalizeText(item.textContent).includes(text),
    );
    if (!button) {
        throw new Error(`Unable to find button with text: ${text}`);
    }

    return button;
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

async function click(button: HTMLButtonElement): Promise<void> {
    await act(async () => {
        button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
}

async function renderWorkspace(): Promise<RenderedWorkspace> {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    await act(async () => {
        root.render(<FingerprintLiveDemoWorkspace />);
    });

    await flush();
    return { container, root };
}

async function unmountWorkspace(root: Root): Promise<void> {
    await act(async () => {
        root.unmount();
    });
}

const topCandidate: IdentifyCandidate = {
    rank: 1,
    random_id: "live_identity_001",
    full_name: "Alex Demo",
    national_id_masked: "***-1234",
    created_at: "2026-05-01T12:00:00Z",
    capture: "plain",
    retrieval_score: 0.8421,
    rerank_score: 0.9132,
    decision: true,
};

const identifyResponse: IdentifyResponse = {
    retrieval_method: "dl",
    rerank_method: "sift",
    threshold: 0.45,
    decision: true,
    total_enrolled: 12,
    candidate_pool_size: 12,
    shortlist_size: 10,
    hints_applied: {},
    top_candidate: topCandidate,
    candidates: [topCandidate],
    latency_ms: {
        total_ms: 42.5,
        probe_embed_ms: 12,
        shortlist_scan_ms: 16,
        rerank_ms: 14.5,
    },
    storage_layout: {},
};

function installFetchMock() {
    let submittedFormData: FormData | null = null;

    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const parsed = new URL(String(input), "http://localhost");

        if (parsed.pathname === "/api/identify/search") {
            submittedFormData = init?.body instanceof FormData ? init.body : null;
            return createJsonResponse(identifyResponse);
        }

        throw new Error(`Unexpected fetch call: ${String(input)}`);
    }));

    return {
        getSubmittedFormData: () => submittedFormData,
    };
}

describe("Fingerprint Live Demo workspace", () => {
    it("runs the manual-upload Identify 1:N fallback through the existing identify API", async () => {
        const controls = installFetchMock();
        const { container, root } = await renderWorkspace();

        expect(normalizeText(container.textContent)).toContain("Fingerprint biometrics");
        expect(getButtonByText(container, "Enroll").disabled).toBe(true);
        expect(getButtonByText(container, "Verify 1:1").disabled).toBe(true);

        const fileInput = container.querySelector<HTMLInputElement>('input[type="file"]');
        if (!fileInput) {
            throw new Error("Could not find manual upload input.");
        }

        const file = new File([new Blob(["fingerprint"], { type: "image/png" })], "fingerprint.png", { type: "image/png" });
        await uploadFile(fileInput, file);
        await click(getButtonByText(container, "Run Identify 1:N"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Match candidate accepted");
            expect(normalizeText(container.textContent)).toContain("Alex Demo");
            expect(normalizeText(container.textContent)).toContain("Benchmark before rollout");
            expect(normalizeText(container.textContent)).toContain("Prefer templates over raw captures");
        });

        const formData = controls.getSubmittedFormData();
        expect(formData).not.toBeNull();
        expect((formData?.get("img") as File).name).toBe("fingerprint.png");
        expect(formData?.get("capture")).toBe("plain");
        expect(formData?.get("retrieval_method")).toBe("dl");
        expect(formData?.get("rerank_method")).toBe("sift");
        expect(formData?.get("shortlist_size")).toBe("10");

        await unmountWorkspace(root);
    });
});
