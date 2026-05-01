import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import FingerprintLiveDemoWorkspace from "../src/features/live-demo/FingerprintLiveDemoWorkspace.tsx";
import type { EnrollFingerprintResponse, IdentifyCandidate, IdentifyResponse } from "../src/types/index.ts";

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

async function changeInput(input: HTMLInputElement, value: string): Promise<void> {
    await act(async () => {
        const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
        valueSetter?.call(input, value);
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
}

async function changeCheckbox(input: HTMLInputElement, checked: boolean): Promise<void> {
    await act(async () => {
        if (input.checked !== checked) {
            input.dispatchEvent(new MouseEvent("click", { bubbles: true }));
        }
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

const enrollResponse: EnrollFingerprintResponse = {
    random_id: "live_identity_001",
    created_at: "2026-05-01T12:00:00Z",
    vector_methods: ["dl", "vit"],
    image_sha256: "abc123",
    storage_layout: {},
};

function installFetchMock() {
    let submittedIdentifyFormData: FormData | null = null;
    let submittedEnrollFormData: FormData | null = null;

    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const parsed = new URL(String(input), "http://localhost");

        if (parsed.pathname === "/api/identify/enroll") {
            submittedEnrollFormData = init?.body instanceof FormData ? init.body : null;
            return createJsonResponse(enrollResponse);
        }

        if (parsed.pathname === "/api/identify/search") {
            submittedIdentifyFormData = init?.body instanceof FormData ? init.body : null;
            return createJsonResponse(identifyResponse);
        }

        throw new Error(`Unexpected fetch call: ${String(input)}`);
    }));

    return {
        getSubmittedIdentifyFormData: () => submittedIdentifyFormData,
        getSubmittedEnrollFormData: () => submittedEnrollFormData,
    };
}

describe("Fingerprint Live Demo workspace", () => {
    it("disables Identify until a probe fingerprint exists", async () => {
        const controls = installFetchMock();
        const { container, root } = await renderWorkspace();

        expect(normalizeText(container.textContent)).toContain("Fingerprint biometrics");
        expect(normalizeText(container.textContent)).toContain("Step 1Enrollment capture");
        expect(normalizeText(container.textContent)).toContain("Step 2Probe capture");
        expect(normalizeText(container.textContent)).toContain("Step 3Identify 1:N result");
        expect(normalizeText(container.textContent)).toContain("Gallery readiness");
        expect(normalizeText(container.textContent)).toContain("Upload a probe fingerprint to search the enrolled gallery.");
        expect(normalizeText(container.textContent)).toContain("You can search an existing gallery, but for a clean demo enroll an identity first.");
        expect(normalizeText(container.textContent)).toContain("Available in Verify tab");
        expect(getButtonByText(container, "Verify 1:1").disabled).toBe(true);
        expect(getButtonByText(container, "Run Identify 1:N").disabled).toBe(true);

        const fileInputs = Array.from(container.querySelectorAll<HTMLInputElement>('input[type="file"]'));
        const enrollmentInput = fileInputs[0];
        const probeInput = fileInputs[1];
        if (!enrollmentInput || !probeInput) {
            throw new Error("Could not find enrollment and probe upload inputs.");
        }

        const enrollmentFile = new File([new Blob(["enrollment"], { type: "image/png" })], "enrollment.png", { type: "image/png" });
        await uploadFile(enrollmentInput, enrollmentFile);
        expect(getButtonByText(container, "Run Identify 1:N").disabled).toBe(true);

        const probeFile = new File([new Blob(["probe"], { type: "image/png" })], "probe.png", { type: "image/png" });
        await uploadFile(probeInput, probeFile);
        expect(getButtonByText(container, "Run Identify 1:N").disabled).toBe(false);
        expect(normalizeText(container.textContent)).toContain("You can search an existing gallery, but for a clean demo enroll an identity first.");
        expect(controls.getSubmittedIdentifyFormData()).toBeNull();

        await unmountWorkspace(root);
    });

    it("submits enrollmentFile for Enroll and probeFile for Identify", async () => {
        const controls = installFetchMock();
        const { container, root } = await renderWorkspace();

        const fileInputs = Array.from(container.querySelectorAll<HTMLInputElement>('input[type="file"]'));
        const enrollmentInput = fileInputs[0];
        const probeInput = fileInputs[1];
        if (!enrollmentInput || !probeInput) {
            throw new Error("Could not find enrollment and probe upload inputs.");
        }

        const enrollmentFile = new File([new Blob(["enrollment"], { type: "image/png" })], "enrollment.png", { type: "image/png" });
        await uploadFile(enrollmentInput, enrollmentFile);
        expect(normalizeText(container.textContent)).toContain("Enrollment sourceManual upload ready");

        const textInputs = Array.from(container.querySelectorAll<HTMLInputElement>('input[type="text"], input:not([type])'));
        const fullNameInput = textInputs[0];
        const nationalIdInput = textInputs[1];
        if (!fullNameInput || !nationalIdInput) {
            throw new Error("Could not find enrollment identity inputs.");
        }

        await changeInput(fullNameInput, "Alex Demo");
        await changeInput(nationalIdInput, "123456789");

        const replaceExistingInput = container.querySelector<HTMLInputElement>('input[type="checkbox"]');
        if (!replaceExistingInput) {
            throw new Error("Could not find replace existing toggle.");
        }
        await changeCheckbox(replaceExistingInput, true);

        await click(getButtonByText(container, "Enroll identity"));

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Enrollment completed");
            expect(text).toContain("Ready for Identify 1:N");
            expect(text).toContain("Alex Demo");
            expect(text).toContain("live_identity_001");
            expect(text).toContain("Plain");
            expect(text).toContain("Deep Learning (ResNet50)");
            expect(text).toContain("Deep Learning (ViT)");
        });

        const enrollFormData = controls.getSubmittedEnrollFormData();
        expect(enrollFormData).not.toBeNull();
        expect((enrollFormData?.get("img") as File).name).toBe("enrollment.png");
        expect(enrollFormData?.get("full_name")).toBe("Alex Demo");
        expect(enrollFormData?.get("national_id")).toBe("123456789");
        expect(enrollFormData?.get("capture")).toBe("plain");
        expect(enrollFormData?.get("vector_methods")).toBe("dl,vit");
        expect(enrollFormData?.get("replace_existing")).toBe("true");

        const probeFile = new File([new Blob(["probe"], { type: "image/png" })], "probe.png", { type: "image/png" });
        await uploadFile(probeInput, probeFile);
        expect(normalizeText(container.textContent)).toContain("Enrollment completed");
        expect(normalizeText(container.textContent)).toContain("live_identity_001");
        expect(normalizeText(container.textContent)).toContain("Probe sourceManual upload ready");

        await click(getButtonByText(container, "Run Identify 1:N"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Match candidate accepted");
            expect(normalizeText(container.textContent)).toContain("Alex Demo");
            expect(normalizeText(container.textContent)).toContain("Enrollment source file: enrollment.png");
            expect(normalizeText(container.textContent)).toContain("Probe source file: probe.png");
            expect(normalizeText(container.textContent)).toContain("Benchmark before rollout");
            expect(normalizeText(container.textContent)).toContain("This demo separates enrollment from probe capture to avoid same-image matching.");
            expect(normalizeText(container.textContent)).toContain("Prefer templates over raw captures");
        });

        const formData = controls.getSubmittedIdentifyFormData();
        expect(formData).not.toBeNull();
        expect((formData?.get("img") as File).name).toBe("probe.png");
        expect(formData?.get("capture")).toBe("plain");
        expect(formData?.get("retrieval_method")).toBe("dl");
        expect(formData?.get("rerank_method")).toBe("sift");
        expect(formData?.get("shortlist_size")).toBe("10");

        await unmountWorkspace(root);
    });

    it("keeps the last enrollment visible when probe or enrollment captures change", async () => {
        installFetchMock();
        const { container, root } = await renderWorkspace();

        const fileInputs = Array.from(container.querySelectorAll<HTMLInputElement>('input[type="file"]'));
        const enrollmentInput = fileInputs[0];
        const probeInput = fileInputs[1];
        if (!enrollmentInput || !probeInput) {
            throw new Error("Could not find enrollment and probe upload inputs.");
        }

        const enrollmentFile = new File([new Blob(["enrollment"], { type: "image/png" })], "enrollment.png", { type: "image/png" });
        await uploadFile(enrollmentInput, enrollmentFile);

        const textInputs = Array.from(container.querySelectorAll<HTMLInputElement>('input[type="text"], input:not([type])'));
        await changeInput(textInputs[0], "Alex Demo");
        await changeInput(textInputs[1], "123456789");
        await click(getButtonByText(container, "Enroll identity"));

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Enrollment completed");
            expect(text).toContain("enrollment.png");
        });

        const probeFile = new File([new Blob(["probe"], { type: "image/png" })], "probe.png", { type: "image/png" });
        await uploadFile(probeInput, probeFile);
        expect(normalizeText(container.textContent)).toContain("Enrollment completed");
        expect(normalizeText(container.textContent)).toContain("Alex Demo");

        const changedEnrollmentFile = new File([new Blob(["changed"], { type: "image/png" })], "changed-enrollment.png", { type: "image/png" });
        await uploadFile(enrollmentInput, changedEnrollmentFile);

        const text = normalizeText(container.textContent);
        expect(text).toContain("Enrollment completed");
        expect(text).toContain("Alex Demo");
        expect(text).toContain("Enrollment capture changed — enroll again to update this identity.");

        await unmountWorkspace(root);
    });
});
