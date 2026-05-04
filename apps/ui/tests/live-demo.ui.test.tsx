import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import FingerprintLiveDemoWorkspace from "../src/features/live-demo/FingerprintLiveDemoWorkspace.tsx";
import type { EnrollFingerprintResponse, IdentifyCandidate, IdentifyResponse } from "../src/types/index.ts";

type RenderedWorkspace = {
    container: HTMLDivElement;
    root: Root;
};

function createJsonResponse(payload: unknown, status = 200): Response {
    return new Response(JSON.stringify(payload), {
        status,
        headers: { "content-type": "application/json" },
    });
}

function createPngResponse(): Response {
    return new Response(new Blob(["normalized scanner png"], { type: "image/png" }), {
        status: 200,
        headers: { "content-type": "image/png" },
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

async function changeSelect(select: HTMLSelectElement, value: string): Promise<void> {
    await act(async () => {
        const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value")?.set;
        valueSetter?.call(select, value);
        select.dispatchEvent(new Event("change", { bubbles: true }));
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

interface FetchMockOptions {
    scannerImport?: "success" | "no-capture";
    scannerOriginalFilename?: string;
    scannerNormalizedFilename?: string;
}

function scannerImportPayload(options: FetchMockOptions = {}) {
    const normalizedFilename = options.scannerNormalizedFilename ?? "scanner_20260502_120000_abcd1234.png";
    const captureId = normalizedFilename.replace(/\.png$/i, "");
    return {
        capture_id: captureId,
        original_filename: options.scannerOriginalFilename ?? "umpi_capture.tif",
        normalized_filename: normalizedFilename,
        normalized_url: `/api/scanner/captures/${captureId}`,
        mime_type: "image/png",
        size_bytes: 2048,
        modified_at: "2026-05-02T12:00:00Z",
        age_seconds: 5,
    };
}

function installFetchMock(options: FetchMockOptions = {}) {
    let submittedIdentifyFormData: FormData | null = null;
    let submittedEnrollFormData: FormData | null = null;
    let scannerImportCalls = 0;

    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const parsed = new URL(String(input), "http://localhost");

        if (parsed.pathname === "/api/scanner/import-latest") {
            scannerImportCalls += 1;
            if (options.scannerImport === "no-capture") {
                return createJsonResponse({ detail: "No saved scanner capture found in the configured folder." }, 404);
            }

            return createJsonResponse(scannerImportPayload(options));
        }

        if (parsed.pathname.startsWith("/api/scanner/captures/")) {
            return createPngResponse();
        }

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
        getScannerImportCalls: () => scannerImportCalls,
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
        expect(normalizeText(container.textContent)).toContain("Shared capture profile");
        expect(normalizeText(container.textContent)).toContain("Used for both enrollment and probe in this demo.");
        expect(normalizeText(container.textContent)).toContain("Upload a probe fingerprint to search the enrolled gallery.");
        expect(normalizeText(container.textContent)).toContain("You can search an existing gallery, but for a clean demo enroll an identity first.");
        expect(normalizeText(container.textContent)).toContain("Available in Verify tab");
        expect(getButtonByText(container, "Verify 1:1").disabled).toBe(true);
        expect(getButtonByText(container, "Run Identify 1:N").disabled).toBe(true);
        expect(container.querySelectorAll("select")).toHaveLength(1);
        expect(container.querySelector<HTMLSelectElement>('select[aria-label="Shared capture profile"]')?.value).toBe("plain");

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

        const sharedCaptureSelect = container.querySelector<HTMLSelectElement>('select[aria-label="Shared capture profile"]');
        if (!sharedCaptureSelect) {
            throw new Error("Could not find shared capture profile selector.");
        }
        await changeSelect(sharedCaptureSelect, "contactless");

        const fileInputs = Array.from(container.querySelectorAll<HTMLInputElement>('input[type="file"]'));
        const enrollmentInput = fileInputs[0];
        const probeInput = fileInputs[1];
        if (!enrollmentInput || !probeInput) {
            throw new Error("Could not find enrollment and probe upload inputs.");
        }

        const enrollmentFile = new File([new Blob(["enrollment"], { type: "image/png" })], "enrollment.png", { type: "image/png" });
        await uploadFile(enrollmentInput, enrollmentFile);
        expect(normalizeText(container.textContent)).toContain("Enrollment sourceImage source ready");

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
            expect(text).toContain("Contactless");
            expect(text).toContain("Deep Learning (ResNet50)");
            expect(text).toContain("Deep Learning (ViT)");
        });

        const enrollFormData = controls.getSubmittedEnrollFormData();
        expect(enrollFormData).not.toBeNull();
        expect((enrollFormData?.get("img") as File).name).toBe("enrollment.png");
        expect(enrollFormData?.get("full_name")).toBe("Alex Demo");
        expect(enrollFormData?.get("national_id")).toBe("123456789");
        expect(enrollFormData?.get("capture")).toBe("contactless");
        expect(enrollFormData?.get("vector_methods")).toBe("dl,vit");
        expect(enrollFormData?.get("replace_existing")).toBe("true");

        const probeFile = new File([new Blob(["probe"], { type: "image/png" })], "probe.png", { type: "image/png" });
        await uploadFile(probeInput, probeFile);
        expect(normalizeText(container.textContent)).toContain("Enrollment completed");
        expect(normalizeText(container.textContent)).toContain("live_identity_001");
        expect(normalizeText(container.textContent)).toContain("Probe sourceImage source ready");

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
        expect(formData?.get("capture")).toBe("contactless");
        expect(formData?.get("retrieval_method")).toBe("dl");
        expect(formData?.get("rerank_method")).toBe("sift");
        expect(formData?.get("shortlist_size")).toBe("10");

        await unmountWorkspace(root);
    });

    it("imports the latest saved UMPI capture as enrollment and enrolls with that normalized file", async () => {
        const controls = installFetchMock({
            scannerOriginalFilename: "umpi_enrollment.tif",
            scannerNormalizedFilename: "scanner_20260502_120000_enroll01.png",
        });
        const { container, root } = await renderWorkspace();

        expect(normalizeText(container.textContent)).toContain("Scanner bridge");
        expect(normalizeText(container.textContent)).toContain(
            "Use the UMPI Diagnostic Tool to capture a fingerprint, save the .tif file into the configured scanner capture folder, then import the latest saved capture here.",
        );
        expect(normalizeText(container.textContent)).toContain("Manual upload remains available.");
        expect(normalizeText(container.textContent)).toContain("Direct SDK capture is a future milestone.");

        await click(getButtonByText(container, "Import latest saved UMPI capture as enrollment"));

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Imported umpi_enrollment.tif as enrollment capture");
            expect(text).toContain("scanner_20260502_120000_enroll01.png");
            expect(text).toContain("Enrollment sourceImage source ready");
            expect(getButtonByText(container, "Enroll identity").disabled).toBe(false);
        });

        const textInputs = Array.from(container.querySelectorAll<HTMLInputElement>('input[type="text"], input:not([type])'));
        await changeInput(textInputs[0], "Alex Demo");
        await changeInput(textInputs[1], "123456789");
        await click(getButtonByText(container, "Enroll identity"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Enrollment completed");
        });

        const enrollFormData = controls.getSubmittedEnrollFormData();
        expect(enrollFormData).not.toBeNull();
        expect((enrollFormData?.get("img") as File).name).toBe("scanner_20260502_120000_enroll01.png");
        expect(controls.getScannerImportCalls()).toBe(1);

        await unmountWorkspace(root);
    });

    it("imports the latest saved UMPI capture as probe and identifies with that normalized file", async () => {
        const controls = installFetchMock({
            scannerOriginalFilename: "umpi_probe.tif",
            scannerNormalizedFilename: "scanner_20260502_120500_probe001.png",
        });
        const { container, root } = await renderWorkspace();

        expect(getButtonByText(container, "Run Identify 1:N").disabled).toBe(true);

        await click(getButtonByText(container, "Import latest saved UMPI capture as probe"));

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Imported umpi_probe.tif as probe capture");
            expect(text).toContain("scanner_20260502_120500_probe001.png");
            expect(text).toContain("Probe sourceImage source ready");
            expect(getButtonByText(container, "Run Identify 1:N").disabled).toBe(false);
        });

        await click(getButtonByText(container, "Run Identify 1:N"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Match candidate accepted");
        });

        const identifyFormData = controls.getSubmittedIdentifyFormData();
        expect(identifyFormData).not.toBeNull();
        expect((identifyFormData?.get("img") as File).name).toBe("scanner_20260502_120500_probe001.png");
        expect(identifyFormData?.get("capture")).toBe("plain");

        await unmountWorkspace(root);
    });

    it("shows the scanner no-capture import error without removing manual upload", async () => {
        installFetchMock({ scannerImport: "no-capture" });
        const { container, root } = await renderWorkspace();

        await click(getButtonByText(container, "Import latest saved UMPI capture as enrollment"));

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("No saved scanner capture found in the configured folder");
        });

        const fileInputs = Array.from(container.querySelectorAll<HTMLInputElement>('input[type="file"]'));
        const enrollmentInput = fileInputs[0];
        if (!enrollmentInput) {
            throw new Error("Could not find enrollment upload input.");
        }

        const manualFile = new File([new Blob(["manual"], { type: "image/png" })], "manual-enrollment.png", { type: "image/png" });
        await uploadFile(enrollmentInput, manualFile);
        expect(normalizeText(container.textContent)).toContain("manual-enrollment.png");
        expect(getButtonByText(container, "Enroll identity").disabled).toBe(false);

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
