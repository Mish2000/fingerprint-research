import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import FingerprintLiveDemoWorkspace from "../src/features/live-demo/FingerprintLiveDemoWorkspace.tsx";
import { IDENTIFICATION_RETRIEVAL_METHOD_VALUES } from "../src/types/index.ts";
import type { EnrollFingerprintResponse, IdentifyCandidate, IdentifyResponse } from "../src/types/index.ts";

type RenderedWorkspace = {
    container: HTMLDivElement;
    root: Root;
};

afterEach(() => {
    vi.useRealTimers();
});

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

function getButtonsByText(container: HTMLElement, text: string): HTMLButtonElement[] {
    const buttons = Array.from(container.querySelectorAll<HTMLButtonElement>("button")).filter((item) =>
        normalizeText(item.textContent).includes(text),
    );
    if (buttons.length === 0) {
        throw new Error(`Unable to find buttons with text: ${text}`);
    }

    return buttons;
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

async function advanceTimersByTime(ms: number): Promise<void> {
    await act(async () => {
        await vi.advanceTimersByTimeAsync(ms);
    });
    await flush();
}

function expectScannerCaptureButtonsDisabled(container: HTMLElement, disabled: boolean): void {
    for (const buttonText of ["Capture from scanner", "Import latest saved scan", "Capture with scanner UI"]) {
        for (const button of getButtonsByText(container, buttonText)) {
            expect(button.disabled).toBe(disabled);
        }
    }
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
    vector_methods: [...IDENTIFICATION_RETRIEVAL_METHOD_VALUES],
    image_sha256: "abc123",
    storage_layout: {},
};

interface FetchMockOptions {
    scannerImport?: "success" | "no-capture";
    scannerStatus?: "direct" | "fallback" | "unavailable";
    scannerCapture?: "success" | "failure";
    scannerOriginalFilename?: string;
    scannerNormalizedFilename?: string;
    scannerCaptureId?: string;
    scannerCaptureDelayMs?: number;
}

function scannerStatusPayload(options: FetchMockOptions = {}) {
    const mode = options.scannerStatus ?? "direct";
    const directAvailable = mode === "direct";
    const fallbackAvailable = mode === "direct" || mode === "fallback";
    return {
        active_mode: directAvailable ? "twain" : fallbackAvailable ? "saved_file_bridge" : "unavailable",
        available_modes: [
            ...(directAvailable ? ["twain"] : []),
            ...(fallbackAvailable ? ["saved_file_bridge"] : []),
        ],
        direct_capture_available: directAvailable,
        saved_file_bridge_available: fallbackAvailable,
        device_detected: null,
        device_name: directAvailable ? "TWAIN Biometrika Driver" : null,
        driver_detected: directAvailable,
        twain_source_detected: directAvailable,
        last_error: directAvailable ? null : "TWAIN direct capture is unavailable and fallback is not allowed.",
        diagnostics: {},
        configured: true,
        enabled: directAvailable || fallbackAvailable,
        direct_capture_enabled: directAvailable,
        saved_file_bridge_enabled: fallbackAvailable,
        capture_dir_display: "data/scanner_captures/incoming",
        normalized_dir_display: "data/scanner_captures/normalized",
    };
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

function scannerCapturePayload(request: Record<string, unknown>, options: FetchMockOptions = {}) {
    const captureId = options.scannerCaptureId ?? "scanner_20260502_121500_direct01";
    const modeUsed = request.mode === "saved_file_bridge" ? "saved_file_bridge" : "twain";
    const directCapture = modeUsed === "twain";
    return {
        ok: true,
        mode_used: modeUsed,
        direct_capture: directCapture,
        normalized_url: `/api/scanner/captures/${captureId}`,
        capture_id: captureId,
        raw_file: { size_bytes: 1024, format: "bmp" },
        normalized_file: { path: `data/scanner_captures/normalized/${captureId}.png`, size_bytes: 2048, format: "png", mime_type: "image/png" },
        duration_ms: directCapture ? 42 : 9,
        device: { name: directCapture ? "TWAIN Biometrika Driver" : null, provider: modeUsed },
        warning: null,
        metadata: {},
    };
}

function installFetchMock(options: FetchMockOptions = {}) {
    let submittedIdentifyFormData: FormData | null = null;
    let submittedEnrollFormData: FormData | null = null;
    let scannerImportCalls = 0;
    let scannerStatusCalls = 0;
    const scannerCaptureRequests: Record<string, unknown>[] = [];

    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const parsed = new URL(String(input), "http://localhost");

        if (parsed.pathname === "/api/scanner/status") {
            scannerStatusCalls += 1;
            return createJsonResponse(scannerStatusPayload(options));
        }

        if (parsed.pathname === "/api/scanner/capture") {
            const request = typeof init?.body === "string"
                ? JSON.parse(init.body) as Record<string, unknown>
                : {};
            scannerCaptureRequests.push(request);
            if (options.scannerCaptureDelayMs != null) {
                await new Promise((resolve) => setTimeout(resolve, options.scannerCaptureDelayMs));
            }
            if (options.scannerCapture === "failure") {
                return createJsonResponse({
                    ok: false,
                    error_code: "twain_unavailable",
                    message: "TWAIN direct capture is unavailable and fallback is not allowed.",
                    mode_requested: String(request.mode ?? "auto"),
                    fallback_available: true,
                    diagnostics: {},
                });
            }

            return createJsonResponse(scannerCapturePayload(request, options));
        }

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
        getScannerStatusCalls: () => scannerStatusCalls,
        getScannerCaptureRequests: () => scannerCaptureRequests,
    };
}

describe("Fingerprint Live Demo workspace", () => {
    it("renders direct scanner capture controls when direct capture is available", async () => {
        const controls = installFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Direct scanner capture available");
            expect(text).toContain("TWAIN Biometrika Driver");
            expect(text).toContain("Capture from scanner");
            expect(text).toContain("Import latest saved scan");
        });

        expect(controls.getScannerStatusCalls()).toBe(1);
        expect(getButtonsByText(container, "Capture from scanner")[0].disabled).toBe(false);

        await unmountWorkspace(root);
    });

    it("waits through the direct scanner countdown before capture and displays direct-capture metadata", async () => {
        const controls = installFetchMock({
            scannerCaptureId: "scanner_20260502_121500_direct01",
            scannerCaptureDelayMs: 1000,
        });
        const { container, root } = await renderWorkspace();

        const captureButton = getButtonsByText(container, "Capture from scanner")[0];
        await waitFor(() => {
            expect(captureButton.disabled).toBe(false);
        });

        vi.useFakeTimers();
        await click(captureButton);

        expect(controls.getScannerCaptureRequests()).toEqual([]);
        expectScannerCaptureButtonsDisabled(container, true);
        expect(normalizeText(container.textContent)).toContain("Place finger on scanner and keep holding still during capture.");
        expect(normalizeText(container.textContent)).toContain("Capturing in 3...");

        await advanceTimersByTime(1000);
        expect(controls.getScannerCaptureRequests()).toEqual([]);
        expect(normalizeText(container.textContent)).toContain("Capturing in 2...");

        await advanceTimersByTime(1000);
        expect(controls.getScannerCaptureRequests()).toEqual([]);
        expect(normalizeText(container.textContent)).toContain("Capturing in 1...");

        await advanceTimersByTime(1000);
        expect(controls.getScannerCaptureRequests()).toEqual([
            {
                mode: "auto",
                timeout_ms: 15000,
                fallback_allowed: false,
                normalize: true,
                show_ui: false,
                settle_after_enable_ms: 1500,
            },
        ]);
        expectScannerCaptureButtonsDisabled(container, true);
        expect(normalizeText(container.textContent)).toContain("Scanner is active — keep finger still.");

        await advanceTimersByTime(1000);
        vi.useRealTimers();

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Direct TWAIN capture");
            expect(text).toContain("scanner_20260502_121500_direct01.png");
            expect(text).toContain("mode_usedtwain");
            expect(text).toContain("direct_capturetrue");
            expect(text).toContain("duration_ms42");
            expect(text).toContain("device.nameTWAIN Biometrika Driver");
            expect(container.querySelector('img[alt="Uploaded fingerprint preview"]')).not.toBeNull();
        });

        expectScannerCaptureButtonsDisabled(container, false);

        await unmountWorkspace(root);
    });

    it("shows backend ok=false scanner capture errors without success metadata", async () => {
        installFetchMock({ scannerCapture: "failure" });
        const { container, root } = await renderWorkspace();

        const captureButton = getButtonsByText(container, "Capture from scanner")[0];
        await waitFor(() => {
            expect(captureButton.disabled).toBe(false);
        });

        vi.useFakeTimers();
        await click(captureButton);
        expect(normalizeText(container.textContent)).toContain("Capturing in 3...");
        await advanceTimersByTime(3000);
        vi.useRealTimers();

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Capture failed");
            expect(text).toContain("TWAIN direct capture is unavailable and fallback is not allowed.");
            expect(text).toContain("Error code: twain_unavailable.");
            expect(text).toContain("Use Import latest saved scan.");
            expect(text).not.toContain("Direct TWAIN capture");
            expect(text).not.toContain("mode_usedtwain");
        });

        await unmountWorkspace(root);
    });

    it("does not call direct scanner capture after unmount during the countdown", async () => {
        const controls = installFetchMock();
        const { container, root } = await renderWorkspace();

        const captureButton = getButtonsByText(container, "Capture from scanner")[0];
        await waitFor(() => {
            expect(captureButton.disabled).toBe(false);
        });

        vi.useFakeTimers();
        await click(captureButton);
        expect(normalizeText(container.textContent)).toContain("Capturing in 3...");

        await unmountWorkspace(root);
        await advanceTimersByTime(3000);

        expect(controls.getScannerCaptureRequests()).toEqual([]);
    });

    it("renders saved-file fallback controls when direct capture is unavailable", async () => {
        installFetchMock({ scannerStatus: "fallback" });
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Direct capture unavailable. Saved-file import fallback available.");
        });

        expect(getButtonsByText(container, "Capture from scanner")[0].disabled).toBe(true);
        expect(getButtonsByText(container, "Import latest saved scan")[0].disabled).toBe(false);

        await unmountWorkspace(root);
    });

    it("sends show_ui true and a longer timeout for scanner UI capture", async () => {
        const controls = installFetchMock({ scannerCaptureId: "scanner_20260502_122000_ui01" });
        const { container, root } = await renderWorkspace();

        const scannerUiButton = getButtonsByText(container, "Capture with scanner UI")[0];
        await waitFor(() => {
            expect(scannerUiButton.disabled).toBe(false);
        });

        await click(scannerUiButton);

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("scanner_20260502_122000_ui01.png");
        });

        expect(controls.getScannerCaptureRequests()).toEqual([
            {
                mode: "twain",
                timeout_ms: 60000,
                fallback_allowed: false,
                normalize: true,
                show_ui: true,
                settle_after_enable_ms: 0,
            },
        ]);

        await unmountWorkspace(root);
    });

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
            expect(text).toContain("Deep Learning (ResNet18)");
            expect(text).toContain("Deep Learning (ViT)");
        });

        const enrollFormData = controls.getSubmittedEnrollFormData();
        expect(enrollFormData).not.toBeNull();
        expect((enrollFormData?.get("img") as File).name).toBe("enrollment.png");
        expect(enrollFormData?.get("full_name")).toBe("Alex Demo");
        expect(enrollFormData?.get("national_id")).toBe("123456789");
        expect(enrollFormData?.get("capture")).toBe("contactless");
        expect(enrollFormData?.get("vector_methods")).toBe(IDENTIFICATION_RETRIEVAL_METHOD_VALUES.join(","));
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

    it("imports the latest saved scan as enrollment and enrolls with that normalized file", async () => {
        const controls = installFetchMock({
            scannerOriginalFilename: "umpi_enrollment.tif",
            scannerNormalizedFilename: "scanner_20260502_120000_enroll01.png",
        });
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Direct scanner capture available");
            expect(getButtonsByText(container, "Import latest saved scan")[0].disabled).toBe(false);
        });

        await click(getButtonsByText(container, "Import latest saved scan")[0]);

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Saved-file fallback");
            expect(text).toContain("scanner_20260502_120000_enroll01.png");
            expect(text).toContain("mode_usedsaved_file_bridge");
            expect(text).toContain("direct_capturefalse");
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

    it("imports the latest saved scan as probe and identifies with that normalized file", async () => {
        const controls = installFetchMock({
            scannerOriginalFilename: "umpi_probe.tif",
            scannerNormalizedFilename: "scanner_20260502_120500_probe001.png",
        });
        const { container, root } = await renderWorkspace();

        expect(getButtonByText(container, "Run Identify 1:N").disabled).toBe(true);

        await waitFor(() => {
            expect(getButtonsByText(container, "Import latest saved scan")[1].disabled).toBe(false);
        });
        await click(getButtonsByText(container, "Import latest saved scan")[1]);

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Saved-file fallback");
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

        await waitFor(() => {
            expect(getButtonsByText(container, "Import latest saved scan")[0].disabled).toBe(false);
        });
        await click(getButtonsByText(container, "Import latest saved scan")[0]);

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
