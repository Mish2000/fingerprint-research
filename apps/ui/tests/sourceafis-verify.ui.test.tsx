import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import SourceAfisVerifyPanel from "../src/features/verify/components/SourceAfisVerifyPanel.tsx";

let root: Root | null = null;

afterEach(async () => {
    await act(async () => root?.unmount());
    root = null;
});

function engineResponse(status = 200) {
    return new Response(JSON.stringify(status === 200
        ? { provider: "sourceafis_open", score: 0, provider_version: "3.18.1", latency_ms: 12 }
        : { detail: "SourceAFIS is unavailable" }), { status, headers: { "Content-Type": "application/json" } });
}

async function renderPanel(verify: (form: FormData) => Promise<Response> = async () => engineResponse()) {
    const submitted: FormData[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input) === "/api/demo/synthetic/0.png") {
            return new Response(new Uint8Array([1, 2, 3]), { headers: { "Content-Type": "image/png" } });
        }
        if (String(input) === "/api/fingerprint-engine/verify") {
            const form = init?.body as FormData;
            submitted.push(form);
            return verify(form);
        }
        throw new Error(`Unexpected URL: ${input}`);
    }));
    const container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => root?.render(<SourceAfisVerifyPanel />));
    await click(container, "Use synthetic SourceAFIS pair");
    return { container, submitted };
}

async function click(container: HTMLElement, text: string) {
    const button = [...container.querySelectorAll("button")].find(element => element.textContent === text);
    expect(button).toBeDefined();
    await act(async () => button!.click());
}

function dpiField(container: HTMLElement, label: string) {
    return container.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
}

async function changeDpi(container: HTMLElement, label: string, value: string) {
    const input = dpiField(container, label);
    await act(async () => {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
}

describe("SourceAFIS result input binding", () => {
    it.each([
        ["Probe DPI", "dpi_a"],
        ["Reference DPI", "dpi_b"],
    ])("invalidates a completed result when %s changes", async (label, formKey) => {
        const { container, submitted } = await renderPanel();
        await click(container, "Compare with SourceAFIS");
        expect(container.textContent).toContain("Raw score: 0.0000");
        expect(container.textContent).toContain("Computed with probe DPI 500 · reference DPI 500");
        await changeDpi(container, label, "1000");
        expect(container.querySelector('[role="status"]')).toBeNull();
        await click(container, "Compare with SourceAFIS");
        expect(submitted).toHaveLength(2);
        expect(submitted[0].get(formKey)).toBe("500");
        expect(submitted[1].get(formKey)).toBe("1000");
        expect(container.textContent).toContain(label === "Probe DPI"
            ? "Computed with probe DPI 1000 · reference DPI 500"
            : "Computed with probe DPI 500 · reference DPI 1000");
    });

    it("locks both DPI fields during an outstanding request and records the submitted values", async () => {
        let complete!: (response: Response) => void;
        const pending = new Promise<Response>(resolve => { complete = resolve; });
        const { container, submitted } = await renderPanel(async () => pending);
        await changeDpi(container, "Probe DPI", "1000");
        await changeDpi(container, "Reference DPI", "2000");
        await click(container, "Compare with SourceAFIS");
        expect(submitted[0].get("dpi_a")).toBe("1000");
        expect(submitted[0].get("dpi_b")).toBe("2000");
        expect(dpiField(container, "Probe DPI").disabled).toBe(true);
        expect(dpiField(container, "Reference DPI").disabled).toBe(true);
        expect(container.querySelector('[role="status"]')).toBeNull();
        await act(async () => complete(engineResponse()));
        expect(dpiField(container, "Probe DPI").disabled).toBe(false);
        expect(dpiField(container, "Reference DPI").disabled).toBe(false);
        expect(container.textContent).toContain("Computed with probe DPI 1000 · reference DPI 2000");
        expect(container.textContent).toContain("Raw score: 0.0000");
    });

    it("unlocks DPI after a service error without displaying a score", async () => {
        const { container } = await renderPanel(async () => engineResponse(503));
        await click(container, "Compare with SourceAFIS");
        expect(container.querySelector('[role="alert"]')?.textContent).toContain("SourceAFIS is unavailable");
        expect(container.querySelector('[role="status"]')).toBeNull();
        expect(dpiField(container, "Probe DPI").disabled).toBe(false);
        expect(dpiField(container, "Reference DPI").disabled).toBe(false);
    });
});
