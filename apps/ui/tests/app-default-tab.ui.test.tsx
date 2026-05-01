import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../src/App.tsx";
import { createDefaultAppPreferences } from "../src/shared/preferences/defaults.ts";
import {
    writePersistedAppPreferences,
    writePersistedLastActiveTab,
} from "../src/shared/preferences/persistence.ts";

type RenderedApp = {
    container: HTMLDivElement;
    root: Root;
};

function createJsonResponse(payload: unknown): Response {
    return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "content-type": "application/json" },
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

async function renderApp(initialUrl = "/"): Promise<RenderedApp> {
    window.history.replaceState(window.history.state, "", initialUrl);
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    await act(async () => {
        root.render(<App />);
    });

    await flush();
    return { container, root };
}

async function unmountApp(root: Root): Promise<void> {
    await act(async () => {
        root.unmount();
    });
}

function activeNavButton(container: HTMLElement): HTMLButtonElement | undefined {
    return Array.from(container.querySelectorAll<HTMLButtonElement>("nav button")).find((button) =>
        button.getAttribute("aria-current") === "page",
    );
}

function expectBenchmarkActive(container: HTMLElement): void {
    expect(container.querySelector("main h2")?.textContent).toBe("Benchmarks");
    expect(activeNavButton(container)?.textContent).toContain("Benchmarks");
}

function expectBenchmarkUrl(): void {
    expect(new URLSearchParams(window.location.search).get("tab")).toBe("benchmark");
}

function expectActiveTab(container: HTMLElement, title: string): void {
    expect(container.querySelector("main h2")?.textContent).toBe(title);
    expect(activeNavButton(container)?.textContent).toContain(title);
}

function expectTabUrl(tab: string): void {
    expect(new URLSearchParams(window.location.search).get("tab")).toBe(tab);
}

function installFetchMock() {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        const parsed = new URL(url, "http://localhost");

        if (parsed.pathname === "/api/benchmark/summary") {
            return createJsonResponse({
                dataset: "nist_sd300b",
                split: "test",
                view_mode: "canonical",
                dataset_info: { key: "nist_sd300b", label: "NIST SD300b", summary: "" },
                split_info: { key: "test", label: "Test", summary: "" },
                view_info: { key: "canonical", label: "Canonical", summary: "" },
                validation_state: "validated",
                selection_note: "Showing curated full benchmark results from validated showcase runs.",
                selection_policy: "Curated full benchmark showcase restricted to validated canonical families.",
                result_count: 1,
                method_count: 1,
                run_count: 1,
                available_datasets: [{ key: "nist_sd300b", label: "NIST SD300b", summary: "" }],
                available_splits: [{ key: "test", label: "Test", summary: "" }],
                available_view_modes: [{ key: "canonical", label: "Canonical", summary: "" }],
                current_run_families: ["full_nist_sd300b_h6"],
                artifact_note: "Artifacts available.",
            });
        }

        if (parsed.pathname === "/api/benchmark/comparison") {
            return createJsonResponse({
                rows: [{
                    dataset: "nist_sd300b",
                    run: "full_nist_sd300b_h6",
                    split: "test",
                    method: "sift",
                    auc: 0.9,
                    eer: 0.1,
                    n_pairs: 100,
                    tar_at_far_1e_2: null,
                    tar_at_far_1e_3: null,
                    latency_ms: 4.2,
                    latency_source: "reported",
                    auc_rank: 1,
                    eer_rank: 1,
                    latency_rank: 1,
                    run_family: "full_nist_sd300b_h6",
                    run_label: "Canonical full benchmark",
                    run_kind: "full",
                    view_mode: "canonical",
                    status: "validated",
                    validation_state: "validated",
                    artifact_count: 2,
                    available_artifacts: ["summary_csv", "scores_csv"],
                    summary_text: "Canonical full benchmark on test with 100 pairs.",
                    artifacts: [
                        { key: "summary_csv", label: "Summary CSV", available: true, url: "/api/benchmark/artifacts/full_nist_sd300b_h6/results_summary.csv" },
                        { key: "scores_csv", label: "Scores CSV", available: true, url: "/api/benchmark/artifacts/full_nist_sd300b_h6/scores_sift_test.csv" },
                    ],
                    provenance: {
                        run: "full_nist_sd300b_h6",
                        run_label: "Canonical full benchmark",
                        run_kind: "full",
                        view_mode: "canonical",
                        status: "validated",
                        validation_state: "validated",
                        source_type: "summary_csv",
                        artifact_source: "results_summary.csv",
                        methods_in_run: ["sift"],
                        available_artifacts: ["summary_csv", "scores_csv"],
                    },
                }],
                datasets: ["nist_sd300b"],
                splits: ["test"],
                default_dataset: "nist_sd300b",
                default_split: "test",
                view_mode: "canonical",
                view_info: { key: "canonical", label: "Canonical", summary: "" },
                dataset_info: { nist_sd300b: { key: "nist_sd300b", label: "NIST SD300b", summary: "" } },
                split_info: { test: { key: "test", label: "Test", summary: "" } },
            });
        }

        if (parsed.pathname === "/api/benchmark/best") {
            return createJsonResponse({
                dataset: "nist_sd300b",
                split: "test",
                view_mode: "canonical",
                entries: [
                    {
                        dataset: "nist_sd300b",
                        split: "test",
                        metric: "best_auc",
                        method: "sift",
                        run: "full_nist_sd300b_h6",
                        value: 0.9,
                        run_family: "full_nist_sd300b_h6",
                        run_label: "Canonical full benchmark",
                        view_mode: "canonical",
                        status: "validated",
                        validation_state: "validated",
                    },
                    {
                        dataset: "nist_sd300b",
                        split: "test",
                        metric: "best_eer",
                        method: "sift",
                        run: "full_nist_sd300b_h6",
                        value: 0.1,
                        run_family: "full_nist_sd300b_h6",
                        run_label: "Canonical full benchmark",
                        view_mode: "canonical",
                        status: "validated",
                        validation_state: "validated",
                    },
                    {
                        dataset: "nist_sd300b",
                        split: "test",
                        metric: "best_latency",
                        method: "sift",
                        run: "full_nist_sd300b_h6",
                        value: 4.2,
                        run_family: "full_nist_sd300b_h6",
                        run_label: "Canonical full benchmark",
                        view_mode: "canonical",
                        status: "validated",
                        validation_state: "validated",
                    },
                ],
            });
        }

        if (parsed.pathname === "/api/catalog/verify-cases") {
            return createJsonResponse({
                items: [],
                total: 0,
                limit: 20,
                offset: 0,
                has_more: false,
            });
        }

        if (parsed.pathname === "/api/catalog/datasets") {
            return createJsonResponse({ items: [] });
        }

        throw new Error(`Unexpected fetch call: ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);
}

afterEach(() => {
    window.history.replaceState(window.history.state, "", "/");
});

describe("App default tab", () => {
    it("opens the Benchmarks tab by default", async () => {
        installFetchMock();
        const { container, root } = await renderApp("/");

        await waitFor(() => {
            expectBenchmarkActive(container);
            expectBenchmarkUrl();
        });

        await unmountApp(root);
    });

    it("only shows the active product tabs in the main navigation", async () => {
        installFetchMock();
        const { container, root } = await renderApp("/");

        await waitFor(() => {
            const navText = container.querySelector("nav")?.textContent ?? "";
            expect(navText).toContain("Verification (1:1)");
            expect(navText).toContain("Identification (1:N)");
            expect(navText).toContain("Benchmarks");
            expect(navText).not.toContain("Contracts");
            expect(navText).not.toContain("QA & release");
        });

        await unmountApp(root);
    });

    it("normalizes legacy contracts links to Benchmarks", async () => {
        installFetchMock();
        const { container, root } = await renderApp("/?tab=contracts");

        await waitFor(() => {
            expectBenchmarkActive(container);
            expectBenchmarkUrl();
        });

        await unmountApp(root);
    });

    it("normalizes legacy release links to Benchmarks", async () => {
        installFetchMock();
        const { container, root } = await renderApp("/?tab=release");

        await waitFor(() => {
            expectBenchmarkActive(container);
            expectBenchmarkUrl();
        });

        await unmountApp(root);
    });

    it("uses explicit URL tabs before remembered or default tabs", async () => {
        installFetchMock();
        writePersistedAppPreferences({
            ...createDefaultAppPreferences(),
            defaultTab: "benchmark",
            rememberLastTab: true,
        });
        writePersistedLastActiveTab("benchmark");

        const { container, root } = await renderApp("/?tab=verify");

        await waitFor(() => {
            expectActiveTab(container, "Verification (1:1)");
            expectTabUrl("verify");
        });

        await unmountApp(root);
    });

    it("uses the remembered tab before the default tab when enabled", async () => {
        installFetchMock();
        writePersistedAppPreferences({
            ...createDefaultAppPreferences(),
            defaultTab: "verify",
            rememberLastTab: true,
        });
        writePersistedLastActiveTab("benchmark");

        const { container, root } = await renderApp("/");

        await waitFor(() => {
            expectBenchmarkActive(container);
            expectBenchmarkUrl();
        });

        await unmountApp(root);
    });

    it("uses the default tab preference when no URL or remembered tab applies", async () => {
        installFetchMock();
        writePersistedAppPreferences({
            ...createDefaultAppPreferences(),
            defaultTab: "verify",
            rememberLastTab: true,
        });

        const { container, root } = await renderApp("/");

        await waitFor(() => {
            expectActiveTab(container, "Verification (1:1)");
            expectTabUrl("verify");
        });

        await unmountApp(root);
    });
});
