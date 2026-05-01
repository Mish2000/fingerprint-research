import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import BenchmarkWorkspace from "../src/features/benchmark/BenchmarkWorkspace.tsx";
import { formatMethodLabel as formatProductMethodLabel } from "../src/shared/storytelling.ts";

type RenderedWorkspace = {
    container: HTMLDivElement;
    root: Root;
};

const datasetInfos = {
    nist_sd300b: {
        key: "nist_sd300b",
        label: "NIST SD300b",
        summary: "Rolled versus plain legacy benchmark at 1000 ppi.",
    },
    polyu_cross: {
        key: "polyu_cross",
        label: "PolyU Cross",
        summary: "Cross-modality evaluation between contactless and contact-based fingerprints.",
    },
} as const;

const splitInfos = {
    val: {
        key: "val",
        label: "Validation",
        summary: "Validation split used to compare methods before final reporting.",
    },
    test: {
        key: "test",
        label: "Test",
        summary: "Locked evaluation split used for final reporting.",
    },
} as const;

function createJsonResponse(payload: unknown): Response {
    return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "content-type": "application/json" },
    });
}

function normalizeText(value: string | null | undefined): string {
    return (value ?? "").replace(/\s+/g, " ").trim();
}

function availableArtifacts(
    run: string,
    method: string,
    split: string,
    options: {
        summary?: boolean;
        scores?: boolean;
        meta?: boolean;
        roc?: boolean;
        markdown?: boolean;
        manifest?: boolean;
        log?: boolean;
    } = {},
) {
    const enabled = {
        summary: options.summary ?? true,
        scores: options.scores ?? true,
        meta: options.meta ?? true,
        roc: options.roc ?? true,
        markdown: options.markdown ?? true,
        manifest: options.manifest ?? true,
        log: options.log ?? true,
    };

    return [
        {
            key: "summary_csv",
            label: "Summary CSV",
            available: enabled.summary,
            url: enabled.summary ? `/api/benchmark/artifacts/${run}/results_summary.csv` : null,
        },
        {
            key: "scores_csv",
            label: "Scores CSV",
            available: enabled.scores,
            url: enabled.scores ? `/api/benchmark/artifacts/${run}/scores_${method}_${split}.csv` : null,
        },
        {
            key: "meta_json",
            label: "Meta JSON",
            available: enabled.meta,
            url: enabled.meta ? `/api/benchmark/artifacts/${run}/run_${method}_${split}.meta.json` : null,
        },
        {
            key: "roc_png",
            label: "ROC Preview",
            available: enabled.roc,
            url: enabled.roc ? `/api/benchmark/artifacts/${run}/roc_${method}_${split}.png` : null,
        },
        {
            key: "markdown_summary",
            label: "Markdown Summary",
            available: enabled.markdown,
            url: enabled.markdown ? `/api/benchmark/artifacts/${run}/results_summary.md` : null,
        },
        {
            key: "run_manifest",
            label: "Run Manifest",
            available: enabled.manifest,
            url: enabled.manifest ? `/api/benchmark/artifacts/${run}/run_manifest.json` : null,
        },
        {
            key: "run_log",
            label: "Run Log",
            available: enabled.log,
            url: enabled.log ? `/api/benchmark/artifacts/${run}/run.log` : null,
        },
    ];
}

function createRow({
    dataset,
    split,
    method,
    benchmarkMethod,
    run,
    runLabel,
    auc,
    eer,
    latency,
    nPairs,
    aucRank,
    eerRank,
    latencyRank,
    artifacts,
}: {
    dataset: "nist_sd300b" | "polyu_cross";
    split: "val" | "test";
    method: string;
    benchmarkMethod?: string;
    run: string;
    runLabel: string;
    auc: number;
    eer: number;
    latency: number;
    nPairs: number;
    aucRank: number;
    eerRank: number;
    latencyRank: number;
    artifacts?: ReturnType<typeof availableArtifacts>;
}) {
    const rawBenchmarkMethod = benchmarkMethod ?? method;
    const resolvedArtifacts = artifacts ?? availableArtifacts(run, rawBenchmarkMethod, split);
    const methodLabel = formatProductMethodLabel(method);
    const available = resolvedArtifacts.filter((item) => item.available).map((item) => item.key);

    return {
        dataset,
        run,
        split,
        method,
        benchmark_method: rawBenchmarkMethod,
        method_label: methodLabel,
        auc,
        eer,
        n_pairs: nPairs,
        tar_at_far_1e_2: null,
        tar_at_far_1e_3: null,
        latency_ms: latency,
        latency_source: "wall",
        auc_rank: aucRank,
        eer_rank: eerRank,
        latency_rank: latencyRank,
        run_family: run,
        run_label: runLabel,
        run_kind: "full",
        view_mode: "canonical",
        status: "validated",
        validation_state: "validated",
        artifact_count: available.length,
        available_artifacts: available,
        summary_text: `${runLabel} on ${split} with ${nPairs} pairs.`,
        artifacts: resolvedArtifacts,
        provenance: {
            run,
            run_label: runLabel,
            run_kind: "full",
            view_mode: "canonical",
            status: "validated",
            validation_state: "validated",
            source_type: "summary_csv",
            artifact_source: "results_summary.csv",
            methods_in_run: ["sift", "dl", "vit"],
            benchmark_methods_in_run: ["sift", "dl_quick", "vit"],
            canonical_method: method,
            benchmark_method: rawBenchmarkMethod,
            method_label: methodLabel,
            timestamp_utc: "2026-04-01T00:00:00Z",
            limit: 0,
            pairs_path: `C:\\pairs_${split}.csv`,
            manifest_path: "C:\\manifest.csv",
            data_dir: "C:\\data\\manifests\\nist_sd300b",
            git_commit: "deadbeef",
            available_artifacts: available,
        },
    };
}

const canonicalBTestRows = [
    createRow({
        dataset: "nist_sd300b",
        split: "test",
        method: "sift",
        run: "full_nist_sd300b_h6",
        runLabel: "Canonical full benchmark",
        auc: 0.6621,
        eer: 0.3398,
        latency: 32.15,
        nPairs: 2800,
        aucRank: 1,
        eerRank: 1,
        latencyRank: 3,
    }),
    createRow({
        dataset: "nist_sd300b",
        split: "test",
        method: "vit",
        run: "full_nist_sd300b_h6",
        runLabel: "Canonical full benchmark",
        auc: 0.5956,
        eer: 0.4360,
        latency: 0.36,
        nPairs: 2800,
        aucRank: 3,
        eerRank: 3,
        latencyRank: 1,
        artifacts: availableArtifacts("full_nist_sd300b_h6", "vit", "test", { meta: false, roc: false }),
    }),
    createRow({
        dataset: "nist_sd300b",
        split: "test",
        method: "dl",
        benchmarkMethod: "dl_quick",
        run: "full_nist_sd300b_h6",
        runLabel: "Canonical full benchmark",
        auc: 0.6045,
        eer: 0.4257,
        latency: 0.39,
        nPairs: 2800,
        aucRank: 2,
        eerRank: 2,
        latencyRank: 2,
    }),
];

const canonicalBValRows = [
    createRow({
        dataset: "nist_sd300b",
        split: "val",
        method: "sift",
        run: "full_nist_sd300b_h6",
        runLabel: "Canonical full benchmark",
        auc: 0.6544,
        eer: 0.3479,
        latency: 31.84,
        nPairs: 2800,
        aucRank: 1,
        eerRank: 1,
        latencyRank: 3,
    }),
    createRow({
        dataset: "nist_sd300b",
        split: "val",
        method: "dl",
        benchmarkMethod: "dl_quick",
        run: "full_nist_sd300b_h6",
        runLabel: "Canonical full benchmark",
        auc: 0.6055,
        eer: 0.4243,
        latency: 2.83,
        nPairs: 2800,
        aucRank: 2,
        eerRank: 2,
        latencyRank: 1,
    }),
];

const canonicalPolyuTestRows = [
    createRow({
        dataset: "polyu_cross",
        split: "test",
        method: "dl",
        benchmarkMethod: "dl_quick",
        run: "full_polyu_cross_h5",
        runLabel: "Canonical full benchmark",
        auc: 0.5310,
        eer: 0.4798,
        latency: 0.27,
        nPairs: 1224,
        aucRank: 1,
        eerRank: 1,
        latencyRank: 2,
    }),
    createRow({
        dataset: "polyu_cross",
        split: "test",
        method: "classic_gftt_orb",
        benchmarkMethod: "classic_v2",
        run: "full_polyu_cross_h5",
        runLabel: "Canonical full benchmark",
        auc: 0.5016,
        eer: 0.4984,
        latency: 7.62,
        nPairs: 1224,
        aucRank: 2,
        eerRank: 2,
        latencyRank: 3,
    }),
];

function selectionRows(dataset: string, split: string) {
    if (dataset === "polyu_cross") {
        return canonicalPolyuTestRows;
    }
    if (split === "val") {
        return canonicalBValRows;
    }
    return canonicalBTestRows;
}

function sortRows(rows: ReturnType<typeof selectionRows>, sortMode: string) {
    const items = [...rows];
    if (sortMode === "lowest_eer") {
        return items.sort((a, b) => a.eer_rank - b.eer_rank);
    }
    if (sortMode === "lowest_latency") {
        return items.sort((a, b) => a.latency_rank - b.latency_rank);
    }
    return items.sort((a, b) => a.auc_rank - b.auc_rank);
}

function bestEntriesFor(dataset: string, split: string) {
    const rows = selectionRows(dataset, split);
    const byRank = (metric: "auc_rank" | "eer_rank" | "latency_rank") => rows.find((row) => row[metric] === 1) ?? null;
    const bestAuc = byRank("auc_rank");
    const bestEer = byRank("eer_rank");
    const bestLatency = byRank("latency_rank");

    return [
        bestAuc ? {
            dataset: bestAuc.dataset,
            split: bestAuc.split,
            metric: "best_auc",
            method: bestAuc.method,
            benchmark_method: bestAuc.benchmark_method,
            method_label: bestAuc.method_label,
            run: bestAuc.run,
            value: bestAuc.auc,
            run_family: bestAuc.run_family,
            run_label: bestAuc.run_label,
            view_mode: bestAuc.view_mode,
            status: bestAuc.status,
            validation_state: bestAuc.validation_state,
        } : null,
        bestEer ? {
            dataset: bestEer.dataset,
            split: bestEer.split,
            metric: "best_eer",
            method: bestEer.method,
            benchmark_method: bestEer.benchmark_method,
            method_label: bestEer.method_label,
            run: bestEer.run,
            value: bestEer.eer,
            run_family: bestEer.run_family,
            run_label: bestEer.run_label,
            view_mode: bestEer.view_mode,
            status: bestEer.status,
            validation_state: bestEer.validation_state,
        } : null,
        bestLatency ? {
            dataset: bestLatency.dataset,
            split: bestLatency.split,
            metric: "best_latency",
            method: bestLatency.method,
            benchmark_method: bestLatency.benchmark_method,
            method_label: bestLatency.method_label,
            run: bestLatency.run,
            value: bestLatency.latency_ms,
            run_family: bestLatency.run_family,
            run_label: bestLatency.run_label,
            view_mode: bestLatency.view_mode,
            status: bestLatency.status,
            validation_state: bestLatency.validation_state,
        } : null,
    ].filter((entry) => entry != null);
}

function summaryPayload(dataset: string, split: string) {
    const effectiveDataset = dataset === "polyu_cross" ? "polyu_cross" : "nist_sd300b";
    const effectiveSplit = effectiveDataset === "polyu_cross" ? "test" : (split === "val" ? "val" : "test");
    const rows = selectionRows(effectiveDataset, effectiveSplit);

    return {
        dataset: effectiveDataset,
        split: effectiveSplit,
        view_mode: "canonical",
        dataset_info: datasetInfos[effectiveDataset as keyof typeof datasetInfos],
        split_info: splitInfos[effectiveSplit as keyof typeof splitInfos],
        view_info: {
            key: "canonical",
            label: "Canonical",
            summary: "Validated showcase runs.",
        },
        validation_state: "validated",
        selection_note: "Showing curated full benchmark results from validated showcase runs.",
        selection_policy: "Curated full benchmark showcase restricted to validated canonical families with usable evidence.",
        result_count: rows.length,
        method_count: new Set(rows.map((row) => row.method)).size,
        run_count: new Set(rows.map((row) => row.run)).size,
        available_datasets: Object.values(datasetInfos),
        available_splits: effectiveDataset === "polyu_cross"
            ? [splitInfos.test]
            : [splitInfos.val, splitInfos.test],
        available_view_modes: [{
            key: "canonical",
            label: "Canonical",
            summary: "Validated showcase runs.",
        }],
        current_run_families: [...new Set(rows.map((row) => row.run))],
        artifact_note: "Artifact links surface stored benchmark evidence when files are available.",
    };
}

function comparisonPayload(dataset: string, split: string, sortMode: string) {
    const effectiveDataset = dataset === "polyu_cross" ? "polyu_cross" : "nist_sd300b";
    const effectiveSplit = effectiveDataset === "polyu_cross" ? "test" : (split === "val" ? "val" : "test");
    const rows = sortRows(selectionRows(effectiveDataset, effectiveSplit), sortMode);

    return {
        rows,
        datasets: Object.keys(datasetInfos),
        splits: effectiveDataset === "polyu_cross" ? ["test"] : ["val", "test"],
        default_dataset: effectiveDataset,
        default_split: effectiveSplit,
        view_mode: "canonical",
        view_info: {
            key: "canonical",
            label: "Canonical",
            summary: "Validated showcase runs.",
        },
        dataset_info: datasetInfos,
        split_info: splitInfos,
    };
}

function bestPayload(dataset: string, split: string) {
    const effectiveDataset = dataset === "polyu_cross" ? "polyu_cross" : "nist_sd300b";
    const effectiveSplit = effectiveDataset === "polyu_cross" ? "test" : (split === "val" ? "val" : "test");
    return {
        dataset: effectiveDataset,
        split: effectiveSplit,
        view_mode: "canonical",
        entries: bestEntriesFor(effectiveDataset, effectiveSplit),
    };
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

async function renderWorkspace(initialUrl = "/"): Promise<RenderedWorkspace> {
    window.history.replaceState(window.history.state, "", initialUrl);
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    await act(async () => {
        root.render(<BenchmarkWorkspace />);
    });

    await flush();
    return { container, root };
}

async function unmountWorkspace(root: Root): Promise<void> {
    await act(async () => {
        root.unmount();
    });
}

function getButtonByText(container: HTMLElement, text: string): HTMLButtonElement {
    const match = Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find((button) =>
        normalizeText(button.textContent).includes(text),
    );
    if (!match) {
        throw new Error(`Unable to find button with text: ${text}`);
    }
    return match;
}

function getLabelField<T extends HTMLSelectElement>(container: HTMLElement, label: string): T {
    const match = Array.from(container.querySelectorAll("label")).find((field) =>
        normalizeText(field.textContent).includes(label),
    );
    if (!match) {
        throw new Error(`Unable to find field with label: ${label}`);
    }

    const control = match.querySelector("select");
    if (!control) {
        throw new Error(`Unable to find control for label: ${label}`);
    }

    return control as T;
}

async function click(element: HTMLElement): Promise<void> {
    await act(async () => {
        element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
}

async function changeSelect(field: HTMLSelectElement, value: string): Promise<void> {
    await act(async () => {
        field.value = value;
        field.dispatchEvent(new Event("input", { bubbles: true }));
        field.dispatchEvent(new Event("change", { bubbles: true }));
    });
}

async function clickRowByText(container: HTMLElement, text: string): Promise<void> {
    const row = Array.from(container.querySelectorAll("tbody tr")).find((item) =>
        normalizeText(item.textContent).includes(text),
    );
    if (!row) {
        throw new Error(`Unable to find row with text: ${text}`);
    }
    await click(row as HTMLElement);
}

function installBenchmarkFetchMock() {
    const requests: string[] = [];

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        requests.push(url);
        const parsed = new URL(url, "http://localhost");
        const dataset = parsed.searchParams.get("dataset") ?? "nist_sd300b";
        const split = parsed.searchParams.get("split") ?? "test";
        const sortMode = parsed.searchParams.get("sort_mode") ?? "best_accuracy";

        if (parsed.pathname === "/api/benchmark/summary") {
            return createJsonResponse(summaryPayload(dataset, split));
        }

        if (parsed.pathname === "/api/benchmark/comparison") {
            return createJsonResponse(comparisonPayload(dataset, split, sortMode));
        }

        if (parsed.pathname === "/api/benchmark/best") {
            return createJsonResponse(bestPayload(dataset, split));
        }

        throw new Error(`Unexpected fetch call: ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);
    return { fetchMock, requests };
}

afterEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    window.history.replaceState(window.history.state, "", "/");
});

describe("Benchmark workspace showcase", () => {
    it("renders showcase-only controls and populates evidence immediately", async () => {
        installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Curated full benchmark results");
            expect(normalizeText(container.textContent)).toContain("Showcase winners and trade-offs");
            expect(normalizeText(container.textContent)).toContain("Classic (SIFT)");
            expect(normalizeText(container.textContent)).toContain("Selected method evidence");
            expect(normalizeText(container.textContent)).toContain("full_nist_sd300b_h6");
        });

        expect(normalizeText(container.textContent)).not.toContain("View mode");
        expect(normalizeText(container.textContent)).not.toContain("Smoke");
        expect(normalizeText(container.textContent)).not.toContain("Archive");

        const datasetField = getLabelField<HTMLSelectElement>(container, "Dataset");
        const splitField = getLabelField<HTMLSelectElement>(container, "Split");

        expect(Array.from(datasetField.options).map((option) => option.textContent)).toEqual([
            "NIST SD300b",
            "PolyU Cross",
        ]);
        expect(Array.from(splitField.options).map((option) => option.textContent)).toEqual([
            "Validation",
            "Test",
        ]);

        await unmountWorkspace(root);
    });

    it("lists only valid splits for the selected showcase dataset", async () => {
        installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace();

        await changeSelect(getLabelField<HTMLSelectElement>(container, "Dataset"), "polyu_cross");
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("full_polyu_cross_h5");
            expect(normalizeText(container.textContent)).toContain("Deep Learning (ResNet50)");
            expect(normalizeText(container.textContent)).toContain("Classic (ROI GFTT+ORB)");
        });

        const splitField = getLabelField<HTMLSelectElement>(container, "Split");
        expect(Array.from(splitField.options).map((option) => option.textContent)).toEqual(["Test"]);
        expect(normalizeText(container.textContent)).not.toContain("No curated full benchmark results");

        await unmountWorkspace(root);
    });

    it("normalizes legacy benchmarkView urls into the canonical showcase flow", async () => {
        const { requests } = installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace("/?benchmarkView=archive&benchmarkDataset=polyu_cross&benchmarkSplit=test");

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("full_polyu_cross_h5");
        });

        expect(window.location.search).not.toContain("benchmarkView");
        expect(requests.some((url) => url.includes("view_mode="))).toBe(false);

        await unmountWorkspace(root);
    });

    it("keeps provenance usable when the selected row is missing some artifacts", async () => {
        installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("full_nist_sd300b_h6");
            expect(normalizeText(container.textContent)).toContain("Deep Learning (ViT)");
            expect(normalizeText(container.textContent)).not.toContain("dl_quick");
        });

        await clickRowByText(container, "Deep Learning (ViT)");
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("ROC preview is not available for this row");
            expect(normalizeText(container.textContent)).toContain("Meta JSON - N/A");
        });

        await click(getButtonByText(container, "Open provenance"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Methods in run");
            expect(normalizeText(container.textContent)).toContain("Benchmark method");
            expect(normalizeText(container.textContent)).toContain("vit");
            expect(normalizeText(container.textContent)).toContain("deadbeef");
        });

        await unmountWorkspace(root);
    });
});
