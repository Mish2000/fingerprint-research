import assert from "node:assert/strict";
import fs from "node:fs";
import {
    fetchCatalogDatasetBrowser,
    fetchCatalogDatasets,
    fetchCatalogVerifyCases,
    loadCatalogBrowserPairFiles,
    loadCatalogVerifyCaseFiles,
} from "../src/api/matchService.ts";
import {
    createBrowserPairKey,
    createDefaultBrowserFilters,
    createPairPreviewState,
    toCatalogBrowserQuery,
} from "../src/features/verify/browserModel.ts";
import {
    buildDemoRunConfiguration,
    filterDemoCases,
    readVerifyModeFromSessionStorage,
} from "../src/features/verify/model.ts";
import type { CatalogBrowserItem, CatalogDatasetSummary, CatalogVerifyCase } from "../src/types/index.ts";

type TestCase = {
    name: string;
    run: () => void | Promise<void>;
};

const catalogCase: CatalogVerifyCase = {
    case_id: "case_demo",
    title: "Catalog case",
    description: "Curated verify case.",
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
    selection_reason: "Selected for demo coverage.",
    image_a_url: "/api/demo/cases/case_demo/a/probe.png",
    image_b_url: "/api/demo/cases/case_demo/b/reference.png",
    availability_status: "available",
    asset_a_id: "asset_a",
    asset_b_id: "asset_b",
};

const catalogDataset: CatalogDatasetSummary = {
    dataset: "nist_sd300b",
    dataset_label: "NIST SD300B",
    has_verify_cases: true,
    has_identify_gallery: false,
    has_browser_assets: true,
    verify_case_count: 1,
    identify_identity_count: 0,
    browser_item_count: 2,
    browser_validation_status: "pass",
    selection_policy: "deterministic_round_robin",
    available_features: ["verify_cases", "dataset_browser"],
};

const browserItemA: CatalogBrowserItem = {
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
};

const browserItemB: CatalogBrowserItem = {
    ...browserItemA,
    asset_id: "asset_browser_b",
    split: "val",
    subject_id: "100002",
    finger: "2",
    capture: "roll",
    preview_url: "/api/catalog/assets/nist_sd300b/asset_browser_b/preview",
    thumbnail_url: "/api/catalog/assets/nist_sd300b/asset_browser_b/thumbnail",
};

const tests: TestCase[] = [
    {
        name: "verify mode defaults to demo",
        run: () => {
            const originalWindow = globalThis.window;

            try {
                Object.defineProperty(globalThis, "window", {
                    configurable: true,
                    value: {
                        sessionStorage: {
                            getItem: () => null,
                        },
                    },
                });

                assert.equal(readVerifyModeFromSessionStorage(), "demo");
            } finally {
                Object.defineProperty(globalThis, "window", {
                    configurable: true,
                    value: originalWindow,
                });
            }
        },
    },
    {
        name: "demo runs default to recommended method",
        run: () => {
            const config = buildDemoRunConfiguration(
                {
                    method: "vit",
                    captureA: "contactless",
                    captureB: "contact_based",
                    thresholdMode: "default",
                    thresholdText: "0.45",
                },
                catalogCase,
                false,
            );

            assert.equal(config.method, "sift");
            assert.equal(config.captureA, "plain");
            assert.equal(config.captureB, "roll");
            assert.equal(config.returnOverlay, true);
        },
    },
    {
        name: "demo runs preserve explicit user override",
        run: () => {
            const config = buildDemoRunConfiguration(
                {
                    method: "dl",
                    captureA: "contactless",
                    captureB: "contact_based",
                    thresholdMode: "custom",
                    thresholdText: "0.77",
                },
                catalogCase,
                true,
            );

            assert.equal(config.method, "dl");
            assert.equal(config.captureA, "contactless");
            assert.equal(config.captureB, "contact_based");
            assert.equal(config.thresholdText, "0.77");
            assert.equal(config.returnOverlay, false);
        },
    },
    {
        name: "demo filters use catalog metadata only",
        run: () => {
            const cases: CatalogVerifyCase[] = [
                catalogCase,
                {
                    ...catalogCase,
                    case_id: "case_hard_non_match",
                    difficulty: "hard",
                    ground_truth: "non_match",
                },
            ];

            assert.deepEqual(filterDemoCases(cases, "hard").map((item) => item.case_id), ["case_hard_non_match"]);
            assert.deepEqual(filterDemoCases(cases, "genuine").map((item) => item.case_id), ["case_demo"]);
            assert.deepEqual(filterDemoCases(cases, "impostor").map((item) => item.case_id), ["case_hard_non_match"]);
        },
    },
    {
        name: "catalog verify cases load from the catalog endpoint",
        run: async () => {
            const originalFetch = globalThis.fetch;
            const calls: string[] = [];

            globalThis.fetch = async (input: string | URL | Request) => {
                calls.push(String(input));
                return new Response(
                    JSON.stringify({
                        items: [catalogCase],
                        total: 1,
                        limit: 20,
                        offset: 0,
                        has_more: false,
                    }),
                    {
                        status: 200,
                        headers: { "content-type": "application/json" },
                    },
                );
            };

            try {
                const response = await fetchCatalogVerifyCases();
                assert.deepEqual(calls, ["/api/catalog/verify-cases"]);
                assert.equal(response.items[0].case_id, "case_demo");
            } finally {
                globalThis.fetch = originalFetch;
            }
        },
    },
    {
        name: "demo asset loading uses server URLs directly",
        run: async () => {
            const originalFetch = globalThis.fetch;
            const calls: string[] = [];

            globalThis.fetch = async (input: string | URL | Request) => {
                const url = String(input);
                calls.push(url);

                if (url === catalogCase.image_a_url) {
                    return new Response(new Blob(["probe"], { type: "image/png" }), {
                        status: 200,
                        headers: { "content-type": "image/png" },
                    });
                }

                if (url === catalogCase.image_b_url) {
                    return new Response(new Blob(["reference"], { type: "image/png" }), {
                        status: 200,
                        headers: { "content-type": "image/png" },
                    });
                }

                throw new Error(`Unexpected fetch: ${url}`);
            };

            try {
                const files = await loadCatalogVerifyCaseFiles(catalogCase);
                assert.deepEqual(calls, [catalogCase.image_a_url, catalogCase.image_b_url]);
                assert.equal(files.fileA.name, "probe.png");
                assert.equal(files.fileB.name, "reference.png");
            } finally {
                globalThis.fetch = originalFetch;
            }
        },
    },
    {
        name: "demo asset failures remain tied to the server asset request",
        run: async () => {
            const originalFetch = globalThis.fetch;

            globalThis.fetch = async () => new Response(JSON.stringify({ detail: "Asset missing" }), {
                status: 404,
                headers: { "content-type": "application/json" },
            });

            try {
                await assert.rejects(
                    () => loadCatalogVerifyCaseFiles(catalogCase),
                    /Failed to load demo asset \(404\): Asset missing/,
                );
            } finally {
                globalThis.fetch = originalFetch;
            }
        },
    },
    {
        name: "browser datasets load from the catalog datasets endpoint",
        run: async () => {
            const originalFetch = globalThis.fetch;
            const calls: string[] = [];

            globalThis.fetch = async (input: string | URL | Request) => {
                calls.push(String(input));
                return new Response(JSON.stringify({ items: [catalogDataset] }), {
                    status: 200,
                    headers: { "content-type": "application/json" },
                });
            };

            try {
                const response = await fetchCatalogDatasets();
                assert.deepEqual(calls, ["/api/catalog/datasets"]);
                assert.equal(response.items[0].dataset, "nist_sd300b");
                assert.equal(response.items[0].has_browser_assets, true);
            } finally {
                globalThis.fetch = originalFetch;
            }
        },
    },
    {
        name: "browser items load from the paginated dataset-browser endpoint",
        run: async () => {
            const originalFetch = globalThis.fetch;
            const calls: string[] = [];

            globalThis.fetch = async (input: string | URL | Request) => {
                calls.push(String(input));
                return new Response(JSON.stringify({
                    dataset: "nist_sd300b",
                    dataset_label: "NIST SD300B",
                    selection_policy: "deterministic_round_robin",
                    validation_status: "pass",
                    total: 1,
                    limit: 24,
                    offset: 24,
                    has_more: true,
                    generated_at: "2026-03-31T00:00:00Z",
                    generator_version: "1.0.0",
                    warning_count: 0,
                    summary: {},
                    items: [browserItemA],
                }), {
                    status: 200,
                    headers: { "content-type": "application/json" },
                });
            };

            try {
                const response = await fetchCatalogDatasetBrowser({
                    dataset: "nist_sd300b",
                    split: "test",
                    ui_eligible: true,
                    limit: 24,
                    offset: 24,
                    sort: "split_subject_asset",
                });
                assert.deepEqual(calls, [
                    "/api/catalog/dataset-browser?dataset=nist_sd300b&split=test&ui_eligible=true&limit=24&offset=24&sort=split_subject_asset",
                ]);
                assert.equal(response.items[0].asset_id, "asset_browser_a");
                assert.equal(response.offset, 24);
            } finally {
                globalThis.fetch = originalFetch;
            }
        },
    },
    {
        name: "browser pair loading uses server preview URLs directly",
        run: async () => {
            const originalFetch = globalThis.fetch;
            const calls: string[] = [];

            globalThis.fetch = async (input: string | URL | Request) => {
                const url = String(input);
                calls.push(url);

                if (url === browserItemA.preview_url || url === browserItemB.preview_url) {
                    return new Response(new Blob([url], { type: "image/png" }), {
                        status: 200,
                        headers: { "content-type": "image/png" },
                    });
                }

                throw new Error(`Unexpected fetch: ${url}`);
            };

            try {
                const files = await loadCatalogBrowserPairFiles(browserItemA, browserItemB);
                assert.deepEqual(calls, [browserItemA.preview_url, browserItemB.preview_url]);
                assert.equal(files.fileA.name, "asset_browser_a_preview.png");
                assert.equal(files.fileB.name, "asset_browser_b_preview.png");
            } finally {
                globalThis.fetch = originalFetch;
            }
        },
    },
    {
        name: "browser model builds query, pair key, and readiness without filesystem logic",
        run: () => {
            const query = toCatalogBrowserQuery("nist_sd300b", {
                ...createDefaultBrowserFilters(),
                split: "test",
                capture: "plain",
                subjectId: "100001",
                uiEligible: "eligible",
            });

            assert.deepEqual(query, {
                dataset: "nist_sd300b",
                split: "test",
                capture: "plain",
                modality: undefined,
                subject_id: "100001",
                finger: undefined,
                ui_eligible: true,
                limit: 48,
                offset: 0,
                sort: "default",
            });
            assert.equal(createBrowserPairKey(browserItemA, browserItemB), "asset_browser_a::asset_browser_b");
            assert.equal(createPairPreviewState(browserItemA, browserItemB).status, "ready");

            const sourcePaths = [
                "../src/features/verify/VerifyWorkspaceProductScreen.tsx",
                "../src/features/verify/hooks/useCatalogBrowser.ts",
                "../src/api/matchService.ts",
            ];
            for (const sourcePath of sourcePaths) {
                const source = fs.readFileSync(new URL(sourcePath, import.meta.url), "utf-8");
                assert.equal(source.includes("data/processed"), false);
                assert.equal(source.includes("ui_assets/index.json"), false);
            }
        },
    },
];

let failures = 0;

for (const testCase of tests) {
    try {
        await testCase.run();
        console.log(`PASS ${testCase.name}`);
    } catch (error) {
        failures += 1;
        console.error(`FAIL ${testCase.name}`);
        console.error(error);
    }
}

if (failures > 0) {
    process.exitCode = 1;
    throw new Error(`${failures} test(s) failed.`);
}

console.log(`PASS ${tests.length} test(s)`);
