import assert from "node:assert/strict";
import test from "node:test";
import {
    fetchCatalogDatasets,
    fetchCatalogVerifyCases,
    loadCatalogVerifyCaseFiles,
} from "../src/api/matchService.ts";
import type { CatalogDatasetSummary, CatalogVerifyCase } from "../src/types/index.ts";

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
    evidence_quality: {
        selection_driver: "heuristic_fallback",
        benchmark_backed_selection: false,
        heuristic_fallback_used: true,
        benchmark_discovery_outcome: "dataset_fallback_no_benchmark_evidence",
        evidence_status: "degraded",
        evidence_note: "Benchmark evidence was unavailable, so this case used a heuristic fallback.",
    },
};

const datasetSummary: CatalogDatasetSummary = {
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
        benchmark_backed_cases: 1,
        heuristic_fallback_cases: 0,
        missing_benchmark_evidence: false,
        status: "healthy",
        note: "1 curated verify case(s) are benchmark-backed and demo-ready.",
    },
};

test("Catalog verify cases are loaded from the catalog endpoint", async () => {
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
                catalog_build_health: {
                    catalog_build_status: "degraded",
                    total_verify_cases: 1,
                    benchmark_backed_case_count: 0,
                    heuristic_fallback_case_count: 1,
                    datasets_with_missing_benchmark_evidence: ["nist_sd300b"],
                    summary_message: "1 of 1 curated verify case(s) use heuristic fallback across 1 dataset(s).",
                },
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
        assert.equal(response.items[0].recommended_method, "sift");
        assert.equal(response.items[0].evidence_quality?.evidence_status, "degraded");
        assert.equal(response.catalog_build_health?.catalog_build_status, "degraded");
    } finally {
        globalThis.fetch = originalFetch;
    }
});

test("Catalog dataset summaries expose compact demo health and top-level build health", async () => {
    const originalFetch = globalThis.fetch;

    globalThis.fetch = async () => new Response(
        JSON.stringify({
            items: [datasetSummary],
            catalog_build_health: {
                catalog_build_status: "healthy",
                total_verify_cases: 1,
                benchmark_backed_case_count: 1,
                heuristic_fallback_case_count: 0,
                datasets_with_missing_benchmark_evidence: [],
                summary_message: "All 1 curated verify case(s) are backed by benchmark evidence.",
            },
        }),
        {
            status: 200,
            headers: { "content-type": "application/json" },
        },
    );

    try {
        const response = await fetchCatalogDatasets();
        assert.equal(response.items[0].demo_health?.status, "healthy");
        assert.equal(response.catalog_build_health?.catalog_build_status, "healthy");
    } finally {
        globalThis.fetch = originalFetch;
    }
});

test("Demo asset loading uses the server URLs directly and returns File objects", async () => {
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
        assert.equal(files.fileA.type, "image/png");
        assert.equal(files.fileB.type, "image/png");
    } finally {
        globalThis.fetch = originalFetch;
    }
});

test("Demo asset failures stay attached to the requested server asset", async () => {
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
});
