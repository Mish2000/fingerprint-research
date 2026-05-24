import { describe, expect, it } from "vitest";
import {
    normalizeIdentificationAdminLayoutResponse,
    normalizeIdentificationHealthResponse,
    normalizeIdentifyResponse,
} from "../src/api/contracts.ts";
import { IDENTIFICATION_RETRIEVAL_METHOD_VALUES, METHOD_VALUES } from "../src/types/index.ts";

const DIRECT_RETRIEVAL_METHODS = [
    "classic_orb",
    "classic_gftt_orb",
    "minutiae",
    "harris",
    "sift",
    "dl",
    "vit",
] as const;

function methodCapabilities() {
    return Object.fromEntries(
        METHOD_VALUES.map((method) => [
            method,
            {
                method,
                display_label: method,
                supports_pairwise_rerank: true,
                supports_direct_vector_retrieval: DIRECT_RETRIEVAL_METHODS.includes(
                    method as (typeof DIRECT_RETRIEVAL_METHODS)[number],
                ),
            },
        ]),
    );
}

function healthPayload(overrides: Record<string, unknown> = {}) {
    return {
        ok: true,
        error: null,
        status: "ready",
        identify_ok: true,
        identify_error: null,
        identify_status: "ready",
        identify_browser_ok: true,
        identify_browser_initialized: false,
        identify_browser_error: null,
        identify_browser_status: "lazy_not_initialized",
        methods: {},
        method_capabilities: methodCapabilities(),
        retrieval_capabilities: methodCapabilities(),
        direct_vector_retrieval_methods: [...DIRECT_RETRIEVAL_METHODS],
        rerank_only_methods: ["dedicated"],
        ...overrides,
    };
}

function adminLayoutPayload(overrides: Record<string, unknown> = {}) {
    return {
        backend: "postgresql",
        layout_version: "v4_dual_database_identity_profile_split",
        dual_database_enabled: true,
        table_prefix: "",
        redacted_database_urls: {
            biometric_db: "postgresql://admin:***@127.0.0.1:5432/biometric_db",
            identity_db: "postgresql://admin:***@127.0.0.1:5433/identity_db",
        },
        resolved_table_names: {
            person: "biometric_db.person_directory",
            identity: "identity_db.identity_map",
            raw: "biometric_db.raw_fingerprints",
            vectors: "biometric_db.feature_vectors",
            generic_vectors: "biometric_db.method_retrieval_vectors",
        },
        table_presence: {
            biometric_db: {
                person: true,
                identity: false,
                raw: true,
                vectors: true,
                generic_vectors: true,
            },
            identity_db: {
                person: false,
                identity: true,
                raw: false,
                vectors: false,
                generic_vectors: false,
            },
        },
        row_counts: {
            people: 2,
            identity: 2,
            raw: 2,
            vectors_by_method: {},
            legacy_vectors_by_method: {},
            generic_vectors_by_method_kind: {},
        },
        vector_extension_present_in_biometric_db: true,
        unexpected_vector_methods: {},
        method_capabilities: methodCapabilities(),
        retrieval_capabilities: methodCapabilities(),
        direct_vector_retrieval_methods: [...DIRECT_RETRIEVAL_METHODS],
        rerank_only_methods: ["dedicated"],
        retrieval_vector_coverage_by_method: {},
        retrieval_methods_missing_vectors: [...DIRECT_RETRIEVAL_METHODS],
        retrieval_methods_with_zero_coverage: [...DIRECT_RETRIEVAL_METHODS],
        coverage_recommendation: "",
        vector_storage_schema: {},
        schema_hardening: {},
        reconciliation: {},
        integrity_warnings: [],
        overall_ok: true,
        readiness: {
            ready: true,
            status: "ready",
            error_count: 0,
            warning_count: 0,
        },
        errors: [],
        warnings: [],
        issues: [],
        ...overrides,
    };
}

describe("identification API contract normalizers", () => {
    it("accepts health direct retrieval methods declared by the registry", () => {
        const payload = normalizeIdentificationHealthResponse(healthPayload());

        expect(payload.direct_vector_retrieval_methods).toEqual([...DIRECT_RETRIEVAL_METHODS]);
        expect(payload.direct_vector_retrieval_methods).toEqual([...IDENTIFICATION_RETRIEVAL_METHOD_VALUES]);
    });

    it("accepts admin layout retrieval coverage methods declared by the registry", () => {
        const payload = normalizeIdentificationAdminLayoutResponse(adminLayoutPayload());

        expect(payload.direct_vector_retrieval_methods).toEqual([...DIRECT_RETRIEVAL_METHODS]);
        expect(payload.retrieval_methods_missing_vectors).toEqual([...DIRECT_RETRIEVAL_METHODS]);
        expect(payload.retrieval_methods_with_zero_coverage).toEqual([...DIRECT_RETRIEVAL_METHODS]);
    });

    it("accepts identify responses for non-embedding direct retrieval methods", () => {
        const payload = normalizeIdentifyResponse({
            retrieval_method: "classic_gftt_orb",
            rerank_method: "dedicated",
            threshold: 0.42,
            decision: false,
            total_enrolled: 0,
            candidate_pool_size: 0,
            shortlist_size: 0,
            hints_applied: {},
            top_candidate: null,
            candidates: [],
            latency_ms: {},
            storage_layout: {},
        });

        expect(payload.retrieval_method).toBe("classic_gftt_orb");
        expect(payload.rerank_method).toBe("dedicated");
    });

    it("rejects dedicated as a direct retrieval method", () => {
        expect(() => normalizeIdentificationHealthResponse(healthPayload({
            direct_vector_retrieval_methods: [...DIRECT_RETRIEVAL_METHODS, "dedicated"],
        }))).toThrow(/direct_vector_retrieval_methods\[7\] must be one of/);

        expect(() => normalizeIdentifyResponse({
            retrieval_method: "dedicated",
            rerank_method: "dedicated",
            threshold: 0.42,
            decision: false,
            total_enrolled: 0,
            candidate_pool_size: 0,
            shortlist_size: 0,
            hints_applied: {},
            top_candidate: null,
            candidates: [],
            latency_ms: {},
            storage_layout: {},
        })).toThrow(/retrieval_method must be one of/);
    });
});
