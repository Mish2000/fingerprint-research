import {isObject} from "../utils/error.ts";
import {
    BENCHMARK_VIEW_MODE_VALUES,
    BENCHMARK_BEST_METRIC_VALUES,
    BENCHMARK_RUN_KIND_VALUES,
    type BenchmarkBestMetric,
    type BenchmarkArtifactLink,
    type BenchmarkProvenance,
    type BenchmarkRunInfo,
    type BenchmarkRunKind,
    type BenchmarkRunsResponse,
    type BenchmarkSummaryResponse,
    type BenchmarkViewMode,
    type BestMethodEntry,
    type BestMethodsResponse,
    type CatalogBrowserItem,
    type CatalogBuildHealth,
    type CatalogDatasetDemoHealth,
    type CatalogDatasetBrowserResponse,
    type CatalogDatasetSummary,
    type CatalogDatasetsResponse,
    type CatalogIdentityItem,
    type CatalogIdentifyExemplar,
    type CatalogIdentifyDemoIdentity,
    type CatalogIdentifyGalleryResponse,
    type CatalogIdentifyProbeCase,
    type CatalogVerifyCase,
    type CatalogVerifyCasesResponse,
    type EvidenceQuality,
    type IdentifyBrowserResetResponse,
    type IdentifyBrowserSeedSelectionResponse,
    type Capture,
    CAPTURE_VALUES,
    type ComparisonResponse,
    type ComparisonRow,
    type DeleteIdentityResponse,
    type DemoCase,
    type DemoCasesResponse,
    type EnrollFingerprintResponse,
    IDENTIFICATION_RETRIEVAL_METHOD_VALUES,
    type IdentificationAdminDatabaseUrls,
    type IdentificationAdminInspectionResponse,
    type IdentificationAdminIssue,
    type IdentificationAdminReadiness,
    type IdentificationAdminReconciliationResponse,
    type IdentificationAdminReconciliationSummary,
    type IdentificationAdminResolvedTableNames,
    type IdentificationAdminRoleTablePresence,
    type IdentificationAdminRowCounts,
    type IdentificationAdminTablePresence,
    type IdentificationHealthMethodAvailability,
    type IdentificationHealthResponse,
    type IdentifyDemoResetResponse,
    type IdentifyDemoSeedResponse,
    type IdentificationRetrievalMethod,
    type IdentificationStatsResponse,
    type IdentifyCandidate,
    type IdentifyResponse,
    type JsonRecord,
    type LatencyBreakdown,
    type MatchMeta,
    type MatchResponse,
    type Method,
    METHOD_VALUES,
    normalizeMethodValue,
    type NamedInfo,
    type Overlay,
    OVERLAY_MATCH_KIND_VALUES,
    type OverlayMatch,
    type OverlayMatchKind,
    type StorageLayout
} from "../types/index.ts";


type UnknownRecord = Record<string, unknown>;

function expectObject(payload: unknown, label: string): UnknownRecord {
    if (!isObject(payload)) {
        throw new Error(`${label} must be an object.`);
    }
    return payload;
}

function expectArray(payload: unknown, label: string): unknown[] {
    if (!Array.isArray(payload)) {
        throw new Error(`${label} must be an array.`);
    }
    return payload;
}

function expectString(record: UnknownRecord, key: string, label: string): string {
    const value = record[key];
    if (typeof value !== "string") {
        throw new Error(`${label}.${key} must be a string.`);
    }
    return value;
}

function maybeString(record: UnknownRecord, key: string): string | null {
    const value = record[key];
    return typeof value === "string" ? value : null;
}

function expectBoolean(record: UnknownRecord, key: string, label: string): boolean {
    const value = record[key];
    if (typeof value !== "boolean") {
        throw new Error(`${label}.${key} must be a boolean.`);
    }
    return value;
}

function maybeBoolean(record: UnknownRecord, key: string): boolean | null {
    const value = record[key];
    return typeof value === "boolean" ? value : null;
}

function expectNumber(record: UnknownRecord, key: string, label: string): number {
    const value = record[key];
    if (typeof value !== "number" || Number.isNaN(value)) {
        throw new Error(`${label}.${key} must be a number.`);
    }
    return value;
}

function maybeNumber(record: UnknownRecord, key: string): number | null {
    const value = record[key];
    if (typeof value !== "number" || Number.isNaN(value)) {
        return null;
    }
    return value;
}

function expectStringArray(payload: unknown, label: string): string[] {
    return expectArray(payload, label).map((item, index) => {
        if (typeof item !== "string") {
            throw new Error(`${label}[${index}] must be a string.`);
        }
        return item;
    });
}

function normalizeEnumValue<T extends string>(
    value: unknown,
    allowed: readonly T[],
    label: string,
): T {
    if (typeof value !== "string") {
        throw new Error(`${label} must be a string.`);
    }
    if (!allowed.includes(value as T)) {
        throw new Error(`${label} must be one of: ${allowed.join(", ")}.`);
    }
    return value as T;
}

function normalizeMethod(value: unknown, label = "method"): Method {
    const normalized = normalizeMethodValue(value);
    if (normalized === null) {
        throw new Error(`${label} must be one of: ${METHOD_VALUES.join(", ")}, classic, classic_v2.`);
    }
    return normalized;
}

function normalizeCapture(value: unknown, label = "capture"): Capture {
    return normalizeEnumValue(value, CAPTURE_VALUES, label);
}

function normalizeOverlayMatchKind(value: unknown, label = "overlay.kind"): OverlayMatchKind {
    return normalizeEnumValue(value, OVERLAY_MATCH_KIND_VALUES, label);
}

function normalizeBenchmarkRunKind(value: unknown, label = "run_kind"): BenchmarkRunKind {
    return normalizeEnumValue(value, BENCHMARK_RUN_KIND_VALUES, label);
}

function normalizeBenchmarkViewMode(value: unknown, label = "view_mode"): BenchmarkViewMode {
    return normalizeEnumValue(value, BENCHMARK_VIEW_MODE_VALUES, label);
}

function normalizeBenchmarkBestMetric(value: unknown, label = "metric"): BenchmarkBestMetric {
    return normalizeEnumValue(value, BENCHMARK_BEST_METRIC_VALUES, label);
}

function normalizeRetrievalMethod(value: unknown, label = "retrieval_method"): IdentificationRetrievalMethod {
    return normalizeEnumValue(value, IDENTIFICATION_RETRIEVAL_METHOD_VALUES, label);
}

function normalizeEvidenceSelectionDriver(value: unknown, label = "selection_driver"): EvidenceQuality["selection_driver"] {
    return normalizeEnumValue(value, ["benchmark_driven", "heuristic_fallback"] as const, label);
}

function normalizeEvidenceStatus(value: unknown, label = "evidence_status"): EvidenceQuality["evidence_status"] {
    return normalizeEnumValue(value, ["strong", "fallback", "degraded"] as const, label);
}

function normalizeCatalogDatasetDemoHealthStatus(
    value: unknown,
    label = "CatalogDatasetDemoHealth.status",
): CatalogDatasetDemoHealth["status"] {
    return normalizeEnumValue(value, ["healthy", "degraded", "incomplete"] as const, label);
}

function normalizeCatalogBuildStatus(value: unknown, label = "CatalogBuildHealth.catalog_build_status"): CatalogBuildHealth["catalog_build_status"] {
    return normalizeEnumValue(value, ["healthy", "degraded", "incomplete"] as const, label);
}

function normalizeNamedInfo(payload: unknown, label = "NamedInfo"): NamedInfo {
    const record = expectObject(payload, label);
    return {
        key: expectString(record, "key", label),
        label: expectString(record, "label", label),
        summary: maybeString(record, "summary") ?? "",
    };
}

function normalizeBenchmarkArtifactLink(payload: unknown): BenchmarkArtifactLink {
    const record = expectObject(payload, "BenchmarkArtifactLink");
    return {
        key: expectString(record, "key", "BenchmarkArtifactLink"),
        label: expectString(record, "label", "BenchmarkArtifactLink"),
        available: expectBoolean(record, "available", "BenchmarkArtifactLink"),
        url: maybeString(record, "url"),
    };
}

function normalizeBenchmarkProvenance(payload: unknown): BenchmarkProvenance {
    const record = expectObject(payload, "BenchmarkProvenance");
    return {
        run: expectString(record, "run", "BenchmarkProvenance"),
        run_label: expectString(record, "run_label", "BenchmarkProvenance"),
        run_kind: normalizeBenchmarkRunKind(record.run_kind, "BenchmarkProvenance.run_kind"),
        view_mode: normalizeBenchmarkViewMode(record.view_mode, "BenchmarkProvenance.view_mode"),
        status: expectString(record, "status", "BenchmarkProvenance"),
        validation_state: expectString(record, "validation_state", "BenchmarkProvenance"),
        source_type: expectString(record, "source_type", "BenchmarkProvenance"),
        artifact_source: expectString(record, "artifact_source", "BenchmarkProvenance"),
        methods_in_run: expectStringArray(record.methods_in_run ?? [], "BenchmarkProvenance.methods_in_run"),
        benchmark_methods_in_run: expectStringArray(record.benchmark_methods_in_run ?? [], "BenchmarkProvenance.benchmark_methods_in_run"),
        canonical_method: maybeString(record, "canonical_method"),
        benchmark_method: maybeString(record, "benchmark_method"),
        method_label: maybeString(record, "method_label"),
        timestamp_utc: maybeString(record, "timestamp_utc"),
        limit: maybeNumber(record, "limit"),
        pairs_path: maybeString(record, "pairs_path"),
        manifest_path: maybeString(record, "manifest_path"),
        data_dir: maybeString(record, "data_dir"),
        git_commit: maybeString(record, "git_commit"),
        available_artifacts: expectStringArray(record.available_artifacts ?? [], "BenchmarkProvenance.available_artifacts"),
    };
}

function normalizeStringRecord(payload: unknown, label: string): StorageLayout {
    const record = expectObject(payload, label);
    return Object.fromEntries(
        Object.entries(record).map(([key, value]) => {
            if (typeof value !== "string") {
                throw new Error(`${label}.${key} must be a string.`);
            }
            return [key, value];
        }),
    );
}

function normalizeNumberRecord(payload: unknown, label: string): LatencyBreakdown {
    const record = expectObject(payload, label);
    return Object.fromEntries(
        Object.entries(record).map(([key, value]) => {
            if (typeof value !== "number" || Number.isNaN(value)) {
                throw new Error(`${label}.${key} must be a number.`);
            }
            return [key, value];
        }),
    );
}

function normalizeJsonRecord(payload: unknown): JsonRecord {
    if (!isObject(payload)) {
        return {};
    }
    return { ...payload };
}

function normalizeNullableNumberRecord(payload: unknown, label: string): Record<string, number | null> {
    const record = expectObject(payload, label);
    return Object.fromEntries(
        Object.entries(record).map(([key, value]) => {
            if (value !== null && (typeof value !== "number" || Number.isNaN(value))) {
                throw new Error(`${label}.${key} must be a number or null.`);
            }
            return [key, value];
        }),
    );
}

function normalizeIdentificationHealthMethodAvailability(
    payload: unknown,
    label = "IdentificationHealthMethodAvailability",
): IdentificationHealthMethodAvailability {
    const record = expectObject(payload, label);
    return {
        ...record,
        available: maybeBoolean(record, "available"),
        error: maybeString(record, "error"),
    };
}

function normalizeIdentificationAdminDatabaseUrls(payload: unknown): IdentificationAdminDatabaseUrls {
    const record = expectObject(payload, "IdentificationAdminDatabaseUrls");
    return {
        biometric_db: expectString(record, "biometric_db", "IdentificationAdminDatabaseUrls"),
        identity_db: expectString(record, "identity_db", "IdentificationAdminDatabaseUrls"),
    };
}

function normalizeIdentificationAdminResolvedTableNames(payload: unknown): IdentificationAdminResolvedTableNames {
    const record = expectObject(payload, "IdentificationAdminResolvedTableNames");
    return {
        person: expectString(record, "person", "IdentificationAdminResolvedTableNames"),
        identity: expectString(record, "identity", "IdentificationAdminResolvedTableNames"),
        raw: expectString(record, "raw", "IdentificationAdminResolvedTableNames"),
        vectors: expectString(record, "vectors", "IdentificationAdminResolvedTableNames"),
    };
}

function normalizeIdentificationAdminRoleTablePresence(
    payload: unknown,
    label = "IdentificationAdminRoleTablePresence",
): IdentificationAdminRoleTablePresence {
    const record = expectObject(payload, label);
    return {
        person: expectBoolean(record, "person", label),
        identity: expectBoolean(record, "identity", label),
        raw: expectBoolean(record, "raw", label),
        vectors: expectBoolean(record, "vectors", label),
    };
}

function normalizeIdentificationAdminTablePresence(payload: unknown): IdentificationAdminTablePresence {
    const record = expectObject(payload, "IdentificationAdminTablePresence");
    return {
        biometric_db: normalizeIdentificationAdminRoleTablePresence(
            record.biometric_db,
            "IdentificationAdminTablePresence.biometric_db",
        ),
        identity_db: normalizeIdentificationAdminRoleTablePresence(
            record.identity_db,
            "IdentificationAdminTablePresence.identity_db",
        ),
    };
}

function normalizeIdentificationAdminRowCounts(payload: unknown): IdentificationAdminRowCounts {
    const record = expectObject(payload, "IdentificationAdminRowCounts");
    return {
        people: maybeNumber(record, "people"),
        identity: maybeNumber(record, "identity"),
        raw: maybeNumber(record, "raw"),
        vectors_by_method: normalizeNullableNumberRecord(
            record.vectors_by_method ?? {},
            "IdentificationAdminRowCounts.vectors_by_method",
        ),
    };
}

function normalizeIdentificationAdminIssue(payload: unknown): IdentificationAdminIssue {
    const record = expectObject(payload, "IdentificationAdminIssue");
    return {
        ...record,
        code: expectString(record, "code", "IdentificationAdminIssue"),
        severity: expectString(record, "severity", "IdentificationAdminIssue"),
        database_role: expectString(record, "database_role", "IdentificationAdminIssue"),
        message: expectString(record, "message", "IdentificationAdminIssue"),
    };
}

function normalizeIdentificationAdminReadiness(payload: unknown): IdentificationAdminReadiness {
    const record = expectObject(payload, "IdentificationAdminReadiness");
    return {
        ready: expectBoolean(record, "ready", "IdentificationAdminReadiness"),
        status: expectString(record, "status", "IdentificationAdminReadiness"),
        error_count: expectNumber(record, "error_count", "IdentificationAdminReadiness"),
        warning_count: expectNumber(record, "warning_count", "IdentificationAdminReadiness"),
    };
}

function normalizeIdentificationAdminInspectionResponse(
    payload: unknown,
    label = "IdentificationAdminInspectionResponse",
): IdentificationAdminInspectionResponse {
    const record = expectObject(payload, label);
    return {
        backend: expectString(record, "backend", label),
        layout_version: expectString(record, "layout_version", label),
        dual_database_enabled: expectBoolean(record, "dual_database_enabled", label),
        table_prefix: maybeString(record, "table_prefix") ?? "",
        redacted_database_urls: normalizeIdentificationAdminDatabaseUrls(record.redacted_database_urls),
        resolved_table_names: normalizeIdentificationAdminResolvedTableNames(record.resolved_table_names),
        table_presence: normalizeIdentificationAdminTablePresence(record.table_presence),
        row_counts: normalizeIdentificationAdminRowCounts(record.row_counts),
        vector_extension_present_in_biometric_db: maybeBoolean(record, "vector_extension_present_in_biometric_db"),
        unexpected_vector_methods: normalizeNumberRecord(
            record.unexpected_vector_methods ?? {},
            `${label}.unexpected_vector_methods`,
        ),
        schema_hardening: normalizeJsonRecord(record.schema_hardening),
        reconciliation: normalizeJsonRecord(record.reconciliation),
        integrity_warnings: expectStringArray(record.integrity_warnings ?? [], `${label}.integrity_warnings`),
        overall_ok: expectBoolean(record, "overall_ok", label),
        readiness: normalizeIdentificationAdminReadiness(record.readiness),
        errors: expectArray(record.errors ?? [], `${label}.errors`).map(normalizeIdentificationAdminIssue),
        warnings: expectArray(record.warnings ?? [], `${label}.warnings`).map(normalizeIdentificationAdminIssue),
        issues: expectArray(record.issues ?? [], `${label}.issues`).map(normalizeIdentificationAdminIssue),
    };
}

function normalizeIdentificationAdminReconciliationSummary(
    payload: unknown,
): IdentificationAdminReconciliationSummary {
    const record = expectObject(payload, "IdentificationAdminReconciliationSummary");
    return {
        severity: normalizeNumberRecord(record.severity ?? {}, "IdentificationAdminReconciliationSummary.severity"),
        repairability: normalizeNumberRecord(
            record.repairability ?? {},
            "IdentificationAdminReconciliationSummary.repairability",
        ),
        manual_reconciliation_required: expectBoolean(
            record,
            "manual_reconciliation_required",
            "IdentificationAdminReconciliationSummary",
        ),
        overall_ok: expectBoolean(record, "overall_ok", "IdentificationAdminReconciliationSummary"),
        readiness: normalizeIdentificationAdminReadiness(record.readiness),
    };
}

function normalizeMatchMeta(payload: unknown): MatchMeta {
    return normalizeJsonRecord(payload) as MatchMeta;
}

function normalizeOverlayMatch(payload: unknown): OverlayMatch {
    const record = expectObject(payload, "OverlayMatch");
    const a = expectArray(record.a, "OverlayMatch.a");
    const b = expectArray(record.b, "OverlayMatch.b");

    if (a.length !== 2 || b.length !== 2) {
        throw new Error("OverlayMatch coordinates must contain exactly two numbers.");
    }

    const ax = a[0];
    const ay = a[1];
    const bx = b[0];
    const by = b[1];

    if (
        typeof ax !== "number" || Number.isNaN(ax) ||
        typeof ay !== "number" || Number.isNaN(ay) ||
        typeof bx !== "number" || Number.isNaN(bx) ||
        typeof by !== "number" || Number.isNaN(by)
    ) {
        throw new Error("OverlayMatch coordinates must be numeric.");
    }

    return {
        a: [ax, ay],
        b: [bx, by],
        kind: normalizeOverlayMatchKind(record.kind),
        sim: maybeNumber(record, "sim"),
    };
}

function normalizeOverlay(payload: unknown): Overlay {
    const record = expectObject(payload, "Overlay");
    return {
        matches: expectArray(record.matches, "Overlay.matches").map(normalizeOverlayMatch),
    };
}

export function normalizeMatchResponse(payload: unknown): MatchResponse {
    const record = expectObject(payload, "MatchResponse");
    return {
        method: normalizeMethod(record.method),
        score: expectNumber(record, "score", "MatchResponse"),
        decision: expectBoolean(record, "decision", "MatchResponse"),
        threshold: expectNumber(record, "threshold", "MatchResponse"),
        latency_ms: expectNumber(record, "latency_ms", "MatchResponse"),
        meta: normalizeMatchMeta(record.meta),
        overlay: record.overlay == null ? null : normalizeOverlay(record.overlay),
    };
}

function normalizeBenchmarkRunInfo(payload: unknown): BenchmarkRunInfo {
    const record = expectObject(payload, "BenchmarkRunInfo");
    return {
        run: expectString(record, "run", "BenchmarkRunInfo"),
        dataset: maybeString(record, "dataset"),
        run_kind: normalizeBenchmarkRunKind(record.run_kind),
        view_mode: normalizeBenchmarkViewMode(record.view_mode),
        status: expectString(record, "status", "BenchmarkRunInfo"),
        validation_state: expectString(record, "validation_state", "BenchmarkRunInfo"),
        validated: expectBoolean(record, "validated", "BenchmarkRunInfo"),
        recommended: expectBoolean(record, "recommended", "BenchmarkRunInfo"),
        run_label: maybeString(record, "run_label"),
        artifact_count: expectNumber(record, "artifact_count", "BenchmarkRunInfo"),
        summary_note: expectString(record, "summary_note", "BenchmarkRunInfo"),
        methods: expectStringArray(record.methods, "BenchmarkRunInfo.methods"),
        benchmark_methods: expectStringArray(record.benchmark_methods ?? [], "BenchmarkRunInfo.benchmark_methods"),
        splits: expectStringArray(record.splits, "BenchmarkRunInfo.splits"),
        dataset_info: record.dataset_info == null ? null : normalizeNamedInfo(record.dataset_info, "BenchmarkRunInfo.dataset_info"),
    };
}

export function normalizeBenchmarkRunsResponse(payload: unknown): BenchmarkRunsResponse {
    const record = expectObject(payload, "BenchmarkRunsResponse");
    return {
        default_run: maybeString(record, "default_run"),
        default_dataset: maybeString(record, "default_dataset"),
        default_split: maybeString(record, "default_split"),
        default_view_mode: normalizeBenchmarkViewMode(record.default_view_mode ?? "canonical", "BenchmarkRunsResponse.default_view_mode"),
        runs: expectArray(record.runs, "BenchmarkRunsResponse.runs").map(normalizeBenchmarkRunInfo),
    };
}

export function normalizeBenchmarkSummaryResponse(payload: unknown): BenchmarkSummaryResponse {
    const record = expectObject(payload, "BenchmarkSummaryResponse");
    return {
        dataset: expectString(record, "dataset", "BenchmarkSummaryResponse"),
        split: expectString(record, "split", "BenchmarkSummaryResponse"),
        view_mode: normalizeBenchmarkViewMode(record.view_mode, "BenchmarkSummaryResponse.view_mode"),
        dataset_info: record.dataset_info == null ? null : normalizeNamedInfo(record.dataset_info, "BenchmarkSummaryResponse.dataset_info"),
        split_info: record.split_info == null ? null : normalizeNamedInfo(record.split_info, "BenchmarkSummaryResponse.split_info"),
        view_info: record.view_info == null ? null : normalizeNamedInfo(record.view_info, "BenchmarkSummaryResponse.view_info"),
        validation_state: expectString(record, "validation_state", "BenchmarkSummaryResponse"),
        selection_note: expectString(record, "selection_note", "BenchmarkSummaryResponse"),
        selection_policy: expectString(record, "selection_policy", "BenchmarkSummaryResponse"),
        result_count: expectNumber(record, "result_count", "BenchmarkSummaryResponse"),
        method_count: expectNumber(record, "method_count", "BenchmarkSummaryResponse"),
        run_count: expectNumber(record, "run_count", "BenchmarkSummaryResponse"),
        available_datasets: expectArray(record.available_datasets ?? [], "BenchmarkSummaryResponse.available_datasets")
            .map((item) => normalizeNamedInfo(item, "BenchmarkSummaryResponse.available_datasets[]")),
        available_splits: expectArray(record.available_splits ?? [], "BenchmarkSummaryResponse.available_splits")
            .map((item) => normalizeNamedInfo(item, "BenchmarkSummaryResponse.available_splits[]")),
        available_view_modes: expectArray(record.available_view_modes ?? [], "BenchmarkSummaryResponse.available_view_modes")
            .map((item) => normalizeNamedInfo(item, "BenchmarkSummaryResponse.available_view_modes[]")),
        current_run_families: expectStringArray(record.current_run_families ?? [], "BenchmarkSummaryResponse.current_run_families"),
        artifact_note: expectString(record, "artifact_note", "BenchmarkSummaryResponse"),
    };
}

function normalizeComparisonRow(payload: unknown): ComparisonRow {
    const record = expectObject(payload, "ComparisonRow");
    return {
        dataset: expectString(record, "dataset", "ComparisonRow"),
        run: expectString(record, "run", "ComparisonRow"),
        split: expectString(record, "split", "ComparisonRow"),
        method: expectString(record, "method", "ComparisonRow"),
        benchmark_method: expectString(record, "benchmark_method", "ComparisonRow"),
        method_label: maybeString(record, "method_label"),
        auc: expectNumber(record, "auc", "ComparisonRow"),
        eer: expectNumber(record, "eer", "ComparisonRow"),
        n_pairs: maybeNumber(record, "n_pairs"),
        tar_at_far_1e_2: maybeNumber(record, "tar_at_far_1e_2"),
        tar_at_far_1e_3: maybeNumber(record, "tar_at_far_1e_3"),
        latency_ms: maybeNumber(record, "latency_ms"),
        latency_source: maybeString(record, "latency_source") as "reported" | "wall" | null,
        auc_rank: maybeNumber(record, "auc_rank"),
        eer_rank: maybeNumber(record, "eer_rank"),
        latency_rank: maybeNumber(record, "latency_rank"),
        run_family: maybeString(record, "run_family"),
        run_label: maybeString(record, "run_label"),
        run_kind: normalizeBenchmarkRunKind(record.run_kind, "ComparisonRow.run_kind"),
        view_mode: normalizeBenchmarkViewMode(record.view_mode, "ComparisonRow.view_mode"),
        status: expectString(record, "status", "ComparisonRow"),
        validation_state: expectString(record, "validation_state", "ComparisonRow"),
        artifact_count: expectNumber(record, "artifact_count", "ComparisonRow"),
        available_artifacts: expectStringArray(record.available_artifacts ?? [], "ComparisonRow.available_artifacts"),
        summary_text: expectString(record, "summary_text", "ComparisonRow"),
        artifacts: expectArray(record.artifacts ?? [], "ComparisonRow.artifacts").map(normalizeBenchmarkArtifactLink),
        provenance: record.provenance == null ? null : normalizeBenchmarkProvenance(record.provenance),
    };
}

function normalizeNamedInfoMap(payload: unknown, label: string): Record<string, NamedInfo> {
    const record = expectObject(payload, label);
    return Object.fromEntries(
        Object.entries(record).map(([key, value]) => [key, normalizeNamedInfo(value, `${label}.${key}`)]),
    );
}

function normalizeAssetDimensions(payload: unknown, label: string) {
    const record = expectObject(payload, label);
    return {
        width: expectNumber(record, "width", label),
        height: expectNumber(record, "height", label),
    };
}

function normalizeEvidenceQuality(payload: unknown): EvidenceQuality {
    const record = expectObject(payload, "EvidenceQuality");
    return {
        selection_driver: normalizeEvidenceSelectionDriver(record.selection_driver, "EvidenceQuality.selection_driver"),
        benchmark_backed_selection: expectBoolean(record, "benchmark_backed_selection", "EvidenceQuality"),
        heuristic_fallback_used: expectBoolean(record, "heuristic_fallback_used", "EvidenceQuality"),
        benchmark_discovery_outcome: expectString(record, "benchmark_discovery_outcome", "EvidenceQuality"),
        evidence_status: normalizeEvidenceStatus(record.evidence_status, "EvidenceQuality.evidence_status"),
        evidence_note: expectString(record, "evidence_note", "EvidenceQuality"),
    };
}

function normalizeCatalogDatasetDemoHealth(payload: unknown): CatalogDatasetDemoHealth {
    const record = expectObject(payload, "CatalogDatasetDemoHealth");
    return {
        planned_verify_cases: expectNumber(record, "planned_verify_cases", "CatalogDatasetDemoHealth"),
        built_verify_cases: expectNumber(record, "built_verify_cases", "CatalogDatasetDemoHealth"),
        benchmark_backed_cases: expectNumber(record, "benchmark_backed_cases", "CatalogDatasetDemoHealth"),
        heuristic_fallback_cases: expectNumber(record, "heuristic_fallback_cases", "CatalogDatasetDemoHealth"),
        missing_benchmark_evidence: expectBoolean(record, "missing_benchmark_evidence", "CatalogDatasetDemoHealth"),
        status: normalizeCatalogDatasetDemoHealthStatus(record.status, "CatalogDatasetDemoHealth.status"),
        note: expectString(record, "note", "CatalogDatasetDemoHealth"),
    };
}

function normalizeCatalogBuildHealth(payload: unknown): CatalogBuildHealth {
    const record = expectObject(payload, "CatalogBuildHealth");
    return {
        catalog_build_status: normalizeCatalogBuildStatus(record.catalog_build_status, "CatalogBuildHealth.catalog_build_status"),
        total_verify_cases: expectNumber(record, "total_verify_cases", "CatalogBuildHealth"),
        benchmark_backed_case_count: expectNumber(record, "benchmark_backed_case_count", "CatalogBuildHealth"),
        heuristic_fallback_case_count: expectNumber(record, "heuristic_fallback_case_count", "CatalogBuildHealth"),
        datasets_with_missing_benchmark_evidence: expectStringArray(
            record.datasets_with_missing_benchmark_evidence ?? [],
            "CatalogBuildHealth.datasets_with_missing_benchmark_evidence",
        ),
        summary_message: expectString(record, "summary_message", "CatalogBuildHealth"),
    };
}

export function normalizeComparisonResponse(payload: unknown): ComparisonResponse {
    const record = expectObject(payload, "ComparisonResponse");
    return {
        rows: expectArray(record.rows, "ComparisonResponse.rows").map(normalizeComparisonRow),
        datasets: expectStringArray(record.datasets, "ComparisonResponse.datasets"),
        splits: expectStringArray(record.splits, "ComparisonResponse.splits"),
        default_dataset: maybeString(record, "default_dataset"),
        default_split: maybeString(record, "default_split"),
        view_mode: normalizeBenchmarkViewMode(record.view_mode ?? "canonical", "ComparisonResponse.view_mode"),
        view_info: record.view_info == null ? null : normalizeNamedInfo(record.view_info, "ComparisonResponse.view_info"),
        dataset_info: normalizeNamedInfoMap(record.dataset_info ?? {}, "ComparisonResponse.dataset_info"),
        split_info: normalizeNamedInfoMap(record.split_info ?? {}, "ComparisonResponse.split_info"),
    };
}

function normalizeBestMethodEntry(payload: unknown): BestMethodEntry {
    const record = expectObject(payload, "BestMethodEntry");
    return {
        dataset: expectString(record, "dataset", "BestMethodEntry"),
        split: expectString(record, "split", "BestMethodEntry"),
        metric: normalizeBenchmarkBestMetric(record.metric),
        method: expectString(record, "method", "BestMethodEntry"),
        benchmark_method: maybeString(record, "benchmark_method"),
        method_label: maybeString(record, "method_label"),
        run: expectString(record, "run", "BestMethodEntry"),
        value: expectNumber(record, "value", "BestMethodEntry"),
        run_family: maybeString(record, "run_family"),
        run_label: maybeString(record, "run_label"),
        view_mode: normalizeBenchmarkViewMode(record.view_mode ?? "canonical", "BestMethodEntry.view_mode"),
        status: expectString(record, "status", "BestMethodEntry"),
        validation_state: expectString(record, "validation_state", "BestMethodEntry"),
    };
}

export function normalizeBestMethodsResponse(payload: unknown): BestMethodsResponse {
    const record = expectObject(payload, "BestMethodsResponse");
    return {
        dataset: maybeString(record, "dataset"),
        split: maybeString(record, "split"),
        view_mode: normalizeBenchmarkViewMode(record.view_mode ?? "canonical", "BestMethodsResponse.view_mode"),
        entries: expectArray(record.entries, "BestMethodsResponse.entries").map(normalizeBestMethodEntry),
    };
}

function normalizeDemoCase(payload: unknown): DemoCase {
    const record = expectObject(payload, "DemoCase");
    return {
        id: expectString(record, "id", "DemoCase"),
        title: expectString(record, "title", "DemoCase"),
        description: expectString(record, "description", "DemoCase"),
        difficulty: expectString(record, "difficulty", "DemoCase"),
        dataset: expectString(record, "dataset", "DemoCase"),
        dataset_label: expectString(record, "dataset_label", "DemoCase"),
        split: expectString(record, "split", "DemoCase"),
        split_label: expectString(record, "split_label", "DemoCase"),
        label: expectNumber(record, "label", "DemoCase"),
        ground_truth: expectString(record, "ground_truth", "DemoCase"),
        recommended_method: normalizeMethod(record.recommended_method, "DemoCase.recommended_method"),
        benchmark_method: maybeString(record, "benchmark_method"),
        benchmark_run: maybeString(record, "benchmark_run"),
        curation_rule: expectString(record, "curation_rule", "DemoCase"),
        benchmark_score: maybeNumber(record, "benchmark_score"),
        capture_a: normalizeCapture(record.capture_a, "DemoCase.capture_a"),
        capture_b: normalizeCapture(record.capture_b, "DemoCase.capture_b"),
        image_a_url: expectString(record, "image_a_url", "DemoCase"),
        image_b_url: expectString(record, "image_b_url", "DemoCase"),
        asset_a_id: maybeString(record, "asset_a_id"),
        asset_b_id: maybeString(record, "asset_b_id"),
        case_type: maybeString(record, "case_type"),
        availability_status: maybeString(record, "availability_status"),
        selection_policy: maybeString(record, "selection_policy"),
        tags: expectStringArray(record.tags ?? [], "DemoCase.tags"),
        modality_relation: maybeString(record, "modality_relation"),
        evidence_quality: record.evidence_quality == null ? null : normalizeEvidenceQuality(record.evidence_quality),
    };
}

export function normalizeDemoCasesResponse(payload: unknown): DemoCasesResponse {
    const record = expectObject(payload, "DemoCasesResponse");
    return {
        cases: expectArray(record.cases, "DemoCasesResponse.cases").map(normalizeDemoCase),
        catalog_build_health: record.catalog_build_health == null ? null : normalizeCatalogBuildHealth(record.catalog_build_health),
    };
}

function normalizeCatalogVerifyCase(payload: unknown): CatalogVerifyCase {
    const record = expectObject(payload, "CatalogVerifyCase");
    return {
        case_id: expectString(record, "case_id", "CatalogVerifyCase"),
        title: expectString(record, "title", "CatalogVerifyCase"),
        description: expectString(record, "description", "CatalogVerifyCase"),
        dataset: expectString(record, "dataset", "CatalogVerifyCase"),
        dataset_label: expectString(record, "dataset_label", "CatalogVerifyCase"),
        split: expectString(record, "split", "CatalogVerifyCase"),
        difficulty: expectString(record, "difficulty", "CatalogVerifyCase"),
        case_type: expectString(record, "case_type", "CatalogVerifyCase"),
        ground_truth: expectString(record, "ground_truth", "CatalogVerifyCase"),
        recommended_method: normalizeMethod(record.recommended_method, "CatalogVerifyCase.recommended_method"),
        capture_a: normalizeCapture(record.capture_a, "CatalogVerifyCase.capture_a"),
        capture_b: normalizeCapture(record.capture_b, "CatalogVerifyCase.capture_b"),
        modality_relation: maybeString(record, "modality_relation"),
        tags: expectStringArray(record.tags ?? [], "CatalogVerifyCase.tags"),
        selection_policy: expectString(record, "selection_policy", "CatalogVerifyCase"),
        selection_reason: expectString(record, "selection_reason", "CatalogVerifyCase"),
        image_a_url: expectString(record, "image_a_url", "CatalogVerifyCase"),
        image_b_url: expectString(record, "image_b_url", "CatalogVerifyCase"),
        availability_status: expectString(record, "availability_status", "CatalogVerifyCase"),
        asset_a_id: maybeString(record, "asset_a_id"),
        asset_b_id: maybeString(record, "asset_b_id"),
        evidence_quality: record.evidence_quality == null ? null : normalizeEvidenceQuality(record.evidence_quality),
    };
}

export function normalizeCatalogVerifyCasesResponse(payload: unknown): CatalogVerifyCasesResponse {
    const record = expectObject(payload, "CatalogVerifyCasesResponse");
    return {
        items: expectArray(record.items, "CatalogVerifyCasesResponse.items").map(normalizeCatalogVerifyCase),
        total: expectNumber(record, "total", "CatalogVerifyCasesResponse"),
        limit: expectNumber(record, "limit", "CatalogVerifyCasesResponse"),
        offset: expectNumber(record, "offset", "CatalogVerifyCasesResponse"),
        has_more: expectBoolean(record, "has_more", "CatalogVerifyCasesResponse"),
        catalog_build_health: record.catalog_build_health == null ? null : normalizeCatalogBuildHealth(record.catalog_build_health),
    };
}

function normalizeCatalogIdentifyDemoIdentity(payload: unknown): CatalogIdentifyDemoIdentity {
    const record = expectObject(payload, "CatalogIdentifyDemoIdentity");
    return {
        id: expectString(record, "id", "CatalogIdentifyDemoIdentity"),
        dataset: expectString(record, "dataset", "CatalogIdentifyDemoIdentity"),
        dataset_label: expectString(record, "dataset_label", "CatalogIdentifyDemoIdentity"),
        display_label: expectString(record, "display_label", "CatalogIdentifyDemoIdentity"),
        capture: maybeString(record, "capture"),
        thumbnail_url: expectString(record, "thumbnail_url", "CatalogIdentifyDemoIdentity"),
        preview_url: expectString(record, "preview_url", "CatalogIdentifyDemoIdentity"),
        subject_id: maybeString(record, "subject_id"),
        gallery_role: maybeString(record, "gallery_role"),
        tags: expectStringArray(record.tags ?? [], "CatalogIdentifyDemoIdentity.tags"),
        recommended_enrollment_asset_id: maybeString(record, "recommended_enrollment_asset_id"),
        recommended_probe_asset_id: maybeString(record, "recommended_probe_asset_id"),
    };
}

function normalizeCatalogIdentifyExemplar(payload: unknown): CatalogIdentifyExemplar {
    const record = expectObject(payload, "CatalogIdentifyExemplar");
    return {
        asset_id: expectString(record, "asset_id", "CatalogIdentifyExemplar"),
        capture: maybeString(record, "capture"),
        finger: maybeString(record, "finger"),
        recommended_usage: maybeString(record, "recommended_usage"),
        asset_reference: expectString(record, "asset_reference", "CatalogIdentifyExemplar"),
        has_servable_asset: expectBoolean(record, "has_servable_asset", "CatalogIdentifyExemplar"),
        availability_status: maybeString(record, "availability_status"),
    };
}

function normalizeCatalogIdentityItem(payload: unknown): CatalogIdentityItem {
    const record = expectObject(payload, "CatalogIdentityItem");
    return {
        identity_id: expectString(record, "identity_id", "CatalogIdentityItem"),
        dataset: expectString(record, "dataset", "CatalogIdentityItem"),
        dataset_label: expectString(record, "dataset_label", "CatalogIdentityItem"),
        display_name: expectString(record, "display_name", "CatalogIdentityItem"),
        subject_id: expectString(record, "subject_id", "CatalogIdentityItem"),
        gallery_role: expectString(record, "gallery_role", "CatalogIdentityItem"),
        tags: expectStringArray(record.tags ?? [], "CatalogIdentityItem.tags"),
        is_demo_safe: expectBoolean(record, "is_demo_safe", "CatalogIdentityItem"),
        enrollment_candidates: expectArray(record.enrollment_candidates ?? [], "CatalogIdentityItem.enrollment_candidates")
            .map(normalizeCatalogIdentifyExemplar),
        probe_candidates: expectArray(record.probe_candidates ?? [], "CatalogIdentityItem.probe_candidates")
            .map(normalizeCatalogIdentifyExemplar),
        preview_url: maybeString(record, "preview_url"),
        thumbnail_url: maybeString(record, "thumbnail_url"),
        recommended_enrollment_asset_id: maybeString(record, "recommended_enrollment_asset_id"),
        recommended_probe_asset_id: maybeString(record, "recommended_probe_asset_id"),
        recommended_enrollment_capture: maybeString(record, "recommended_enrollment_capture"),
        recommended_probe_capture: maybeString(record, "recommended_probe_capture"),
    };
}

function normalizeCatalogIdentifyProbeCase(payload: unknown): CatalogIdentifyProbeCase {
    const record = expectObject(payload, "CatalogIdentifyProbeCase");
    return {
        id: expectString(record, "id", "CatalogIdentifyProbeCase"),
        title: expectString(record, "title", "CatalogIdentifyProbeCase"),
        description: expectString(record, "description", "CatalogIdentifyProbeCase"),
        dataset: expectString(record, "dataset", "CatalogIdentifyProbeCase"),
        dataset_label: expectString(record, "dataset_label", "CatalogIdentifyProbeCase"),
        capture: maybeString(record, "capture"),
        difficulty: expectString(record, "difficulty", "CatalogIdentifyProbeCase"),
        probe_thumbnail_url: expectString(record, "probe_thumbnail_url", "CatalogIdentifyProbeCase"),
        probe_preview_url: expectString(record, "probe_preview_url", "CatalogIdentifyProbeCase"),
        probe_asset_url: expectString(record, "probe_asset_url", "CatalogIdentifyProbeCase"),
        expected_outcome: maybeString(record, "expected_outcome"),
        expected_top_identity_id: maybeString(record, "expected_top_identity_id"),
        expected_top_identity_label: maybeString(record, "expected_top_identity_label"),
        recommended_retrieval_method: record.recommended_retrieval_method == null
            ? null
            : normalizeRetrievalMethod(record.recommended_retrieval_method, "CatalogIdentifyProbeCase.recommended_retrieval_method"),
        recommended_rerank_method: record.recommended_rerank_method == null
            ? null
            : normalizeMethod(record.recommended_rerank_method, "CatalogIdentifyProbeCase.recommended_rerank_method"),
        recommended_shortlist_size: maybeNumber(record, "recommended_shortlist_size"),
        scenario_type: maybeString(record, "scenario_type"),
        tags: expectStringArray(record.tags ?? [], "CatalogIdentifyProbeCase.tags"),
    };
}

export function normalizeCatalogIdentifyGalleryResponse(payload: unknown): CatalogIdentifyGalleryResponse {
    const record = expectObject(payload, "CatalogIdentifyGalleryResponse");
    return {
        items: expectArray(record.items ?? [], "CatalogIdentifyGalleryResponse.items")
            .map(normalizeCatalogIdentityItem),
        demo_identities: expectArray(record.demo_identities ?? [], "CatalogIdentifyGalleryResponse.demo_identities")
            .map(normalizeCatalogIdentifyDemoIdentity),
        probe_cases: expectArray(record.probe_cases ?? [], "CatalogIdentifyGalleryResponse.probe_cases")
            .map(normalizeCatalogIdentifyProbeCase),
        total: expectNumber(record, "total", "CatalogIdentifyGalleryResponse"),
        limit: expectNumber(record, "limit", "CatalogIdentifyGalleryResponse"),
        offset: expectNumber(record, "offset", "CatalogIdentifyGalleryResponse"),
        has_more: expectBoolean(record, "has_more", "CatalogIdentifyGalleryResponse"),
        total_probe_cases: expectNumber(record, "total_probe_cases", "CatalogIdentifyGalleryResponse"),
    };
}

function normalizeCatalogDatasetSummary(payload: unknown): CatalogDatasetSummary {
    const record = expectObject(payload, "CatalogDatasetSummary");
    return {
        dataset: expectString(record, "dataset", "CatalogDatasetSummary"),
        dataset_label: expectString(record, "dataset_label", "CatalogDatasetSummary"),
        has_verify_cases: expectBoolean(record, "has_verify_cases", "CatalogDatasetSummary"),
        has_identify_gallery: expectBoolean(record, "has_identify_gallery", "CatalogDatasetSummary"),
        has_browser_assets: expectBoolean(record, "has_browser_assets", "CatalogDatasetSummary"),
        verify_case_count: expectNumber(record, "verify_case_count", "CatalogDatasetSummary"),
        identify_identity_count: expectNumber(record, "identify_identity_count", "CatalogDatasetSummary"),
        browser_item_count: expectNumber(record, "browser_item_count", "CatalogDatasetSummary"),
        browser_validation_status: maybeString(record, "browser_validation_status"),
        selection_policy: maybeString(record, "selection_policy"),
        available_features: expectStringArray(record.available_features ?? [], "CatalogDatasetSummary.available_features"),
        demo_health: record.demo_health == null ? null : normalizeCatalogDatasetDemoHealth(record.demo_health),
    };
}

export function normalizeCatalogDatasetsResponse(payload: unknown): CatalogDatasetsResponse {
    const record = expectObject(payload, "CatalogDatasetsResponse");
    return {
        items: expectArray(record.items, "CatalogDatasetsResponse.items").map(normalizeCatalogDatasetSummary),
        catalog_build_health: record.catalog_build_health == null ? null : normalizeCatalogBuildHealth(record.catalog_build_health),
    };
}

function normalizeCatalogBrowserItem(payload: unknown): CatalogBrowserItem {
    const record = expectObject(payload, "CatalogBrowserItem");
    return {
        asset_id: expectString(record, "asset_id", "CatalogBrowserItem"),
        dataset: expectString(record, "dataset", "CatalogBrowserItem"),
        split: expectString(record, "split", "CatalogBrowserItem"),
        subject_id: maybeString(record, "subject_id"),
        finger: maybeString(record, "finger"),
        capture: maybeString(record, "capture"),
        modality: maybeString(record, "modality"),
        ui_eligible: expectBoolean(record, "ui_eligible", "CatalogBrowserItem"),
        selection_reason: expectString(record, "selection_reason", "CatalogBrowserItem"),
        selection_policy: expectString(record, "selection_policy", "CatalogBrowserItem"),
        thumbnail_url: expectString(record, "thumbnail_url", "CatalogBrowserItem"),
        preview_url: expectString(record, "preview_url", "CatalogBrowserItem"),
        availability_status: expectString(record, "availability_status", "CatalogBrowserItem"),
        original_dimensions: normalizeAssetDimensions(record.original_dimensions, "CatalogBrowserItem.original_dimensions"),
        thumbnail_dimensions: normalizeAssetDimensions(record.thumbnail_dimensions, "CatalogBrowserItem.thumbnail_dimensions"),
        preview_dimensions: normalizeAssetDimensions(record.preview_dimensions, "CatalogBrowserItem.preview_dimensions"),
    };
}

export function normalizeCatalogDatasetBrowserResponse(payload: unknown): CatalogDatasetBrowserResponse {
    const record = expectObject(payload, "CatalogDatasetBrowserResponse");
    return {
        dataset: expectString(record, "dataset", "CatalogDatasetBrowserResponse"),
        dataset_label: expectString(record, "dataset_label", "CatalogDatasetBrowserResponse"),
        selection_policy: expectString(record, "selection_policy", "CatalogDatasetBrowserResponse"),
        validation_status: expectString(record, "validation_status", "CatalogDatasetBrowserResponse"),
        total: expectNumber(record, "total", "CatalogDatasetBrowserResponse"),
        limit: expectNumber(record, "limit", "CatalogDatasetBrowserResponse"),
        offset: expectNumber(record, "offset", "CatalogDatasetBrowserResponse"),
        has_more: expectBoolean(record, "has_more", "CatalogDatasetBrowserResponse"),
        generated_at: maybeString(record, "generated_at"),
        generator_version: maybeString(record, "generator_version"),
        warning_count: expectNumber(record, "warning_count", "CatalogDatasetBrowserResponse"),
        summary: normalizeJsonRecord(record.summary),
        items: expectArray(record.items, "CatalogDatasetBrowserResponse.items").map(normalizeCatalogBrowserItem),
    };
}

export function normalizeEnrollFingerprintResponse(payload: unknown): EnrollFingerprintResponse {
    const record = expectObject(payload, "EnrollFingerprintResponse");
    return {
        random_id: expectString(record, "random_id", "EnrollFingerprintResponse"),
        created_at: expectString(record, "created_at", "EnrollFingerprintResponse"),
        vector_methods: expectStringArray(record.vector_methods, "EnrollFingerprintResponse.vector_methods"),
        image_sha256: expectString(record, "image_sha256", "EnrollFingerprintResponse"),
        storage_layout: normalizeStringRecord(record.storage_layout ?? {}, "EnrollFingerprintResponse.storage_layout"),
    };
}

function normalizeIdentifyCandidate(payload: unknown): IdentifyCandidate {
    const record = expectObject(payload, "IdentifyCandidate");
    return {
        rank: expectNumber(record, "rank", "IdentifyCandidate"),
        random_id: expectString(record, "random_id", "IdentifyCandidate"),
        full_name: expectString(record, "full_name", "IdentifyCandidate"),
        national_id_masked: expectString(record, "national_id_masked", "IdentifyCandidate"),
        created_at: expectString(record, "created_at", "IdentifyCandidate"),
        capture: expectString(record, "capture", "IdentifyCandidate"),
        retrieval_score: expectNumber(record, "retrieval_score", "IdentifyCandidate"),
        rerank_score: maybeNumber(record, "rerank_score"),
        decision: maybeBoolean(record, "decision"),
    };
}

export function normalizeIdentifyResponse(payload: unknown): IdentifyResponse {
    const record = expectObject(payload, "IdentifyResponse");
    return {
        retrieval_method: normalizeRetrievalMethod(record.retrieval_method),
        rerank_method: normalizeMethod(record.rerank_method, "IdentifyResponse.rerank_method"),
        threshold: expectNumber(record, "threshold", "IdentifyResponse"),
        decision: expectBoolean(record, "decision", "IdentifyResponse"),
        total_enrolled: expectNumber(record, "total_enrolled", "IdentifyResponse"),
        candidate_pool_size: expectNumber(record, "candidate_pool_size", "IdentifyResponse"),
        shortlist_size: expectNumber(record, "shortlist_size", "IdentifyResponse"),
        hints_applied: normalizeStringRecord(record.hints_applied ?? {}, "IdentifyResponse.hints_applied"),
        top_candidate: record.top_candidate == null ? null : normalizeIdentifyCandidate(record.top_candidate),
        candidates: expectArray(record.candidates, "IdentifyResponse.candidates").map(normalizeIdentifyCandidate),
        latency_ms: normalizeNumberRecord(record.latency_ms ?? {}, "IdentifyResponse.latency_ms"),
        storage_layout: normalizeStringRecord(record.storage_layout ?? {}, "IdentifyResponse.storage_layout"),
    };
}

export function normalizeIdentificationStatsResponse(payload: unknown): IdentificationStatsResponse {
    const record = expectObject(payload, "IdentificationStatsResponse");
    return {
        total_enrolled: expectNumber(record, "total_enrolled", "IdentificationStatsResponse"),
        demo_seeded_count: expectNumber(record, "demo_seeded_count", "IdentificationStatsResponse"),
        browser_seeded_count: maybeNumber(record, "browser_seeded_count") ?? 0,
        storage_layout: normalizeStringRecord(record.storage_layout ?? {}, "IdentificationStatsResponse.storage_layout"),
    };
}

export function normalizeIdentificationHealthResponse(payload: unknown): IdentificationHealthResponse {
    const record = expectObject(payload, "IdentificationHealthResponse");
    const methods = expectObject(record.methods ?? {}, "IdentificationHealthResponse.methods");

    return {
        ok: expectBoolean(record, "ok", "IdentificationHealthResponse"),
        error: maybeString(record, "error"),
        status: expectString(record, "status", "IdentificationHealthResponse"),
        identify_ok: expectBoolean(record, "identify_ok", "IdentificationHealthResponse"),
        identify_error: maybeString(record, "identify_error"),
        identify_status: expectString(record, "identify_status", "IdentificationHealthResponse"),
        identify_browser_ok: expectBoolean(record, "identify_browser_ok", "IdentificationHealthResponse"),
        identify_browser_initialized: expectBoolean(record, "identify_browser_initialized", "IdentificationHealthResponse"),
        identify_browser_error: maybeString(record, "identify_browser_error"),
        identify_browser_status: expectString(record, "identify_browser_status", "IdentificationHealthResponse"),
        methods: Object.fromEntries(
            Object.entries(methods).map(([key, value]) => [
                key,
                normalizeIdentificationHealthMethodAvailability(
                    value,
                    `IdentificationHealthResponse.methods.${key}`,
                ),
            ]),
        ),
    };
}

export function normalizeIdentificationAdminLayoutResponse(
    payload: unknown,
): IdentificationAdminInspectionResponse {
    return normalizeIdentificationAdminInspectionResponse(payload);
}

export function normalizeIdentificationAdminReconciliationResponse(
    payload: unknown,
): IdentificationAdminReconciliationResponse {
    const record = expectObject(payload, "IdentificationAdminReconciliationResponse");
    return {
        generated_at: expectString(record, "generated_at", "IdentificationAdminReconciliationResponse"),
        report_mode: expectString(record, "report_mode", "IdentificationAdminReconciliationResponse"),
        requested_repairs: expectStringArray(
            record.requested_repairs ?? [],
            "IdentificationAdminReconciliationResponse.requested_repairs",
        ),
        available_repairs: expectStringArray(
            record.available_repairs ?? [],
            "IdentificationAdminReconciliationResponse.available_repairs",
        ),
        applied_repairs: expectArray(
            record.applied_repairs ?? [],
            "IdentificationAdminReconciliationResponse.applied_repairs",
        ).map((item, index) => normalizeJsonRecord(expectObject(
            item,
            `IdentificationAdminReconciliationResponse.applied_repairs[${index}]`,
        ))),
        summary: normalizeIdentificationAdminReconciliationSummary(record.summary),
        commands: normalizeStringRecord(record.commands ?? {}, "IdentificationAdminReconciliationResponse.commands"),
        inspection: normalizeIdentificationAdminInspectionResponse(
            record.inspection,
            "IdentificationAdminReconciliationResponse.inspection",
        ),
        issues: expectArray(record.issues ?? [], "IdentificationAdminReconciliationResponse.issues")
            .map(normalizeIdentificationAdminIssue),
        inspection_before_repairs: record.inspection_before_repairs == null
            ? null
            : normalizeIdentificationAdminInspectionResponse(
                record.inspection_before_repairs,
                "IdentificationAdminReconciliationResponse.inspection_before_repairs",
            ),
    };
}

export function normalizeDeleteIdentityResponse(payload: unknown): DeleteIdentityResponse {
    const record = expectObject(payload, "DeleteIdentityResponse");
    return {
        random_id: expectString(record, "random_id", "DeleteIdentityResponse"),
        removed: expectBoolean(record, "removed", "DeleteIdentityResponse"),
        storage_layout: normalizeStringRecord(record.storage_layout ?? {}, "DeleteIdentityResponse.storage_layout"),
    };
}

export function normalizeIdentifyDemoSeedResponse(payload: unknown): IdentifyDemoSeedResponse {
    const record = expectObject(payload, "IdentifyDemoSeedResponse");
    return {
        seeded_count: expectNumber(record, "seeded_count", "IdentifyDemoSeedResponse"),
        updated_count: expectNumber(record, "updated_count", "IdentifyDemoSeedResponse"),
        skipped_count: expectNumber(record, "skipped_count", "IdentifyDemoSeedResponse"),
        total_enrolled: expectNumber(record, "total_enrolled", "IdentifyDemoSeedResponse"),
        demo_seeded_count: expectNumber(record, "demo_seeded_count", "IdentifyDemoSeedResponse"),
        storage_layout: normalizeStringRecord(record.storage_layout ?? {}, "IdentifyDemoSeedResponse.storage_layout"),
        notice: maybeString(record, "notice"),
    };
}

export function normalizeIdentifyDemoResetResponse(payload: unknown): IdentifyDemoResetResponse {
    const record = expectObject(payload, "IdentifyDemoResetResponse");
    return {
        removed_count: expectNumber(record, "removed_count", "IdentifyDemoResetResponse"),
        total_enrolled: expectNumber(record, "total_enrolled", "IdentifyDemoResetResponse"),
        demo_seeded_count: expectNumber(record, "demo_seeded_count", "IdentifyDemoResetResponse"),
        storage_layout: normalizeStringRecord(record.storage_layout ?? {}, "IdentifyDemoResetResponse.storage_layout"),
        notice: maybeString(record, "notice"),
    };
}

export function normalizeIdentifyBrowserSeedSelectionResponse(payload: unknown): IdentifyBrowserSeedSelectionResponse {
    const record = expectObject(payload, "IdentifyBrowserSeedSelectionResponse");
    return {
        dataset: expectString(record, "dataset", "IdentifyBrowserSeedSelectionResponse"),
        selected_count: expectNumber(record, "selected_count", "IdentifyBrowserSeedSelectionResponse"),
        seeded_count: expectNumber(record, "seeded_count", "IdentifyBrowserSeedSelectionResponse"),
        updated_count: expectNumber(record, "updated_count", "IdentifyBrowserSeedSelectionResponse"),
        skipped_count: expectNumber(record, "skipped_count", "IdentifyBrowserSeedSelectionResponse"),
        total_enrolled: expectNumber(record, "total_enrolled", "IdentifyBrowserSeedSelectionResponse"),
        browser_seeded_count: expectNumber(record, "browser_seeded_count", "IdentifyBrowserSeedSelectionResponse"),
        store_ready: expectBoolean(record, "store_ready", "IdentifyBrowserSeedSelectionResponse"),
        seeded_identity_ids: expectStringArray(record.seeded_identity_ids ?? [], "IdentifyBrowserSeedSelectionResponse.seeded_identity_ids"),
        storage_layout: normalizeStringRecord(record.storage_layout ?? {}, "IdentifyBrowserSeedSelectionResponse.storage_layout"),
        warnings: expectStringArray(record.warnings ?? [], "IdentifyBrowserSeedSelectionResponse.warnings"),
        errors: expectStringArray(record.errors ?? [], "IdentifyBrowserSeedSelectionResponse.errors"),
        notice: maybeString(record, "notice"),
    };
}

export function normalizeIdentifyBrowserResetResponse(payload: unknown): IdentifyBrowserResetResponse {
    const record = expectObject(payload, "IdentifyBrowserResetResponse");
    return {
        removed_count: expectNumber(record, "removed_count", "IdentifyBrowserResetResponse"),
        total_enrolled: expectNumber(record, "total_enrolled", "IdentifyBrowserResetResponse"),
        browser_seeded_count: expectNumber(record, "browser_seeded_count", "IdentifyBrowserResetResponse"),
        storage_layout: normalizeStringRecord(record.storage_layout ?? {}, "IdentifyBrowserResetResponse.storage_layout"),
        notice: maybeString(record, "notice"),
    };
}
