export const METHOD_VALUES = ["classic_orb", "classic_gftt_orb", "harris", "sift", "dl", "vit", "dedicated"] as const;
export type Method = (typeof METHOD_VALUES)[number];

const METHOD_ALIAS_MAP = {
    classic: "classic_orb",
    classic_v2: "classic_gftt_orb",
} as const satisfies Record<string, Method>;

export function normalizeMethodValue(value: unknown): Method | null {
    if (typeof value !== "string") {
        return null;
    }

    const normalized = value.trim().toLowerCase();
    if (!normalized) {
        return null;
    }

    const canonical = METHOD_ALIAS_MAP[normalized as keyof typeof METHOD_ALIAS_MAP] ?? normalized;
    return METHOD_VALUES.includes(canonical as Method) ? (canonical as Method) : null;
}

export const CAPTURE_VALUES = ["plain", "roll", "contactless", "contact_based"] as const;
export type Capture = (typeof CAPTURE_VALUES)[number];

export const OVERLAY_MATCH_KIND_VALUES = ["tentative", "inlier", "outlier"] as const;
export type OverlayMatchKind = (typeof OVERLAY_MATCH_KIND_VALUES)[number];

export const BENCHMARK_RUN_KIND_VALUES = ["full", "smoke", "legacy"] as const;
export type BenchmarkRunKind = (typeof BENCHMARK_RUN_KIND_VALUES)[number];

export const BENCHMARK_VIEW_MODE_VALUES = ["canonical", "smoke", "archive"] as const;
export type BenchmarkViewMode = (typeof BENCHMARK_VIEW_MODE_VALUES)[number];

export const BENCHMARK_SORT_MODE_VALUES = ["best_accuracy", "lowest_eer", "lowest_latency"] as const;
export type BenchmarkSortMode = (typeof BENCHMARK_SORT_MODE_VALUES)[number];

export const BENCHMARK_BEST_METRIC_VALUES = ["best_auc", "best_eer", "best_latency"] as const;
export type BenchmarkBestMetric = (typeof BENCHMARK_BEST_METRIC_VALUES)[number];

export const IDENTIFICATION_RETRIEVAL_METHOD_VALUES = ["dl", "vit"] as const;
export type IdentificationRetrievalMethod = (typeof IDENTIFICATION_RETRIEVAL_METHOD_VALUES)[number];

export type JsonRecord = Record<string, unknown>;
export type StorageLayout = Record<string, string>;
export type LatencyBreakdown = Record<string, number>;

export type OverlayMatch = {
    a: [number, number];
    b: [number, number];
    kind: OverlayMatchKind;
    sim?: number | null;
};

export type Overlay = {
    matches: OverlayMatch[];
};

export type MatchMeta = JsonRecord & {
    dl_config?: JsonRecord;
    embed_ms_a?: number;
    embed_ms_b?: number;
    inliers?: number;
    matches?: number;
    k1?: number;
    k2?: number;
    tentative_count?: number;
    inliers_count?: number;
    stats?: Record<string, number>;
    latency_breakdown_ms?: LatencyBreakdown;
};

export type MatchResponse = {
    method: Method;
    score: number;
    decision: boolean;
    threshold: number;
    latency_ms: number;
    meta: MatchMeta;
    overlay: Overlay | null;
};

export type MatchRequest = {
    method: Method;
    fileA: File;
    fileB: File;
    captureA: Capture;
    captureB: Capture;
    returnOverlay: boolean;
    threshold?: number | string;
};

export type NamedInfo = {
    key: string;
    label: string;
    summary: string;
};

export type BenchmarkArtifactLink = {
    key: string;
    label: string;
    available: boolean;
    url?: string | null;
};

export type BenchmarkProvenance = {
    run: string;
    run_label: string;
    run_kind: BenchmarkRunKind;
    view_mode: BenchmarkViewMode;
    status: string;
    validation_state: string;
    source_type: string;
    artifact_source: string;
    methods_in_run: string[];
    benchmark_methods_in_run: string[];
    canonical_method?: string | null;
    benchmark_method?: string | null;
    method_label?: string | null;
    timestamp_utc?: string | null;
    limit?: number | null;
    pairs_path?: string | null;
    manifest_path?: string | null;
    data_dir?: string | null;
    git_commit?: string | null;
    available_artifacts: string[];
};

export type BenchmarkRow = {
    method: string;
    split: string;
    n_pairs: number;
    auc: number;
    eer: number;
    tar_at_far_1e_2?: number | null;
    tar_at_far_1e_3?: number | null;
    avg_ms_pair_reported?: number | null;
    avg_ms_pair_wall?: number | null;
};

export type BenchmarkRunInfo = {
    run: string;
    dataset?: string | null;
    run_kind: BenchmarkRunKind;
    view_mode: BenchmarkViewMode;
    status: string;
    validation_state: string;
    validated: boolean;
    recommended: boolean;
    run_label?: string | null;
    artifact_count: number;
    summary_note: string;
    methods: string[];
    benchmark_methods: string[];
    splits: string[];
    dataset_info?: NamedInfo | null;
};

export type BenchmarkRunsResponse = {
    default_run?: string | null;
    default_dataset?: string | null;
    default_split?: string | null;
    default_view_mode: BenchmarkViewMode;
    runs: BenchmarkRunInfo[];
};

export type BenchmarkSummaryResponse = {
    dataset: string;
    split: string;
    view_mode: BenchmarkViewMode;
    dataset_info?: NamedInfo | null;
    split_info?: NamedInfo | null;
    view_info?: NamedInfo | null;
    validation_state: string;
    selection_note: string;
    selection_policy: string;
    result_count: number;
    method_count: number;
    run_count: number;
    available_datasets: NamedInfo[];
    available_splits: NamedInfo[];
    available_view_modes: NamedInfo[];
    current_run_families: string[];
    artifact_note: string;
};

export type BenchmarkComparisonQuery = {
    dataset?: string;
    split?: string;
    view_mode?: BenchmarkViewMode;
    sort_mode?: BenchmarkSortMode;
};

export type BenchmarkSummaryQuery = {
    dataset?: string;
    split?: string;
    view_mode?: BenchmarkViewMode;
};

export type BenchmarkBestQuery = {
    dataset?: string;
    split?: string;
    view_mode?: BenchmarkViewMode;
};

export type ComparisonRow = {
    dataset: string;
    run: string;
    split: string;
    method: string;
    benchmark_method: string;
    method_label?: string | null;
    auc: number;
    eer: number;
    n_pairs?: number | null;
    tar_at_far_1e_2?: number | null;
    tar_at_far_1e_3?: number | null;
    latency_ms?: number | null;
    latency_source?: "reported" | "wall" | null;
    auc_rank?: number | null;
    eer_rank?: number | null;
    latency_rank?: number | null;
    run_family?: string | null;
    run_label?: string | null;
    run_kind: BenchmarkRunKind;
    view_mode: BenchmarkViewMode;
    status: string;
    validation_state: string;
    artifact_count: number;
    available_artifacts: string[];
    summary_text: string;
    artifacts: BenchmarkArtifactLink[];
    provenance?: BenchmarkProvenance | null;
};

export type ComparisonResponse = {
    rows: ComparisonRow[];
    datasets: string[];
    splits: string[];
    default_dataset?: string | null;
    default_split?: string | null;
    view_mode: BenchmarkViewMode;
    view_info?: NamedInfo | null;
    dataset_info: Record<string, NamedInfo>;
    split_info: Record<string, NamedInfo>;
};

export type BestMethodEntry = {
    dataset: string;
    split: string;
    metric: BenchmarkBestMetric;
    method: string;
    benchmark_method?: string | null;
    method_label?: string | null;
    run: string;
    value: number;
    run_family?: string | null;
    run_label?: string | null;
    view_mode: BenchmarkViewMode;
    status: string;
    validation_state: string;
};

export type BestMethodsResponse = {
    dataset?: string | null;
    split?: string | null;
    view_mode: BenchmarkViewMode;
    entries: BestMethodEntry[];
};

export type EvidenceSelectionDriver = "benchmark_driven" | "heuristic_fallback";
export type EvidenceStatus = "strong" | "fallback" | "degraded";

export type EvidenceQuality = {
    selection_driver: EvidenceSelectionDriver;
    benchmark_backed_selection: boolean;
    heuristic_fallback_used: boolean;
    benchmark_discovery_outcome: string;
    evidence_status: EvidenceStatus;
    evidence_note: string;
};

export type CatalogDatasetDemoHealthStatus = "healthy" | "degraded" | "incomplete";

export type CatalogDatasetDemoHealth = {
    planned_verify_cases: number;
    built_verify_cases: number;
    benchmark_backed_cases: number;
    heuristic_fallback_cases: number;
    missing_benchmark_evidence: boolean;
    status: CatalogDatasetDemoHealthStatus;
    note: string;
};

export type CatalogBuildStatus = "healthy" | "degraded" | "incomplete";

export type CatalogBuildHealth = {
    catalog_build_status: CatalogBuildStatus;
    total_verify_cases: number;
    benchmark_backed_case_count: number;
    heuristic_fallback_case_count: number;
    datasets_with_missing_benchmark_evidence: string[];
    summary_message: string;
};

export type CatalogDatasetSummary = {
    dataset: string;
    dataset_label: string;
    has_verify_cases: boolean;
    has_identify_gallery: boolean;
    has_browser_assets: boolean;
    verify_case_count: number;
    identify_identity_count: number;
    browser_item_count: number;
    browser_validation_status?: string | null;
    selection_policy?: string | null;
    available_features: string[];
    demo_health?: CatalogDatasetDemoHealth | null;
};

export type CatalogDatasetsResponse = {
    items: CatalogDatasetSummary[];
    catalog_build_health?: CatalogBuildHealth | null;
};

export type DemoCase = {
    id: string;
    title: string;
    description: string;
    difficulty: string;
    dataset: string;
    dataset_label: string;
    split: string;
    split_label: string;
    label: number;
    ground_truth: string;
    recommended_method: Method;
    benchmark_method?: string | null;
    benchmark_run?: string | null;
    curation_rule: string;
    benchmark_score?: number | null;
    capture_a: Capture;
    capture_b: Capture;
    image_a_url: string;
    image_b_url: string;
    asset_a_id?: string | null;
    asset_b_id?: string | null;
    case_type?: string | null;
    availability_status?: string | null;
    selection_policy?: string | null;
    tags: string[];
    modality_relation?: string | null;
    evidence_quality?: EvidenceQuality | null;
};

export type DemoCasesResponse = {
    cases: DemoCase[];
    catalog_build_health?: CatalogBuildHealth | null;
};

export type CatalogVerifyCase = {
    case_id: string;
    title: string;
    description: string;
    dataset: string;
    dataset_label: string;
    split: string;
    difficulty: string;
    case_type: string;
    ground_truth: string;
    recommended_method: Method;
    capture_a: Capture;
    capture_b: Capture;
    modality_relation?: string | null;
    tags: string[];
    selection_policy: string;
    selection_reason: string;
    image_a_url: string;
    image_b_url: string;
    availability_status: string;
    asset_a_id?: string | null;
    asset_b_id?: string | null;
    evidence_quality?: EvidenceQuality | null;
};

export type CatalogVerifyCasesResponse = {
    items: CatalogVerifyCase[];
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
    catalog_build_health?: CatalogBuildHealth | null;
};

export type CatalogIdentifyDemoIdentity = {
    id: string;
    dataset: string;
    dataset_label: string;
    display_label: string;
    capture?: string | null;
    thumbnail_url: string;
    preview_url: string;
    subject_id?: string | null;
    gallery_role?: string | null;
    tags: string[];
    recommended_enrollment_asset_id?: string | null;
    recommended_probe_asset_id?: string | null;
};

export type CatalogIdentifyExemplar = {
    asset_id: string;
    capture?: string | null;
    finger?: string | null;
    recommended_usage?: string | null;
    asset_reference: string;
    has_servable_asset: boolean;
    availability_status?: string | null;
};

export type CatalogIdentityItem = {
    identity_id: string;
    dataset: string;
    dataset_label: string;
    display_name: string;
    subject_id: string;
    gallery_role: string;
    tags: string[];
    is_demo_safe: boolean;
    enrollment_candidates: CatalogIdentifyExemplar[];
    probe_candidates: CatalogIdentifyExemplar[];
    preview_url?: string | null;
    thumbnail_url?: string | null;
    recommended_enrollment_asset_id?: string | null;
    recommended_probe_asset_id?: string | null;
    recommended_enrollment_capture?: string | null;
    recommended_probe_capture?: string | null;
};

export type CatalogIdentifyProbeCase = {
    id: string;
    title: string;
    description: string;
    dataset: string;
    dataset_label: string;
    capture?: string | null;
    difficulty: string;
    probe_thumbnail_url: string;
    probe_preview_url: string;
    probe_asset_url: string;
    expected_outcome?: string | null;
    expected_top_identity_id?: string | null;
    expected_top_identity_label?: string | null;
    recommended_retrieval_method?: IdentificationRetrievalMethod | null;
    recommended_rerank_method?: Method | null;
    recommended_shortlist_size?: number | null;
    scenario_type?: string | null;
    tags: string[];
};

export type CatalogIdentifyGalleryResponse = {
    items: CatalogIdentityItem[];
    demo_identities: CatalogIdentifyDemoIdentity[];
    probe_cases: CatalogIdentifyProbeCase[];
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
    total_probe_cases: number;
};

export type AssetDimensions = {
    width: number;
    height: number;
};

export type CatalogBrowserItem = {
    asset_id: string;
    dataset: string;
    split: string;
    subject_id?: string | null;
    finger?: string | null;
    capture?: string | null;
    modality?: string | null;
    ui_eligible: boolean;
    selection_reason: string;
    selection_policy: string;
    thumbnail_url: string;
    preview_url: string;
    availability_status: string;
    original_dimensions: AssetDimensions;
    thumbnail_dimensions: AssetDimensions;
    preview_dimensions: AssetDimensions;
};

export const CATALOG_BROWSER_SORT_VALUES = ["default", "split_subject_asset"] as const;
export type CatalogBrowserSort = (typeof CATALOG_BROWSER_SORT_VALUES)[number];

export type CatalogDatasetBrowserQuery = {
    dataset: string;
    split?: string;
    capture?: string;
    modality?: string;
    subject_id?: string;
    finger?: string;
    ui_eligible?: boolean;
    limit?: number;
    offset?: number;
    sort?: CatalogBrowserSort;
};

export type CatalogDatasetBrowserResponse = {
    dataset: string;
    dataset_label: string;
    selection_policy: string;
    validation_status: string;
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
    generated_at?: string | null;
    generator_version?: string | null;
    warning_count: number;
    summary: JsonRecord;
    items: CatalogBrowserItem[];
};

export type LoadedDemoCaseFiles = {
    fileA: File;
    fileB: File;
};

export type EnrollFingerprintRequest = {
    file: File;
    fullName: string;
    nationalId: string;
    capture: Capture;
    vectorMethods?: string[];
    replaceExisting?: boolean;
};

export type EnrollFingerprintResponse = {
    random_id: string;
    created_at: string;
    vector_methods: string[];
    image_sha256: string;
    storage_layout: StorageLayout;
};

export type IdentifyCandidate = {
    rank: number;
    random_id: string;
    full_name: string;
    national_id_masked: string;
    created_at: string;
    capture: string;
    retrieval_score: number;
    rerank_score?: number | null;
    decision?: boolean | null;
};

export type IdentifyFingerprintRequest = {
    file: File;
    capture: Capture;
    retrievalMethod?: IdentificationRetrievalMethod;
    rerankMethod?: Method;
    shortlistSize?: number;
    threshold?: number;
    namePattern?: string;
    nationalIdPattern?: string;
    createdFrom?: string;
    createdTo?: string;
    storeScope?: "operational" | "browser";
};

export type IdentifyResponse = {
    retrieval_method: IdentificationRetrievalMethod;
    rerank_method: Method;
    threshold: number;
    decision: boolean;
    total_enrolled: number;
    candidate_pool_size: number;
    shortlist_size: number;
    hints_applied: Record<string, string>;
    top_candidate?: IdentifyCandidate | null;
    candidates: IdentifyCandidate[];
    latency_ms: LatencyBreakdown;
    storage_layout: StorageLayout;
};

export type IdentificationStatsResponse = {
    total_enrolled: number;
    demo_seeded_count: number;
    browser_seeded_count: number;
    storage_layout: StorageLayout;
};

export type IdentificationHealthMethodAvailability = JsonRecord & {
    available?: boolean | null;
    error?: string | null;
};

export type IdentificationHealthResponse = {
    ok: boolean;
    error?: string | null;
    status: string;
    identify_ok: boolean;
    identify_error?: string | null;
    identify_status: string;
    identify_browser_ok: boolean;
    identify_browser_initialized: boolean;
    identify_browser_error?: string | null;
    identify_browser_status: string;
    methods: Record<string, IdentificationHealthMethodAvailability>;
};

export type IdentificationAdminDatabaseUrls = {
    biometric_db: string;
    identity_db: string;
};

export type IdentificationAdminResolvedTableNames = {
    person: string;
    identity: string;
    raw: string;
    vectors: string;
};

export type IdentificationAdminRoleTablePresence = {
    person: boolean;
    identity: boolean;
    raw: boolean;
    vectors: boolean;
};

export type IdentificationAdminTablePresence = {
    biometric_db: IdentificationAdminRoleTablePresence;
    identity_db: IdentificationAdminRoleTablePresence;
};

export type IdentificationAdminRowCounts = {
    people?: number | null;
    identity?: number | null;
    raw?: number | null;
    vectors_by_method: Record<string, number | null>;
};

export type IdentificationAdminIssue = JsonRecord & {
    code: string;
    severity: string;
    database_role: string;
    message: string;
};

export type IdentificationAdminReadiness = {
    ready: boolean;
    status: string;
    error_count: number;
    warning_count: number;
};

export type IdentificationAdminInspectionResponse = {
    backend: string;
    layout_version: string;
    dual_database_enabled: boolean;
    table_prefix: string;
    redacted_database_urls: IdentificationAdminDatabaseUrls;
    resolved_table_names: IdentificationAdminResolvedTableNames;
    table_presence: IdentificationAdminTablePresence;
    row_counts: IdentificationAdminRowCounts;
    vector_extension_present_in_biometric_db?: boolean | null;
    unexpected_vector_methods: Record<string, number>;
    schema_hardening: JsonRecord;
    reconciliation: JsonRecord;
    integrity_warnings: string[];
    overall_ok: boolean;
    readiness: IdentificationAdminReadiness;
    errors: IdentificationAdminIssue[];
    warnings: IdentificationAdminIssue[];
    issues: IdentificationAdminIssue[];
};

export type IdentificationAdminReconciliationSummary = {
    severity: Record<string, number>;
    repairability: Record<string, number>;
    manual_reconciliation_required: boolean;
    overall_ok: boolean;
    readiness: IdentificationAdminReadiness;
};

export type IdentificationAdminReconciliationResponse = {
    generated_at: string;
    report_mode: string;
    requested_repairs: string[];
    available_repairs: string[];
    applied_repairs: JsonRecord[];
    summary: IdentificationAdminReconciliationSummary;
    commands: Record<string, string>;
    inspection: IdentificationAdminInspectionResponse;
    issues: IdentificationAdminIssue[];
    inspection_before_repairs?: IdentificationAdminInspectionResponse | null;
};

export type DeleteIdentityResponse = {
    random_id: string;
    removed: boolean;
    storage_layout: StorageLayout;
};

export type IdentifyDemoSeedResponse = {
    seeded_count: number;
    updated_count: number;
    skipped_count: number;
    total_enrolled: number;
    demo_seeded_count: number;
    storage_layout: StorageLayout;
    notice?: string | null;
};

export type IdentifyDemoResetResponse = {
    removed_count: number;
    total_enrolled: number;
    demo_seeded_count: number;
    storage_layout: StorageLayout;
    notice?: string | null;
};

export type IdentifyBrowserSeedSelectionRequest = {
    dataset: string;
    selected_identity_ids: string[];
    overwrite?: boolean;
    metadata?: JsonRecord;
};

export type IdentifyBrowserSeedSelectionResponse = {
    dataset: string;
    selected_count: number;
    seeded_count: number;
    updated_count: number;
    skipped_count: number;
    total_enrolled: number;
    browser_seeded_count: number;
    store_ready: boolean;
    seeded_identity_ids: string[];
    storage_layout: StorageLayout;
    warnings: string[];
    errors: string[];
    notice?: string | null;
};

export type IdentifyBrowserResetResponse = {
    removed_count: number;
    total_enrolled: number;
    browser_seeded_count: number;
    storage_layout: StorageLayout;
    notice?: string | null;
};

export type ContractFlowStatus = "contract-ready" | "pending-ui-integration";

export type ContractMatrixEntry = {
    id: string;
    label: string;
    requestEndpoint: string;
    responseType: string;
    notes: string;
    status: ContractFlowStatus;
};
