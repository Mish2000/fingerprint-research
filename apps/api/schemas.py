from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class MatchMethod(str, Enum):
    classic_orb = "classic_orb"
    classic_gftt_orb = "classic_gftt_orb"
    harris = "harris"
    sift = "sift"
    dl = "dl"
    dedicated = "dedicated"
    vit = "vit"


class ResolvedMethodMetadata(BaseModel):
    canonical_method: str
    requested_method: str
    benchmark_method: str
    method_label: str
    family: str
    status: Optional[str] = None
    embedding_dim: Optional[int] = None
    aliases: List[str] = Field(default_factory=list)
    resolved_from_alias: bool = False


class MethodThresholds(BaseModel):
    decision: float


class MethodIdentificationRoles(BaseModel):
    retrieval_capable: bool = False
    rerank_capable: bool = False
    notes: List[str] = Field(default_factory=list)


class MethodAvailability(BaseModel):
    available: bool = False
    error: Optional[str] = None


class MethodCatalogEntry(BaseModel):
    id: str
    canonical_api_name: str
    label: str
    benchmark_name: str
    aliases: List[str] = Field(default_factory=list)
    family: str
    status: str
    thresholds: MethodThresholds
    runtime_defaults: Dict[str, Any] = Field(default_factory=dict)
    benchmark_defaults: Dict[str, Any] = Field(default_factory=dict)
    runtime_notes: List[str] = Field(default_factory=list)
    identification_roles: MethodIdentificationRoles
    availability: MethodAvailability


class MethodsCatalogResponse(BaseModel):
    methods: List[MethodCatalogEntry] = Field(default_factory=list)


class OverlayMatch(BaseModel):
    a: Tuple[float, float]
    b: Tuple[float, float]
    kind: str = Field(..., description="tentative|inlier|outlier")
    sim: Optional[float] = None


class Overlay(BaseModel):
    matches: List[OverlayMatch] = Field(default_factory=list)


class MatchResponse(BaseModel):
    method: MatchMethod
    score: float
    decision: bool
    threshold: float
    latency_ms: float
    meta: Dict[str, Any] = Field(default_factory=dict)
    overlay: Optional[Overlay] = None
    method_metadata: Optional[ResolvedMethodMetadata] = None


class NamedInfo(BaseModel):
    key: str
    label: str
    summary: str = ""


class BenchmarkArtifactLink(BaseModel):
    key: str
    label: str
    available: bool = False
    url: Optional[str] = None


class BenchmarkProvenance(BaseModel):
    run: str
    run_label: str
    run_kind: str
    view_mode: str
    status: str
    validation_state: str
    source_type: str = "summary_csv"
    artifact_source: str
    methods_in_run: List[str] = Field(default_factory=list)
    benchmark_methods_in_run: List[str] = Field(default_factory=list)
    canonical_method: Optional[str] = None
    benchmark_method: Optional[str] = None
    method_label: Optional[str] = None
    timestamp_utc: Optional[str] = None
    limit: Optional[int] = None
    pairs_path: Optional[str] = None
    manifest_path: Optional[str] = None
    data_dir: Optional[str] = None
    git_commit: Optional[str] = None
    available_artifacts: List[str] = Field(default_factory=list)


class BenchmarkRow(BaseModel):
    method: str
    split: str
    n_pairs: int
    auc: float
    eer: float
    tar_at_far_1e_2: Optional[float] = None
    tar_at_far_1e_3: Optional[float] = None
    avg_ms_pair_reported: Optional[float] = None
    avg_ms_pair_wall: Optional[float] = None


class BenchmarkRunInfo(BaseModel):
    run: str
    dataset: Optional[str] = None
    run_kind: str = "legacy"
    view_mode: str = "archive"
    status: str = "archived"
    validation_state: str = "archived"
    validated: bool = False
    recommended: bool = False
    run_label: Optional[str] = None
    artifact_count: int = 0
    summary_note: str = ""
    methods: List[str] = Field(default_factory=list)
    benchmark_methods: List[str] = Field(default_factory=list)
    splits: List[str] = Field(default_factory=list)
    dataset_info: Optional[NamedInfo] = None


class BenchmarkRunsResponse(BaseModel):
    default_run: Optional[str] = None
    default_dataset: Optional[str] = None
    default_split: Optional[str] = None
    default_view_mode: str = "canonical"
    runs: List[BenchmarkRunInfo] = Field(default_factory=list)


class BenchmarkSummaryResponse(BaseModel):
    dataset: str
    split: str = "all"
    view_mode: str = "canonical"
    dataset_info: Optional[NamedInfo] = None
    split_info: Optional[NamedInfo] = None
    view_info: Optional[NamedInfo] = None
    validation_state: str = "snapshot"
    selection_note: str = ""
    selection_policy: str = ""
    result_count: int = 0
    method_count: int = 0
    run_count: int = 0
    available_datasets: List[NamedInfo] = Field(default_factory=list)
    available_splits: List[NamedInfo] = Field(default_factory=list)
    available_view_modes: List[NamedInfo] = Field(default_factory=list)
    current_run_families: List[str] = Field(default_factory=list)
    artifact_note: str = ""


class ComparisonRow(BaseModel):
    dataset: str
    run: str
    split: str
    method: str
    benchmark_method: str
    method_label: Optional[str] = None
    auc: float
    eer: float
    n_pairs: Optional[int] = None
    tar_at_far_1e_2: Optional[float] = None
    tar_at_far_1e_3: Optional[float] = None
    latency_ms: Optional[float] = None
    latency_source: Optional[str] = None
    auc_rank: Optional[int] = None
    eer_rank: Optional[int] = None
    latency_rank: Optional[int] = None
    run_family: Optional[str] = None
    run_label: Optional[str] = None
    run_kind: str = "legacy"
    view_mode: str = "archive"
    status: str = "archived"
    validation_state: str = "archived"
    artifact_count: int = 0
    available_artifacts: List[str] = Field(default_factory=list)
    summary_text: str = ""
    artifacts: List[BenchmarkArtifactLink] = Field(default_factory=list)
    provenance: Optional[BenchmarkProvenance] = None


class ComparisonResponse(BaseModel):
    rows: List[ComparisonRow]
    datasets: List[str]
    splits: List[str]
    default_dataset: Optional[str] = None
    default_split: Optional[str] = None
    view_mode: str = "canonical"
    view_info: Optional[NamedInfo] = None
    dataset_info: Dict[str, NamedInfo] = Field(default_factory=dict)
    split_info: Dict[str, NamedInfo] = Field(default_factory=dict)


class BestMethodEntry(BaseModel):
    dataset: str
    split: str
    metric: str
    method: str
    benchmark_method: Optional[str] = None
    method_label: Optional[str] = None
    run: str
    value: float
    run_family: Optional[str] = None
    run_label: Optional[str] = None
    view_mode: str = "canonical"
    status: str = "validated"
    validation_state: str = "validated"


class BestMethodsResponse(BaseModel):
    dataset: Optional[str] = None
    split: Optional[str] = None
    view_mode: str = "canonical"
    entries: List[BestMethodEntry]


class EvidenceQualitySummary(BaseModel):
    selection_driver: str
    benchmark_backed_selection: bool
    heuristic_fallback_used: bool
    benchmark_discovery_outcome: str
    evidence_status: str
    evidence_note: str


class CatalogDatasetDemoHealth(BaseModel):
    planned_verify_cases: int
    built_verify_cases: int
    benchmark_backed_cases: int
    heuristic_fallback_cases: int
    missing_benchmark_evidence: bool
    status: str
    note: str


class CatalogBuildHealthSummary(BaseModel):
    catalog_build_status: str
    total_verify_cases: int
    benchmark_backed_case_count: int
    heuristic_fallback_case_count: int
    datasets_with_missing_benchmark_evidence: List[str] = Field(default_factory=list)
    summary_message: str


class DemoCase(BaseModel):
    id: str
    title: str
    description: str
    difficulty: str
    dataset: str
    dataset_label: str
    split: str
    split_label: str
    label: int
    ground_truth: str
    recommended_method: MatchMethod
    benchmark_method: Optional[str] = None
    benchmark_run: Optional[str] = None
    curation_rule: str
    benchmark_score: Optional[float] = None
    capture_a: str
    capture_b: str
    image_a_url: str
    image_b_url: str
    asset_a_id: Optional[str] = None
    asset_b_id: Optional[str] = None
    case_type: Optional[str] = None
    availability_status: Optional[str] = None
    selection_policy: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    modality_relation: Optional[str] = None
    evidence_quality: Optional[EvidenceQualitySummary] = None


class DemoCasesResponse(BaseModel):
    cases: List[DemoCase] = Field(default_factory=list)
    catalog_build_health: Optional[CatalogBuildHealthSummary] = None


class CatalogDatasetSummary(BaseModel):
    dataset: str
    dataset_label: str
    has_verify_cases: bool
    has_identify_gallery: bool
    has_browser_assets: bool
    verify_case_count: int
    identify_identity_count: int
    browser_item_count: int
    browser_validation_status: Optional[str] = None
    selection_policy: Optional[str] = None
    available_features: List[str] = Field(default_factory=list)
    demo_health: Optional[CatalogDatasetDemoHealth] = None


class CatalogDatasetsResponse(BaseModel):
    items: List[CatalogDatasetSummary] = Field(default_factory=list)
    catalog_build_health: Optional[CatalogBuildHealthSummary] = None


class CatalogVerifyCaseItem(BaseModel):
    case_id: str
    title: str
    description: str
    dataset: str
    dataset_label: str
    split: str
    difficulty: str
    case_type: str
    ground_truth: str
    recommended_method: MatchMethod
    capture_a: str
    capture_b: str
    modality_relation: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    selection_policy: str
    selection_reason: str
    image_a_url: str
    image_b_url: str
    availability_status: str
    asset_a_id: Optional[str] = None
    asset_b_id: Optional[str] = None
    evidence_quality: Optional[EvidenceQualitySummary] = None


class CatalogVerifyCaseDetail(CatalogVerifyCaseItem):
    benchmark_context: Dict[str, Any] = Field(default_factory=dict)
    traceability_summary: Dict[str, Any] = Field(default_factory=dict)
    additional_notes: Dict[str, Any] = Field(default_factory=dict)


class CatalogVerifyCasesResponse(BaseModel):
    items: List[CatalogVerifyCaseItem] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    has_more: bool
    catalog_build_health: Optional[CatalogBuildHealthSummary] = None


class CatalogIdentifyExemplar(BaseModel):
    asset_id: str
    capture: Optional[str] = None
    finger: Optional[str] = None
    recommended_usage: Optional[str] = None
    asset_reference: str
    has_servable_asset: bool
    availability_status: Optional[str] = None


class CatalogIdentityItem(BaseModel):
    identity_id: str
    dataset: str
    dataset_label: str
    display_name: str
    subject_id: str
    gallery_role: str
    tags: List[str] = Field(default_factory=list)
    is_demo_safe: bool
    enrollment_candidates: List[CatalogIdentifyExemplar] = Field(default_factory=list)
    probe_candidates: List[CatalogIdentifyExemplar] = Field(default_factory=list)
    preview_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    recommended_enrollment_asset_id: Optional[str] = None
    recommended_probe_asset_id: Optional[str] = None
    recommended_enrollment_capture: Optional[str] = None
    recommended_probe_capture: Optional[str] = None


class CatalogIdentifyDemoIdentity(BaseModel):
    id: str
    dataset: str
    dataset_label: str
    display_label: str
    capture: Optional[str] = None
    thumbnail_url: str
    preview_url: str
    subject_id: Optional[str] = None
    gallery_role: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    recommended_enrollment_asset_id: Optional[str] = None
    recommended_probe_asset_id: Optional[str] = None


class CatalogIdentifyProbeCase(BaseModel):
    id: str
    title: str
    description: str
    dataset: str
    dataset_label: str
    capture: Optional[str] = None
    difficulty: str
    probe_thumbnail_url: str
    probe_preview_url: str
    probe_asset_url: str
    expected_outcome: Optional[str] = None
    expected_top_identity_id: Optional[str] = None
    expected_top_identity_label: Optional[str] = None
    recommended_retrieval_method: Optional[str] = None
    recommended_rerank_method: Optional[MatchMethod] = None
    recommended_shortlist_size: Optional[int] = None
    scenario_type: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class CatalogIdentifyGalleryResponse(BaseModel):
    items: List[CatalogIdentityItem] = Field(default_factory=list)
    demo_identities: List[CatalogIdentifyDemoIdentity] = Field(default_factory=list)
    probe_cases: List[CatalogIdentifyProbeCase] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    has_more: bool
    total_probe_cases: int = 0


class AssetDimensions(BaseModel):
    width: int
    height: int


class CatalogBrowserItem(BaseModel):
    asset_id: str
    dataset: str
    split: str
    subject_id: Optional[str] = None
    finger: Optional[str] = None
    capture: Optional[str] = None
    modality: Optional[str] = None
    ui_eligible: bool
    selection_reason: str
    selection_policy: str
    thumbnail_url: str
    preview_url: str
    availability_status: str
    original_dimensions: AssetDimensions
    thumbnail_dimensions: AssetDimensions
    preview_dimensions: AssetDimensions


class CatalogDatasetBrowserResponse(BaseModel):
    dataset: str
    dataset_label: str
    selection_policy: str
    validation_status: str
    total: int
    limit: int
    offset: int
    has_more: bool
    generated_at: Optional[str] = None
    generator_version: Optional[str] = None
    warning_count: int = 0
    summary: Dict[str, Any] = Field(default_factory=dict)
    items: List[CatalogBrowserItem] = Field(default_factory=list)


# =========================================================
# 1:N Identification / enrollment (new step after Stage 3)
# =========================================================
class EnrollFingerprintResponse(BaseModel):
    random_id: str
    created_at: str
    vector_methods: List[str] = Field(default_factory=list)
    image_sha256: str
    storage_layout: Dict[str, str] = Field(default_factory=dict)


class IdentifyCandidate(BaseModel):
    rank: int
    random_id: str
    full_name: str
    national_id_masked: str
    created_at: str
    capture: str
    retrieval_score: float
    rerank_score: Optional[float] = None
    decision: Optional[bool] = None


class IdentifyResponse(BaseModel):
    retrieval_method: str
    rerank_method: MatchMethod
    threshold: float
    decision: bool
    total_enrolled: int
    candidate_pool_size: int
    shortlist_size: int
    hints_applied: Dict[str, str] = Field(default_factory=dict)
    top_candidate: Optional[IdentifyCandidate] = None
    candidates: List[IdentifyCandidate] = Field(default_factory=list)
    latency_ms: Dict[str, float] = Field(default_factory=dict)
    storage_layout: Dict[str, str] = Field(default_factory=dict)
    retrieval_method_metadata: Optional[ResolvedMethodMetadata] = None
    rerank_method_metadata: Optional[ResolvedMethodMetadata] = None


class IdentificationStatsResponse(BaseModel):
    total_enrolled: int
    demo_seeded_count: int = 0
    browser_seeded_count: int = 0
    storage_layout: Dict[str, str] = Field(default_factory=dict)


class IdentificationAdminDatabaseUrls(BaseModel):
    biometric_db: str
    identity_db: str


class IdentificationAdminResolvedTableNames(BaseModel):
    person: str
    identity: str
    raw: str
    vectors: str


class IdentificationAdminRoleTablePresence(BaseModel):
    person: bool = False
    identity: bool = False
    raw: bool = False
    vectors: bool = False


class IdentificationAdminTablePresence(BaseModel):
    biometric_db: IdentificationAdminRoleTablePresence
    identity_db: IdentificationAdminRoleTablePresence


class IdentificationAdminRowCounts(BaseModel):
    people: Optional[int] = None
    identity: Optional[int] = None
    raw: Optional[int] = None
    vectors_by_method: Dict[str, Optional[int]] = Field(default_factory=dict)


class IdentificationAdminIssue(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    severity: str
    database_role: str
    message: str


class IdentificationAdminReadiness(BaseModel):
    ready: bool = False
    status: str = "not_ready"
    error_count: int = 0
    warning_count: int = 0


class IdentificationAdminInspectionResponse(BaseModel):
    backend: str
    layout_version: str
    dual_database_enabled: bool
    table_prefix: str = ""
    redacted_database_urls: IdentificationAdminDatabaseUrls
    resolved_table_names: IdentificationAdminResolvedTableNames
    table_presence: IdentificationAdminTablePresence
    row_counts: IdentificationAdminRowCounts
    vector_extension_present_in_biometric_db: Optional[bool] = None
    unexpected_vector_methods: Dict[str, int] = Field(default_factory=dict)
    schema_hardening: Dict[str, Any] = Field(default_factory=dict)
    reconciliation: Dict[str, Any] = Field(default_factory=dict)
    integrity_warnings: List[str] = Field(default_factory=list)
    overall_ok: bool = False
    readiness: IdentificationAdminReadiness
    errors: List[IdentificationAdminIssue] = Field(default_factory=list)
    warnings: List[IdentificationAdminIssue] = Field(default_factory=list)
    issues: List[IdentificationAdminIssue] = Field(default_factory=list)


class IdentificationAdminReconciliationSummary(BaseModel):
    severity: Dict[str, int] = Field(default_factory=dict)
    repairability: Dict[str, int] = Field(default_factory=dict)
    manual_reconciliation_required: bool = False
    overall_ok: bool = False
    readiness: IdentificationAdminReadiness = Field(default_factory=IdentificationAdminReadiness)


class IdentificationAdminReconciliationResponse(BaseModel):
    generated_at: str
    report_mode: str
    requested_repairs: List[str] = Field(default_factory=list)
    available_repairs: List[str] = Field(default_factory=list)
    applied_repairs: List[Dict[str, Any]] = Field(default_factory=list)
    summary: IdentificationAdminReconciliationSummary
    commands: Dict[str, str] = Field(default_factory=dict)
    inspection: IdentificationAdminInspectionResponse
    issues: List[IdentificationAdminIssue] = Field(default_factory=list)
    inspection_before_repairs: Optional[IdentificationAdminInspectionResponse] = None


class DeleteIdentityResponse(BaseModel):
    random_id: str
    removed: bool
    storage_layout: Dict[str, str] = Field(default_factory=dict)


class IdentifyDemoSeedResponse(BaseModel):
    seeded_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    total_enrolled: int
    demo_seeded_count: int = 0
    storage_layout: Dict[str, str] = Field(default_factory=dict)
    notice: Optional[str] = None


class IdentifyDemoResetResponse(BaseModel):
    removed_count: int = 0
    total_enrolled: int
    demo_seeded_count: int = 0
    storage_layout: Dict[str, str] = Field(default_factory=dict)
    notice: Optional[str] = None


class IdentifyBrowserSeedSelectionRequest(BaseModel):
    dataset: str
    selected_identity_ids: List[str] = Field(default_factory=list)
    overwrite: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IdentifyBrowserSeedSelectionResponse(BaseModel):
    dataset: str
    selected_count: int = 0
    seeded_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    total_enrolled: int
    browser_seeded_count: int = 0
    store_ready: bool = False
    seeded_identity_ids: List[str] = Field(default_factory=list)
    storage_layout: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    notice: Optional[str] = None


class IdentifyBrowserResetResponse(BaseModel):
    removed_count: int = 0
    total_enrolled: int
    browser_seeded_count: int = 0
    storage_layout: Dict[str, str] = Field(default_factory=dict)
    notice: Optional[str] = None
