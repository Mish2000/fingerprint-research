import { useState } from "react";
import { ExternalLink } from "lucide-react";
import type { BenchmarkArtifactLink, ComparisonRow, NamedInfo, ResearchRunGroupInfo } from "../../types";
import { CompactEmptyState, MetricTile } from "../../shared/ui/presentation.tsx";
import {
    formatLatency,
    formatMethodLabel,
    formatMetric,
    formatOperatingPointActualFar,
    formatOperatingPointCounts,
    formatOperatingPointFrr,
    formatOperatingPointTarget,
    formatOperatingPointTar,
    formatOperatingPointThreshold,
    formatPercentFromFraction,
    formatApproxEqualEer,
    formatPairs,
    methodStatusBadges,
    operatingPointsForRow,
    researchRunSourceLabel,
    statusLabel,
    statusToneClassName,
} from "./benchmarkPresentation.ts";

type Props = {
    row: ComparisonRow | null;
    datasetInfo: Record<string, NamedInfo>;
    splitInfo: Record<string, NamedInfo>;
    researchGroupInfo?: ResearchRunGroupInfo | null;
    showResearchHistory?: boolean;
    onShowResearchHistory?: () => void;
};

const ARTIFACT_GROUPS = [
    { title: "Data", keys: ["summary_csv", "thresholds_csv", "threshold_sweep_csv", "tar_far_distribution_csv", "scores_csv", "latency_summary"] },
    { title: "Metrics", keys: ["positive_only_metrics", "negative_only_metrics"] },
    { title: "Metadata", keys: ["meta_json", "run_manifest", "failures_csv"] },
    { title: "Report", keys: ["final_markdown", "markdown_summary", "run_log"] },
] as const;

function artifactByKey(row: ComparisonRow, key: string) {
    return row.artifacts.find((item) => item.key === key) ?? null;
}

function basenameFromPath(value: string | null | undefined): string {
    const text = (value ?? "").trim();
    if (!text) {
        return "N/A";
    }

    const normalized = text.replace(/\\/g, "/");
    return normalized.split("/").filter(Boolean).pop() ?? text;
}

function artifactSourceLabel(row: ComparisonRow): string {
    if (row.provenance?.benchmark_source_label) {
        return row.provenance.benchmark_source_label;
    }
    if (row.provenance?.benchmark_source_root === "reference") {
        return "Reference artifacts";
    }
    if (row.provenance?.benchmark_source_root === "live") {
        return "Live artifacts";
    }
    return "Benchmark artifacts";
}

function TrustField({ label, value, title }: { label: string; value: string; title?: string }) {
    return (
        <div className="min-w-0 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] p-3">
            <p className="text-[11px] font-semibold uppercase text-[var(--app-text-muted)]">{label}</p>
            <p className="mt-1 safe-truncate text-sm font-medium text-[var(--app-text)]" title={title ?? value}>
                {value}
            </p>
        </div>
    );
}

function MethodStatusBadges({ row }: { row: ComparisonRow }) {
    const badges = methodStatusBadges(row);
    if (badges.length === 0) {
        return null;
    }
    return (
        <div className="flex flex-wrap gap-2">
            {badges.map((badge) => (
                <span
                    key={badge}
                    className={`status-pill ${
                        badge === "Not showcase eligible" ? "status-pill--warning" : "status-pill--info"
                    }`}
                >
                    {badge}
                </span>
            ))}
        </div>
    );
}

function ArtifactLink({ artifact }: { artifact: BenchmarkArtifactLink | null }) {
    if (!artifact) {
        return null;
    }

    if (artifact.available && artifact.url) {
        return (
            <a
                href={artifact.url}
                target="_blank"
                rel="noreferrer"
                className="safe-truncate rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] px-3 py-2 text-sm text-[var(--app-text-soft)] transition hover:border-[var(--app-brand-border)] hover:text-[var(--app-brand-text)]"
                title={artifact.label}
            >
                {artifact.label}
            </a>
        );
    }

    return (
        <div
            className="safe-truncate rounded-lg border border-[var(--app-border-muted)] bg-[var(--app-surface-muted)] px-3 py-2 text-sm text-[var(--app-text-muted)]"
            title={`${artifact.label} - unavailable`}
        >
            {artifact.label} unavailable
        </div>
    );
}

function ArtifactGroup({ title, row, keys }: { title: string; row: ComparisonRow; keys: readonly string[] }) {
    const artifacts = keys.map((key) => artifactByKey(row, key)).filter((item): item is BenchmarkArtifactLink => item != null);

    return (
        <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
            <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">{title}</p>
            <div className="mt-3 grid gap-2">
                {artifacts.length > 0 ? (
                    artifacts.map((artifact) => <ArtifactLink key={artifact.key} artifact={artifact} />)
                ) : (
                    <div className="rounded-lg border border-[var(--app-border-muted)] bg-[var(--app-surface-muted)] px-3 py-2 text-sm text-[var(--app-text-muted)]">
                        No artifacts listed.
                    </div>
                )}
            </div>
        </div>
    );
}

function RocPreview({ artifact, row }: { artifact: BenchmarkArtifactLink | null; row: ComparisonRow }) {
    const [failedArtifactKey, setFailedArtifactKey] = useState<string | null>(null);
    const artifactKey = JSON.stringify([artifact?.url ?? null, row.run, row.benchmark_method, row.split]);
    const failed = failedArtifactKey === artifactKey;
    const hasUrl = Boolean(artifact?.available && artifact.url);

    const openButton = artifact?.url ? (
        <a
            href={artifact.url}
            target="_blank"
            rel="noreferrer"
            className="app-button app-button--secondary"
        >
            <ExternalLink className="mr-2 h-4 w-4" />
            Open ROC artifact
        </a>
    ) : null;

    if (!hasUrl) {
        return (
            <div className="space-y-3">
                <div className="flex h-56 items-center justify-center rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] px-6 text-center text-sm text-[var(--app-text-muted)]">
                    ROC preview is not available for this row.
                </div>
                {openButton}
            </div>
        );
    }

    if (failed) {
        return (
            <div className="space-y-3">
                <div className="flex h-56 items-center justify-center rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] px-6 text-center text-sm text-[var(--app-text-muted)]">
                    ROC preview could not be rendered. Open artifact instead.
                </div>
                {openButton}
            </div>
        );
    }

    return (
        <div className="space-y-3">
            <div className="overflow-hidden rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)]">
                <img
                    src={artifact?.url ?? ""}
                    alt=""
                    aria-label={`${formatMethodLabel(row.method, row.method_label)} ROC preview`}
                    loading="lazy"
                    decoding="async"
                    onError={() => setFailedArtifactKey(artifactKey)}
                    className="h-56 w-full object-contain"
                />
            </div>
            {openButton}
        </div>
    );
}

function OperatingPointTile({ point }: { point: ReturnType<typeof operatingPointsForRow>[number] }) {
    const calibrationCounts =
        point.calibration_negatives != null || point.calibration_positives != null
            ? [
                point.calibration_false_accepts != null && point.calibration_negatives != null
                    ? `${point.calibration_false_accepts}/${point.calibration_negatives} FA`
                    : null,
                point.calibration_positives != null ? `${point.calibration_positives} positives` : null,
            ].filter((item): item is string => Boolean(item)).join("; ")
            : null;
    const calibrationDetail =
        point.calibration_far != null
            ? `VAL FAR ${formatPercentFromFraction(point.calibration_far)}${calibrationCounts ? ` (${calibrationCounts})` : ""}`
            : point.calibration_false_accepts != null && point.calibration_negatives != null
                ? `VAL FA ${point.calibration_false_accepts}/${point.calibration_negatives}${point.calibration_positives != null ? `; ${point.calibration_positives} positives` : ""}`
                : null;
    const details = [
        point.test_far != null ? `Actual FAR ${formatOperatingPointActualFar(point)}` : null,
        point.test_frr != null ? `FRR ${formatOperatingPointFrr(point)}` : null,
        point.threshold != null ? `Threshold ${formatOperatingPointThreshold(point)}` : null,
        calibrationDetail,
        formatOperatingPointCounts(point),
    ].filter((item): item is string => Boolean(item));

    return (
        <MetricTile
            label={`TAR @ ${formatOperatingPointTarget(point)}`}
            value={formatOperatingPointTar(point)}
            detail={details.join(" / ") || "Target operating point"}
        />
    );
}

function DistributionCell({ value, format = "metric" }: { value: number | null | undefined; format?: "metric" | "percent" | "count" }) {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return <td className="px-3 py-2 text-[var(--app-text-muted)]">N/A</td>;
    }

    const text = format === "percent"
        ? formatPercentFromFraction(value)
        : format === "count"
            ? value.toLocaleString()
            : formatMetric(value, 4);

    return <td className="whitespace-nowrap px-3 py-2">{text}</td>;
}

function ExpertTarFarDistribution({ row }: { row: ComparisonRow }) {
    const distribution = row.tar_far_distribution ?? [];
    if (distribution.length === 0) {
        return null;
    }

    return (
        <div className="mt-6 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-5">
            <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Expert TAR/FAR Distribution</p>
            <div className="mt-3 space-y-2 text-sm text-[var(--app-text-muted)]">
                <p>
                    This distribution is a threshold sweep. Calibrated operating points remain the official
                    VAL-to-TEST evidence above.
                </p>
                <p>
                    TA/FR are computed only from positive pairs. FA/TR are computed only from negative pairs.
                    FA means a negative pair was incorrectly accepted as a match. TR means a negative pair was
                    correctly rejected.
                </p>
            </div>
            <div className="mt-4 overflow-x-auto rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)]">
                <table className="min-w-full text-left text-sm text-[var(--app-text-soft)]">
                    <thead className="bg-[var(--app-surface-muted)] text-xs uppercase text-[var(--app-text-muted)]">
                        <tr>
                            <th scope="col" className="px-3 py-2 font-semibold">FAR ceiling</th>
                            <th scope="col" className="px-3 py-2 font-semibold">Threshold</th>
                            <th scope="col" className="px-3 py-2 font-semibold">Actual FAR</th>
                            <th scope="col" className="px-3 py-2 font-semibold">TAR</th>
                            <th scope="col" className="px-3 py-2 font-semibold">FRR</th>
                            <th scope="col" className="px-3 py-2 font-semibold">TNR</th>
                            <th scope="col" className="px-3 py-2 font-semibold">TA</th>
                            <th scope="col" className="px-3 py-2 font-semibold">FR</th>
                            <th scope="col" className="px-3 py-2 font-semibold">FA</th>
                            <th scope="col" className="px-3 py-2 font-semibold">TR</th>
                            <th scope="col" className="px-3 py-2 font-semibold">n positive</th>
                            <th scope="col" className="px-3 py-2 font-semibold">n negative</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--app-border-muted)]">
                        {distribution.map((point) => (
                            <tr key={`${point.far_ceiling}_${point.threshold ?? "na"}`}>
                                <DistributionCell value={point.far_ceiling} format="percent" />
                                <DistributionCell value={point.threshold} />
                                <DistributionCell value={point.actual_far} format="percent" />
                                <DistributionCell value={point.tar} format="percent" />
                                <DistributionCell value={point.frr} format="percent" />
                                <DistributionCell value={point.tnr} format="percent" />
                                <DistributionCell value={point.ta} format="count" />
                                <DistributionCell value={point.fr} format="count" />
                                <DistributionCell value={point.fa} format="count" />
                                <DistributionCell value={point.tr} format="count" />
                                <DistributionCell value={point.n_positive} format="count" />
                                <DistributionCell value={point.n_negative} format="count" />
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

export default function BenchmarkEvidencePanel({
    row,
    datasetInfo,
    splitInfo,
    researchGroupInfo = null,
    showResearchHistory = false,
    onShowResearchHistory,
}: Props) {
    const [provenanceOpen, setProvenanceOpen] = useState(false);

    if (!row) {
        return (
            <CompactEmptyState
                title="Selected method evidence"
                description="Select a comparison row to inspect run provenance, artifacts, and validation state."
                className="h-full"
            />
        );
    }

    const datasetLabel = datasetInfo[row.dataset]?.label ?? row.dataset;
    const splitLabel = splitInfo[row.split]?.label ?? row.split;
    const rocArtifact = artifactByKey(row, "roc_png");
    const runManifestArtifact = artifactByKey(row, "run_manifest");
    const provenance = row.provenance;
    const sourceLabel = artifactSourceLabel(row);
    const manifestDisplayPath = provenance?.manifest_path ?? runManifestArtifact?.url;
    const showcaseExclusionNote = row.showcase_exclusion_note ?? provenance?.showcase_exclusion_note ?? null;
    const operatingPoints = operatingPointsForRow(row);
    const baselineOnly = row.presentation_tier === "baseline"
        || (row.not_champion_candidate && row.showcase_eligible !== false && !row.research_track);

    return (
        <section className="surface-card p-6">
            <div className="flex min-w-0 items-start justify-between gap-4">
                <div className="min-w-0 space-y-2">
                    <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Evidence</p>
                    <h3 className="safe-text text-xl font-semibold text-[var(--app-text)]">{formatMethodLabel(row.method, row.method_label)}</h3>
                    <MethodStatusBadges row={row} />
                    <p className="safe-text text-sm text-[var(--app-text-muted)]">
                        {datasetLabel} - {splitLabel}
                    </p>
                </div>
                <span className={`status-pill ${statusToneClassName(row.status)}`}>
                    {statusLabel(row.status)}
                </span>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-2">
                <MetricTile label="AUC" value={formatMetric(row.auc)} tone="success" />
                <MetricTile label="EER" value={formatMetric(row.eer)} />
                <MetricTile label="FAR ~= FRR @ EER" value={formatApproxEqualEer(row.eer)} detail="EER operating point" tone="info" />
                {row.dpi ? <MetricTile label="DPI" value={`${row.dpi}`} detail="Acquisition resolution" tone="info" /> : null}
                {operatingPoints.map((point) => (
                    <OperatingPointTile key={`${point.target_far}_${point.label}`} point={point} />
                ))}
                <MetricTile label="Latency" value={formatLatency(row.latency_ms)} tone="warning" />
                <MetricTile label="Pairs" value={formatPairs(row.n_pairs)} />
            </div>

            <div className="mt-5 inline-banner inline-banner--info">
                <div className="inline-banner__body">
                    <p>
                        EER is the point where FAR and FRR are approximately equal. Calibrated operating points show
                        TEST TAR at calibrated target FARs, plus actual TEST FAR and FRR when the evidence exports them.
                    </p>
                    <p className="mt-1 text-sm">
                        Raw score thresholds and calibration false accepts are shown when final evidence preserves
                        them; older benchmark bundles may only expose TAR at fixed FAR targets.
                    </p>
                </div>
            </div>

            <ExpertTarFarDistribution row={row} />

            {showcaseExclusionNote ? (
                <div className="mt-5 inline-banner inline-banner--warning">
                    <div className="inline-banner__body">
                        <p className="font-semibold">{baselineOnly ? "Baseline evidence" : "Research-only method"}</p>
                        <p>{showcaseExclusionNote}</p>
                    </div>
                </div>
            ) : null}

            {researchGroupInfo && researchGroupInfo.totalCount > 1 ? (
                <div className="mt-5 inline-banner inline-banner--info">
                    <div className="inline-banner__body">
                        <p className="font-semibold">Representative research run</p>
                        <p>
                            {showResearchHistory
                                ? `${researchGroupInfo.totalCount} research runs are visible. Selected source: ${researchRunSourceLabel(row)}.`
                                : `Showing ${researchRunSourceLabel(row)}; ${researchGroupInfo.hiddenCount} archived research runs hidden.`}
                        </p>
                        {!showResearchHistory && researchGroupInfo.hiddenCount > 0 && onShowResearchHistory ? (
                            <button
                                type="button"
                                className="app-button app-button--secondary mt-3"
                                onClick={onShowResearchHistory}
                            >
                                Show archived research runs
                            </button>
                        ) : null}
                    </div>
                </div>
            ) : null}

            <div className="mt-6 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-5">
                <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Trust & provenance</p>
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                    <TrustField label="Source" value={sourceLabel} />
                    <TrustField label="Run family" value={row.run_family ?? row.run} />
                    <TrustField label="Artifact source" value={provenance?.artifact_source ?? "results_summary.csv"} />
                    {row.dpi ? <TrustField label="DPI" value={`${row.dpi}`} /> : null}
                    <TrustField label="Pairs source" value={basenameFromPath(provenance?.pairs_path)} title={provenance?.pairs_path ?? undefined} />
                    <TrustField label="Manifest" value={basenameFromPath(manifestDisplayPath)} title={manifestDisplayPath ?? undefined} />
                    <TrustField label="Validation" value={provenance?.validation_state ?? row.validation_state} />
                </div>
            </div>

            <div className="mt-6 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-5">
                <div className="flex items-center justify-between gap-4">
                    <div>
                        <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Visual evidence</p>
                        <p className="mt-1 text-sm text-[var(--app-text-muted)]">ROC Preview</p>
                    </div>
                </div>
                <div className="mt-4">
                    <RocPreview artifact={rocArtifact} row={row} />
                </div>
            </div>

            <div className="mt-6 space-y-4">
                <div>
                    <p className="text-xs font-semibold uppercase text-[var(--app-text-muted)]">Artifacts</p>
                    <p className="mt-1 text-sm text-[var(--app-text-muted)]">
                        {row.artifact_count} available links for this method row.
                    </p>
                </div>
                <div className="grid gap-4">
                    {ARTIFACT_GROUPS.map((group) => (
                        <ArtifactGroup key={group.title} title={group.title} row={row} keys={group.keys} />
                    ))}
                </div>
            </div>

            <div className="mt-6 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-5">
                <button
                    type="button"
                    onClick={() => setProvenanceOpen((current) => !current)}
                    className="text-sm font-semibold text-[var(--app-brand-text)] transition hover:opacity-80"
                >
                    {provenanceOpen ? "Hide provenance details" : "Open provenance details"}
                </button>
                {provenanceOpen ? (
                    <div className="mt-4 space-y-3 text-sm text-[var(--app-text-soft)]">
                        <TrustField label="API method" value={formatMethodLabel(provenance?.canonical_method ?? row.method, provenance?.method_label ?? row.method_label)} />
                        <TrustField label="Benchmark method" value={provenance?.benchmark_method ?? row.benchmark_method} />
                        <TrustField label="Run" value={provenance?.run ?? row.run} />
                        <TrustField label="Available artifacts" value={(provenance?.available_artifacts ?? row.available_artifacts).join(", ") || "N/A"} />
                        <TrustField label="Methods in run" value={(provenance?.methods_in_run ?? []).map((method) => formatMethodLabel(method)).join(", ") || "N/A"} />
                        <TrustField label="Raw methods in run" value={(provenance?.benchmark_methods_in_run ?? []).join(", ") || "N/A"} />
                        <TrustField label="Pairs path" value={provenance?.pairs_path ?? "N/A"} />
                        <TrustField label="Manifest path" value={provenance?.manifest_path ?? "N/A"} />
                        <TrustField label="Data directory" value={provenance?.data_dir ?? "N/A"} />
                        {showcaseExclusionNote ? <TrustField label="Showcase note" value={showcaseExclusionNote} /> : null}
                        {provenance?.git_commit ? <TrustField label="Git commit" value={provenance.git_commit} /> : null}
                    </div>
                ) : null}
            </div>
        </section>
    );
}
