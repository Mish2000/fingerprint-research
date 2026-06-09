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
    nist_sd300c: {
        key: "nist_sd300c",
        label: "NIST SD300c",
        summary: "Rolled versus plain benchmark at 2000 ppi.",
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

const viewInfos = {
    canonical: {
        key: "canonical",
        label: "Canonical",
        summary: "Validated showcase runs.",
    },
    smoke: {
        key: "smoke",
        label: "Smoke",
        summary: "Smoke regression anchors.",
    },
    archive: {
        key: "archive",
        label: "Archive",
        summary: "Archived benchmark rows.",
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

function finalBundleArtifacts(run: string, dataset: string, method: string) {
    const prefix = `/api/benchmark/artifacts/${run}/plain_roll_final_baselines_v1`;
    return [
        {
            key: "summary_csv",
            label: "Summary CSV",
            available: true,
            url: `${prefix}/plain_roll_final_metrics.csv`,
        },
        {
            key: "thresholds_csv",
            label: "Thresholds CSV",
            available: true,
            url: `${prefix}/plain_roll_final_thresholds.csv`,
        },
        {
            key: "threshold_sweep_csv",
            label: "Threshold Sweep CSV",
            available: true,
            url: `${prefix}/plain_roll_final_threshold_sweep.csv`,
        },
        {
            key: "tar_far_distribution_csv",
            label: "TAR/FAR Distribution CSV",
            available: true,
            url: `${prefix}/plain_roll_final_tar_far_distribution.csv`,
        },
        {
            key: "scores_csv",
            label: "Scores CSV",
            available: true,
            url: `${prefix}/scores_${dataset}_${method}_test.csv`,
        },
        {
            key: "meta_json",
            label: "Meta JSON",
            available: true,
            url: `${prefix}/run_${dataset}_${method}_test.meta.json`,
        },
        {
            key: "roc_png",
            label: "ROC Preview",
            available: true,
            url: `${prefix}/roc_${dataset}_${method}_test.png`,
        },
        {
            key: "markdown_summary",
            label: "Markdown Summary",
            available: true,
            url: `${prefix}/plain_roll_final_summary.md`,
        },
        {
            key: "final_markdown",
            label: "Final Markdown Evidence",
            available: true,
            url: `${prefix}/final_markdown/${dataset}_${method}_plain_roll_final.md`,
        },
        {
            key: "run_manifest",
            label: "Run Manifest",
            available: true,
            url: `${prefix}/plain_roll_final_manifest.json`,
        },
        {
            key: "latency_summary",
            label: "Latency Summary",
            available: true,
            url: `${prefix}/plain_roll_final_latency_summary.csv`,
        },
        {
            key: "positive_only_metrics",
            label: "Positive-only Metrics",
            available: true,
            url: `${prefix}/plain_roll_final_positive_only_metrics.csv`,
        },
        {
            key: "negative_only_metrics",
            label: "Negative-only Metrics",
            available: true,
            url: `${prefix}/plain_roll_final_negative_only_metrics.csv`,
        },
        {
            key: "failures_csv",
            label: "Failures CSV",
            available: true,
            url: `${prefix}/plain_roll_final_failures.csv`,
        },
    ];
}

function sourceAfisFinalBundleArtifacts(run: string, dataset: string) {
    const prefix = `/api/benchmark/artifacts/${run}/plain_roll_final_sourceafis_v1`;
    return [
        {
            key: "summary_csv",
            label: "Summary CSV",
            available: true,
            url: `${prefix}/plain_roll_final_metrics.csv`,
        },
        {
            key: "thresholds_csv",
            label: "Thresholds CSV",
            available: true,
            url: `${prefix}/plain_roll_final_thresholds.csv`,
        },
        {
            key: "threshold_sweep_csv",
            label: "Threshold Sweep CSV",
            available: true,
            url: `${prefix}/plain_roll_final_threshold_sweep.csv`,
        },
        {
            key: "tar_far_distribution_csv",
            label: "TAR/FAR Distribution CSV",
            available: true,
            url: `${prefix}/plain_roll_final_tar_far_distribution.csv`,
        },
        {
            key: "scores_csv",
            label: "Scores CSV",
            available: true,
            url: `${prefix}/scores/scores_${dataset}_sourceafis_open_test.csv`,
        },
        {
            key: "meta_json",
            label: "Meta JSON",
            available: true,
            url: `${prefix}/run_meta/run_${dataset}_sourceafis_open_test.meta.json`,
        },
        {
            key: "roc_png",
            label: "ROC Preview",
            available: false,
            url: null,
        },
        {
            key: "markdown_summary",
            label: "Markdown Summary",
            available: true,
            url: `${prefix}/plain_roll_final_summary.md`,
        },
        {
            key: "final_markdown",
            label: "Final Markdown Evidence",
            available: true,
            url: `${prefix}/final_markdown/${dataset}_sourceafis_open_plain_roll_final.md`,
        },
        {
            key: "run_manifest",
            label: "Run Manifest",
            available: true,
            url: `${prefix}/plain_roll_final_manifest.json`,
        },
        {
            key: "latency_summary",
            label: "Latency Summary",
            available: true,
            url: `${prefix}/plain_roll_final_latency_summary.csv`,
        },
        {
            key: "positive_only_metrics",
            label: "Positive-only Metrics",
            available: true,
            url: `${prefix}/plain_roll_final_positive_only_metrics.csv`,
        },
        {
            key: "negative_only_metrics",
            label: "Negative-only Metrics",
            available: true,
            url: `${prefix}/plain_roll_final_negative_only_metrics.csv`,
        },
        {
            key: "failures_csv",
            label: "Failures CSV",
            available: true,
            url: `${prefix}/plain_roll_final_failures.csv`,
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
    tarAtFar1e2 = null,
    tarAtFar1e3 = null,
    aucRank,
    eerRank,
    latencyRank,
    artifacts,
    viewMode = "canonical",
    status = "validated",
    validationState = "validated",
}: {
    dataset: "nist_sd300b" | "nist_sd300c" | "polyu_cross";
    split: "val" | "test";
    method: string;
    benchmarkMethod?: string;
    run: string;
    runLabel: string;
    auc: number;
    eer: number;
    latency: number;
    nPairs: number;
    tarAtFar1e2?: number | null;
    tarAtFar1e3?: number | null;
    aucRank: number;
    eerRank: number;
    latencyRank: number;
    artifacts?: ReturnType<typeof availableArtifacts>;
    viewMode?: "canonical" | "smoke" | "archive";
    status?: "validated" | "smoke" | "archived" | "partial";
    validationState?: "validated" | "snapshot" | "archived" | "partial";
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
        tar_at_far_1e_2: tarAtFar1e2,
        tar_at_far_1e_3: tarAtFar1e3,
        latency_ms: latency,
        latency_source: "wall",
        auc_rank: aucRank,
        eer_rank: eerRank,
        latency_rank: latencyRank,
        run_family: run,
        run_label: runLabel,
        run_kind: "full",
        view_mode: viewMode,
        status,
        validation_state: validationState,
        artifact_count: available.length,
        available_artifacts: available,
        summary_text: `${runLabel} on ${split} with ${nPairs} pairs.`,
        artifacts: resolvedArtifacts,
        provenance: {
            run,
            run_label: runLabel,
            run_kind: "full",
            view_mode: viewMode,
            status,
            validation_state: validationState,
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
            benchmark_source_root: viewMode === "canonical" ? "reference" : "live",
            benchmark_source_label: viewMode === "canonical" ? "Reference artifacts" : "Live artifacts",
        },
    };
}

function finalClassicalRow({
    dataset,
    method,
    benchmarkMethod,
    runLabel,
    auc,
    eer,
    latency,
    operatingPoints,
    note,
}: {
    dataset: "nist_sd300b" | "nist_sd300c";
    method: string;
    benchmarkMethod?: string;
    runLabel: string;
    auc: number;
    eer: number;
    latency: number;
    operatingPoints: ReturnType<typeof createOperatingPoints>;
    note: string;
}) {
    const rawBenchmarkMethod = benchmarkMethod ?? method;
    const run = `plain_roll_final_baselines_v1_${dataset}_final`;
    const base = createRow({
        dataset,
        split: "test",
        method,
        benchmarkMethod: rawBenchmarkMethod,
        run,
        runLabel,
        auc,
        eer,
        latency,
        nPairs: 1400,
        tarAtFar1e2: operatingPoints[0]?.test_tar ?? null,
        aucRank: 9,
        eerRank: 9,
        latencyRank: 9,
        artifacts: finalBundleArtifacts(run, dataset, rawBenchmarkMethod),
    });
    const isSiftV2 = rawBenchmarkMethod === "sift_plain_roll_v2";
    const finalMarkdown = `final_markdown/${dataset}_${rawBenchmarkMethod}_plain_roll_final.md`;
    return {
        ...base,
        method_status: isSiftV2 ? "experimental" : "active",
        presentation_tier: isSiftV2 ? "research" : "baseline",
        showcase_eligible: !isSiftV2,
        research_track: isSiftV2,
        not_champion_candidate: true,
        showcase_exclusion_note: note,
        operating_points: operatingPoints,
        tar_far_distribution: createTarFarDistribution(operatingPoints),
        latency_ms: latency,
        latency_source: "reported",
        auc_rank: null,
        eer_rank: null,
        latency_rank: null,
        run_family: "plain_roll_final_baselines_v1",
        summary_text: `${note} TEST pair audit passed with 700 positive / 700 negative pairs. VAL calibration used 700 positive / 700 negative pairs. 0 recorded failures. Positive-only and negative-only metrics are available as separate final artifacts.`,
        provenance: {
            ...base.provenance,
            source_type: "plain_roll_final",
            artifact_source: finalMarkdown,
            methods_in_run: ["sift_plain_roll_v2", "sift", "minutiae", "harris", "classic_gftt_orb"],
            benchmark_methods_in_run: ["sift_plain_roll_v2", "sift", "minutiae", "harris", "classic_v2"],
            method_status: isSiftV2 ? "experimental" : "active",
            presentation_tier: isSiftV2 ? "research" : "baseline",
            showcase_eligible: !isSiftV2,
            research_track: isSiftV2,
            not_champion_candidate: true,
            run_family: "plain_roll_final_baselines_v1",
            timestamp_utc: "2026-06-03T07:39:15Z",
            manifest_path: "plain_roll_final_manifest.json",
            data_dir: "artifacts/reports/benchmark/plain_roll_final_baselines_v1",
            git_commit: "6d2d9fc3f28f167e6c3a24d162bed20de56457a7",
            available_artifacts: base.available_artifacts,
            benchmark_source_root: "live",
            benchmark_source_label: "Final curated evidence",
            showcase_exclusion_note: note,
        },
    };
}

function finalSourceAfisRow({
    dataset,
    dpi,
    operatingPoints,
    auc,
    eer,
    latency,
}: {
    dataset: "nist_sd300b" | "nist_sd300c";
    dpi: number;
    operatingPoints: ReturnType<typeof createOperatingPoints>;
    auc: number;
    eer: number;
    latency: number;
}) {
    const run = `plain_roll_final_sourceafis_v1_${dataset}_final`;
    const runLabel = `Final SourceAFIS plain-vs-roll evidence (${dataset})`;
    const base = createRow({
        dataset,
        split: "test",
        method: "sourceafis_open",
        run,
        runLabel,
        auc,
        eer,
        latency,
        nPairs: 1400,
        tarAtFar1e2: operatingPoints[0]?.test_tar ?? null,
        aucRank: 1,
        eerRank: 1,
        latencyRank: 1,
        artifacts: sourceAfisFinalBundleArtifacts(run, dataset),
    });
    const finalMarkdown = `final_markdown/${dataset}_sourceafis_open_plain_roll_final.md`;

    return {
        ...base,
        dpi,
        method_status: "optional_external",
        presentation_tier: "production_candidate",
        showcase_eligible: true,
        research_track: false,
        not_champion_candidate: false,
        operating_points: operatingPoints,
        tar_far_distribution: createTarFarDistribution(operatingPoints),
        latency_source: "reported",
        run_family: "plain_roll_final_sourceafis_v1",
        summary_text: "SourceAFIS champion evidence under the same audited plain-vs-roll selected pairs. 700 positive / 700 negative TEST pairs. VAL calibration used 700 positive / 700 negative pairs. 0 recorded failures. Expert TAR/FAR distribution plus positive-only and negative-only metrics are available as final artifacts.",
        provenance: {
            ...base.provenance,
            source_type: "plain_roll_final_sourceafis",
            artifact_source: finalMarkdown,
            methods_in_run: ["sourceafis_open"],
            benchmark_methods_in_run: ["sourceafis_open"],
            showcase_methods_in_run: ["sourceafis_open"],
            showcase_benchmark_methods_in_run: ["sourceafis_open"],
            research_methods_in_run: [],
            research_benchmark_methods_in_run: [],
            method_status: "optional_external",
            presentation_tier: "production_candidate",
            showcase_eligible: true,
            research_track: false,
            not_champion_candidate: false,
            run_family: "plain_roll_final_sourceafis_v1",
            timestamp_utc: "2026-06-04T07:39:15Z",
            pairs_path: `artifacts/reports/benchmark/plain_roll_final_sourceafis_v1/selected_pairs/pairs_${dataset}_test.csv`,
            manifest_path: "plain_roll_final_manifest.json",
            data_dir: "artifacts/reports/benchmark/plain_roll_final_sourceafis_v1",
            git_commit: "sourceafis-final",
            available_artifacts: base.available_artifacts,
            benchmark_source_root: "live",
            benchmark_source_label: "Final curated evidence",
        },
    };
}

type OperatingPointSeed = {
    threshold?: number;
    tar: number;
    far: number;
    frr: number;
    ta: number;
    fr: number;
    fa: number;
    tr: number;
    calibrationFar?: number;
    calibrationFalseAccepts?: number;
};

function createOperatingPoints(
    onePct: OperatingPointSeed,
    halfPct?: Partial<OperatingPointSeed>,
) {
    const second = {
        threshold: onePct.threshold,
        tar: Math.max(0, onePct.tar * 0.8),
        far: Math.min(onePct.far, 0.005),
        frr: Math.min(1, onePct.frr + 0.08),
        ta: Math.floor(onePct.ta * 0.8),
        fr: 700 - Math.floor(onePct.ta * 0.8),
        fa: Math.min(onePct.fa, 3),
        tr: 700 - Math.min(onePct.fa, 3),
        calibrationFar: Math.min(onePct.calibrationFar ?? 0.01, 0.0043),
        calibrationFalseAccepts: Math.min(onePct.calibrationFalseAccepts ?? 7, 3),
        ...halfPct,
    };
    return [
        {
            target_far: 0.01,
            label: "1.00% FAR",
            threshold: onePct.threshold,
            test_tar: onePct.tar,
            test_far: onePct.far,
            test_frr: onePct.frr,
            ta: onePct.ta,
            fr: onePct.fr,
            fa: onePct.fa,
            tr: onePct.tr,
            calibration_far: onePct.calibrationFar ?? 0.01,
            calibration_false_accepts: onePct.calibrationFalseAccepts ?? 7,
            calibration_negatives: 700,
            calibration_positives: 700,
        },
        {
            target_far: 0.005,
            label: "0.50% FAR",
            threshold: second.threshold,
            test_tar: second.tar,
            test_far: second.far,
            test_frr: second.frr,
            ta: second.ta,
            fr: second.fr,
            fa: second.fa,
            tr: second.tr,
            calibration_far: second.calibrationFar,
            calibration_false_accepts: second.calibrationFalseAccepts,
            calibration_negatives: 700,
            calibration_positives: 700,
        },
    ];
}

function createTarFarDistribution(points: ReturnType<typeof createOperatingPoints>) {
    return points.map((point) => {
        const nNegative = point.calibration_negatives ?? 700;
        const ceiling = point.target_far;
        const maxFalseAccepts = Math.floor(ceiling * nNegative);
        const falseAccepts = Math.min(point.fa ?? maxFalseAccepts, maxFalseAccepts);
        const actualFar = Math.min(point.test_far ?? ceiling, ceiling);

        return {
            far_ceiling: ceiling,
            threshold: point.threshold,
            actual_far: actualFar,
            tar: point.test_tar,
            frr: point.test_frr,
            tnr: 1 - actualFar,
            ta: point.ta,
            fr: point.fr,
            fa: falseAccepts,
            tr: nNegative - falseAccepts,
            n_positive: point.calibration_positives ?? 700,
            n_negative: nNegative,
        };
    }).sort((a, b) => a.far_ceiling - b.far_ceiling);
}

const sourceAfisBOperatingPoints = [
    {
        target_far: 0.01,
        label: "1.00% FAR",
        threshold: 14.72326764987426,
        test_tar: 0.7729,
        test_far: 0.0086,
        test_frr: 0.2271,
        ta: 541,
        fr: 159,
        fa: 6,
        tr: 694,
        calibration_far: 0.01,
        calibration_false_accepts: 7,
        calibration_negatives: 700,
    },
    {
        target_far: 0.005,
        label: "0.50% FAR",
        threshold: 17.393218350729448,
        test_tar: 0.76,
        test_far: 0.0043,
        test_frr: 0.24,
        ta: 532,
        fr: 168,
        fa: 3,
        tr: 697,
        calibration_far: 0.0043,
        calibration_false_accepts: 3,
        calibration_negatives: 700,
    },
];

const sourceAfisCOperatingPoints = [
    {
        target_far: 0.01,
        label: "1.00% FAR",
        threshold: 14.483463789540309,
        test_tar: 0.78,
        test_far: 0.0129,
        test_frr: 0.22,
        ta: 546,
        fr: 154,
        fa: 9,
        tr: 691,
        calibration_far: 0.01,
        calibration_false_accepts: 7,
        calibration_negatives: 700,
    },
    {
        target_far: 0.005,
        label: "0.50% FAR",
        threshold: 20.06041975470194,
        test_tar: 0.7529,
        test_far: 0.0057,
        test_frr: 0.2471,
        ta: 527,
        fr: 173,
        fa: 4,
        tr: 696,
        calibration_far: 0.0043,
        calibration_false_accepts: 3,
        calibration_negatives: 700,
    },
];

const siftV2BOperatingPoints = createOperatingPoints(
    { threshold: 6.6542, tar: 0.5, far: 0.0157, frr: 0.5, ta: 350, fr: 350, fa: 11, tr: 689 },
    { threshold: 8.1246, tar: 0.4318, far: 0.0033, frr: 0.5682, ta: 302, fr: 398, fa: 3, tr: 697 },
);

const siftBOperatingPoints = createOperatingPoints(
    { threshold: 0.0087, tar: 0.28, far: 0.0071, frr: 0.72, ta: 196, fr: 504, fa: 5, tr: 695 },
    { threshold: 0.0107, tar: 0.2271, far: 0.0057, frr: 0.7729, ta: 159, fr: 541, fa: 4, tr: 696 },
);

const minutiaeBOperatingPoints = createOperatingPoints(
    { threshold: 0.2432, tar: 0.0086, far: 0, frr: 0.9914, ta: 6, fr: 694, fa: 0, tr: 700 },
);

const harrisBOperatingPoints = createOperatingPoints(
    { threshold: 0.0104, tar: 0.0057, far: 0.0029, frr: 0.9943, ta: 4, fr: 696, fa: 2, tr: 698 },
);

const classicBOperatingPoints = createOperatingPoints(
    { threshold: 0.0067, tar: 0.0043, far: 0.0029, frr: 0.9957, ta: 3, fr: 697, fa: 2, tr: 698, calibrationFar: 0.0057, calibrationFalseAccepts: 4 },
);

const siftV2COperatingPoints = createOperatingPoints(
    { threshold: 7.8922, tar: 0.4314, far: 0.0043, frr: 0.5686, ta: 302, fr: 398, fa: 3, tr: 697 },
    { threshold: 10.2580, tar: 0.38, far: 0.0029, frr: 0.62, ta: 266, fr: 434, fa: 2, tr: 698 },
);

const siftCOperatingPoints = createOperatingPoints(
    { threshold: 0.0087, tar: 0.2957, far: 0.0086, frr: 0.7043, ta: 207, fr: 493, fa: 6, tr: 694 },
);

const minutiaeCOperatingPoints = createOperatingPoints(
    { threshold: 0.2353, tar: 0.0143, far: 0.01, frr: 0.9857, ta: 10, fr: 690, fa: 7, tr: 693 },
);

const harrisCOperatingPoints = createOperatingPoints(
    { threshold: 0.0098, tar: 0.0043, far: 0.0043, frr: 0.9957, ta: 3, fr: 697, fa: 3, tr: 697 },
);

const classicCOperatingPoints = createOperatingPoints(
    { threshold: 0.0047, tar: 0.0057, far: 0.01, frr: 0.9943, ta: 4, fr: 696, fa: 7, tr: 693 },
);

const canonicalBTestRows = [
    finalSourceAfisRow({
        dataset: "nist_sd300b",
        dpi: 1000,
        operatingPoints: sourceAfisBOperatingPoints,
        auc: 0.8902,
        eer: 0.17,
        latency: 272.902,
    }),
    finalClassicalRow({
        dataset: "nist_sd300b",
        method: "sift_plain_roll_v2",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300b)",
        auc: 0.7882,
        eer: 0.2957,
        latency: 121.3818579294353,
        operatingPoints: siftV2BOperatingPoints,
        note: "SIFT Plain/Roll v2 is the strongest custom research baseline with exported latency.",
    }),
    finalClassicalRow({
        dataset: "nist_sd300b",
        method: "sift",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300b)",
        auc: 0.8049,
        eer: 0.2757,
        latency: 61.30681642904944,
        operatingPoints: siftBOperatingPoints,
        note: "Standard feature baseline under the strict pair-audited plain-vs-roll protocol.",
    }),
    finalClassicalRow({
        dataset: "nist_sd300b",
        method: "minutiae",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300b)",
        auc: 0.5230,
        eer: 0.4807,
        latency: 1135.2125217152727,
        operatingPoints: minutiaeBOperatingPoints,
        note: "Weak custom minutiae baseline under the strict plain-vs-roll protocol.",
    }),
    finalClassicalRow({
        dataset: "nist_sd300b",
        method: "harris",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300b)",
        auc: 0.5034,
        eer: 0.4936,
        latency: 561.3511191424083,
        operatingPoints: harrisBOperatingPoints,
        note: "Weak classical baseline under the strict plain-vs-roll protocol.",
    }),
    finalClassicalRow({
        dataset: "nist_sd300b",
        method: "classic_gftt_orb",
        benchmarkMethod: "classic_v2",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300b)",
        auc: 0.5048,
        eer: 0.4971,
        latency: 29.041971928257095,
        operatingPoints: classicBOperatingPoints,
        note: "Weak classical baseline under the strict plain-vs-roll protocol.",
    }),
];

const canonicalCTestRows = [
    finalSourceAfisRow({
        dataset: "nist_sd300c",
        dpi: 2000,
        operatingPoints: sourceAfisCOperatingPoints,
        auc: 0.8815,
        eer: 0.1743,
        latency: 249.966,
    }),
    finalClassicalRow({
        dataset: "nist_sd300c",
        method: "sift_plain_roll_v2",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300c)",
        auc: 0.7859,
        eer: 0.2879,
        latency: 155.3244766428335,
        operatingPoints: siftV2COperatingPoints,
        note: "SIFT Plain/Roll v2 is the strongest custom research baseline with exported latency.",
    }),
    finalClassicalRow({
        dataset: "nist_sd300c",
        method: "sift",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300c)",
        auc: 0.7912,
        eer: 0.2936,
        latency: 86.19373192669757,
        operatingPoints: siftCOperatingPoints,
        note: "Standard feature baseline under the strict pair-audited plain-vs-roll protocol.",
    }),
    finalClassicalRow({
        dataset: "nist_sd300c",
        method: "minutiae",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300c)",
        auc: 0.5285,
        eer: 0.4764,
        latency: 1160.7680758554488,
        operatingPoints: minutiaeCOperatingPoints,
        note: "Weak custom minutiae baseline under the strict plain-vs-roll protocol.",
    }),
    finalClassicalRow({
        dataset: "nist_sd300c",
        method: "harris",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300c)",
        auc: 0.4960,
        eer: 0.5071,
        latency: 576.1259360012731,
        operatingPoints: harrisCOperatingPoints,
        note: "Weak classical baseline under the strict plain-vs-roll protocol.",
    }),
    finalClassicalRow({
        dataset: "nist_sd300c",
        method: "classic_gftt_orb",
        benchmarkMethod: "classic_v2",
        runLabel: "Final classical plain-vs-roll evidence (nist_sd300c)",
        auc: 0.4776,
        eer: 0.5250,
        latency: 51.953797144150094,
        operatingPoints: classicCOperatingPoints,
        note: "Weak classical baseline under the strict plain-vs-roll protocol.",
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
        tarAtFar1e2: 0.0327,
        tarAtFar1e3: 0.0082,
        aucRank: 1,
        eerRank: 1,
        latencyRank: 2,
        artifacts: availableArtifacts("full_polyu_cross_h5", "dl_quick", "test", { meta: false, roc: false }),
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

function applyViewMode(rows: ReturnType<typeof selectionRowsBase>, viewMode: string) {
    if (viewMode === "smoke") {
        return rows.map((row) => ({
            ...row,
            run: row.run.replace("full_", "smoke_"),
            run_family: row.run_family.replace("full_", "smoke_"),
            run_label: "Smoke benchmark",
            run_kind: "smoke",
            view_mode: "smoke",
            status: "smoke",
            validation_state: "snapshot",
            split: "val",
            n_pairs: Math.min(row.n_pairs, 200),
            provenance: {
                ...row.provenance,
                run: row.provenance.run.replace("full_", "smoke_"),
                run_label: "Smoke benchmark",
                run_kind: "smoke",
                view_mode: "smoke",
                status: "smoke",
                validation_state: "snapshot",
                benchmark_source_root: "live",
                benchmark_source_label: "Live artifacts",
            },
        }));
    }

    if (viewMode === "archive") {
        return rows.map((row) => ({
            ...row,
            run: "current",
            run_family: "current",
            run_label: "Archived benchmark",
            run_kind: "legacy",
            view_mode: "archive",
            status: "archived",
            validation_state: "archived",
            provenance: {
                ...row.provenance,
                run: "current",
                run_label: "Archived benchmark",
                run_kind: "legacy",
                view_mode: "archive",
                status: "archived",
                validation_state: "archived",
                benchmark_source_root: "live",
                benchmark_source_label: "Live artifacts",
            },
        }));
    }

    return rows;
}

function selectionRowsBase(dataset: string, split: string) {
    if (dataset === "nist_sd300c") {
        return canonicalCTestRows;
    }
    if (dataset === "polyu_cross") {
        return canonicalPolyuTestRows;
    }
    if (split === "val") {
        return canonicalBValRows;
    }
    return canonicalBTestRows;
}

function selectionRows(dataset: string, split: string, viewMode = "canonical") {
    return applyViewMode(selectionRowsBase(dataset, split), viewMode);
}

function sortRows(rows: ReturnType<typeof selectionRows>, sortMode: string) {
    const items = [...rows];
    const rank = (row: (typeof items)[number], key: "auc_rank" | "eer_rank" | "latency_rank") =>
        typeof row[key] === "number" ? row[key] : 999;
    if (sortMode === "lowest_eer") {
        return items.sort((a, b) => rank(a, "eer_rank") - rank(b, "eer_rank"));
    }
    if (sortMode === "lowest_latency") {
        return items.sort((a, b) => rank(a, "latency_rank") - rank(b, "latency_rank"));
    }
    return items.sort((a, b) => rank(a, "auc_rank") - rank(b, "auc_rank"));
}

function bestEntriesFor(dataset: string, split: string, viewMode = "canonical") {
    const rows = selectionRows(dataset, split, viewMode);
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

function summaryPayload(dataset: string, split: string, viewMode = "canonical") {
    const effectiveDataset = dataset === "polyu_cross" || dataset === "nist_sd300c" ? dataset : "nist_sd300b";
    const effectiveSplit = viewMode === "smoke"
        ? "val"
        : effectiveDataset === "polyu_cross" || effectiveDataset === "nist_sd300c"
            ? "test"
            : (split === "val" ? "val" : "test");
    const rows = selectionRows(effectiveDataset, effectiveSplit, viewMode);

    return {
        dataset: effectiveDataset,
        split: effectiveSplit,
        view_mode: viewMode,
        dataset_info: datasetInfos[effectiveDataset as keyof typeof datasetInfos],
        split_info: splitInfos[effectiveSplit as keyof typeof splitInfos],
        view_info: viewInfos[viewMode as keyof typeof viewInfos],
        validation_state: viewMode === "smoke" ? "snapshot" : viewMode === "archive" ? "archived" : "validated",
        selection_note: "Showing curated full benchmark results from validated showcase runs.",
        selection_policy: "Curated full benchmark showcase restricted to validated canonical families with usable evidence.",
        result_count: rows.length,
        method_count: new Set(rows.map((row) => row.method)).size,
        run_count: new Set(rows.map((row) => row.run)).size,
        available_datasets: Object.values(datasetInfos),
        available_splits: viewMode === "smoke"
            ? [splitInfos.val]
            : effectiveDataset === "polyu_cross" || effectiveDataset === "nist_sd300c"
            ? [splitInfos.test]
            : [splitInfos.val, splitInfos.test],
        available_view_modes: Object.values(viewInfos),
        current_run_families: [...new Set(rows.map((row) => row.run))],
        artifact_note: "Artifact links surface stored benchmark evidence when files are available.",
    };
}

function comparisonPayload(dataset: string, split: string, sortMode: string, viewMode = "canonical") {
    const effectiveDataset = dataset === "polyu_cross" || dataset === "nist_sd300c" ? dataset : "nist_sd300b";
    const effectiveSplit = viewMode === "smoke"
        ? "val"
        : effectiveDataset === "polyu_cross" || effectiveDataset === "nist_sd300c"
            ? "test"
            : (split === "val" ? "val" : "test");
    const rows = sortRows(selectionRows(effectiveDataset, effectiveSplit, viewMode), sortMode);

    return {
        rows,
        datasets: Object.keys(datasetInfos),
        splits: viewMode === "smoke" ? ["val"] : effectiveDataset === "polyu_cross" || effectiveDataset === "nist_sd300c" ? ["test"] : ["val", "test"],
        default_dataset: effectiveDataset,
        default_split: effectiveSplit,
        view_mode: viewMode,
        view_info: viewInfos[viewMode as keyof typeof viewInfos],
        dataset_info: datasetInfos,
        split_info: splitInfos,
    };
}

function bestPayload(dataset: string, split: string, viewMode = "canonical") {
    const effectiveDataset = dataset === "polyu_cross" || dataset === "nist_sd300c" ? dataset : "nist_sd300b";
    const effectiveSplit = viewMode === "smoke"
        ? "val"
        : effectiveDataset === "polyu_cross" || effectiveDataset === "nist_sd300c"
            ? "test"
            : (split === "val" ? "val" : "test");
    return {
        dataset: effectiveDataset,
        split: effectiveSplit,
        view_mode: viewMode,
        entries: bestEntriesFor(effectiveDataset, effectiveSplit, viewMode),
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
        const viewMode = parsed.searchParams.get("view_mode") ?? "canonical";
        const sortMode = parsed.searchParams.get("sort_mode") ?? "best_accuracy";

        if (parsed.pathname === "/api/benchmark/summary") {
            return createJsonResponse(summaryPayload(dataset, split, viewMode));
        }

        if (parsed.pathname === "/api/benchmark/comparison") {
            return createJsonResponse(comparisonPayload(dataset, split, sortMode, viewMode));
        }

        if (parsed.pathname === "/api/benchmark/best") {
            return createJsonResponse(bestPayload(dataset, split, viewMode));
        }

        throw new Error(`Unexpected fetch call: ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);
    return { fetchMock, requests };
}

function installDedicatedBenchmarkFetchMock() {
    const baseDedicatedRow = createRow({
        dataset: "nist_sd300b",
        split: "val",
        method: "dedicated",
        run: "full_nist_sd300b_dedicated_audit",
        runLabel: "Archived benchmark",
        auc: 0.4676,
        eer: 0.5075,
        latency: 243,
        nPairs: 1200,
        aucRank: 1,
        eerRank: 1,
        latencyRank: 1,
        viewMode: "archive",
        status: "archived",
        validationState: "archived",
    });
    const dedicatedNote = "Dedicated Patch AI remains available as an experimental research method, but it is not showcase eligible.";
    const dedicatedRow = {
        ...baseDedicatedRow,
        method_status: "experimental",
        presentation_tier: "research",
        showcase_eligible: false,
        benchmark_default: false,
        canonical_default: false,
        research_track: true,
        not_champion_candidate: true,
        showcase_exclusion_note: dedicatedNote,
        auc_rank: null,
        eer_rank: null,
        latency_rank: null,
        provenance: {
            ...baseDedicatedRow.provenance,
            method_status: "experimental",
            presentation_tier: "research",
            showcase_eligible: false,
            benchmark_default: false,
            canonical_default: false,
            research_track: true,
            not_champion_candidate: true,
            showcase_exclusion_note: dedicatedNote,
            methods_in_run: ["dedicated"],
            benchmark_methods_in_run: ["dedicated"],
            showcase_methods_in_run: [],
            showcase_benchmark_methods_in_run: [],
            research_methods_in_run: ["dedicated"],
            research_benchmark_methods_in_run: ["dedicated"],
        },
    };

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
        const parsed = new URL(String(input), "http://localhost");
        if (parsed.pathname === "/api/benchmark/summary") {
            return createJsonResponse({
                dataset: "nist_sd300b",
                split: "val",
                view_mode: "archive",
                dataset_info: datasetInfos.nist_sd300b,
                split_info: splitInfos.val,
                view_info: viewInfos.archive,
                validation_state: "archived",
                selection_note: "Showing archived benchmark rows for provenance review.",
                selection_policy: "Archive view preserves research rows without champion promotion.",
                result_count: 1,
                method_count: 1,
                run_count: 1,
                available_datasets: [datasetInfos.nist_sd300b],
                available_splits: [splitInfos.val],
                available_view_modes: Object.values(viewInfos),
                current_run_families: ["full_nist_sd300b_dedicated_audit"],
                artifact_note: "Artifact links surface stored benchmark evidence when files are available.",
            });
        }
        if (parsed.pathname === "/api/benchmark/comparison") {
            return createJsonResponse({
                rows: [dedicatedRow],
                datasets: ["nist_sd300b"],
                splits: ["val"],
                default_dataset: "nist_sd300b",
                default_split: "val",
                view_mode: "archive",
                view_info: viewInfos.archive,
                dataset_info: datasetInfos,
                split_info: splitInfos,
            });
        }
        if (parsed.pathname === "/api/benchmark/best") {
            return createJsonResponse({
                dataset: "nist_sd300b",
                split: "val",
                view_mode: "archive",
                entries: [],
            });
        }
        throw new Error(`Unexpected fetch call: ${String(input)}`);
    });

    vi.stubGlobal("fetch", fetchMock);
    return { fetchMock };
}

function dedicatedResearchRow(
    row: ReturnType<typeof createRow>,
    {
        note,
        timestamp,
        sourceRoot,
        sourceLabel,
        runKind,
    }: {
        note: string;
        timestamp: string;
        sourceRoot: "live" | "reference";
        sourceLabel: string;
        runKind?: "full" | "smoke" | "legacy";
    },
) {
    const normalizedRunKind = runKind ?? row.run_kind;
    return {
        ...row,
        method_label: "Dedicated Patch AI (Experimental)",
        method_status: "experimental",
        presentation_tier: "research",
        showcase_eligible: false,
        benchmark_default: false,
        canonical_default: false,
        research_track: true,
        not_champion_candidate: true,
        showcase_exclusion_note: note,
        auc_rank: null,
        eer_rank: null,
        latency_rank: null,
        run_kind: normalizedRunKind,
        provenance: {
            ...row.provenance,
            method_label: "Dedicated Patch AI (Experimental)",
            method_status: "experimental",
            presentation_tier: "research",
            showcase_eligible: false,
            benchmark_default: false,
            canonical_default: false,
            research_track: true,
            not_champion_candidate: true,
            showcase_exclusion_note: note,
            methods_in_run: ["dedicated"],
            benchmark_methods_in_run: ["dedicated"],
            showcase_methods_in_run: [],
            showcase_benchmark_methods_in_run: [],
            research_methods_in_run: ["dedicated"],
            research_benchmark_methods_in_run: ["dedicated"],
            run_kind: normalizedRunKind,
            timestamp_utc: timestamp,
            benchmark_source_root: sourceRoot,
            benchmark_source_label: sourceLabel,
        },
    };
}

function installMultipleDedicatedBenchmarkFetchMock() {
    const dedicatedNote = "Dedicated Patch AI remains available as an experimental research method, but it is not showcase eligible.";
    const canonicalRow = createRow({
        dataset: "nist_sd300b",
        split: "val",
        method: "sift",
        run: "full_nist_sd300b_h6",
        runLabel: "Canonical full benchmark",
        auc: 0.6544,
        eer: 0.3479,
        latency: 31.84,
        nPairs: 2800,
        tarAtFar1e2: 0.1124,
        tarAtFar1e3: 0.0391,
        aucRank: 1,
        eerRank: 1,
        latencyRank: 1,
        viewMode: "archive",
        status: "archived",
        validationState: "archived",
    });
    const currentDedicated = dedicatedResearchRow(createRow({
        dataset: "nist_sd300b",
        split: "val",
        method: "dedicated",
        run: "current_dedicated_audit",
        runLabel: "Current dedicated audit",
        auc: 0.4676,
        eer: 0.5075,
        latency: 243,
        nPairs: 1200,
        aucRank: 4,
        eerRank: 4,
        latencyRank: 4,
        viewMode: "archive",
        status: "archived",
        validationState: "archived",
    }), {
        note: dedicatedNote,
        timestamp: "2026-05-01T00:00:00Z",
        sourceRoot: "live",
        sourceLabel: "Live artifacts",
        runKind: "legacy",
    });
    const archivedFullDedicated = dedicatedResearchRow(createRow({
        dataset: "nist_sd300b",
        split: "val",
        method: "dedicated",
        run: "full_nist_sd300b_dedicated_audit",
        runLabel: "Archived full dedicated audit",
        auc: 0.56,
        eer: 0.49,
        latency: 220,
        nPairs: 1200,
        aucRank: 2,
        eerRank: 2,
        latencyRank: 2,
        viewMode: "archive",
        status: "archived",
        validationState: "archived",
    }), {
        note: dedicatedNote,
        timestamp: "2026-04-15T00:00:00Z",
        sourceRoot: "reference",
        sourceLabel: "Reference artifacts",
        runKind: "full",
    });
    const archivedSmokeDedicated = dedicatedResearchRow(createRow({
        dataset: "nist_sd300b",
        split: "val",
        method: "dedicated",
        run: "smoke_nist_sd300b_dedicated_audit",
        runLabel: "Archived smoke dedicated audit",
        auc: 0.58,
        eer: 0.48,
        latency: 210,
        nPairs: 200,
        aucRank: 3,
        eerRank: 3,
        latencyRank: 3,
        viewMode: "archive",
        status: "archived",
        validationState: "archived",
    }), {
        note: dedicatedNote,
        timestamp: "2026-04-20T00:00:00Z",
        sourceRoot: "reference",
        sourceLabel: "Reference artifacts",
        runKind: "smoke",
    });
    const rows = [canonicalRow, archivedSmokeDedicated, archivedFullDedicated, currentDedicated];

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
        const parsed = new URL(String(input), "http://localhost");
        if (parsed.pathname === "/api/benchmark/summary") {
            return createJsonResponse({
                dataset: "nist_sd300b",
                split: "val",
                view_mode: "archive",
                dataset_info: datasetInfos.nist_sd300b,
                split_info: splitInfos.val,
                view_info: viewInfos.archive,
                validation_state: "archived",
                selection_note: "Showing archived benchmark rows for provenance review.",
                selection_policy: "Archive view preserves research rows without champion promotion.",
                result_count: rows.length,
                method_count: new Set(rows.map((row) => row.method)).size,
                run_count: new Set(rows.map((row) => row.run)).size,
                available_datasets: [datasetInfos.nist_sd300b],
                available_splits: [splitInfos.val],
                available_view_modes: Object.values(viewInfos),
                current_run_families: rows.map((row) => row.run),
                artifact_note: "Artifact links surface stored benchmark evidence when files are available.",
            });
        }
        if (parsed.pathname === "/api/benchmark/comparison") {
            return createJsonResponse({
                rows,
                datasets: ["nist_sd300b"],
                splits: ["val"],
                default_dataset: "nist_sd300b",
                default_split: "val",
                view_mode: "archive",
                view_info: viewInfos.archive,
                dataset_info: datasetInfos,
                split_info: splitInfos,
            });
        }
        if (parsed.pathname === "/api/benchmark/best") {
            return createJsonResponse({
                dataset: "nist_sd300b",
                split: "val",
                view_mode: "archive",
                entries: [],
            });
        }
        throw new Error(`Unexpected fetch call: ${String(input)}`);
    });

    vi.stubGlobal("fetch", fetchMock);
    return { fetchMock };
}

afterEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    window.history.replaceState(window.history.state, "", "/");
});

describe("Benchmark workspace showcase", () => {
    it("renders view controls and populates evidence immediately", async () => {
        installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Validated fingerprint matching benchmark");
            expect(normalizeText(container.textContent)).toContain("Benchmark Story");
            expect(normalizeText(container.textContent)).toContain("Current benchmark finding");
            expect(normalizeText(container.textContent)).toContain("SourceAFIS remains the strongest validated plain-vs-roll evidence");
            expect(normalizeText(container.textContent)).toContain("SIFT v2 is now the strongest custom research baseline with exported latency");
            expect(normalizeText(container.textContent)).toContain("final classical baselines produced under the same strict pair-audited VAL-to-TEST protocol");
            expect(normalizeText(container.textContent)).toContain("Positive-only and negative-only evidence is reported separately");
            expect(normalizeText(container.textContent)).toContain("not a default interactive runtime method");
            expect(normalizeText(container.textContent)).not.toContain("SIFT is currently the strongest verified method");
            expect(normalizeText(container.textContent)).toContain("SourceAFIS Open Matcher");
            expect(normalizeText(container.textContent)).toContain("SIFT Plain/Roll v2 (Experimental)");
            expect(normalizeText(container.textContent)).toContain("Baselines / Research");
            expect(normalizeText(container.textContent)).toContain("Classic (ROI GFTT+ORB)");
            expect(normalizeText(container.textContent)).toContain("Baseline");
            expect(normalizeText(container.textContent)).toContain("121.38 ms");
            expect(normalizeText(container.textContent)).toContain("Trust & provenance");
            expect(normalizeText(container.textContent)).toContain("plain_roll_final_sourceafis_v1");
            expect(normalizeText(container.textContent)).toContain("DPI");
            expect(normalizeText(container.textContent)).toContain("1000");
            expect(normalizeText(container.textContent)).toContain("FAR ~= FRR @ EER");
            expect(normalizeText(container.textContent)).toContain("EER is the point where FAR and FRR are approximately equal");
            expect(normalizeText(container.textContent)).toContain("Calibrated operating points show TEST TAR");
            expect(normalizeText(container.textContent)).toContain("Expert TAR/FAR Distribution");
            expect(normalizeText(container.textContent)).toContain("This distribution is a threshold sweep");
            expect(normalizeText(container.textContent)).toContain("TA/FR are computed only from positive pairs");
            expect(normalizeText(container.textContent)).toContain("FA/TR are computed only from negative pairs");
            expect(normalizeText(container.textContent)).toContain("TAR @ 1.00% FAR");
            expect(normalizeText(container.textContent)).toContain("TAR @ 0.50% FAR");
            expect(normalizeText(container.textContent)).toContain("Actual FAR 0.86%");
            expect(normalizeText(container.textContent)).toContain("FRR 22.71%");
            expect(normalizeText(container.textContent)).toContain("VAL FAR 1.00%");
            expect(normalizeText(container.textContent)).toContain("7/700");
            expect(normalizeText(container.textContent)).toContain("3/700");
            expect(normalizeText(container.textContent)).not.toContain("Not exported");
            expect(normalizeText(container.textContent)).toContain("Final Markdown Evidence");
            expect(normalizeText(container.textContent)).toContain("nist_sd300b_sourceafis_open_plain_roll_final.md");
            expect(normalizeText(container.textContent)).toContain("Threshold Sweep CSV");
            expect(normalizeText(container.textContent)).toContain("TAR/FAR Distribution CSV");
            expect(normalizeText(container.textContent)).toContain("Positive-only Metrics");
            expect(normalizeText(container.textContent)).toContain("Negative-only Metrics");
            expect(normalizeText(container.textContent)).not.toContain("TAR @ 0.10% FAR");
            expect(normalizeText(container.textContent)).not.toContain("TAR@FAR=1e-2");
            expect(normalizeText(container.textContent)).not.toContain("TAR@FAR=1e-3");
            expect(normalizeText(container.textContent)).not.toContain("TAR@1e-2");
            expect(normalizeText(container.textContent)).not.toContain("TAR@1e-3");
            expect(normalizeText(container.textContent)).toContain("Operating points");
            expect(normalizeText(container.querySelector("tbody tr.cursor-pointer")?.textContent)).toContain("SourceAFIS Open Matcher");
        });

        await clickRowByText(container, "SIFT Plain/Roll v2");
        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("121.38 ms");
            expect(text).toContain("Expert TAR/FAR Distribution");
            expect(text).toContain("This distribution is a threshold sweep");
            expect(text).toContain("Calibrated operating points remain the official VAL-to-TEST evidence above");
            expect(text).toContain("TA/FR are computed only from positive pairs");
            expect(text).toContain("FA/TR are computed only from negative pairs");
            expect(text).toContain("FA means a negative pair was incorrectly accepted as a match");
            expect(text).toContain("TR means a negative pair was correctly rejected");
            for (const heading of ["FAR ceiling", "Threshold", "Actual FAR", "TAR", "FRR", "TNR", "TA", "FR", "FA", "TR", "n positive", "n negative"]) {
                expect(text).toContain(heading);
            }
            expect(text).toContain("700 positive / 700 negative pairs");
            expect(text).toContain("Threshold Sweep CSV");
            expect(text).toContain("TAR/FAR Distribution CSV");
            expect(text).toContain("Positive-only Metrics");
            expect(text).toContain("Negative-only Metrics");
            expect(text).toContain("Failures CSV");
            expect(text).toContain("nist_sd300b_sift_plain_roll_v2_plain_roll_final.md");
            expect(text).toContain("0 recorded failures");
        });

        const viewField = getLabelField<HTMLSelectElement>(container, "View");
        expect(Array.from(viewField.options).map((option) => option.textContent)).toEqual([
            "Showcase",
            "Smoke",
            "Archive",
        ]);

        const datasetField = getLabelField<HTMLSelectElement>(container, "Dataset");
        const splitField = getLabelField<HTMLSelectElement>(container, "Split");

        expect(Array.from(datasetField.options).map((option) => option.textContent)).toEqual([
            "NIST SD300b",
            "NIST SD300c",
            "PolyU Cross",
        ]);
        expect(Array.from(splitField.options).map((option) => option.textContent)).toEqual([
            "Validation",
            "Test",
        ]);

        await changeSelect(datasetField, "nist_sd300c");
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("SourceAFIS champion evidence");
            expect(normalizeText(container.textContent)).toContain("DPI");
            expect(normalizeText(container.textContent)).toContain("2000");
            expect(normalizeText(container.textContent)).toContain("Actual FAR 1.29%");
            expect(normalizeText(container.textContent)).toContain("Final Markdown Evidence");
        });

        await unmountWorkspace(root);
    });

    it("lists only valid splits for the selected showcase dataset", async () => {
        installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace();

        await changeSelect(getLabelField<HTMLSelectElement>(container, "Dataset"), "polyu_cross");
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("full_polyu_cross_h5");
            expect(normalizeText(container.textContent)).toContain("Deep Learning (ResNet18)");
            expect(normalizeText(container.textContent)).toContain("Classic (ROI GFTT+ORB)");
        });

        const splitField = getLabelField<HTMLSelectElement>(container, "Split");
        expect(Array.from(splitField.options).map((option) => option.textContent)).toEqual(["Test"]);
        expect(normalizeText(container.textContent)).not.toContain("No curated full benchmark results");

        await unmountWorkspace(root);
    });

    it("preserves benchmarkView urls and sends view_mode to benchmark endpoints", async () => {
        const { requests } = installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace("/?benchmarkView=archive&benchmarkDataset=polyu_cross&benchmarkSplit=test");

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("current");
            expect(getLabelField<HTMLSelectElement>(container, "View").value).toBe("archive");
        });

        expect(window.location.search).toContain("benchmarkView=archive");
        expect(requests.some((url) => url.includes("view_mode=archive"))).toBe(true);

        await unmountWorkspace(root);
    });

    it("changing view mode reloads benchmark endpoints with view_mode", async () => {
        const { requests } = installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace();

        await waitFor(() => {
            expect(getLabelField<HTMLSelectElement>(container, "View").value).toBe("canonical");
        });

        await changeSelect(getLabelField<HTMLSelectElement>(container, "View"), "smoke");
        await waitFor(() => {
            expect(getLabelField<HTMLSelectElement>(container, "View").value).toBe("smoke");
            expect(normalizeText(container.textContent)).toContain("Smoke benchmark");
        });

        expect(window.location.search).toContain("benchmarkView=smoke");
        expect(requests.some((url) => url.includes("/api/benchmark/summary") && url.includes("view_mode=smoke"))).toBe(true);
        expect(requests.some((url) => url.includes("/api/benchmark/comparison") && url.includes("view_mode=smoke"))).toBe(true);
        expect(requests.some((url) => url.includes("/api/benchmark/best") && url.includes("view_mode=smoke"))).toBe(true);

        await unmountWorkspace(root);
    });

    it("keeps provenance usable when the selected row is missing some artifacts", async () => {
        installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace();

        await changeSelect(getLabelField<HTMLSelectElement>(container, "Dataset"), "polyu_cross");
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Deep Learning (ResNet18)");
            expect(normalizeText(container.textContent)).not.toContain("dl_quick");
        });

        await clickRowByText(container, "Deep Learning (ResNet18)");
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("ROC preview is not available for this row");
            expect(normalizeText(container.textContent)).toContain("Meta JSON unavailable");
            expect(normalizeText(container.textContent)).toContain("N/A");
        });

        await click(getButtonByText(container, "Open provenance details"));
        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("Methods in run");
            expect(normalizeText(container.textContent)).toContain("Benchmark method");
            expect(normalizeText(container.textContent)).toContain("vit");
            expect(normalizeText(container.textContent)).toContain("deadbeef");
        });

        await unmountWorkspace(root);
    });

    it("falls back cleanly when the ROC image cannot render", async () => {
        installBenchmarkFetchMock();
        const { container, root } = await renderWorkspace();

        await clickRowByText(container, "SIFT Plain/Roll v2");
        await waitFor(() => {
            expect(container.querySelector("img[aria-label*='ROC preview']")).not.toBeNull();
        });

        const image = container.querySelector("img[aria-label*='ROC preview']");
        if (!image) {
            throw new Error("ROC preview image was not rendered.");
        }

        await act(async () => {
            image.dispatchEvent(new Event("error", { bubbles: false }));
        });

        await waitFor(() => {
            expect(normalizeText(container.textContent)).toContain("ROC preview could not be rendered. Open artifact instead.");
            expect(normalizeText(container.textContent)).toContain("Open ROC artifact");
        });

        await unmountWorkspace(root);
    });

    it("marks dedicated rows as research and keeps them out of champion treatment", async () => {
        installDedicatedBenchmarkFetchMock();
        const { container, root } = await renderWorkspace("/?benchmarkView=archive&benchmarkDataset=nist_sd300b&benchmarkSplit=val");

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Dedicated Patch AI");
            expect(text).toContain("Baselines / Research");
            expect(text).toContain("Experimental");
            expect(text).toContain("Research");
            expect(text).toContain("Not showcase eligible");
            expect(text).toContain("Research-only method");
            expect(text).not.toContain("#1 Accuracy");
            expect(text).not.toContain("Fastest method, slightly lower AUC.");
        });

        await unmountWorkspace(root);
    });

    it("groups duplicate dedicated research rows by default and can reveal history", async () => {
        installMultipleDedicatedBenchmarkFetchMock();
        const { container, root } = await renderWorkspace("/?benchmarkView=archive&benchmarkDataset=nist_sd300b&benchmarkSplit=val");

        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Classic (SIFT)");
            expect(text).toContain("Dedicated Patch AI (Experimental)");
            expect(text).toContain("Experimental");
            expect(text).toContain("Research");
            expect(text).toContain("Not showcase eligible");
            expect(text).toContain("3 research runs available");
            expect(text).toContain("Showing representative research run; 2 archived runs hidden");
            expect(text).toContain("Current dedicated audit");
            expect(text).toContain("methods in comparison");
            expect(text).not.toContain("validated benchmark methods");
            expect(text).not.toContain("Archived full dedicated audit");
            expect(text).not.toContain("Archived smoke dedicated audit");
        });

        expect(container.querySelectorAll("tbody tr.cursor-pointer")).toHaveLength(2);

        await clickRowByText(container, "Dedicated Patch AI");
        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Representative research run");
            expect(text).toContain("Showing current");
            expect(text).toContain("2 archived research runs hidden");
        });

        await click(getButtonByText(container, "Show archived research runs"));
        await waitFor(() => {
            const text = normalizeText(container.textContent);
            expect(text).toContain("Hide archived research runs");
            expect(text).toContain("Archived full dedicated audit");
            expect(text).toContain("Archived smoke dedicated audit");
            expect(text).toContain("archived full benchmark");
            expect(text).toContain("archived smoke benchmark");
        });

        expect(container.querySelectorAll("tbody tr.cursor-pointer")).toHaveLength(4);

        await unmountWorkspace(root);
    });
});
