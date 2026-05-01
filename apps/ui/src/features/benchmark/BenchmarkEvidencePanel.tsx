import { useState } from "react";
import type { ComparisonRow, NamedInfo } from "../../types";
import {
    formatLatency,
    formatMethodLabel,
    formatMetric,
    formatPairs,
    statusLabel,
    statusToneClassName,
} from "./benchmarkPresentation.ts";

type Props = {
    row: ComparisonRow | null;
    datasetInfo: Record<string, NamedInfo>;
    splitInfo: Record<string, NamedInfo>;
};

function artifactByKey(row: ComparisonRow, key: string) {
    return row.artifacts.find((item) => item.key === key) ?? null;
}

export default function BenchmarkEvidencePanel({ row, datasetInfo, splitInfo }: Props) {
    const [provenanceOpen, setProvenanceOpen] = useState(false);

    if (!row) {
        return (
            <section className="rounded-[1.75rem] border border-slate-800 bg-slate-950/80 p-6 text-slate-300 shadow-[0_20px_80px_rgba(2,6,23,0.35)]">
                <div className="space-y-3">
                    <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Evidence</p>
                    <h3 className="text-xl font-semibold text-slate-100">Selected method evidence</h3>
                    <p className="text-sm leading-6 text-slate-400">
                        Preparing benchmark evidence for the top-ranked showcase method.
                    </p>
                </div>
            </section>
        );
    }

    const datasetLabel = datasetInfo[row.dataset]?.label ?? row.dataset;
    const splitLabel = splitInfo[row.split]?.label ?? row.split;
    const rocArtifact = artifactByKey(row, "roc_png");
    const rawMetaArtifact = artifactByKey(row, "meta_json");

    return (
        <section className="rounded-[1.75rem] border border-slate-800 bg-slate-950/80 p-6 text-slate-200 shadow-[0_20px_80px_rgba(2,6,23,0.35)]">
            <div className="flex items-start justify-between gap-4">
                <div className="space-y-2">
                    <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Evidence</p>
                    <h3 className="text-xl font-semibold text-slate-50">{formatMethodLabel(row.method, row.method_label)}</h3>
                    <p className="text-sm text-slate-400">
                        {datasetLabel} - {splitLabel} - {row.run_family ?? row.run}
                    </p>
                </div>
                <span className={`inline-flex rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] ${statusToneClassName(row.status)}`}>
                    {statusLabel(row.status)}
                </span>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">AUC</p>
                    <p className="mt-2 text-lg font-semibold text-slate-50">{formatMetric(row.auc)}</p>
                </div>
                <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">EER</p>
                    <p className="mt-2 text-lg font-semibold text-slate-50">{formatMetric(row.eer)}</p>
                </div>
                <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Latency</p>
                    <p className="mt-2 text-lg font-semibold text-slate-50">{formatLatency(row.latency_ms)}</p>
                </div>
                <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Pairs</p>
                    <p className="mt-2 text-lg font-semibold text-slate-50">{formatPairs(row.n_pairs)}</p>
                </div>
            </div>

            <div className="mt-6 overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/70">
                {rocArtifact?.available && rocArtifact.url ? (
                    <img
                        src={rocArtifact.url}
                        alt={`${formatMethodLabel(row.method, row.method_label)} ROC preview`}
                        className="h-56 w-full object-contain bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.14),_transparent_55%),linear-gradient(180deg,rgba(15,23,42,0.96),rgba(2,6,23,0.98))]"
                    />
                ) : (
                    <div className="flex h-56 items-center justify-center bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.14),_transparent_55%),linear-gradient(180deg,rgba(15,23,42,0.96),rgba(2,6,23,0.98))] px-6 text-center text-sm text-slate-500">
                        ROC preview is not available for this row. The rest of the evidence remains usable.
                    </div>
                )}
            </div>

            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Summary</p>
                <p className="mt-3 text-sm leading-6 text-slate-300">{row.summary_text}</p>
                <p className="mt-3 text-sm leading-6 text-slate-400">
                    Selected method evidence for the active curated full benchmark comparison.
                </p>
            </div>

            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="flex items-center justify-between gap-4">
                    <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Artifacts</p>
                        <p className="mt-1 text-sm text-slate-400">{row.artifact_count} artifact links available for this method row.</p>
                    </div>
                </div>
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                    {row.artifacts.map((artifact) => (
                        artifact.available && artifact.url ? (
                            <a
                                key={artifact.key}
                                href={artifact.url}
                                target="_blank"
                                rel="noreferrer"
                                className="rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3 text-sm text-slate-200 transition hover:border-sky-500/40 hover:text-sky-200"
                            >
                                {artifact.label}
                            </a>
                        ) : (
                            <div
                                key={artifact.key}
                                className="rounded-2xl border border-slate-900 bg-slate-950/60 px-4 py-3 text-sm text-slate-500"
                            >
                                {artifact.label} - N/A
                            </div>
                        )
                    ))}
                </div>
            </div>

            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <button
                    type="button"
                    onClick={() => setProvenanceOpen((current) => !current)}
                    className="text-sm font-semibold text-slate-200 transition hover:text-sky-200"
                >
                    {provenanceOpen ? "Hide provenance" : "Open provenance"}
                </button>
                {provenanceOpen ? (
                    <div className="mt-4 space-y-3 text-sm text-slate-300">
                        <div className="grid gap-3 sm:grid-cols-2">
                            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">API method</p>
                                <p className="mt-2 font-medium text-slate-100">
                                    {formatMethodLabel(row.provenance?.canonical_method ?? row.method, row.provenance?.method_label ?? row.method_label)}
                                </p>
                            </div>
                            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Benchmark method</p>
                                <p className="mt-2 font-mono text-slate-300">
                                    {row.provenance?.benchmark_method ?? row.benchmark_method}
                                </p>
                            </div>
                            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Run</p>
                                <p className="mt-2 font-medium text-slate-100">{row.provenance?.run ?? row.run}</p>
                            </div>
                            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Source</p>
                                <p className="mt-2 font-medium text-slate-100">{row.provenance?.source_type ?? "summary_csv"}</p>
                            </div>
                            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Validation</p>
                                <p className="mt-2 font-medium text-slate-100">{row.provenance?.validation_state ?? row.validation_state}</p>
                            </div>
                            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Artifacts available</p>
                                <p className="mt-2 font-medium text-slate-100">
                                    {(row.provenance?.available_artifacts ?? row.available_artifacts).join(", ") || "N/A"}
                                </p>
                            </div>
                        </div>
                        <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Methods in run</p>
                            <p className="mt-2 font-medium text-slate-100">
                                {(row.provenance?.methods_in_run ?? []).map((method) => formatMethodLabel(method)).join(", ") || "N/A"}
                            </p>
                        </div>
                        {(row.provenance?.benchmark_methods_in_run ?? []).length > 0 ? (
                            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Raw benchmark methods in run</p>
                                <p className="mt-2 font-mono text-slate-300">
                                    {(row.provenance?.benchmark_methods_in_run ?? []).join(", ")}
                                </p>
                            </div>
                        ) : null}
                        {row.provenance?.git_commit ? (
                            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Git commit</p>
                                <p className="mt-2 break-all text-slate-300">{row.provenance.git_commit}</p>
                            </div>
                        ) : null}
                        {row.provenance?.pairs_path ? (
                            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Pair source</p>
                                <p className="mt-2 break-all text-slate-300">{row.provenance.pairs_path}</p>
                            </div>
                        ) : null}
                        {rawMetaArtifact?.available && rawMetaArtifact.url ? (
                            <div className="pt-1">
                                <a
                                    href={rawMetaArtifact.url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-sm font-medium text-sky-300 transition hover:text-sky-200"
                                >
                                    Show raw metadata
                                </a>
                            </div>
                        ) : null}
                    </div>
                ) : null}
            </div>
        </section>
    );
}
