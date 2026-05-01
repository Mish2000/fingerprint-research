import { LoaderCircle, Play, Sparkles } from "lucide-react";
import type { CatalogVerifyCase } from "../types/index.ts";
import { formatCaseMetadataLabel, formatGroundTruthLabel } from "../features/verify/model.ts";
import { formatMethodLabel } from "../shared/storytelling.ts";

interface VerifyDemoCasesPanelProps {
    cases: CatalogVerifyCase[];
    loading?: boolean;
    busy?: boolean;
    selectedCaseId?: string | null;
    runningCaseId?: string | null;
    onSelectDemo: (demoCase: CatalogVerifyCase) => void;
    onRunDemo: (demoCase: CatalogVerifyCase) => void;
}

function formatCapturePair(demoCase: CatalogVerifyCase): string {
    return `${formatCaseMetadataLabel(demoCase.capture_a)} to ${formatCaseMetadataLabel(demoCase.capture_b)}`;
}

function formatTagLabel(value: string): string {
    return formatCaseMetadataLabel(value).replace("Non Match", "Impostor");
}

function evidenceStatusLabel(status: NonNullable<CatalogVerifyCase["evidence_quality"]>["evidence_status"]): string {
    if (status === "strong") {
        return "Strong evidence";
    }
    if (status === "fallback") {
        return "Fallback evidence";
    }
    return "Degraded evidence";
}

function evidenceStatusBadgeClass(status: NonNullable<CatalogVerifyCase["evidence_quality"]>["evidence_status"]): string {
    if (status === "strong") {
        return "border-emerald-200 bg-emerald-50 text-emerald-700";
    }
    if (status === "fallback") {
        return "border-amber-200 bg-amber-50 text-amber-700";
    }
    return "border-orange-200 bg-orange-50 text-orange-700";
}

function selectionDriverLabel(driver: NonNullable<CatalogVerifyCase["evidence_quality"]>["selection_driver"]): string {
    return driver === "benchmark_driven" ? "Benchmark-backed" : "Heuristic fallback";
}

export default function VerifyDemoCasesPanel({
    cases,
    loading = false,
    busy = false,
    selectedCaseId = null,
    runningCaseId = null,
    onSelectDemo,
    onRunDemo,
}: VerifyDemoCasesPanelProps) {
    if (loading) {
        return (
            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-6 py-10 text-center text-sm text-slate-500">
                Loading curated verify cases...
            </div>
        );
    }

    if (cases.length === 0) {
        return (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-6 py-10 text-center text-sm text-slate-500">
                No curated cases match the current narrowing.
            </div>
        );
    }

    return (
        <div className="grid gap-4 lg:grid-cols-2">
            {cases.map((demoCase) => {
                const isSelected = selectedCaseId === demoCase.case_id;
                const isRunning = runningCaseId === demoCase.case_id;

                return (
                    <article
                        key={demoCase.case_id}
                        className={[
                            "rounded-2xl border bg-white p-5 shadow-sm transition-all",
                            isSelected ? "border-brand-300 ring-2 ring-brand-100" : "border-slate-200 hover:border-slate-300",
                        ].join(" ")}
                    >
                        <button
                            type="button"
                            onClick={() => onSelectDemo(demoCase)}
                            className="w-full text-left"
                            aria-pressed={isSelected}
                        >
                            <div className="flex flex-wrap items-start justify-between gap-3">
                                <div className="space-y-2">
                                    <div className="flex flex-wrap items-center gap-2">
                                        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600">
                                            {demoCase.dataset_label}
                                        </span>
                                        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600">
                                            {demoCase.split}
                                        </span>
                                        <span className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-700">
                                            {formatCaseMetadataLabel(demoCase.difficulty)}
                                        </span>
                                        {demoCase.evidence_quality ? (
                                            <span
                                                className={[
                                                    "rounded-full border px-2.5 py-1 text-xs font-semibold",
                                                    evidenceStatusBadgeClass(demoCase.evidence_quality.evidence_status),
                                                ].join(" ")}
                                            >
                                                {evidenceStatusLabel(demoCase.evidence_quality.evidence_status)}
                                            </span>
                                        ) : null}
                                    </div>
                                    <div>
                                        <h3 className="text-base font-semibold text-slate-900">{demoCase.title}</h3>
                                        <p className="mt-1 text-sm leading-6 text-slate-600">{demoCase.description}</p>
                                    </div>
                                </div>

                                <div className="flex flex-wrap gap-2">
                                    {isSelected ? (
                                        <span className="rounded-full border border-brand-200 bg-brand-50 px-2.5 py-1 text-xs font-semibold text-brand-700">
                                            Selected
                                        </span>
                                    ) : null}
                                    {isRunning ? (
                                        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
                                            Running
                                        </span>
                                    ) : null}
                                </div>
                            </div>
                        </button>

                        <div className="mt-4 grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
                            <div>
                                <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Ground Truth</p>
                                <p className="mt-1 font-medium text-slate-900">{formatGroundTruthLabel(demoCase.ground_truth)}</p>
                            </div>
                            <div>
                                <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Recommended Method</p>
                                <p className="mt-1 font-medium text-slate-900">{formatMethodLabel(demoCase.recommended_method)}</p>
                            </div>
                            <div>
                                <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Capture Types</p>
                                <p className="mt-1 font-medium text-slate-900">{formatCapturePair(demoCase)}</p>
                            </div>
                            <div>
                                <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Case Type</p>
                                <p className="mt-1 font-medium text-slate-900">{formatCaseMetadataLabel(demoCase.case_type)}</p>
                            </div>
                            {demoCase.modality_relation ? (
                                <div className="sm:col-span-2">
                                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Modality Relation</p>
                                    <p className="mt-1 font-medium text-slate-900">{formatCaseMetadataLabel(demoCase.modality_relation)}</p>
                                </div>
                            ) : null}
                            {demoCase.evidence_quality ? (
                                <div className="sm:col-span-2">
                                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Evidence Quality</p>
                                    <div className="mt-1 flex flex-wrap gap-2">
                                        <span
                                            className={[
                                                "rounded-full border px-2.5 py-1 text-xs font-semibold",
                                                evidenceStatusBadgeClass(demoCase.evidence_quality.evidence_status),
                                            ].join(" ")}
                                        >
                                            {evidenceStatusLabel(demoCase.evidence_quality.evidence_status)}
                                        </span>
                                        <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-600">
                                            {selectionDriverLabel(demoCase.evidence_quality.selection_driver)}
                                        </span>
                                    </div>
                                    <p className="mt-2 text-sm leading-6 text-slate-700">{demoCase.evidence_quality.evidence_note}</p>
                                </div>
                            ) : null}
                        </div>

                        <div className="mt-4 rounded-2xl border border-slate-100 bg-slate-50 p-4">
                            <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Why This Case Is Here</p>
                            <p className="mt-2 text-sm leading-6 text-slate-700">{demoCase.selection_reason}</p>
                            <p className="mt-2 text-xs font-medium uppercase tracking-[0.16em] text-slate-400">
                                {formatTagLabel(demoCase.selection_policy)}
                            </p>
                        </div>

                        {demoCase.tags.length > 0 ? (
                            <div className="mt-4 flex flex-wrap gap-2">
                                {demoCase.tags.map((tag) => (
                                    <span
                                        key={tag}
                                        className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-600"
                                    >
                                        {formatTagLabel(tag)}
                                    </span>
                                ))}
                            </div>
                        ) : null}

                        <div className="mt-5 flex items-center justify-between gap-3">
                            <span className="inline-flex items-center gap-2 text-sm text-slate-500">
                                <Sparkles className="h-4 w-4 text-brand-600" />
                                One-click server-backed verify
                            </span>

                            <button
                                type="button"
                                onClick={() => onRunDemo(demoCase)}
                                disabled={busy}
                                className="inline-flex items-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-brand-300"
                            >
                                {isRunning ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                                {isRunning ? "Running..." : isSelected ? "Run Selected" : "Run Case"}
                            </button>
                        </div>
                    </article>
                );
            })}
        </div>
    );
}
