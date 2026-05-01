import { FileImage, FolderOpen, LoaderCircle, Play } from "lucide-react";
import type { DemoCase } from "../types";
import { formatMethodLabel } from "../shared/storytelling.ts";

interface Props {
    cases: DemoCase[];
    loading?: boolean;
    busy?: boolean;
    runningCaseId?: string | null;
    selectedCaseId?: string | null;
    onRunDemo: (demo: DemoCase) => void;
}

function formatDifficulty(value: string): string {
    return value
        .split(/[_-]/g)
        .filter(Boolean)
        .map((item) => item[0].toUpperCase() + item.slice(1))
        .join(" ");
}

function formatCapture(value: string): string {
    return value
        .split(/[_-]/g)
        .filter(Boolean)
        .map((item) => item[0].toUpperCase() + item.slice(1))
        .join(" ");
}

function getFileLabel(url: string): string {
    return url.split("/").pop() || url;
}

function evidenceBadgeLabel(status: NonNullable<DemoCase["evidence_quality"]>["evidence_status"]): string {
    if (status === "strong") {
        return "Strong evidence";
    }
    if (status === "fallback") {
        return "Fallback evidence";
    }
    return "Degraded evidence";
}

function evidenceBadgeClass(status: NonNullable<DemoCase["evidence_quality"]>["evidence_status"]): string {
    if (status === "strong") {
        return "border-emerald-200 bg-emerald-50 text-emerald-700";
    }
    if (status === "fallback") {
        return "border-amber-200 bg-amber-50 text-amber-700";
    }
    return "border-orange-200 bg-orange-50 text-orange-700";
}

export default function DemoCasesPanel({
    cases,
    loading = false,
    busy = false,
    runningCaseId = null,
    selectedCaseId = null,
    onRunDemo,
}: Props) {
    if (loading) {
        return (
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center text-slate-500">
                Loading demo cases...
            </div>
        );
    }

    if (cases.length === 0) {
        return (
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center text-slate-500">
                No curated demo cases are currently available.
            </div>
        );
    }

    return (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            {cases.map((demo) => {
                const isRunning = runningCaseId === demo.id;
                const isSelected = selectedCaseId === demo.id;

                return (
                    <div
                        key={demo.id}
                        className={[
                            "overflow-hidden rounded-xl border bg-white shadow-sm transition-all",
                            isSelected ? "border-brand-200 ring-2 ring-brand-100" : "border-slate-200",
                        ].join(" ")}
                    >
                        <div className="border-b border-slate-100 bg-slate-50/80 px-6 py-5">
                            <div className="flex items-start justify-between gap-4">
                                <div>
                                    <h3 className="text-lg font-semibold text-slate-800">{demo.title}</h3>
                                    <p className="mt-1 text-sm text-slate-500">{demo.description}</p>
                                </div>
                                <span className="rounded-full border border-brand-100 bg-brand-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-brand-700">
                                    {formatDifficulty(demo.difficulty)}
                                </span>
                            </div>
                        </div>

                        <div className="space-y-4 p-6 text-sm text-slate-700">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <p className="text-slate-500">Dataset</p>
                                    <p className="font-medium text-slate-800">{demo.dataset_label}</p>
                                </div>
                                <div>
                                    <p className="text-slate-500">Split</p>
                                    <p className="font-medium text-slate-800">{demo.split_label}</p>
                                </div>
                                <div>
                                    <p className="text-slate-500">Ground truth</p>
                                    <p className="font-medium text-slate-800">{demo.ground_truth}</p>
                                </div>
                                {demo.case_type ? (
                                    <div>
                                        <p className="text-slate-500">Case type</p>
                                        <p className="font-medium text-slate-800">{formatDifficulty(demo.case_type)}</p>
                                    </div>
                                ) : null}
                                {demo.modality_relation ? (
                                    <div>
                                        <p className="text-slate-500">Relation</p>
                                        <p className="font-medium text-slate-800">{formatDifficulty(demo.modality_relation)}</p>
                                    </div>
                                ) : null}
                                {typeof demo.benchmark_score === "number" ? (
                                    <div>
                                        <p className="text-slate-500">Benchmark score</p>
                                        <p className="font-medium text-slate-800">{demo.benchmark_score.toFixed(4)}</p>
                                    </div>
                                ) : null}
                                {demo.benchmark_method ? (
                                    <div>
                                        <p className="text-slate-500">Benchmark method</p>
                                        <p className="font-medium text-slate-800">{formatMethodLabel(demo.benchmark_method)}</p>
                                        <p className="mt-1 text-xs uppercase tracking-wide text-slate-400">{demo.benchmark_method}</p>
                                    </div>
                                ) : null}
                                {demo.benchmark_run ? (
                                    <div>
                                        <p className="text-slate-500">Benchmark context</p>
                                        <p className="font-medium text-slate-800">{demo.benchmark_run}</p>
                                    </div>
                                ) : null}
                                {demo.evidence_quality ? (
                                    <div className="col-span-2">
                                        <p className="text-slate-500">Evidence quality</p>
                                        <div className="mt-2 flex flex-wrap gap-2">
                                            <span
                                                className={[
                                                    "rounded-full border px-3 py-1 text-xs font-semibold",
                                                    evidenceBadgeClass(demo.evidence_quality.evidence_status),
                                                ].join(" ")}
                                            >
                                                {evidenceBadgeLabel(demo.evidence_quality.evidence_status)}
                                            </span>
                                            <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
                                                {demo.evidence_quality.selection_driver === "benchmark_driven"
                                                    ? "Benchmark-backed"
                                                    : "Heuristic fallback"}
                                            </span>
                                        </div>
                                        <p className="mt-2 text-sm text-slate-600">{demo.evidence_quality.evidence_note}</p>
                                    </div>
                                ) : null}
                            </div>

                            <div className="rounded-lg border border-slate-100 bg-slate-50 p-4 space-y-2">
                                <p className="flex items-center font-medium text-slate-800">
                                    <FolderOpen className="mr-2 h-4 w-4 text-brand-600" />
                                    Server assets
                                </p>
                                <div className="grid grid-cols-1 gap-2 text-slate-600 md:grid-cols-2">
                                    <span className="flex items-center">
                                        <FileImage className="mr-2 h-3.5 w-3.5" />
                                        {getFileLabel(demo.image_a_url)} ({formatCapture(demo.capture_a)})
                                    </span>
                                    <span className="flex items-center">
                                        <FileImage className="mr-2 h-3.5 w-3.5" />
                                        {getFileLabel(demo.image_b_url)} ({formatCapture(demo.capture_b)})
                                    </span>
                                </div>
                            </div>

                            <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
                                <p className="text-slate-500">Selection rationale</p>
                                <p className="mt-1 text-slate-700">{demo.curation_rule}</p>
                                {demo.selection_policy ? (
                                    <p className="mt-2 text-xs uppercase tracking-wide text-slate-400">{demo.selection_policy}</p>
                                ) : null}
                            </div>

                            <div className="flex items-center justify-between gap-4 pt-2">
                                <div>
                                    <p className="text-slate-500">Recommended method</p>
                                    <p className="font-semibold text-slate-800">
                                        {formatMethodLabel(demo.recommended_method)}
                                    </p>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => onRunDemo(demo)}
                                    disabled={busy}
                                    className="flex items-center rounded-lg bg-brand-600 px-4 py-2.5 font-medium text-white transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-brand-300"
                                >
                                    {isRunning ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                                    {isRunning ? "Running case" : isSelected ? "Run again" : "Run case"}
                                </button>
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
