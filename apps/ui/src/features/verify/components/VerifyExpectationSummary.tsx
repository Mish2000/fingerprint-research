import type { VerifyExpectationState } from "../storyModel.ts";

interface VerifyExpectationSummaryProps {
    expectation: VerifyExpectationState;
}

export default function VerifyExpectationSummary({ expectation }: VerifyExpectationSummaryProps) {
    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Expected vs actual</p>
            <div className="mt-3 space-y-2 text-sm text-slate-700">
                <div className="flex items-start justify-between gap-4 rounded-xl bg-slate-50 px-4 py-3">
                    <span className="font-medium text-slate-500">Expected</span>
                    <span className="text-right font-semibold text-slate-900">{expectation.expectedLabel ?? "Unavailable"}</span>
                </div>
                <div className="flex items-start justify-between gap-4 rounded-xl bg-slate-50 px-4 py-3">
                    <span className="font-medium text-slate-500">Actual</span>
                    <span className="text-right font-semibold text-slate-900">{expectation.actualLabel ?? "Pending run"}</span>
                </div>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-600">{expectation.summary}</p>
        </section>
    );
}
