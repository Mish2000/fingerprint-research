import type { VerifyCaseContextState } from "../storyModel.ts";

interface VerifyCaseContextSummaryProps {
    context: VerifyCaseContextState;
}

export default function VerifyCaseContextSummary({ context }: VerifyCaseContextSummaryProps) {
    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Probe context</p>
            <p className="mt-2 text-base font-semibold text-slate-900">{context.label}</p>
            <p className="mt-2 text-sm leading-6 text-slate-600">{context.summary}</p>
            {context.details.length > 0 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                    {context.details.map((detail) => (
                        <span key={detail} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-700">
                            {detail}
                        </span>
                    ))}
                </div>
            ) : null}
        </section>
    );
}
