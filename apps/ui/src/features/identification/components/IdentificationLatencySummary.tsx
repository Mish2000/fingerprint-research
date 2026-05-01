import type { IdentificationLatencyState } from "../storyModel.ts";

interface IdentificationLatencySummaryProps {
    latency: IdentificationLatencyState | null;
}

export default function IdentificationLatencySummary({ latency }: IdentificationLatencySummaryProps) {
    if (!latency) {
        return (
            <section className="rounded-2xl border border-slate-200 bg-white p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Latency</p>
                <p className="mt-2 text-sm text-slate-600">Latency details will appear once the backend returns a completed result.</p>
            </section>
        );
    }

    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Latency</p>
            <p className="mt-2 text-base font-semibold text-slate-900">{latency.totalLabel}</p>
            <p className="mt-2 text-sm leading-6 text-slate-600">{latency.summary}</p>
            {latency.breakdown.length > 0 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                    {latency.breakdown.map((item) => (
                        <span key={`${item.label}_${item.value}`} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-700">
                            {item.label}: {item.value}
                        </span>
                    ))}
                </div>
            ) : null}
        </section>
    );
}
