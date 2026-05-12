import type { VerifyLatencyState } from "../storyModel.ts";

interface VerifyLatencySummaryProps {
    latency: VerifyLatencyState | null;
}

export default function VerifyLatencySummary({ latency }: VerifyLatencySummaryProps) {
    if (!latency) {
        return (
            <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Latency</p>
                <p className="mt-2 text-sm text-[var(--app-text-soft)]">Latency will appear once the backend returns a completed result.</p>
            </section>
        );
    }

    return (
        <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Latency</p>
            <p className="mt-2 text-base font-semibold text-[var(--app-text)]">{latency.totalLabel}</p>
            <p className="mt-2 text-sm leading-6 text-[var(--app-text-soft)]">{latency.summary}</p>
            {latency.breakdown.length > 0 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                    {latency.breakdown.map((item) => (
                        <span key={`${item.label}_${item.value}`} className="status-pill">
                            {item.label}: {item.value}
                        </span>
                    ))}
                </div>
            ) : null}
        </section>
    );
}
