import type { VerifyCaseContextState } from "../storyModel.ts";

interface VerifyCaseContextSummaryProps {
    context: VerifyCaseContextState;
}

export default function VerifyCaseContextSummary({ context }: VerifyCaseContextSummaryProps) {
    return (
        <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Probe context</p>
            <p className="mt-2 text-base font-semibold text-[var(--app-text)]">{context.label}</p>
            <p className="mt-2 text-sm leading-6 text-[var(--app-text-soft)]">{context.summary}</p>
            {context.details.length > 0 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                    {context.details.map((detail) => (
                        <span key={detail} className="status-pill">
                            {detail}
                        </span>
                    ))}
                </div>
            ) : null}
        </section>
    );
}
