import type { VerifyExpectationState } from "../storyModel.ts";

interface VerifyExpectationSummaryProps {
    expectation: VerifyExpectationState;
}

export default function VerifyExpectationSummary({ expectation }: VerifyExpectationSummaryProps) {
    return (
        <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Expected vs actual</p>
            <div className="mt-3 space-y-2 text-sm text-[var(--app-text-soft)]">
                <div className="flex items-start justify-between gap-4 rounded-xl bg-[var(--app-surface-subtle)] px-4 py-3">
                    <span className="font-medium text-[var(--app-text-muted)]">Expected</span>
                    <span className="text-right font-semibold text-[var(--app-text)]">{expectation.expectedLabel ?? "Unavailable"}</span>
                </div>
                <div className="flex items-start justify-between gap-4 rounded-xl bg-[var(--app-surface-subtle)] px-4 py-3">
                    <span className="font-medium text-[var(--app-text-muted)]">Actual</span>
                    <span className="text-right font-semibold text-[var(--app-text)]">{expectation.actualLabel ?? "Pending run"}</span>
                </div>
            </div>
            <p className="mt-3 text-sm leading-6 text-[var(--app-text-soft)]">{expectation.summary}</p>
        </section>
    );
}
