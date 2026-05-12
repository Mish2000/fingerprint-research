import type { ConfidenceBand } from "../../../shared/storytelling.ts";

interface VerifyConfidenceBandProps {
    band: ConfidenceBand | null;
}

function paletteFor(level: ConfidenceBand["level"]) {
    switch (level) {
        case "strong":
            return "border-[var(--app-success-border)] bg-[var(--app-success-surface)] text-[var(--app-success-text)]";
        case "medium":
            return "border-[var(--app-brand-border)] bg-[var(--app-brand-surface)] text-[var(--app-brand-text)]";
        case "borderline":
            return "border-[var(--app-warning-border)] bg-[var(--app-warning-surface)] text-[var(--app-warning-text)]";
        case "weak":
            return "border-[var(--app-warning-border)] bg-[var(--app-warning-surface)] text-[var(--app-warning-text)]";
        case "negative":
        default:
            return "border-[var(--app-border)] bg-[var(--app-surface-muted)] text-[var(--app-text)]";
    }
}

export default function VerifyConfidenceBand({ band }: VerifyConfidenceBandProps) {
    if (!band) {
        return (
            <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Confidence band</p>
                <p className="mt-2 text-sm text-[var(--app-text-soft)]">Confidence becomes available after a scored verify result.</p>
            </section>
        );
    }

    return (
        <section className={`rounded-2xl border p-5 ${paletteFor(band.level)}`}>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] opacity-70">Confidence band</p>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                <div>
                    <p className="text-xl font-semibold">{band.label}</p>
                    <p className="mt-1 text-sm opacity-90">{band.summary}</p>
                </div>
                <div className="rounded-2xl border border-current/15 bg-[var(--app-surface)] px-4 py-3 text-right text-sm">
                    <div>Score {band.score.toFixed(4)}</div>
                    <div className="mt-1 opacity-80">Threshold {band.threshold.toFixed(4)}</div>
                </div>
            </div>
        </section>
    );
}
