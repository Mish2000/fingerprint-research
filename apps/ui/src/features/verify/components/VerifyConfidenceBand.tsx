import type { ConfidenceBand } from "../../../shared/storytelling.ts";

interface VerifyConfidenceBandProps {
    band: ConfidenceBand | null;
}

function paletteFor(level: ConfidenceBand["level"]) {
    switch (level) {
        case "strong":
            return "border-emerald-200 bg-emerald-50 text-emerald-900";
        case "medium":
            return "border-brand-200 bg-brand-50 text-brand-900";
        case "borderline":
            return "border-amber-200 bg-amber-50 text-amber-900";
        case "weak":
            return "border-amber-200 bg-amber-50 text-amber-900";
        case "negative":
        default:
            return "border-slate-200 bg-slate-100 text-slate-900";
    }
}

export default function VerifyConfidenceBand({ band }: VerifyConfidenceBandProps) {
    if (!band) {
        return (
            <section className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Confidence band</p>
                <p className="mt-2 text-sm text-slate-600">Confidence becomes available after a scored verify result.</p>
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
                <div className="rounded-2xl border border-current/15 bg-white/70 px-4 py-3 text-right text-sm">
                    <div>Score {band.score.toFixed(4)}</div>
                    <div className="mt-1 opacity-80">Threshold {band.threshold.toFixed(4)}</div>
                </div>
            </div>
        </section>
    );
}
