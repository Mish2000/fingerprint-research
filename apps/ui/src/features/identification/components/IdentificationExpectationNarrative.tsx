import type { DemoExpectationSummary } from "../model.ts";

interface IdentificationExpectationNarrativeProps {
    summary: DemoExpectationSummary;
}

export default function IdentificationExpectationNarrative({ summary }: IdentificationExpectationNarrativeProps) {
    const toneClassName = summary.status === "aligned"
        ? "border-emerald-200 bg-emerald-50 text-emerald-900"
        : summary.status === "mismatch"
            ? "border-amber-200 bg-amber-50 text-amber-900"
            : "border-slate-200 bg-white text-slate-900";

    const message = summary.status === "pending"
        ? "Run the probe to compare the expected outcome with the actual result."
        : summary.status === "no_expectation"
            ? "This probe does not carry expected outcome metadata."
            : summary.status === "aligned"
                ? "Expected and actual outcomes are aligned."
                : "Expected and actual outcomes are not aligned.";

    return (
        <section className={`rounded-2xl border p-5 ${toneClassName}`}>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] opacity-70">Expected vs actual</p>
            <div className="mt-3 space-y-2 text-sm">
                <div className="flex items-start justify-between gap-4 rounded-xl bg-white/70 px-4 py-3">
                    <span className="font-medium opacity-80">Expected</span>
                    <span className="text-right font-semibold">
                        {summary.expectedOutcome ?? "Unavailable"}
                        {summary.expectedTopIdentityLabel ? ` / ${summary.expectedTopIdentityLabel}` : ""}
                    </span>
                </div>
                <div className="flex items-start justify-between gap-4 rounded-xl bg-white/70 px-4 py-3">
                    <span className="font-medium opacity-80">Actual</span>
                    <span className="text-right font-semibold">
                        {summary.actualOutcome ?? "Pending run"}
                        {summary.actualTopIdentityLabel ? ` / ${summary.actualTopIdentityLabel}` : ""}
                    </span>
                </div>
            </div>
            <p className="mt-3 text-sm leading-6 opacity-90">{message}</p>
        </section>
    );
}
