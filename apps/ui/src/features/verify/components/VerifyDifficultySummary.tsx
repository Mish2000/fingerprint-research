import type { VerifyDifficultyState } from "../storyModel.ts";

interface VerifyDifficultySummaryProps {
    difficulty: VerifyDifficultyState | null;
}

export default function VerifyDifficultySummary({ difficulty }: VerifyDifficultySummaryProps) {
    if (!difficulty) {
        return null;
    }

    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Difficulty</p>
            <p className="mt-2 text-base font-semibold text-slate-900">{difficulty.label}</p>
            <p className="mt-2 text-sm leading-6 text-slate-600">{difficulty.summary}</p>
        </section>
    );
}
