import { ShieldAlert, ShieldCheck, UserRoundSearch } from "lucide-react";
import { formatCaptureLabel } from "../../../shared/storytelling.ts";
import type { IdentifyCandidate } from "../../../types/index.ts";

interface TopCandidateHeroCardProps {
    candidate: IdentifyCandidate;
    accepted: boolean;
    probeCaptureLabel?: string | null;
}

export default function TopCandidateHeroCard({
    candidate,
    accepted,
    probeCaptureLabel,
}: TopCandidateHeroCardProps) {
    const candidateCapture = formatCaptureLabel(candidate.capture);
    const crossCaptureNote = probeCaptureLabel && candidateCapture !== probeCaptureLabel
        ? `${probeCaptureLabel} probe against ${candidateCapture} candidate capture`
        : `${candidateCapture} gallery capture`;

    return (
        <section
            className={`rounded-3xl border p-6 ${
                accepted
                    ? "border-emerald-200 bg-emerald-50 text-emerald-950"
                    : "border-amber-200 bg-amber-50 text-amber-950"
            }`}
        >
            <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="max-w-3xl">
                    <div className="inline-flex items-center gap-2 rounded-full border border-current/15 bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em]">
                        <UserRoundSearch className="h-3.5 w-3.5" />
                        Top candidate
                    </div>
                    <h3 className="mt-4 text-3xl font-semibold">{candidate.full_name}</h3>
                    <p className="mt-2 text-sm leading-6 opacity-90">
                        Rank {candidate.rank} / {candidate.random_id} / {candidate.national_id_masked}
                    </p>
                    <p className="mt-2 text-sm leading-6 opacity-90">{crossCaptureNote}</p>
                </div>
                <div className="rounded-2xl border border-current/15 bg-white/70 px-4 py-3 text-sm font-semibold uppercase tracking-[0.14em]">
                    {accepted ? "Accepted" : "Rejected"}
                </div>
            </div>

            <div className="mt-5 grid gap-3 md:grid-cols-3">
                <div className="rounded-2xl border border-white/70 bg-white/70 p-4 text-sm">
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] opacity-60">Retrieval score</p>
                    <p className="mt-2 text-lg font-semibold">{candidate.retrieval_score.toFixed(4)}</p>
                </div>
                <div className="rounded-2xl border border-white/70 bg-white/70 p-4 text-sm">
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] opacity-60">Re-rank score</p>
                    <p className="mt-2 text-lg font-semibold">
                        {typeof candidate.rerank_score === "number" ? candidate.rerank_score.toFixed(4) : "-"}
                    </p>
                </div>
                <div className="rounded-2xl border border-white/70 bg-white/70 p-4 text-sm">
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] opacity-60">Decision</p>
                    <p className="mt-2 flex items-center gap-2 text-lg font-semibold">
                        {accepted ? <ShieldCheck className="h-5 w-5" /> : <ShieldAlert className="h-5 w-5" />}
                        {accepted ? "Match" : "No match"}
                    </p>
                </div>
            </div>
        </section>
    );
}
