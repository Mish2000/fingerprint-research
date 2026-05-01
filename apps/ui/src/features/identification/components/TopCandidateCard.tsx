import { ShieldAlert, ShieldCheck } from "lucide-react";
import type { IdentifyCandidate } from "../../../types/index.ts";

interface TopCandidateCardProps {
    candidate: IdentifyCandidate;
    accepted: boolean;
}

export default function TopCandidateCard({ candidate, accepted }: TopCandidateCardProps) {
    const toneClassName = accepted
        ? "border-emerald-200 bg-emerald-50 text-emerald-900"
        : "border-amber-200 bg-amber-50 text-amber-900";

    return (
        <div className={`rounded-2xl border p-5 ${toneClassName}`}>
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] opacity-70">Top candidate</p>
                    <h3 className="mt-1 text-xl font-semibold">{candidate.full_name}</h3>
                    <p className="mt-2 text-sm opacity-80">
                        Rank {candidate.rank} · {candidate.random_id} · {candidate.national_id_masked} · {candidate.capture}
                    </p>
                </div>
                <div className="rounded-full border border-current/20 bg-white/60 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em]">
                    {accepted ? "MATCH" : "NO MATCH"}
                </div>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <div className="rounded-xl bg-white/70 px-4 py-3 text-sm">
                    <p className="text-xs font-medium uppercase tracking-[0.14em] opacity-60">Retrieval score</p>
                    <p className="mt-1 font-semibold">{candidate.retrieval_score.toFixed(4)}</p>
                </div>
                <div className="rounded-xl bg-white/70 px-4 py-3 text-sm">
                    <p className="text-xs font-medium uppercase tracking-[0.14em] opacity-60">Re-rank score</p>
                    <p className="mt-1 font-semibold">
                        {typeof candidate.rerank_score === "number" ? candidate.rerank_score.toFixed(4) : "-"}
                    </p>
                </div>
                <div className="rounded-xl bg-white/70 px-4 py-3 text-sm">
                    <p className="text-xs font-medium uppercase tracking-[0.14em] opacity-60">Decision</p>
                    <p className="mt-1 flex items-center gap-2 font-semibold">
                        {candidate.decision ? <ShieldCheck className="h-4 w-4" /> : <ShieldAlert className="h-4 w-4" />}
                        {candidate.decision ? "Accepted" : "Rejected"}
                    </p>
                </div>
            </div>
        </div>
    );
}
