import RequestState from "../../../components/RequestState.tsx";
import { formatCaptureLabel } from "../../../shared/storytelling.ts";
import type { IdentifyCandidate } from "../../../types/index.ts";

interface CandidateShortlistStoryPanelProps {
    candidates: IdentifyCandidate[];
}

export default function CandidateShortlistStoryPanel({ candidates }: CandidateShortlistStoryPanelProps) {
    if (candidates.length === 0) {
        return (
            <RequestState
                variant="empty"
                title="No shortlist candidates"
                description="The search completed, but the backend did not return any shortlist rows to narrate."
            />
        );
    }

    return (
        <section className="space-y-4">
            <div className="rounded-2xl border border-slate-200 bg-white p-5">
                <h3 className="text-lg font-semibold text-slate-900">Candidate ranking</h3>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                    Story-friendly shortlist of the top returned candidates, including retrieval, re-rank, capture, and decision state.
                </p>
            </div>

            <div className="grid gap-4 xl:grid-cols-2">
                {candidates.map((candidate) => (
                    <article key={`${candidate.random_id}_${candidate.rank}`} className="rounded-2xl border border-slate-200 bg-white p-5">
                        <div className="flex items-start justify-between gap-3">
                            <div>
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Rank {candidate.rank}</p>
                                <h4 className="mt-2 text-lg font-semibold text-slate-900">{candidate.full_name}</h4>
                                <p className="mt-1 text-sm text-slate-600">{candidate.random_id} / {candidate.national_id_masked}</p>
                            </div>
                            <span className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] ${
                                candidate.decision
                                    ? "bg-emerald-100 text-emerald-800"
                                    : "bg-slate-100 text-slate-700"
                            }`}
                            >
                                {candidate.decision ? "Accepted" : "Not accepted"}
                            </span>
                        </div>

                        <div className="mt-4 grid gap-3 sm:grid-cols-3">
                            <div className="rounded-xl bg-slate-50 px-4 py-3 text-sm">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Capture</p>
                                <p className="mt-2 font-semibold text-slate-900">{formatCaptureLabel(candidate.capture)}</p>
                            </div>
                            <div className="rounded-xl bg-slate-50 px-4 py-3 text-sm">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Retrieval</p>
                                <p className="mt-2 font-semibold text-slate-900">{candidate.retrieval_score.toFixed(4)}</p>
                            </div>
                            <div className="rounded-xl bg-slate-50 px-4 py-3 text-sm">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Re-rank</p>
                                <p className="mt-2 font-semibold text-slate-900">
                                    {typeof candidate.rerank_score === "number" ? candidate.rerank_score.toFixed(4) : "-"}
                                </p>
                            </div>
                        </div>
                    </article>
                ))}
            </div>
        </section>
    );
}
