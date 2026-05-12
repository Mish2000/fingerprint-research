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
            <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5">
                <h3 className="text-lg font-semibold text-[var(--app-text)]">Candidate ranking</h3>
                <p className="mt-2 text-sm leading-6 text-[var(--app-text-soft)]">
                    Story-friendly shortlist of the top returned candidates, including retrieval, re-rank, capture, and decision state.
                </p>
            </div>

            <div className="grid gap-4 xl:grid-cols-2">
                {candidates.map((candidate) => (
                    <article key={`${candidate.random_id}_${candidate.rank}`} className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5">
                        <div className="flex items-start justify-between gap-3">
                            <div>
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">Rank {candidate.rank}</p>
                                <h4 className="mt-2 text-lg font-semibold text-[var(--app-text)]">{candidate.full_name}</h4>
                                <p className="mt-1 text-sm text-[var(--app-text-soft)]">{candidate.random_id} / {candidate.national_id_masked}</p>
                            </div>
                            <span className={`status-pill uppercase tracking-[0.14em] ${
                                candidate.decision
                                    ? "status-pill--success"
                                    : ""
                            }`}
                            >
                                {candidate.decision ? "Accepted" : "Not accepted"}
                            </span>
                        </div>

                        <div className="mt-4 grid gap-3 sm:grid-cols-3">
                            <div className="rounded-xl bg-[var(--app-surface-subtle)] px-4 py-3 text-sm">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">Capture</p>
                                <p className="mt-2 font-semibold text-[var(--app-text)]">{formatCaptureLabel(candidate.capture)}</p>
                            </div>
                            <div className="rounded-xl bg-[var(--app-surface-subtle)] px-4 py-3 text-sm">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">Retrieval</p>
                                <p className="mt-2 font-semibold text-[var(--app-text)]">{candidate.retrieval_score.toFixed(4)}</p>
                            </div>
                            <div className="rounded-xl bg-[var(--app-surface-subtle)] px-4 py-3 text-sm">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">Re-rank</p>
                                <p className="mt-2 font-semibold text-[var(--app-text)]">
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
