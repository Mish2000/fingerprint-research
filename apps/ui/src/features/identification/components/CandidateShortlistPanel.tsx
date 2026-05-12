import { ShieldAlert, ShieldCheck } from "lucide-react";
import RequestState from "../../../components/RequestState.tsx";
import type { IdentifyCandidate } from "../../../types/index.ts";

interface CandidateShortlistPanelProps {
    candidates: IdentifyCandidate[];
}

export default function CandidateShortlistPanel({ candidates }: CandidateShortlistPanelProps) {
    if (candidates.length === 0) {
        return (
            <RequestState
                variant="empty"
                title="No shortlist candidates"
                description="The search completed, but the backend did not return any shortlist rows to display."
            />
        );
    }

    return (
        <div className="overflow-hidden rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)]">
            <div className="border-b border-[var(--app-border-muted)] px-6 py-5">
                <h3 className="text-lg font-semibold text-[var(--app-text)]">Candidate shortlist</h3>
                <p className="mt-1 text-sm text-[var(--app-text-muted)]">Top candidates surfaced from the official 1:N response, including retrieval and re-rank scores.</p>
            </div>

            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-[var(--app-border)] text-left text-sm">
                    <thead className="bg-[var(--app-surface-subtle)] text-xs uppercase tracking-wide text-[var(--app-text-muted)]">
                        <tr>
                            <th className="px-4 py-3">Rank</th>
                            <th className="px-4 py-3">Person</th>
                            <th className="px-4 py-3">Masked ID</th>
                            <th className="px-4 py-3">Capture</th>
                            <th className="px-4 py-3 text-right">Retrieval</th>
                            <th className="px-4 py-3 text-right">Re-rank</th>
                            <th className="px-4 py-3 text-center">Decision</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--app-border-muted)] bg-[var(--app-surface)]">
                        {candidates.map((candidate) => (
                            <tr key={`${candidate.random_id}_${candidate.rank}`} className="hover:bg-[var(--app-surface-subtle)]">
                                <td className="px-4 py-3 font-medium text-[var(--app-text)]">{candidate.rank}</td>
                                <td className="px-4 py-3">
                                    <div className="font-medium text-[var(--app-text)]">{candidate.full_name}</div>
                                    <div className="text-xs text-[var(--app-text-muted)]">{candidate.random_id}</div>
                                </td>
                                <td className="px-4 py-3 text-[var(--app-text-soft)]">{candidate.national_id_masked}</td>
                                <td className="px-4 py-3 text-[var(--app-text-soft)]">{candidate.capture}</td>
                                <td className="px-4 py-3 text-right text-[var(--app-text)]">{candidate.retrieval_score.toFixed(4)}</td>
                                <td className="px-4 py-3 text-right text-[var(--app-text)]">
                                    {typeof candidate.rerank_score === "number" ? candidate.rerank_score.toFixed(4) : "-"}
                                </td>
                                <td className="px-4 py-3 text-center">
                                    {candidate.decision === true ? (
                                        <ShieldCheck className="mx-auto h-4 w-4 text-[var(--app-success-text)]" />
                                    ) : candidate.decision === false ? (
                                        <ShieldAlert className="mx-auto h-4 w-4 text-[var(--app-warning-text)]" />
                                    ) : (
                                        <span className="text-[var(--app-text-muted)]">-</span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
