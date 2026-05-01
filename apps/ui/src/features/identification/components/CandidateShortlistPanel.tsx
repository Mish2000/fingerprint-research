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
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
            <div className="border-b border-slate-100 px-6 py-5">
                <h3 className="text-lg font-semibold text-slate-900">Candidate shortlist</h3>
                <p className="mt-1 text-sm text-slate-500">Top candidates surfaced from the official 1:N response, including retrieval and re-rank scores.</p>
            </div>

            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                    <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
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
                    <tbody className="divide-y divide-slate-100 bg-white">
                        {candidates.map((candidate) => (
                            <tr key={`${candidate.random_id}_${candidate.rank}`} className="hover:bg-slate-50">
                                <td className="px-4 py-3 font-medium text-slate-900">{candidate.rank}</td>
                                <td className="px-4 py-3">
                                    <div className="font-medium text-slate-900">{candidate.full_name}</div>
                                    <div className="text-xs text-slate-500">{candidate.random_id}</div>
                                </td>
                                <td className="px-4 py-3 text-slate-600">{candidate.national_id_masked}</td>
                                <td className="px-4 py-3 text-slate-600">{candidate.capture}</td>
                                <td className="px-4 py-3 text-right text-slate-900">{candidate.retrieval_score.toFixed(4)}</td>
                                <td className="px-4 py-3 text-right text-slate-900">
                                    {typeof candidate.rerank_score === "number" ? candidate.rerank_score.toFixed(4) : "-"}
                                </td>
                                <td className="px-4 py-3 text-center">
                                    {candidate.decision === true ? (
                                        <ShieldCheck className="mx-auto h-4 w-4 text-emerald-600" />
                                    ) : candidate.decision === false ? (
                                        <ShieldAlert className="mx-auto h-4 w-4 text-amber-600" />
                                    ) : (
                                        <span className="text-slate-400">-</span>
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
