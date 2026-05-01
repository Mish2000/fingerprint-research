import { BarChart3, GitBranch, ShieldCheck } from "lucide-react";
import { formatMethodLabel } from "../../../shared/storytelling.ts";
import type { IdentificationRetrievalMethod, IdentifyResponse, Method } from "../../../types/index.ts";

interface LiveEvidenceStripProps {
    result: IdentifyResponse | null;
    retrievalMethod: IdentificationRetrievalMethod;
    rerankMethod: Method;
}

function EvidenceItem({
    icon: Icon,
    label,
    value,
    detail,
}: {
    icon: typeof GitBranch;
    label: string;
    value: string;
    detail: string;
}) {
    return (
        <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-3">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-2 text-slate-600">
                    <Icon className="h-4 w-4" />
                </div>
                <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">{label}</p>
                    <p className="text-sm font-semibold text-slate-900">{value}</p>
                </div>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-600">{detail}</p>
        </article>
    );
}

export default function LiveEvidenceStrip({
    result,
    retrievalMethod,
    rerankMethod,
}: LiveEvidenceStripProps) {
    const activeRetrievalMethod = result?.retrieval_method ?? retrievalMethod;
    const activeRerankMethod = result?.rerank_method ?? rerankMethod;
    const candidatePool = result ? `${result.candidate_pool_size} enrolled` : "Pending run";

    return (
        <section className="grid gap-3 md:grid-cols-3" aria-label="Live demo evidence">
            <EvidenceItem
                icon={GitBranch}
                label="Method used"
                value={`${formatMethodLabel(activeRetrievalMethod)} + ${formatMethodLabel(activeRerankMethod)}`}
                detail={`${candidatePool}; shortlist target ${result?.shortlist_size ?? 10}. The demo uses the existing identify service path.`}
            />
            <EvidenceItem
                icon={BarChart3}
                label="Evidence reminder"
                value="Benchmark before rollout"
                detail="Treat live scores as a demo signal and compare operating thresholds against the benchmark workspace before stakeholder commitments."
            />
            <EvidenceItem
                icon={ShieldCheck}
                label="Privacy / template note"
                value="Prefer templates over raw captures"
                detail="Production scanner wiring should keep retention policy explicit and match against generated templates rather than treating raw images as durable records."
            />
        </section>
    );
}
