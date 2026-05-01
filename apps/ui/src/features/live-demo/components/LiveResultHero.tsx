import { CheckCircle2, Clock, Gauge, ShieldAlert, UserRoundSearch, XCircle } from "lucide-react";
import RequestState from "../../../components/RequestState.tsx";
import type { AsyncState } from "../../../shared/request-state/index.ts";
import { formatCaptureLabel, formatLatency, formatMethodLabel } from "../../../shared/storytelling.ts";
import type { IdentifyCandidate, IdentifyResponse, LatencyBreakdown } from "../../../types/index.ts";

interface LiveResultHeroProps {
    resultState: AsyncState<IdentifyResponse>;
    sourceFileName: string | null;
    onRetry: () => void | Promise<void>;
}

function formatNumber(value: number | null | undefined, digits = 4): string {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "-";
    }

    return value.toFixed(digits);
}

function readCandidateScore(candidate: IdentifyCandidate | null | undefined): number | null {
    if (!candidate) {
        return null;
    }

    return typeof candidate.rerank_score === "number" && !Number.isNaN(candidate.rerank_score)
        ? candidate.rerank_score
        : candidate.retrieval_score;
}

function readTotalLatency(latency: LatencyBreakdown): number | null {
    const total = latency.total_ms;
    if (typeof total === "number" && !Number.isNaN(total)) {
        return total;
    }

    const values = Object.values(latency).filter((value) => typeof value === "number" && !Number.isNaN(value));
    if (values.length === 0) {
        return null;
    }

    return values.reduce((sum, value) => sum + value, 0);
}

function StatTile({
    icon: Icon,
    label,
    value,
}: {
    icon: typeof Gauge;
    label: string;
    value: string;
}) {
    return (
        <div className="rounded-2xl border border-white/70 bg-white/75 p-4">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] opacity-65">
                <Icon className="h-4 w-4" />
                {label}
            </div>
            <p className="mt-2 text-lg font-semibold">{value}</p>
        </div>
    );
}

export default function LiveResultHero({ resultState, sourceFileName, onRetry }: LiveResultHeroProps) {
    if (resultState.status === "loading") {
        return (
            <RequestState
                variant="loading"
                title="Running fingerprint identification"
                description="The uploaded fingerprint is being compared against the enrolled gallery."
            />
        );
    }

    if (resultState.status === "error" && resultState.error) {
        return (
            <RequestState
                variant="error"
                title="Live demo run failed"
                description={resultState.error}
                actionLabel="Try again"
                onAction={() => {
                    void onRetry();
                }}
            />
        );
    }

    if (resultState.status !== "success" || !resultState.data) {
        return (
            <RequestState
                variant="empty"
                title="Result hero ready"
                description="Upload a fingerprint and run Identify 1:N to populate score, threshold, decision, latency, and candidate details."
            />
        );
    }

    const result = resultState.data;
    const candidate = result.top_candidate ?? null;
    const score = readCandidateScore(candidate);
    const latency = readTotalLatency(result.latency_ms);
    const isAccepted = result.decision;

    return (
        <section
            className={`rounded-2xl border p-6 shadow-sm ${
                isAccepted
                    ? "border-emerald-200 bg-emerald-50 text-emerald-950"
                    : "border-amber-200 bg-amber-50 text-amber-950"
            }`}
        >
            <div className="flex flex-wrap items-start justify-between gap-5">
                <div className="max-w-3xl">
                    <div className="inline-flex items-center gap-2 rounded-full border border-current/15 bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em]">
                        {isAccepted ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                        Result decision
                    </div>
                    <h3 className="mt-4 text-3xl font-semibold">
                        {isAccepted ? "Match candidate accepted" : "No candidate accepted"}
                    </h3>
                    <p className="mt-2 text-sm leading-6 opacity-85">
                        {candidate
                            ? `Top candidate ${candidate.full_name} ranked #${candidate.rank} from ${result.candidates.length} shortlisted candidate${result.candidates.length === 1 ? "" : "s"}.`
                            : `No top candidate returned from ${result.candidate_pool_size} eligible enrolled record${result.candidate_pool_size === 1 ? "" : "s"}.`}
                    </p>
                    {sourceFileName ? (
                        <p className="mt-1 text-sm leading-6 opacity-75">Source: {sourceFileName}</p>
                    ) : null}
                </div>

                <div className="min-w-40 rounded-2xl border border-current/15 bg-white/75 px-5 py-4 text-right">
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] opacity-60">Score</p>
                    <p className="mt-2 text-4xl font-semibold">{formatNumber(score)}</p>
                </div>
            </div>

            <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <StatTile icon={Gauge} label="Threshold" value={formatNumber(result.threshold, 4)} />
                <StatTile icon={Clock} label="Latency" value={formatLatency(latency)} />
                <StatTile
                    icon={UserRoundSearch}
                    label="Top candidate"
                    value={candidate ? candidate.full_name : "None"}
                />
                <StatTile
                    icon={ShieldAlert}
                    label="Decision"
                    value={isAccepted ? "Accept" : "Review / reject"}
                />
            </div>

            {candidate ? (
                <div className="mt-5 grid gap-3 md:grid-cols-3">
                    <div className="rounded-2xl border border-white/70 bg-white/75 p-4">
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] opacity-60">Identity</p>
                        <p className="mt-2 text-sm font-semibold">{candidate.random_id}</p>
                        <p className="mt-1 text-xs opacity-70">{candidate.national_id_masked}</p>
                    </div>
                    <div className="rounded-2xl border border-white/70 bg-white/75 p-4">
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] opacity-60">Capture</p>
                        <p className="mt-2 text-sm font-semibold">{formatCaptureLabel(candidate.capture)}</p>
                    </div>
                    <div className="rounded-2xl border border-white/70 bg-white/75 p-4">
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] opacity-60">Method</p>
                        <p className="mt-2 text-sm font-semibold">
                            {formatMethodLabel(result.retrieval_method)} + {formatMethodLabel(result.rerank_method)}
                        </p>
                    </div>
                </div>
            ) : null}
        </section>
    );
}
