import CandidateShortlistStoryPanel from "./CandidateShortlistStoryPanel.tsx";
import IdentificationLatencySummary from "./IdentificationLatencySummary.tsx";
import IdentificationOutcomeStoryHeader from "./IdentificationOutcomeStoryHeader.tsx";
import NoMatchNarrativeState from "./NoMatchNarrativeState.tsx";
import VerifyConfidenceBand from "../../verify/components/VerifyConfidenceBand.tsx";
import type { IdentificationStoryState } from "../storyModel.ts";
import type { IdentifyCandidate } from "../../../types/index.ts";

interface IdentificationOutcomeStoryPanelProps {
    story: IdentificationStoryState;
    candidates?: IdentifyCandidate[];
}

export default function IdentificationOutcomeStoryPanel({
    story,
    candidates = [],
}: IdentificationOutcomeStoryPanelProps) {
    return (
        <div className="space-y-4">
            <IdentificationOutcomeStoryHeader
                headline={story.headline}
                meaning={story.meaning}
                contextLabel={story.contextLabel}
            />

            <div className="grid gap-4 xl:grid-cols-2">
                <section className="rounded-2xl border border-slate-200 bg-white p-5">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Probe context</p>
                    <p className="mt-2 text-base font-semibold text-slate-900">{story.probeContext.label}</p>
                    <p className="mt-2 text-sm leading-6 text-slate-600">{story.probeContext.summary}</p>
                    {story.probeContext.details.length > 0 ? (
                        <div className="mt-3 flex flex-wrap gap-2">
                            {story.probeContext.details.map((detail) => (
                                <span key={detail} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-700">
                                    {detail}
                                </span>
                            ))}
                        </div>
                    ) : null}
                </section>

                <VerifyConfidenceBand band={story.confidenceBand} />

                <section className="rounded-2xl border border-slate-200 bg-white p-5">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Method story</p>
                    <p className="mt-2 text-sm leading-6 text-slate-600">{story.methodStory}</p>
                </section>

                <IdentificationLatencySummary latency={story.latency} />
            </div>

            {candidates.length > 0 ? (
                <CandidateShortlistStoryPanel candidates={candidates} />
            ) : (
                <NoMatchNarrativeState />
            )}
        </div>
    );
}
