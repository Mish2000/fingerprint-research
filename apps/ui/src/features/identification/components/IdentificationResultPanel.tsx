import RequestState from "../../../components/RequestState.tsx";
import InlineBanner from "../../../shared/ui/InlineBanner.tsx";
import type { AsyncState } from "../../../shared/request-state/index.ts";
import type { CatalogIdentifyProbeCase, IdentifyResponse } from "../../../types/index.ts";
import { createIdentificationStoryState } from "../storyModel.ts";
import DemoExpectationSummary from "./DemoExpectationSummary.tsx";
import IdentificationOutcomeStoryPanel from "./IdentificationOutcomeStoryPanel.tsx";
import TopCandidateHeroCard from "./TopCandidateHeroCard.tsx";
import { createDemoExpectationSummary } from "../model.ts";

interface IdentificationResultPanelProps {
    resultState: AsyncState<IdentifyResponse>;
    lastProbeCase: CatalogIdentifyProbeCase | null;
    onRetry: () => void | Promise<void>;
}

export default function IdentificationResultPanel({
    resultState,
    lastProbeCase,
    onRetry,
}: IdentificationResultPanelProps) {
    if (resultState.status === "loading") {
        return (
            <RequestState
                variant="loading"
                title="Running guided identification"
                description="Loading the selected probe and waiting for the official identify endpoint."
            />
        );
    }

    if (resultState.status === "error" && resultState.error) {
        return (
            <RequestState
                variant="error"
                title="Guided identification failed"
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
                title="No identification result yet"
                description="Seed the demo store, choose a probe, then run identification to see the story."
            />
        );
    }

    const result = resultState.data;
    const story = createIdentificationStoryState({ resultState, probeCase: lastProbeCase });
    const expectation = createDemoExpectationSummary(lastProbeCase, result);

    return (
        <div className="space-y-5">
            <InlineBanner variant="success" title="Latest demo result">
                The latest guided run is attached to {lastProbeCase?.title ?? "the selected probe"}.
            </InlineBanner>

            {result.top_candidate ? (
                <TopCandidateHeroCard
                    candidate={result.top_candidate}
                    accepted={result.decision}
                    probeCaptureLabel={lastProbeCase?.capture ?? null}
                />
            ) : null}
            <DemoExpectationSummary summary={expectation} />
            <IdentificationOutcomeStoryPanel story={story} candidates={result.candidates} />
        </div>
    );
}
