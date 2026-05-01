import RequestState from "../../../components/RequestState.tsx";
import InlineBanner from "../../../shared/ui/InlineBanner.tsx";
import type { AsyncState } from "../../../shared/request-state/index.ts";
import type { CatalogBrowserItem, IdentifyResponse } from "../../../types/index.ts";
import { createIdentificationStoryState } from "../storyModel.ts";
import IdentificationOutcomeStoryPanel from "./IdentificationOutcomeStoryPanel.tsx";
import TopCandidateHeroCard from "./TopCandidateHeroCard.tsx";

interface IdentificationBrowserResultPanelProps {
    resultState: AsyncState<IdentifyResponse>;
    lastProbeAsset: CatalogBrowserItem | null;
    datasetLabel: string | null;
    onRetry: () => void | Promise<void>;
}

export default function IdentificationBrowserResultPanel({
    resultState,
    lastProbeAsset,
    datasetLabel,
    onRetry,
}: IdentificationBrowserResultPanelProps) {
    if (resultState.status === "loading") {
        return (
            <RequestState
                variant="loading"
                title="Running browser identification"
                description="Seeding the selected gallery and identifying the selected browser probe."
            />
        );
    }

    if (resultState.status === "error" && resultState.error) {
        return (
            <RequestState
                variant="error"
                title="Browser identification failed"
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
                title="No browser identification result yet"
                description="Choose a gallery, choose a probe asset, then seed and run the browser flow."
            />
        );
    }

    const story = createIdentificationStoryState({ resultState, probeCase: null });

    return (
        <div className="space-y-5">
            <InlineBanner variant="success" title="Latest browser result">
                Probe {lastProbeAsset?.asset_id ?? "asset"} from {datasetLabel ?? "the selected dataset"} produced this result.
            </InlineBanner>

            {resultState.data.top_candidate ? (
                <TopCandidateHeroCard
                    candidate={resultState.data.top_candidate}
                    accepted={resultState.data.decision}
                    probeCaptureLabel={lastProbeAsset?.capture ?? null}
                />
            ) : null}
            <IdentificationOutcomeStoryPanel story={story} candidates={resultState.data.candidates} />
        </div>
    );
}
