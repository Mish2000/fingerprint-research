import type { VerifyStoryState } from "../storyModel.ts";
import VerifyCaseContextSummary from "./VerifyCaseContextSummary.tsx";
import VerifyConfidenceBand from "./VerifyConfidenceBand.tsx";
import VerifyDifficultySummary from "./VerifyDifficultySummary.tsx";
import VerifyExpectationSummary from "./VerifyExpectationSummary.tsx";
import VerifyLatencySummary from "./VerifyLatencySummary.tsx";
import VerifyOutcomeNarrative from "./VerifyOutcomeNarrative.tsx";
import VerifyOutcomeStoryHeader from "./VerifyOutcomeStoryHeader.tsx";

interface VerifyOutcomeStoryPanelProps {
    story: VerifyStoryState;
}

export default function VerifyOutcomeStoryPanel({ story }: VerifyOutcomeStoryPanelProps) {
    return (
        <div className="space-y-4">
            <VerifyOutcomeStoryHeader
                headline={story.headline}
                meaning={story.meaning}
                contextLabel={story.contextLabel}
            />
            <div className="grid gap-4 xl:grid-cols-2">
                <VerifyCaseContextSummary context={story.caseContext} />
                <VerifyConfidenceBand band={story.confidenceBand} />
                <VerifyExpectationSummary expectation={story.expectation} />
                <VerifyLatencySummary latency={story.latency} />
                <VerifyDifficultySummary difficulty={story.difficulty} />
                <VerifyOutcomeNarrative methodStory={story.methodStory} />
            </div>
        </div>
    );
}
