import type { AsyncState } from "../../shared/request-state/index.ts";
import {
    deriveConfidenceBand,
    formatCaptureLabel,
    formatLatency,
    formatMethodLabel,
    formatWords,
    humanizeLatencyKey,
    type ConfidenceBand,
} from "../../shared/storytelling.ts";
import type { MatchResponse } from "../../types/index.ts";
import type { VerifyRunContext } from "./model.ts";

export interface VerifyExpectationState {
    expectedLabel: string | null;
    actualLabel: string | null;
    summary: string;
}

export interface VerifyLatencyState {
    totalLabel: string;
    summary: string;
    breakdown: Array<{
        label: string;
        value: string;
    }>;
}

export interface VerifyCaseContextState {
    label: string;
    summary: string;
    details: string[];
}

export interface VerifyDifficultyState {
    label: string;
    summary: string;
}

export interface VerifyStoryState {
    headline: string;
    meaning: string;
    contextLabel: string;
    methodStory: string;
    expectation: VerifyExpectationState;
    latency: VerifyLatencyState | null;
    confidenceBand: ConfidenceBand | null;
    caseContext: VerifyCaseContextState;
    difficulty: VerifyDifficultyState | null;
    storyErrorState: string | null;
}

function actualDecisionLabel(result: MatchResponse | null): string | null {
    if (!result) {
        return null;
    }

    return result.decision ? "Genuine" : "Impostor";
}

function expectedDecisionLabel(context: VerifyRunContext | null): string | null {
    if (!context?.groundTruth) {
        return null;
    }

    const normalized = context.groundTruth.toLowerCase();
    if (normalized === "match") {
        return "Genuine";
    }
    if (normalized === "non_match") {
        return "Impostor";
    }

    return formatWords(context.groundTruth);
}

function createExpectationState(
    result: MatchResponse | null,
    context: VerifyRunContext | null,
): VerifyExpectationState {
    const expectedLabel = expectedDecisionLabel(context);
    const actualLabel = actualDecisionLabel(result);

    if (!expectedLabel) {
        return {
            expectedLabel: null,
            actualLabel,
            summary: "This run does not include catalog expectation metadata.",
        };
    }

    if (!actualLabel) {
        return {
            expectedLabel,
            actualLabel: null,
            summary: "Run the pair to compare the expected label with the actual decision.",
        };
    }

    return {
        expectedLabel,
        actualLabel,
        summary: expectedLabel === actualLabel
            ? "Expected and actual verification outcomes are aligned."
            : "Expected and actual verification outcomes diverge for this run.",
    };
}

function createLatencyState(result: MatchResponse | null): VerifyLatencyState | null {
    if (!result) {
        return null;
    }

    const metaLatency = result.meta.latency_breakdown_ms ?? {};
    const breakdown = Object.entries(metaLatency)
        .filter((entry): entry is [string, number] => typeof entry[1] === "number" && !Number.isNaN(entry[1]))
        .map(([key, value]) => ({
            label: humanizeLatencyKey(key),
            value: formatLatency(value),
        }));

    return {
        totalLabel: formatLatency(result.latency_ms),
        summary: breakdown.length > 0
            ? "Backend returned a total latency plus a method-specific breakdown."
            : "Backend returned total latency for this verification request.",
        breakdown,
    };
}

function createCaseContextState(context: VerifyRunContext | null): VerifyCaseContextState {
    const captureA = formatCaptureLabel(context?.captureA);
    const captureB = formatCaptureLabel(context?.captureB);
    const label = captureA !== "-" || captureB !== "-" ? `${captureA} vs ${captureB}` : "Pair context";

    return {
        label,
        summary: context?.subtitle ?? "The story is attached to the current verify run context.",
        details: [
            context?.datasetLabel ? `Dataset: ${context.datasetLabel}` : null,
            context?.split ? `Split: ${context.split}` : null,
            context?.assetAId && context.assetBId ? `Assets: ${context.assetAId} vs ${context.assetBId}` : null,
            context?.probeFileName ? `Probe: ${context.probeFileName}` : null,
            context?.referenceFileName ? `Reference: ${context.referenceFileName}` : null,
        ].filter((item): item is string => item !== null),
    };
}

function createDifficultyState(context: VerifyRunContext | null): VerifyDifficultyState | null {
    if (!context?.difficulty && !context?.caseType && !context?.selectionReason) {
        return null;
    }

    return {
        label: formatWords(context.difficulty ?? context.caseType ?? "Case context"),
        summary: context.selectionReason ?? context.description ?? "Catalog metadata describes why this pair was selected.",
    };
}

export function createVerifyStoryState({
    resultState,
    context,
}: {
    resultState: AsyncState<MatchResponse>;
    context: VerifyRunContext | null;
}): VerifyStoryState {
    const result = resultState.status === "success" ? resultState.data : null;
    const headline = result?.decision ? "Same finger" : "Different fingers";

    return {
        headline,
        meaning: result
            ? result.decision
                ? "The score cleared the active threshold for this pair."
                : "The score stayed below the active threshold for this pair."
            : "Run verification to turn the selected pair into a narrated result.",
        contextLabel: context?.mode === "demo"
            ? "Demo verify result"
            : context?.mode === "browser"
                ? "Browser verify result"
                : "Verify result",
        methodStory: result
            ? `${formatMethodLabel(result.method)} scored ${result.score.toFixed(4)} against threshold ${result.threshold.toFixed(4)}.`
            : "Method story appears after a completed verification response.",
        expectation: createExpectationState(result, context),
        latency: createLatencyState(result),
        confidenceBand: result ? deriveConfidenceBand(result) : null,
        caseContext: createCaseContextState(context),
        difficulty: createDifficultyState(context),
        storyErrorState: resultState.status === "error" ? resultState.error : null,
    };
}
