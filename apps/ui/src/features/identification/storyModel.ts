import type { AsyncState } from "../../shared/request-state/index.ts";
import {
    deriveConfidenceBand,
    formatLatency,
    formatMethodLabel,
    humanizeLatencyKey,
    type ConfidenceBand,
} from "../../shared/storytelling.ts";
import type { CatalogIdentifyProbeCase, IdentifyResponse } from "../../types/index.ts";

export interface IdentificationLatencyState {
    totalLabel: string;
    summary: string;
    breakdown: Array<{
        label: string;
        value: string;
    }>;
}

export interface IdentificationProbeContextState {
    label: string;
    summary: string;
    details: string[];
}

export interface IdentificationStoryState {
    headline: string;
    meaning: string;
    contextLabel: string;
    methodStory: string;
    latency: IdentificationLatencyState | null;
    confidenceBand: ConfidenceBand | null;
    probeContext: IdentificationProbeContextState;
    storyErrorState: string | null;
}

function topScore(result: IdentifyResponse): number {
    const topCandidate = result.top_candidate ?? result.candidates[0] ?? null;
    return topCandidate?.rerank_score ?? topCandidate?.retrieval_score ?? 0;
}

function createLatencyState(result: IdentifyResponse | null): IdentificationLatencyState | null {
    if (!result) {
        return null;
    }

    const entries = Object.entries(result.latency_ms)
        .filter((entry): entry is [string, number] => typeof entry[1] === "number" && !Number.isNaN(entry[1]));
    const total = result.latency_ms.total_ms
        ?? entries.reduce((sum, [, value]) => sum + value, 0);

    return {
        totalLabel: formatLatency(total),
        summary: entries.length > 0
            ? "Backend returned identification latency details for this run."
            : "Backend returned a completed identification response without detailed latency entries.",
        breakdown: entries.map(([key, value]) => ({
            label: humanizeLatencyKey(key),
            value: formatLatency(value),
        })),
    };
}

function createProbeContext(probeCase: CatalogIdentifyProbeCase | null): IdentificationProbeContextState {
    if (!probeCase) {
        return {
            label: "Operational search",
            summary: "This result was produced outside a curated probe case.",
            details: [],
        };
    }

    return {
        label: probeCase.title,
        summary: probeCase.description,
        details: [
            `Dataset: ${probeCase.dataset_label}`,
            `Difficulty: ${probeCase.difficulty}`,
            probeCase.capture ? `Capture: ${probeCase.capture}` : null,
            probeCase.expected_outcome ? `Expected: ${probeCase.expected_outcome}` : null,
        ].filter((item): item is string => item !== null),
    };
}

export function createIdentificationStoryState({
    resultState,
    probeCase,
}: {
    resultState: AsyncState<IdentifyResponse>;
    probeCase: CatalogIdentifyProbeCase | null;
}): IdentificationStoryState {
    const result = resultState.status === "success" ? resultState.data : null;
    const headline = result?.decision ? "Match found" : "No match";
    const confidenceBand = result
        ? deriveConfidenceBand({
            score: topScore(result),
            threshold: result.threshold,
            decision: result.decision,
        })
        : null;

    return {
        headline,
        meaning: result
            ? result.decision
                ? "The top candidate cleared the active identification threshold."
                : "No returned candidate cleared the active identification threshold."
            : "Run identification to produce a narrated 1:N result.",
        contextLabel: probeCase ? "Demo identification result" : "Identification result",
        methodStory: result
            ? `${formatMethodLabel(result.retrieval_method)} retrieval with ${formatMethodLabel(result.rerank_method)} re-rank returned ${result.candidates.length} candidate(s).`
            : "Method story appears after a completed identification response.",
        latency: createLatencyState(result),
        confidenceBand,
        probeContext: createProbeContext(probeCase),
        storyErrorState: resultState.status === "error" ? resultState.error : null,
    };
}
