import type { CatalogIdentifyProbeCase, IdentifyResponse } from "../../types/index.ts";

export const IDENTIFICATION_MODE_KEY = "fp-research-identification-mode";

export type IdentificationMode = "demo" | "browser" | "operational";

export type DemoExpectationStatus = "pending" | "aligned" | "mismatch" | "no_expectation";

export type DemoExpectationSummary = {
    status: DemoExpectationStatus;
    expectedOutcome: string | null;
    expectedTopIdentityLabel: string | null;
    actualOutcome: string | null;
    actualTopIdentityLabel: string | null;
};

export function readIdentificationModeFromSessionStorage(): IdentificationMode {
    if (typeof window === "undefined") {
        return "demo";
    }

    const stored = window.sessionStorage.getItem(IDENTIFICATION_MODE_KEY);
    return stored === "operational" || stored === "browser" ? stored : "demo";
}

export function persistIdentificationModeInSessionStorage(mode: IdentificationMode): void {
    if (typeof window === "undefined") {
        return;
    }

    window.sessionStorage.setItem(IDENTIFICATION_MODE_KEY, mode);
}

export function clearIdentificationModeInSessionStorage(): void {
    if (typeof window === "undefined") {
        return;
    }

    try {
        window.sessionStorage.removeItem(IDENTIFICATION_MODE_KEY);
    } catch {
        // Ignore session-storage failures and keep the UI usable.
    }
}

export function formatIdentifyDecisionLabel(value: boolean | null | undefined): string {
    return value ? "MATCH" : "NO MATCH";
}

export function createDemoExpectationSummary(
    probeCase: CatalogIdentifyProbeCase | null,
    result: IdentifyResponse | null,
): DemoExpectationSummary {
    if (!probeCase) {
        return {
            status: "pending",
            expectedOutcome: null,
            expectedTopIdentityLabel: null,
            actualOutcome: null,
            actualTopIdentityLabel: null,
        };
    }

    const expectedOutcome = probeCase.expected_outcome ?? null;
    const expectedTopIdentityLabel = probeCase.expected_top_identity_label ?? null;
    const actualOutcome = result ? formatIdentifyDecisionLabel(result.decision) : null;
    const actualTopIdentityLabel = result?.top_candidate?.full_name ?? null;

    if (!expectedOutcome && !expectedTopIdentityLabel) {
        return {
            status: "no_expectation",
            expectedOutcome: null,
            expectedTopIdentityLabel: null,
            actualOutcome,
            actualTopIdentityLabel,
        };
    }

    if (!result) {
        return {
            status: "pending",
            expectedOutcome,
            expectedTopIdentityLabel,
            actualOutcome: null,
            actualTopIdentityLabel: null,
        };
    }

    const normalizedExpectedOutcome = expectedOutcome?.toLowerCase() === "match" ? "MATCH" : expectedOutcome?.toLowerCase() === "no_match" ? "NO MATCH" : expectedOutcome;
    const outcomeMatches = !normalizedExpectedOutcome || normalizedExpectedOutcome === actualOutcome;
    const topIdentityMatches = !expectedTopIdentityLabel || expectedTopIdentityLabel === actualTopIdentityLabel;

    return {
        status: outcomeMatches && topIdentityMatches ? "aligned" : "mismatch",
        expectedOutcome: normalizedExpectedOutcome ?? null,
        expectedTopIdentityLabel,
        actualOutcome,
        actualTopIdentityLabel,
    };
}
