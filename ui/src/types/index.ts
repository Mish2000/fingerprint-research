export type Method = "classic" | "dl" | "dedicated";
export type Capture = "plain" | "roll";

export type OverlayMatch = {
    a: [number, number];
    b: [number, number];
    kind: string; // "tentative" | "inlier" | "outlier"
    sim?: number | null;
};

export type Overlay = {
    matches: OverlayMatch[];
};

export type MatchResponse = {
    method: Method;
    score: number;
    decision: boolean;
    threshold: number;
    latency_ms: number;
    meta: Record<string, unknown>;
    overlay?: Overlay | null;
};