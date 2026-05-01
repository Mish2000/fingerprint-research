export type ConfidenceBandLevel = "strong" | "medium" | "borderline" | "weak" | "negative";

export interface ConfidenceBand {
    level: ConfidenceBandLevel;
    label: string;
    summary: string;
    score: number;
    threshold: number;
}

export function formatWords(value: string | null | undefined): string {
    if (!value) {
        return "-";
    }

    return value
        .split(/[_-]/g)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

export function formatCaptureLabel(value: string | null | undefined): string {
    switch ((value ?? "").trim().toLowerCase()) {
        case "plain":
            return "Plain";
        case "roll":
        case "rolled":
            return "Rolled";
        case "contactless":
            return "Contactless";
        case "contact_based":
        case "contact-based":
        case "contactbased":
            return "Contact-based";
        default:
            return formatWords(value);
    }
}

export function formatMethodLabel(value: string | null | undefined): string {
    switch ((value ?? "").trim().toLowerCase()) {
        case "classic":
        case "classic_orb":
            return "Classic (ORB)";
        case "classic_v2":
        case "classic_gftt_orb":
            return "Classic (ROI GFTT+ORB)";
        case "harris":
            return "Classic (Harris + ORB)";
        case "sift":
            return "Classic (SIFT)";
        case "dl":
        case "dl_quick":
            return "Deep Learning (ResNet50)";
        case "vit":
            return "Deep Learning (ViT)";
        case "dedicated":
            return "Dedicated (Patch AI)";
        default:
            return formatWords(value);
    }
}

export function formatLatency(value: number | null | undefined): string {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "-";
    }

    return `${value.toFixed(1)} ms`;
}

export function humanizeLatencyKey(key: string): string {
    switch (key) {
        case "total_ms":
            return "Total";
        case "probe_embed_ms":
            return "Probe embed";
        case "shortlist_scan_ms":
            return "Shortlist";
        case "rerank_ms":
            return "Re-rank";
        case "embed_ms_a":
            return "Embed A";
        case "embed_ms_b":
            return "Embed B";
        default:
            return formatWords(key.replace(/_ms$/i, ""));
    }
}

function normalizePositiveMargin(score: number, threshold: number): number {
    return (score - threshold) / Math.max(1e-6, 1 - threshold);
}

function negativeRatio(score: number, threshold: number): number {
    return threshold <= 0 ? 0 : score / threshold;
}

export function deriveConfidenceBand(params: {
    score: number | null | undefined;
    threshold: number | null | undefined;
    decision: boolean | null | undefined;
}): ConfidenceBand | null {
    const { score, threshold, decision } = params;
    if (
        typeof score !== "number"
        || Number.isNaN(score)
        || typeof threshold !== "number"
        || Number.isNaN(threshold)
        || typeof decision !== "boolean"
    ) {
        return null;
    }

    if (decision) {
        const margin = score - threshold;
        const normalized = normalizePositiveMargin(score, threshold);

        if (normalized >= 0.35 || margin >= 0.2) {
            return {
                level: "strong",
                label: "Strong",
                summary: "Well above the active threshold.",
                score,
                threshold,
            };
        }

        if (normalized >= 0.12 || margin >= 0.06) {
            return {
                level: "medium",
                label: "Medium",
                summary: "Above threshold with visible room.",
                score,
                threshold,
            };
        }

        return {
            level: "borderline",
            label: "Borderline",
            summary: "Above threshold, but still close to the cutoff.",
            score,
            threshold,
        };
    }

    if (threshold > 0 && threshold < 0.1) {
        const ratio = negativeRatio(score, threshold);
        if (ratio >= 0.9) {
            return {
                level: "borderline",
                label: "Borderline",
                summary: "Just below the active threshold.",
                score,
                threshold,
            };
        }

        if (ratio >= 0.5) {
            return {
                level: "weak",
                label: "Weak",
                summary: "Below threshold, but not far from the cutoff.",
                score,
                threshold,
            };
        }

        return {
            level: "negative",
            label: "Negative",
            summary: "Clearly below the active threshold.",
            score,
            threshold,
        };
    }

    const deficit = threshold - score;
    if (deficit <= 0.02) {
        return {
            level: "borderline",
            label: "Borderline",
            summary: "Just below the active threshold.",
            score,
            threshold,
        };
    }

    if (deficit <= 0.08) {
        return {
            level: "weak",
            label: "Weak",
            summary: "Below threshold, but not far from the cutoff.",
            score,
            threshold,
        };
    }

    return {
        level: "negative",
        label: "Negative",
        summary: "Clearly below the active threshold.",
        score,
        threshold,
    };
}
