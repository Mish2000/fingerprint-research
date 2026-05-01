import type { Capture, Method } from "../../types/index.ts";

export interface MethodProfile {
    value: Method;
    label: string;
    hint: string;
    defaultThreshold: number;
    supportsOverlay: boolean;
    captureMode: "ignored" | "metadata";
    recommendedWarmUp: boolean;
    thresholdHelp: string;
    captureHelp: string;
    overlayHelp: string;
}

export const METHOD_PROFILES: Record<Method, MethodProfile> = {
    classic_orb: {
        value: "classic_orb",
        label: "Classic (ORB)",
        hint: "Fast baseline matcher with visual overlay support.",
        defaultThreshold: 0.01,
        supportsOverlay: true,
        captureMode: "ignored",
        recommendedWarmUp: false,
        thresholdHelp: "Backend default threshold: 0.01.",
        captureHelp: "Capture values stay visible in the UI, but the current classic matcher does not use them on the server.",
        overlayHelp: "Overlay is supported and is usually useful for debugging or demos.",
    },
    classic_gftt_orb: {
        value: "classic_gftt_orb",
        label: "Classic (ROI GFTT+ORB)",
        hint: "Benchmark-style classic matcher with ROI masking, GFTT keypoints, and overlay support.",
        defaultThreshold: 0.01,
        supportsOverlay: true,
        captureMode: "ignored",
        recommendedWarmUp: false,
        thresholdHelp: "Backend default threshold: 0.01.",
        captureHelp: "Capture values stay visible in the UI, but the ROI GFTT+ORB runtime path does not currently use capture metadata on the server.",
        overlayHelp: "Overlay is supported and helps explain the ROI GFTT+ORB decision path visually.",
    },
    harris: {
        value: "harris",
        label: "Classic (Harris + ORB)",
        hint: "Classic matcher with Harris keypoint extraction and overlay support.",
        defaultThreshold: 0.01,
        supportsOverlay: true,
        captureMode: "ignored",
        recommendedWarmUp: false,
        thresholdHelp: "Backend default threshold: 0.01.",
        captureHelp: "Capture values are ignored by the current backend flow for Harris requests.",
        overlayHelp: "Overlay is supported and helps explain the decision visually.",
    },
    sift: {
        value: "sift",
        label: "Classic (SIFT)",
        hint: "Robust classic matcher with tentative / inlier / outlier overlay states.",
        defaultThreshold: 0.01,
        supportsOverlay: true,
        captureMode: "ignored",
        recommendedWarmUp: false,
        thresholdHelp: "Backend default threshold: 0.01.",
        captureHelp: "Capture metadata is currently ignored by the backend for SIFT requests.",
        overlayHelp: "Overlay is supported and often the most informative on SIFT runs.",
    },
    dl: {
        value: "dl",
        label: "Deep Learning (ResNet50)",
        hint: "Embedding-based matcher. Best paired with optional warm-up to soften cold starts.",
        defaultThreshold: 0.45,
        supportsOverlay: false,
        captureMode: "metadata",
        recommendedWarmUp: true,
        thresholdHelp: "Backend default threshold: 0.45.",
        captureHelp: "Capture values are sent to the backend; when masking is enabled, plain/contactless use the plain gate and roll/contact_based use the roll gate.",
        overlayHelp: "The backend returns overlay = null for this matcher.",
    },
    vit: {
        value: "vit",
        label: "Deep Learning (ViT)",
        hint: "Transformer-based embedding matcher with the same cold-start behavior as the ResNet path.",
        defaultThreshold: 0.45,
        supportsOverlay: false,
        captureMode: "metadata",
        recommendedWarmUp: true,
        thresholdHelp: "Backend default threshold: 0.45.",
        captureHelp: "Capture values are sent to the backend and follow the same capture-aware masking policy as the ResNet path when masking is enabled.",
        overlayHelp: "The backend returns overlay = null for this matcher.",
    },
    dedicated: {
        value: "dedicated",
        label: "Dedicated (Patch AI)",
        hint: "Patch-based matcher with overlay support and capture-aware preprocessing on the server.",
        defaultThreshold: 0.85,
        supportsOverlay: true,
        captureMode: "metadata",
        recommendedWarmUp: true,
        thresholdHelp: "Backend default threshold: 0.85.",
        captureHelp: "The backend accepts plain, roll, contactless, and contact_based after normalization; capture metadata drives preprocessing but is not UI-restricted to plain/roll.",
        overlayHelp: "Overlay is supported and includes inlier / outlier similarity data when requested.",
    },
};

export const CAPTURE_OPTIONS: Array<{ value: Capture; label: string }> = [
    { value: "plain", label: "Plain" },
    { value: "roll", label: "Roll" },
    { value: "contactless", label: "Contactless" },
    { value: "contact_based", label: "Contact-based" },
];


export function formatThresholdValue(value: number): string {
    return value.toFixed(2);
}
