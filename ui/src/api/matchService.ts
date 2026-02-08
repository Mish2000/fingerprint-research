import type {Capture, MatchResponse, Method} from "../types";
import { extractApiErrorMessage, isObject, toErrorMessage } from "../utils/error";

interface MatchRequest {
    method: Method;
    fileA: File;
    fileB: File;
    captureA: Capture;
    captureB: Capture;
    returnOverlay: boolean;
    threshold?: number | string; 
}

export async function matchFingerprints(req: MatchRequest): Promise<MatchResponse> {
    const { method, fileA, fileB, captureA, captureB, returnOverlay, threshold } = req;

    const form = new FormData();
    form.append("method", method);
    form.append("img_a", fileA);
    form.append("img_b", fileB);

    if (method === "dedicated" || method === "dl") {
        form.append("capture_a", captureA);
        form.append("capture_b", captureB);
    }

    form.append("return_overlay", String(returnOverlay));

    // Only append threshold if it's a valid number/string and not empty
    if (threshold !== undefined && threshold !== "") {
        form.append("threshold", String(threshold));
    }

    try {
        const r = await fetch("/api/match", { method: "POST", body: form });
        const payload: unknown = await r.json();

        if (!r.ok) {
            throw new Error(extractApiErrorMessage(payload));
        }

        if (isObject(payload) && typeof payload.score === "number") {
            return payload as MatchResponse;
        } else {
            throw new Error("Unexpected response shape from backend.");
        }
    } catch (e) {
        throw new Error(toErrorMessage(e));
    }
}

// Helper to wake up the backend
export async function warmUp(method: Method): Promise<void> {
    // Create a tiny 1x1 transparent PNG blob
    const pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
    const res = await fetch(`data:image/png;base64,${pixel}`);
    const blob = await res.blob();
    const file = new File([blob], "warmup.png", { type: "image/png" });

    // Fire a minimal request
    await matchFingerprints({
        method,
        fileA: file,
        fileB: file,
        captureA: "plain",
        captureB: "plain",
        returnOverlay: false,
    });
}