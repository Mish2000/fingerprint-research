import { readResponsePayload } from "./http.ts";
import { extractApiErrorMessage } from "../utils/error.ts";

function formatAssetError(response: Response, payload: unknown, assetLabel: string): string {
    const extractedMessage = extractApiErrorMessage(payload);
    const statusLabel = response.status ? `${response.status}` : "request";

    if (extractedMessage) {
        return `Failed to load ${assetLabel} (${statusLabel}): ${extractedMessage}`;
    }

    return `Failed to load ${assetLabel} (${statusLabel}).`;
}

export async function loadFileFromUrl(
    url: string,
    fallbackName: string,
    assetLabel: string,
): Promise<File> {
    const response = await fetch(url);

    if (!response.ok) {
        const payload = await readResponsePayload(response);
        throw new Error(formatAssetError(response, payload, assetLabel));
    }

    const blob = await response.blob();
    const urlTail = url.split("/").pop() || "";
    const fileName = urlTail.includes(".") ? urlTail : fallbackName;
    const contentType = response.headers.get("content-type") || blob.type || "";

    if (contentType.includes("json") || fileName.toLowerCase().endsWith(".json")) {
        throw new Error(
            "Catalog asset is a descriptor rather than a runnable binary file. " +
            "Regenerate the demo asset bundle or provide binary assets under data/samples/assets/.",
        );
    }

    return new File([blob], fileName, { type: blob.type || "image/png" });
}
