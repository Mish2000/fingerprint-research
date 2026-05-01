import { extractApiErrorMessage, toErrorMessage } from "../utils/error.ts";

export async function readResponsePayload(response: Response): Promise<unknown> {
    const text = await response.text();
    if (!text) {
        return null;
    }

    try {
        return JSON.parse(text) as unknown;
    } catch {
        return text;
    }
}

function fallbackHttpMessage(response: Response): string {
    const statusText = response.statusText.trim();
    return `${response.status}${statusText ? ` ${statusText}` : ""}`.trim();
}

export async function readJsonOrThrow<T>(
    response: Response,
    normalize: (payload: unknown) => T,
): Promise<T> {
    const payload = await readResponsePayload(response);

    if (!response.ok) {
        const extractedMessage = extractApiErrorMessage(payload);
        throw new Error(extractedMessage || fallbackHttpMessage(response) || "Request failed.");
    }

    try {
        return normalize(payload);
    } catch (error) {
        throw new Error(`Response validation failed: ${toErrorMessage(error)}`);
    }
}