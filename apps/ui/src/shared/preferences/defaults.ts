import type { AppPreferences } from "./types.ts";

export function createDefaultAppPreferences(): AppPreferences {
    return {
        theme: "system",
        language: "en",
        density: "comfortable",
        defaultTab: "benchmark",
        rememberLastTab: true,
        reducedMotion: false,
    };
}
