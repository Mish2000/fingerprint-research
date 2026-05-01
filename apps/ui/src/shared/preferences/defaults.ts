import type { AppPreferences } from "./types.ts";

export function createDefaultAppPreferences(): AppPreferences {
    return {
        theme: "system",
        language: "en",
        density: "comfortable",
        defaultTab: "live-demo",
        rememberLastTab: true,
        reducedMotion: false,
    };
}
