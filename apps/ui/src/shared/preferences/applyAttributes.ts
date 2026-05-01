import type { AppPreferences, EffectiveAppTheme } from "./types.ts";

const LANGUAGE_DIRECTION: Record<AppPreferences["language"], "ltr" | "rtl"> = {
    en: "ltr",
    he: "rtl",
};

export function applyAppPreferenceAttributes(
    preferences: AppPreferences,
    effectiveTheme: EffectiveAppTheme,
): void {
    if (typeof document === "undefined") {
        return;
    }

    const root = document.documentElement;

    root.lang = preferences.language;
    root.dir = LANGUAGE_DIRECTION[preferences.language];

    root.dataset.themePreference = preferences.theme;
    root.dataset.theme = effectiveTheme;
    root.dataset.language = preferences.language;
    root.dataset.density = preferences.density;
    root.dataset.reducedMotion = String(preferences.reducedMotion);

    root.classList.toggle("theme-light", effectiveTheme === "light");
    root.classList.toggle("theme-dark", effectiveTheme === "dark");
    root.classList.toggle("density-compact", preferences.density === "compact");
    root.classList.toggle("density-comfortable", preferences.density === "comfortable");
    root.classList.toggle("motion-reduced", preferences.reducedMotion);
}
