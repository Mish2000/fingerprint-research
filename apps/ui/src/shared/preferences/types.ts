export const APP_THEME_VALUES = ["system", "light", "dark"] as const;
export type AppThemePreference = (typeof APP_THEME_VALUES)[number];
export type EffectiveAppTheme = "light" | "dark";

export const APP_LANGUAGE_VALUES = ["en", "he"] as const;
export type AppLanguage = (typeof APP_LANGUAGE_VALUES)[number];

export const APP_DENSITY_VALUES = ["comfortable", "compact"] as const;
export type AppDensityPreference = (typeof APP_DENSITY_VALUES)[number];

export const APP_TAB_VALUES = ["verify", "identify", "benchmark"] as const;
export type AppTabPreference = (typeof APP_TAB_VALUES)[number];

export interface AppPreferences {
    theme: AppThemePreference;
    language: AppLanguage;
    density: AppDensityPreference;
    defaultTab: AppTabPreference;
    rememberLastTab: boolean;
    reducedMotion: boolean;
}
