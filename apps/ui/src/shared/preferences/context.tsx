import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { applyAppPreferenceAttributes } from "./applyAttributes.ts";
import { AppPreferencesContext, type AppPreferencesContextValue } from "./contextValue.ts";
import { createDefaultAppPreferences } from "./defaults.ts";
import {
    clearPersistedAppPreferences,
    readPersistedAppPreferences,
    sanitizeAppPreferences,
    writePersistedAppPreferences,
} from "./persistence.ts";
import type { AppPreferences, EffectiveAppTheme } from "./types.ts";

function readSystemTheme(): EffectiveAppTheme {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
        return "light";
    }

    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function resolveEffectiveTheme(theme: AppPreferences["theme"], systemTheme: EffectiveAppTheme): EffectiveAppTheme {
    return theme === "system" ? systemTheme : theme;
}

export function AppPreferencesProvider({ children }: { children: ReactNode }) {
    const [preferences, setPreferencesState] = useState<AppPreferences>(() =>
        readPersistedAppPreferences() ?? createDefaultAppPreferences(),
    );
    const [systemTheme, setSystemTheme] = useState<EffectiveAppTheme>(readSystemTheme);
    const effectiveTheme = resolveEffectiveTheme(preferences.theme, systemTheme);

    useEffect(() => {
        if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
            const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
            const handleChange = (): void => {
                setSystemTheme(readSystemTheme());
            };

            handleChange();
            mediaQuery.addEventListener("change", handleChange);
            return () => {
                mediaQuery.removeEventListener("change", handleChange);
            };
        }

        return undefined;
    }, []);

    useEffect(() => {
        writePersistedAppPreferences(preferences);
    }, [preferences]);

    useEffect(() => {
        applyAppPreferenceAttributes(preferences, effectiveTheme);
    }, [effectiveTheme, preferences]);

    const updatePreferences = useCallback((updates: Partial<AppPreferences>): void => {
        setPreferencesState((current) => sanitizeAppPreferences({ ...current, ...updates }) ?? current);
    }, []);

    const setPreference = useCallback(
        <Key extends keyof AppPreferences>(key: Key, value: AppPreferences[Key]): void => {
            updatePreferences({ [key]: value } as Partial<AppPreferences>);
        },
        [updatePreferences],
    );

    const resetPreferences = useCallback((): void => {
        clearPersistedAppPreferences();
        setPreferencesState(createDefaultAppPreferences());
    }, []);

    const value = useMemo<AppPreferencesContextValue>(
        () => ({
            preferences,
            effectiveTheme,
            setPreference,
            updatePreferences,
            resetPreferences,
        }),
        [effectiveTheme, preferences, resetPreferences, setPreference, updatePreferences],
    );

    return <AppPreferencesContext.Provider value={value}>{children}</AppPreferencesContext.Provider>;
}
