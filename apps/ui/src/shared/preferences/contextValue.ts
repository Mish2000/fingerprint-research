import { createContext } from "react";
import type { AppPreferences, EffectiveAppTheme } from "./types.ts";

export interface AppPreferencesContextValue {
    preferences: AppPreferences;
    effectiveTheme: EffectiveAppTheme;
    setPreference: <Key extends keyof AppPreferences>(
        key: Key,
        value: AppPreferences[Key],
    ) => void;
    updatePreferences: (updates: Partial<AppPreferences>) => void;
    resetPreferences: () => void;
}

export const AppPreferencesContext = createContext<AppPreferencesContextValue | null>(null);
