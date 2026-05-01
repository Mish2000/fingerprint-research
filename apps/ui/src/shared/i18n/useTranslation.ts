import { useCallback } from "react";
import { useAppPreferences } from "../preferences/useAppPreferences.ts";
import { enMessages, heMessages, type MessageKey } from "./index.ts";
import { useFeatureDomLocalization } from "./featureDomLocalization.ts";

export function useTranslation() {
    const { preferences } = useAppPreferences();
    const messages = preferences.language === "he" ? heMessages : enMessages;

    useFeatureDomLocalization(preferences.language);

    const t = useCallback(
        (key: MessageKey): string => messages[key] ?? enMessages[key] ?? key,
        [messages],
    );

    return {
        language: preferences.language,
        t,
    };
}
