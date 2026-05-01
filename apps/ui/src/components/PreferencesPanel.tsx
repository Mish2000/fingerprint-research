import { useState } from "react";
import { RotateCcw, Trash2, X } from "lucide-react";
import { useTranslation } from "../shared/i18n/useTranslation.ts";
import { useAppPreferences } from "../shared/preferences/useAppPreferences.ts";
import type {
    AppLanguage,
    AppTabPreference,
    AppThemePreference,
} from "../shared/preferences/types.ts";
import FormField from "../shared/ui/FormField.tsx";
import { CHECKBOX_CLASS_NAME, INPUT_CLASS_NAME } from "../shared/ui/inputClasses.ts";

interface PreferencesPanelProps {
    onClose: () => void;
    onClearWorkspaceState: () => void;
}

type StatusMessage = "preferencesReset" | "workspaceCleared" | null;

function ToggleField({
    id,
    label,
    hint,
    checked,
    onChange,
}: {
    id: string;
    label: string;
    hint: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
}) {
    return (
        <label className="preference-toggle" htmlFor={id}>
            <span>
                <span className="preference-toggle__label">{label}</span>
                <span className="preference-toggle__hint">{hint}</span>
            </span>
            <input
                id={id}
                type="checkbox"
                className={CHECKBOX_CLASS_NAME}
                checked={checked}
                onChange={(event) => onChange(event.target.checked)}
            />
        </label>
    );
}

export default function PreferencesPanel({ onClose, onClearWorkspaceState }: PreferencesPanelProps) {
    const { preferences, setPreference, resetPreferences } = useAppPreferences();
    const { t } = useTranslation();
    const [statusMessage, setStatusMessage] = useState<StatusMessage>(null);

    function handleResetPreferences(): void {
        resetPreferences();
        setStatusMessage("preferencesReset");
    }

    function handleClearWorkspaceState(): void {
        onClearWorkspaceState();
        setStatusMessage("workspaceCleared");
    }

    return (
        <section
            id="app-preferences-panel"
            className="preferences-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="preferences-panel-title"
        >
            <div className="preferences-panel__header">
                <div>
                    <h2 id="preferences-panel-title" className="preferences-panel__title">
                        {t("preferences.title")}
                    </h2>
                </div>
                <button type="button" className="app-icon-button" onClick={onClose} aria-label={t("preferences.close")}>
                    <X className="h-4 w-4" />
                </button>
            </div>

            <div className="preferences-panel__section">
                <div className="preferences-panel__grid">
                    <FormField label={t("preferences.theme")}>
                        <select
                            className={INPUT_CLASS_NAME}
                            value={preferences.theme}
                            onChange={(event) => setPreference("theme", event.target.value as AppThemePreference)}
                        >
                            <option value="light">{t("preferences.theme.light")}</option>
                            <option value="dark">{t("preferences.theme.dark")}</option>
                            <option value="system">{t("preferences.theme.system")}</option>
                        </select>
                    </FormField>

                    <FormField label={t("preferences.language")}>
                        <select
                            className={INPUT_CLASS_NAME}
                            value={preferences.language}
                            onChange={(event) => setPreference("language", event.target.value as AppLanguage)}
                        >
                            <option value="en">{t("preferences.language.en")}</option>
                            <option value="he">{t("preferences.language.he")}</option>
                        </select>
                    </FormField>
                </div>
            </div>

            <div className="preferences-panel__section">
                <div className="preferences-panel__grid">
                    <FormField label={t("preferences.defaultTab")}>
                        <select
                            className={INPUT_CLASS_NAME}
                            value={preferences.defaultTab}
                            onChange={(event) => setPreference("defaultTab", event.target.value as AppTabPreference)}
                        >
                            <option value="verify">{t("tab.verify.title")}</option>
                            <option value="identify">{t("tab.identify.title")}</option>
                            <option value="benchmark">{t("tab.benchmark.title")}</option>
                        </select>
                    </FormField>

                    <ToggleField
                        id="remember-last-tab"
                        label={t("preferences.rememberLastTab")}
                        hint={t("preferences.rememberLastTab.hint")}
                        checked={preferences.rememberLastTab}
                        onChange={(checked) => setPreference("rememberLastTab", checked)}
                    />

                    <ToggleField
                        id="reduced-motion"
                        label={t("preferences.reducedMotion")}
                        hint={t("preferences.reducedMotion.hint")}
                        checked={preferences.reducedMotion}
                        onChange={(checked) => setPreference("reducedMotion", checked)}
                    />
                </div>
            </div>

            <div className="preferences-panel__section">
                <div className="preferences-panel__actions">
                    <div className="preferences-panel__action">
                        <button type="button" className="app-button app-button--secondary" onClick={handleResetPreferences}>
                            <RotateCcw className="h-4 w-4" />
                            <span>{t("preferences.reset")}</span>
                        </button>
                        <p className="preferences-panel__hint">{t("preferences.reset.hint")}</p>
                    </div>

                    <div className="preferences-panel__action">
                        <button type="button" className="app-button app-button--danger" onClick={handleClearWorkspaceState}>
                            <Trash2 className="h-4 w-4" />
                            <span>{t("preferences.clearWorkspace")}</span>
                        </button>
                        <p className="preferences-panel__hint">{t("preferences.clearWorkspace.hint")}</p>
                    </div>
                </div>
            </div>

            <p className="preferences-panel__status" aria-live="polite">
                {statusMessage === "preferencesReset" ? t("preferences.resetDone") : null}
                {statusMessage === "workspaceCleared" ? t("preferences.workspaceCleared") : null}
            </p>
        </section>
    );
}
