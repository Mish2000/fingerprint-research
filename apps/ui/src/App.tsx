import { useEffect, useMemo, useState } from "react";
import type { LucideIcon } from "lucide-react";
import { BarChart3, Fingerprint, ScanFace, Settings } from "lucide-react";
import PreferencesPanel from "./components/PreferencesPanel.tsx";
import BenchmarkWorkspace from "./features/benchmark/BenchmarkWorkspace.tsx";
import { clearAllPersistedIdentificationState } from "./features/identification/persistence.ts";
import IdentificationWorkspace from "./features/identification/IdentificationWorkspace.tsx";
import { clearAllPersistedVerifyState } from "./features/verify/persistence.ts";
import VerifyWorkspace from "./features/verify/VerifyWorkspace.tsx";
import type { MessageKey } from "./shared/i18n";
import { useTranslation } from "./shared/i18n/useTranslation.ts";
import { AppPreferencesProvider } from "./shared/preferences/context.tsx";
import {
    clearPersistedLastActiveTab,
    readPersistedLastActiveTab,
    writePersistedLastActiveTab,
} from "./shared/preferences/persistence.ts";
import { useAppPreferences } from "./shared/preferences/useAppPreferences.ts";
import type { AppPreferences, AppTabPreference } from "./shared/preferences/types.ts";

type Tab = AppTabPreference;

type TabConfig = {
    id: Tab;
    titleKey: MessageKey;
    descriptionKey: MessageKey;
    icon: LucideIcon;
};

const TAB_QUERY_PARAM = "tab";
const LEGACY_TAB_FALLBACKS: Record<string, Tab> = {
    contracts: "benchmark",
    release: "benchmark",
};

const TABS: TabConfig[] = [
    {
        id: "verify",
        titleKey: "tab.verify.title",
        descriptionKey: "tab.verify.description",
        icon: ScanFace,
    },
    {
        id: "identify",
        titleKey: "tab.identify.title",
        descriptionKey: "tab.identify.description",
        icon: Fingerprint,
    },
    {
        id: "benchmark",
        titleKey: "tab.benchmark.title",
        descriptionKey: "tab.benchmark.description",
        icon: BarChart3,
    },
];

function isTab(value: string | null): value is Tab {
    return TABS.some((tab) => tab.id === value);
}

function readTabFromUrl(): Tab | null {
    if (typeof window === "undefined") {
        return null;
    }

    const params = new URLSearchParams(window.location.search);
    const candidate = params.get(TAB_QUERY_PARAM);
    if (isTab(candidate)) {
        return candidate;
    }

    if (candidate && Object.prototype.hasOwnProperty.call(LEGACY_TAB_FALLBACKS, candidate)) {
        return LEGACY_TAB_FALLBACKS[candidate];
    }

    return null;
}

function readInitialActiveTab(preferences: AppPreferences): Tab {
    const tabFromUrl = readTabFromUrl();
    if (tabFromUrl) {
        return tabFromUrl;
    }

    if (preferences.rememberLastTab) {
        const rememberedTab = readPersistedLastActiveTab();
        if (rememberedTab) {
            return rememberedTab;
        }
    }

    return preferences.defaultTab;
}

function syncActiveTabInUrl(tab: Tab): void {
    const params = new URLSearchParams(window.location.search);
    if (params.get(TAB_QUERY_PARAM) === tab) {
        return;
    }

    params.set(TAB_QUERY_PARAM, tab);
    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
    window.history.replaceState(window.history.state, "", nextUrl);
}

function renderTab(tab: Tab) {
    switch (tab) {
        case "verify":
            return <VerifyWorkspace />;
        case "identify":
            return <IdentificationWorkspace />;
        case "benchmark":
            return <BenchmarkWorkspace />;
    }
}

function AppShell() {
    const { preferences } = useAppPreferences();
    const { t } = useTranslation();
    const [activeTab, setActiveTab] = useState<Tab>(() => readInitialActiveTab(preferences));
    const [isPreferencesOpen, setIsPreferencesOpen] = useState(false);

    useEffect(() => {
        document.title = t("app.title");
    }, [t]);

    useEffect(() => {
        syncActiveTabInUrl(activeTab);

        if (preferences.rememberLastTab) {
            writePersistedLastActiveTab(activeTab);
        } else {
            clearPersistedLastActiveTab();
        }
    }, [activeTab, preferences.rememberLastTab]);

    useEffect(() => {
        const handlePopState = (): void => {
            setActiveTab(readInitialActiveTab(preferences));
        };

        window.addEventListener("popstate", handlePopState);
        return () => {
            window.removeEventListener("popstate", handlePopState);
        };
    }, [preferences]);

    useEffect(() => {
        if (!isPreferencesOpen) {
            return undefined;
        }

        const handleKeyDown = (event: KeyboardEvent): void => {
            if (event.key === "Escape") {
                setIsPreferencesOpen(false);
            }
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
        };
    }, [isPreferencesOpen]);

    const activeTabConfig = useMemo(
        () => TABS.find((tab) => tab.id === activeTab) ?? TABS[0],
        [activeTab],
    );

    function clearSavedWorkspaceState(): void {
        clearAllPersistedVerifyState();
        clearAllPersistedIdentificationState();
        clearPersistedLastActiveTab();
    }

    return (
        <div className="app-shell font-sans lg:flex">
            <aside className="app-sidebar lg:sticky lg:top-0 lg:flex lg:h-screen lg:w-72 lg:flex-col">
                <div className="app-sidebar__brand">
                    <div className="flex items-center gap-3">
                        <div className="rounded-lg bg-brand-600 p-2 text-white shadow-sm">
                            <Fingerprint className="h-6 w-6" />
                        </div>
                        <div>
                            <h1 className="app-title">{t("app.title")}</h1>
                            <p className="app-subtitle">{t("app.subtitle")}</p>
                        </div>
                    </div>
                </div>

                <nav className="app-nav" aria-label={t("app.title")}>
                    {TABS.map((tab) => {
                        const Icon = tab.icon;
                        const isActive = tab.id === activeTab;

                        return (
                            <button
                                key={tab.id}
                                type="button"
                                onClick={() => setActiveTab(tab.id)}
                                className={`app-nav-button ${isActive ? "app-nav-button--active" : ""}`.trim()}
                                aria-current={isActive ? "page" : undefined}
                            >
                                <div className="app-nav-button__content">
                                    <div className="app-nav-button__icon">
                                        <Icon className="h-4 w-4" />
                                    </div>
                                    <div className="min-w-0">
                                        <div className="font-semibold">{t(tab.titleKey)}</div>
                                        <div className="app-nav-button__description">{t(tab.descriptionKey)}</div>
                                    </div>
                                </div>
                            </button>
                        );
                    })}
                </nav>

                <div className="app-sidebar__footer">
                    <button
                        type="button"
                        className="app-settings-button"
                        onClick={() => setIsPreferencesOpen((current) => !current)}
                        aria-expanded={isPreferencesOpen}
                        aria-controls="app-preferences-panel"
                    >
                        <Settings className="h-4 w-4" />
                        <span>{t("preferences.open")}</span>
                    </button>
                    <p className="app-sidebar__footer-note">{t("app.subtitle")}</p>
                </div>
            </aside>

            <main className="app-main">
                <header className="app-header">
                    <div>
                        <h2 className="app-header__title">{t(activeTabConfig.titleKey)}</h2>
                        <p className="app-header__description">{t(activeTabConfig.descriptionKey)}</p>
                    </div>
                </header>

                <div className="app-content">
                    <div className="mx-auto max-w-7xl pb-16">{renderTab(activeTab)}</div>
                </div>
            </main>

            {isPreferencesOpen ? (
                <div
                    className="preferences-modal-backdrop"
                    onMouseDown={(event) => {
                        if (event.target === event.currentTarget) {
                            setIsPreferencesOpen(false);
                        }
                    }}
                >
                    <PreferencesPanel
                        onClose={() => setIsPreferencesOpen(false)}
                        onClearWorkspaceState={clearSavedWorkspaceState}
                    />
                </div>
            ) : null}
        </div>
    );
}

export default function App() {
    return (
        <AppPreferencesProvider>
            <AppShell />
        </AppPreferencesProvider>
    );
}
