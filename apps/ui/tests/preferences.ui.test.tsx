import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import App from "../src/App.tsx";
import { IDENTIFICATION_MODE_KEY } from "../src/features/identification/model.ts";
import { VERIFY_MODE_SESSION_KEY } from "../src/features/verify/model.ts";
import { createWorkspaceStorageKey } from "../src/shared/persistence/workspacePersistence.ts";
import {
    APP_LAST_TAB_STORAGE_KEY,
    APP_PREFERENCES_STORAGE_KEY,
    readPersistedAppPreferences,
    writePersistedLastActiveTab,
} from "../src/shared/preferences/persistence.ts";

type RenderedApp = {
    container: HTMLDivElement;
    root: Root;
};

function createJsonResponse(payload: unknown): Response {
    return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "content-type": "application/json" },
    });
}

async function flush(): Promise<void> {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
}

async function renderApp(initialUrl = "/"): Promise<RenderedApp> {
    window.history.replaceState(window.history.state, "", initialUrl);
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    await act(async () => {
        root.render(<App />);
    });

    await flush();
    return { container, root };
}

async function unmountApp(root: Root): Promise<void> {
    await act(async () => {
        root.unmount();
    });
}

function installFetchMock() {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
        const parsed = new URL(String(input), "http://localhost");

        if (parsed.pathname === "/api/benchmark/summary") {
            return createJsonResponse({
                dataset: "nist_sd300b",
                split: "test",
                view_mode: "canonical",
                dataset_info: { key: "nist_sd300b", label: "NIST SD300b", summary: "" },
                split_info: { key: "test", label: "Test", summary: "" },
                view_info: { key: "canonical", label: "Canonical", summary: "" },
                validation_state: "validated",
                selection_note: "Showing curated full benchmark results from validated showcase runs.",
                selection_policy: "Curated full benchmark showcase restricted to validated canonical families.",
                result_count: 0,
                method_count: 0,
                run_count: 0,
                available_datasets: [{ key: "nist_sd300b", label: "NIST SD300b", summary: "" }],
                available_splits: [{ key: "test", label: "Test", summary: "" }],
                available_view_modes: [{ key: "canonical", label: "Canonical", summary: "" }],
                current_run_families: ["full_nist_sd300b_h6"],
                artifact_note: "Artifacts available.",
            });
        }

        if (parsed.pathname === "/api/benchmark/comparison") {
            return createJsonResponse({
                rows: [],
                datasets: ["nist_sd300b"],
                splits: ["test"],
                default_dataset: "nist_sd300b",
                default_split: "test",
                view_mode: "canonical",
                view_info: { key: "canonical", label: "Canonical", summary: "" },
                dataset_info: { nist_sd300b: { key: "nist_sd300b", label: "NIST SD300b", summary: "" } },
                split_info: { test: { key: "test", label: "Test", summary: "" } },
            });
        }

        if (parsed.pathname === "/api/benchmark/best") {
            return createJsonResponse({
                dataset: "nist_sd300b",
                split: "test",
                view_mode: "canonical",
                entries: [],
            });
        }

        throw new Error(`Unexpected fetch call: ${String(input)}`);
    }));
}

function installDarkSystemPreference(): void {
    Object.defineProperty(window, "matchMedia", {
        configurable: true,
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
            matches: query === "(prefers-color-scheme: dark)",
            media: query,
            onchange: null,
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            addListener: vi.fn(),
            removeListener: vi.fn(),
            dispatchEvent: vi.fn(),
        })),
    });
}

function getButton(container: HTMLElement, label: string): HTMLButtonElement {
    const button = Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find((item) =>
        item.textContent?.includes(label),
    );
    if (!button) {
        throw new Error(`Could not find button: ${label}`);
    }

    return button;
}

function getSelectByLabel(container: HTMLElement, label: string): HTMLSelectElement {
    const field = Array.from(container.querySelectorAll<HTMLLabelElement>("label")).find((item) =>
        item.textContent?.includes(label),
    );
    const select = field?.querySelector("select");
    if (!select) {
        throw new Error(`Could not find select: ${label}`);
    }

    return select;
}

async function changeSelect(select: HTMLSelectElement, value: string): Promise<void> {
    await act(async () => {
        select.value = value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
    });
}

describe("global preferences UI", () => {
    it("applies system theme from prefers-color-scheme", async () => {
        installFetchMock();
        installDarkSystemPreference();

        const { root } = await renderApp("/");

        expect(document.documentElement.dataset.themePreference).toBe("system");
        expect(document.documentElement.dataset.theme).toBe("dark");
        expect(document.documentElement.classList.contains("theme-dark")).toBe(true);

        await unmountApp(root);
    });

    it("updates theme, language, RTL direction, motion, modal placement, and reset state", async () => {
        installFetchMock();
        const { container, root } = await renderApp("/");

        await act(async () => {
            getButton(container, "Preferences").click();
        });
        expect(container.querySelector(".preferences-modal-backdrop")).not.toBeNull();

        await changeSelect(getSelectByLabel(container, "Theme"), "dark");
        expect(document.documentElement.dataset.theme).toBe("dark");
        expect(document.documentElement.dataset.themePreference).toBe("dark");

        await changeSelect(getSelectByLabel(container, "Language"), "he");
        await flush();
        expect(document.documentElement.lang).toBe("he");
        expect(document.documentElement.dir).toBe("rtl");
        expect(container.textContent).toContain("העדפות משתמש");
        expect(container.textContent).toContain("מצב חיסכון באנרגיה");
        expect(container.textContent).toContain("תקציר מנהלים");
        expect(container.textContent).not.toContain("צפיפות");
        expect(container.querySelector("main h2")?.textContent).toBe("מדדי ביצועים");
        expect(document.documentElement.dataset.density).toBe("comfortable");
        expect(document.documentElement.classList.contains("density-compact")).toBe(false);

        const reducedMotion = container.querySelector<HTMLInputElement>("#reduced-motion");
        if (!reducedMotion) {
            throw new Error("Could not find reduced motion toggle.");
        }
        await act(async () => {
            reducedMotion.click();
        });
        expect(document.documentElement.dataset.reducedMotion).toBe("true");

        await act(async () => {
            getButton(container, "איפוס העדפות").click();
        });

        expect(document.documentElement.lang).toBe("en");
        expect(document.documentElement.dir).toBe("ltr");
        expect(document.documentElement.dataset.themePreference).toBe("system");
        expect(document.documentElement.dataset.density).toBe("comfortable");
        expect(document.documentElement.dataset.reducedMotion).toBe("false");
        expect(container.textContent).toContain("Preferences reset.");
        expect(readPersistedAppPreferences()).toMatchObject({
            language: "en",
            theme: "system",
            density: "comfortable",
            reducedMotion: false,
        });

        await unmountApp(root);
    });

    it("clears saved workspace state from the preferences panel", async () => {
        installFetchMock();
        const { container, root } = await renderApp("/");
        const verifyLocalKey = createWorkspaceStorageKey("verify", "local");
        const verifySessionKey = createWorkspaceStorageKey("verify", "session");
        const identificationLocalKey = createWorkspaceStorageKey("identification", "local");

        localStorage.setItem(verifyLocalKey, "verify local");
        sessionStorage.setItem(verifySessionKey, "verify session");
        sessionStorage.setItem(VERIFY_MODE_SESSION_KEY, "browser");
        localStorage.setItem(identificationLocalKey, "identification local");
        sessionStorage.setItem(IDENTIFICATION_MODE_KEY, "operational");
        writePersistedLastActiveTab("benchmark");

        await act(async () => {
            getButton(container, "Preferences").click();
        });
        await act(async () => {
            getButton(container, "Clear saved workspace state").click();
        });

        expect(localStorage.getItem(verifyLocalKey)).toBeNull();
        expect(sessionStorage.getItem(verifySessionKey)).toBeNull();
        expect(sessionStorage.getItem(VERIFY_MODE_SESSION_KEY)).toBeNull();
        expect(localStorage.getItem(identificationLocalKey)).toBeNull();
        expect(sessionStorage.getItem(IDENTIFICATION_MODE_KEY)).toBeNull();
        expect(localStorage.getItem(APP_LAST_TAB_STORAGE_KEY)).toBeNull();
        expect(localStorage.getItem(APP_PREFERENCES_STORAGE_KEY)).not.toBeNull();
        expect(container.textContent).toContain("Saved workspace state cleared.");

        await unmountApp(root);
    });
});
