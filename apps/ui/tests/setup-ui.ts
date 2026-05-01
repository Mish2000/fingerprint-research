import { afterEach, vi } from "vitest";

const createObjectUrlMock = vi.fn(() => "blob:verify-test");
const revokeObjectUrlMock = vi.fn();

function installDefaultMatchMedia(): void {
    Object.defineProperty(window, "matchMedia", {
        configurable: true,
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
            matches: false,
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

Object.defineProperty(globalThis, "IS_REACT_ACT_ENVIRONMENT", {
    configurable: true,
    writable: true,
    value: true,
});

Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    writable: true,
    value: createObjectUrlMock,
});

Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    writable: true,
    value: revokeObjectUrlMock,
});

installDefaultMatchMedia();

afterEach(() => {
    document.body.innerHTML = "";
    document.documentElement.removeAttribute("data-theme");
    document.documentElement.removeAttribute("data-theme-preference");
    document.documentElement.removeAttribute("data-language");
    document.documentElement.removeAttribute("data-density");
    document.documentElement.removeAttribute("data-reduced-motion");
    document.documentElement.removeAttribute("lang");
    document.documentElement.removeAttribute("dir");
    document.documentElement.classList.remove("theme-dark", "theme-light", "density-compact");
    document.documentElement.style.removeProperty("color-scheme");
    localStorage.clear();
    sessionStorage.clear();
    createObjectUrlMock.mockClear();
    revokeObjectUrlMock.mockClear();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    installDefaultMatchMedia();
});
