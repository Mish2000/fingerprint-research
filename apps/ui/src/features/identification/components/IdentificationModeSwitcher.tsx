import { Compass, Database, Fingerprint, Sparkles } from "lucide-react";
import type { IdentificationMode } from "../model.ts";

interface IdentificationModeSwitcherProps {
    mode: IdentificationMode;
    onChange: (mode: IdentificationMode) => void;
}

const MODES = [
    {
        value: "demo" as const,
        label: "Demo Mode",
        description: "Seed a curated gallery, pick a probe, and run a guided 1:N flow.",
        icon: Sparkles,
    },
    {
        value: "browser" as const,
        label: "Browser Mode",
        description: "Choose a dataset, seed selected identities into an isolated browser gallery, and run 1:N with a browser-picked probe.",
        icon: Compass,
    },
    {
        value: "operational" as const,
        label: "Operational Mode",
        description: "Keep full control over stats, enroll, manual search, and delete workflows.",
        icon: Database,
    },
];

export default function IdentificationModeSwitcher({ mode, onChange }: IdentificationModeSwitcherProps) {
    return (
        <div className="grid gap-3 lg:grid-cols-3">
            {MODES.map((entry) => {
                const Icon = entry.icon;
                const isActive = mode === entry.value;

                return (
                    <button
                        key={entry.value}
                        type="button"
                        onClick={() => onChange(entry.value)}
                        className={`mode-card ${isActive ? "mode-card--active" : ""}`.trim()}
                        aria-pressed={isActive}
                    >
                        <div className="mode-card__content">
                            <div className="mode-card__icon">
                                <Icon className="h-4 w-4" />
                            </div>
                            <div className="min-w-0">
                                <div className="mode-card__label">{entry.label}</div>
                                <div className="mode-card__description text-clamp-2">{entry.description}</div>
                            </div>
                        </div>
                    </button>
                );
            })}

            <div className="rounded-xl border border-[var(--app-brand-border)] bg-[var(--app-brand-surface)] px-4 py-3 text-sm text-[var(--app-brand-text)] lg:col-span-3">
                <div className="flex items-center gap-2 font-semibold">
                    <Fingerprint className="h-4 w-4" />
                    Recommended first path
                </div>
                <p className="mt-1">Start in Demo Mode, use Browser Mode for guided catalog-backed 1:N, then drop into Operational Mode for manual controls.</p>
            </div>
        </div>
    );
}
