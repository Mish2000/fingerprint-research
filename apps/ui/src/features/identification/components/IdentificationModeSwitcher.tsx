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
        <div className="flex flex-wrap gap-3">
            {MODES.map((entry) => {
                const Icon = entry.icon;
                const isActive = mode === entry.value;

                return (
                    <button
                        key={entry.value}
                        type="button"
                        onClick={() => onChange(entry.value)}
                        className={[
                            "min-w-56 rounded-2xl border px-4 py-3 text-left transition",
                            isActive
                                ? "border-white/30 bg-white/15 shadow-sm"
                                : "border-white/10 bg-black/10 hover:border-white/20 hover:bg-white/10",
                        ].join(" ")}
                        aria-pressed={isActive}
                    >
                        <div className="flex items-start gap-3">
                            <div className="rounded-xl bg-white/10 p-2 text-white">
                                <Icon className="h-4 w-4" />
                            </div>
                            <div>
                                <div className="font-semibold text-white">{entry.label}</div>
                                <div className="mt-1 text-sm text-white/70">{entry.description}</div>
                            </div>
                        </div>
                    </button>
                );
            })}

            <div className="rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm text-white/75">
                <div className="flex items-center gap-2 font-semibold text-white">
                    <Fingerprint className="h-4 w-4" />
                    Recommended first path
                </div>
                <p className="mt-1">Start in Demo Mode, switch to Browser Mode for guided catalog-backed 1:N, then drop into Operational Mode for manual controls.</p>
            </div>
        </div>
    );
}
