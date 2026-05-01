import { Fingerprint, ShieldAlert, ShieldCheck } from "lucide-react";

interface VerifyOutcomeStoryHeaderProps {
    headline: string;
    meaning: string;
    contextLabel: string;
}

export default function VerifyOutcomeStoryHeader({
    headline,
    meaning,
    contextLabel,
}: VerifyOutcomeStoryHeaderProps) {
    const isPositive = headline === "Same finger";
    const Icon = isPositive ? ShieldCheck : ShieldAlert;

    return (
        <section
            className={`rounded-3xl border p-6 ${
                isPositive
                    ? "border-emerald-200 bg-emerald-50 text-emerald-950"
                    : "border-amber-200 bg-amber-50 text-amber-950"
            }`}
        >
            <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="max-w-3xl">
                    <div className="inline-flex items-center gap-2 rounded-full border border-current/15 bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em]">
                        <Fingerprint className="h-3.5 w-3.5" />
                        {contextLabel}
                    </div>
                    <div className="mt-4 flex items-start gap-3">
                        <div className="rounded-2xl border border-current/10 bg-white/70 p-3">
                            <Icon className="h-6 w-6" />
                        </div>
                        <div>
                            <h3 className="text-2xl font-semibold">{headline}</h3>
                            <p className="mt-2 text-sm leading-6 opacity-90">{meaning}</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
