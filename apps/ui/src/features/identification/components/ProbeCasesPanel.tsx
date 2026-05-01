import type { CatalogIdentifyProbeCase } from "../../../types/index.ts";
import ProbeCaseCard from "./ProbeCaseCard.tsx";

interface ProbeCasesPanelProps {
    probeCases: CatalogIdentifyProbeCase[];
    selectedProbeCaseId: string | null;
    onSelect: (probeCase: CatalogIdentifyProbeCase) => void;
}

export default function ProbeCasesPanel({ probeCases, selectedProbeCaseId, onSelect }: ProbeCasesPanelProps) {
    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between gap-3">
                <div>
                    <h3 className="text-lg font-semibold text-slate-900">Probe cases</h3>
                    <p className="mt-1 text-sm text-slate-500">Choose a ready-made probe without touching File Explorer or manual uploads.</p>
                </div>
                <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-slate-600">
                    {probeCases.length} probes
                </div>
            </div>

            {probeCases.length > 0 ? (
                <div className="grid gap-4 md:grid-cols-2">
                    {probeCases.map((probeCase) => (
                        <ProbeCaseCard
                            key={probeCase.id}
                            probeCase={probeCase}
                            selected={probeCase.id === selectedProbeCaseId}
                            onSelect={onSelect}
                        />
                    ))}
                </div>
            ) : (
                <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-5 py-8 text-sm text-slate-600">
                    No probe cases are currently available in the identify gallery.
                </div>
            )}
        </div>
    );
}
