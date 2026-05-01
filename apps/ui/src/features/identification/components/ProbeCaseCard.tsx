import CatalogAssetImage from "../../verify/components/CatalogAssetImage.tsx";
import type { CatalogIdentifyProbeCase } from "../../../types/index.ts";

interface ProbeCaseCardProps {
    probeCase: CatalogIdentifyProbeCase;
    selected: boolean;
    onSelect: (probeCase: CatalogIdentifyProbeCase) => void;
}

export default function ProbeCaseCard({ probeCase, selected, onSelect }: ProbeCaseCardProps) {
    return (
        <button
            type="button"
            onClick={() => onSelect(probeCase)}
            className={`overflow-hidden rounded-2xl border bg-white text-left transition ${selected ? "border-brand-300 shadow-sm ring-2 ring-brand-100" : "border-slate-200 hover:border-slate-300"}`}
            aria-pressed={selected}
        >
            <CatalogAssetImage
                src={probeCase.probe_thumbnail_url}
                alt={probeCase.title}
                fallbackLabel={probeCase.title}
                className="h-32 rounded-none border-0 border-b border-slate-200"
            />

            <div className="space-y-3 p-4">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{probeCase.dataset_label}</p>
                    <h4 className="mt-1 text-base font-semibold text-slate-900">{probeCase.title}</h4>
                    <p className="mt-2 text-sm leading-6 text-slate-600">{probeCase.description}</p>
                </div>

                <div className="flex flex-wrap gap-2 text-xs font-medium text-slate-600">
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">{probeCase.capture ?? "plain"}</span>
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">{probeCase.difficulty}</span>
                    {probeCase.expected_outcome ? (
                        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
                            Expected {probeCase.expected_outcome === "match" ? "match" : "no match"}
                        </span>
                    ) : null}
                </div>
            </div>
        </button>
    );
}
