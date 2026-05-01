import CatalogAssetImage from "../../verify/components/CatalogAssetImage.tsx";
import type { CatalogIdentifyProbeCase } from "../../../types/index.ts";

interface SelectedProbePreviewProps {
    probeCase: CatalogIdentifyProbeCase | null;
    demoStoreReady: boolean;
    pinned: boolean;
    onTogglePinned: () => void;
}

export default function SelectedProbePreview({ probeCase, demoStoreReady, pinned, onTogglePinned }: SelectedProbePreviewProps) {
    if (!probeCase) {
        return (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-5 py-8 text-sm text-slate-600">
                Select a probe case to inspect its preview, capture, and expected outcome before running identification.
            </div>
        );
    }

    return (
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
            <CatalogAssetImage
                src={probeCase.probe_preview_url}
                alt={probeCase.title}
                fallbackLabel={probeCase.title}
                className="h-52 rounded-none border-0 border-b border-slate-200"
            />

            <div className="space-y-4 p-5">
                <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Selected probe</p>
                        <h3 className="mt-1 text-xl font-semibold text-slate-900">{probeCase.title}</h3>
                        <p className="mt-2 text-sm leading-6 text-slate-600">{probeCase.description}</p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <button
                            type="button"
                            onClick={onTogglePinned}
                            className={[
                                "rounded-xl border px-3 py-2 text-sm font-medium transition",
                                pinned
                                    ? "border-amber-200 bg-amber-50 text-amber-800"
                                    : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
                            ].join(" ")}
                        >
                            {pinned ? "Unpin probe" : "Pin probe"}
                        </button>

                        <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${demoStoreReady ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-700"}`}>
                            {demoStoreReady ? "Ready to run" : "Seed demo store first"}
                        </span>
                    </div>
                </div>

                <div className="flex flex-wrap gap-2 text-xs font-medium text-slate-600">
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">{probeCase.dataset_label}</span>
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">{probeCase.capture ?? "plain"}</span>
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">{probeCase.difficulty}</span>
                    {probeCase.expected_outcome ? (
                        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
                            Expected {probeCase.expected_outcome === "match" ? "match" : "no match"}
                        </span>
                    ) : null}
                    {probeCase.expected_top_identity_label ? (
                        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
                            Expected {probeCase.expected_top_identity_label}
                        </span>
                    ) : null}
                </div>
            </div>
        </div>
    );
}
