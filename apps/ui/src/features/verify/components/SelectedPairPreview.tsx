import type { CatalogBrowserItem } from "../../../types/index.ts";
import { formatCaseMetadataLabel } from "../model.ts";
import type { BrowserSelectionSide } from "../browserModel.ts";
import CatalogAssetImage from "./CatalogAssetImage.tsx";

interface SelectedPairPreviewProps {
    side: BrowserSelectionSide;
    asset: CatalogBrowserItem | null;
    datasetLabel: string | null;
    isReplacing: boolean;
    onClear: () => void;
    onReplace: () => void;
    onCancelReplace: () => void;
}

function previewMetadata(label: string, value: string | null | undefined): string {
    return `${label}: ${value ? formatCaseMetadataLabel(value) : "-"}`;
}

export default function SelectedPairPreview({
    side,
    asset,
    datasetLabel,
    isReplacing,
    onClear,
    onReplace,
    onCancelReplace,
}: SelectedPairPreviewProps) {
    if (!asset) {
        return (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Side {side}</p>
                <p className="mt-3 text-sm font-medium text-slate-900">Waiting for selection</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                    Pick a dataset item to assign it to side {side}.
                </p>
            </div>
        );
    }

    return (
        <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Side {side}</p>
                    <p className="mt-1 text-sm font-semibold text-slate-900">{asset.asset_id}</p>
                    <p className="mt-1 text-sm text-slate-500">{datasetLabel ?? asset.dataset}</p>
                </div>
                <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600">
                    {asset.split}
                </span>
            </div>

            <CatalogAssetImage
                src={asset.preview_url}
                alt={`${asset.asset_id} preview`}
                className="aspect-[4/3]"
                fallbackLabel="The pair stays usable even if the larger preview image is unavailable."
            />

            <div className="grid gap-2 text-sm text-slate-700">
                <p>{previewMetadata("Capture", asset.capture)}</p>
                <p>{previewMetadata("Finger", asset.finger)}</p>
                <p>{previewMetadata("Subject", asset.subject_id)}</p>
                <p>{previewMetadata("Modality", asset.modality)}</p>
            </div>

            <div className="flex flex-wrap gap-2">
                <button
                    type="button"
                    onClick={isReplacing ? onCancelReplace : onReplace}
                    className={[
                        "rounded-xl px-3 py-2 text-sm font-medium transition",
                        isReplacing
                            ? "border border-brand-200 bg-brand-50 text-brand-700"
                            : "border border-slate-200 bg-white text-slate-700 hover:border-slate-300",
                    ].join(" ")}
                >
                    {isReplacing ? `Cancel Replace ${side}` : `Replace ${side}`}
                </button>

                <button
                    type="button"
                    onClick={onClear}
                    className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-300"
                >
                    Clear {side}
                </button>
            </div>
        </div>
    );
}
