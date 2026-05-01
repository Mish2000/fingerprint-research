import type { CatalogBrowserItem } from "../../../types/index.ts";
import { formatCaseMetadataLabel } from "../model.ts";
import type { BrowserSelectionSide } from "../browserModel.ts";
import CatalogAssetImage from "./CatalogAssetImage.tsx";

interface BrowserItemCardProps {
    item: CatalogBrowserItem;
    datasetLabel: string;
    selectedSide: BrowserSelectionSide | null;
    nextTarget: BrowserSelectionSide | null;
    replacementTarget: BrowserSelectionSide | null;
    onSelect: (item: CatalogBrowserItem) => void;
}

function metadataLine(label: string, value: string | null | undefined): string {
    return `${label}: ${value ? formatCaseMetadataLabel(value) : "-"}`;
}

export default function BrowserItemCard({
    item,
    datasetLabel,
    selectedSide,
    nextTarget,
    replacementTarget,
    onSelect,
}: BrowserItemCardProps) {
    const isSelected = selectedSide !== null;
    const isSelectable = !isSelected && nextTarget !== null;
    const actionLabel = selectedSide
        ? `Selected as ${selectedSide}`
        : nextTarget
            ? `${replacementTarget === nextTarget ? "Replace" : "Choose"} ${nextTarget}`
            : "Choose A or B first";

    return (
        <article
            className={[
                "overflow-hidden rounded-2xl border bg-white shadow-sm transition",
                isSelected ? "border-brand-300 ring-2 ring-brand-100" : "border-slate-200 hover:border-slate-300",
            ].join(" ")}
        >
            <CatalogAssetImage
                src={item.thumbnail_url}
                alt={`${item.asset_id} thumbnail`}
                className="aspect-[4/3] rounded-none border-0"
                fallbackLabel="This item is still selectable even if the thumbnail could not be loaded."
            />

            <div className="space-y-4 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{datasetLabel}</p>
                        <h3 className="mt-1 text-sm font-semibold text-slate-900">{item.asset_id}</h3>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600">
                            {item.split}
                        </span>
                        {selectedSide ? (
                            <span className="rounded-full border border-brand-200 bg-brand-50 px-2.5 py-1 text-xs font-semibold text-brand-700">
                                {selectedSide}
                            </span>
                        ) : null}
                    </div>
                </div>

                <div className="grid gap-2 text-xs leading-5 text-slate-600">
                    <p>{metadataLine("Capture", item.capture)}</p>
                    <p>{metadataLine("Finger", item.finger)}</p>
                    <p>{metadataLine("Subject", item.subject_id)}</p>
                    <p>{metadataLine("Modality", item.modality)}</p>
                </div>

                <div className="rounded-2xl border border-slate-100 bg-slate-50 p-3">
                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Selection hint</p>
                    <p className="mt-2 text-xs leading-5 text-slate-700">{item.selection_reason}</p>
                </div>

                <button
                    type="button"
                    disabled={!isSelectable}
                    onClick={() => {
                        if (isSelectable) {
                            onSelect(item);
                        }
                    }}
                    className={[
                        "w-full rounded-xl px-3 py-2.5 text-sm font-medium transition",
                        isSelectable
                            ? "bg-slate-900 text-white hover:bg-slate-700"
                            : "cursor-not-allowed border border-slate-200 bg-slate-50 text-slate-500",
                    ].join(" ")}
                >
                    {actionLabel}
                </button>
            </div>
        </article>
    );
}
