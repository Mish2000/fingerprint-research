import CatalogAssetImage from "../../verify/components/CatalogAssetImage.tsx";
import type { CatalogIdentityItem } from "../../../types/index.ts";

interface IdentificationBrowserGalleryPanelProps {
    identities: CatalogIdentityItem[];
    selectedIdentityIds: string[];
    onToggleIdentity: (identity: CatalogIdentityItem) => void;
}

export default function IdentificationBrowserGalleryPanel({
    identities,
    selectedIdentityIds,
    onToggleIdentity,
}: IdentificationBrowserGalleryPanelProps) {
    if (identities.length === 0) {
        return (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-5 py-8 text-sm text-slate-600">
                This dataset does not currently expose any identify-gallery identities for Browser mode.
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between gap-3">
                <div>
                    <h3 className="text-lg font-semibold text-slate-900">Gallery identities</h3>
                    <p className="mt-1 text-sm text-slate-500">Select one or more catalog identities to seed the browser-isolated 1:N gallery.</p>
                </div>
                <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-slate-600">
                    {selectedIdentityIds.length} selected
                </span>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
                {identities.map((identity) => {
                    const isSelected = selectedIdentityIds.includes(identity.identity_id);

                    return (
                        <button
                            key={identity.identity_id}
                            type="button"
                            onClick={() => onToggleIdentity(identity)}
                            aria-pressed={isSelected}
                            className={[
                                "overflow-hidden rounded-2xl border bg-white text-left transition",
                                isSelected
                                    ? "border-brand-300 ring-2 ring-brand-100"
                                    : "border-slate-200 hover:border-slate-300",
                            ].join(" ")}
                        >
                            <CatalogAssetImage
                                src={identity.preview_url ?? identity.thumbnail_url ?? ""}
                                alt={identity.display_name}
                                fallbackLabel={identity.display_name}
                                className="h-36 rounded-none border-0 border-b border-slate-200"
                            />

                            <div className="space-y-3 p-4">
                                <div className="flex items-start justify-between gap-3">
                                    <div>
                                        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{identity.dataset_label}</p>
                                        <h4 className="mt-1 text-base font-semibold text-slate-900">{identity.display_name}</h4>
                                    </div>
                                    {isSelected ? (
                                        <span className="rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700">
                                            In gallery
                                        </span>
                                    ) : null}
                                </div>

                                <div className="flex flex-wrap gap-2 text-xs font-medium text-slate-600">
                                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">{identity.gallery_role}</span>
                                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">Subject {identity.subject_id}</span>
                                    {identity.recommended_enrollment_capture ? (
                                        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
                                            Enroll {identity.recommended_enrollment_capture}
                                        </span>
                                    ) : null}
                                    {identity.recommended_probe_capture ? (
                                        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
                                            Probe {identity.recommended_probe_capture}
                                        </span>
                                    ) : null}
                                </div>
                            </div>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
