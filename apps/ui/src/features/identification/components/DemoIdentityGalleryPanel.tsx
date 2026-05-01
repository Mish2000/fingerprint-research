import CatalogAssetImage from "../../verify/components/CatalogAssetImage.tsx";
import type { CatalogIdentifyDemoIdentity, CatalogIdentifyProbeCase } from "../../../types/index.ts";

interface DemoIdentityGalleryPanelProps {
    identities: CatalogIdentifyDemoIdentity[];
    selectedProbeCase: CatalogIdentifyProbeCase | null;
}

export default function DemoIdentityGalleryPanel({ identities, selectedProbeCase }: DemoIdentityGalleryPanelProps) {
    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between gap-3">
                <div>
                    <h3 className="text-lg font-semibold text-slate-900">Demo identity gallery</h3>
                    <p className="mt-1 text-sm text-slate-500">These are the curated identities that the demo seeding flow will enroll.</p>
                </div>
                <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-slate-600">
                    {identities.length} identities
                </div>
            </div>

            {identities.length > 0 ? (
                <div className="grid gap-4 md:grid-cols-2">
                    {identities.map((identity) => {
                        const isExpected = selectedProbeCase?.expected_top_identity_id === identity.id;

                        return (
                            <article
                                key={identity.id}
                                className={`overflow-hidden rounded-2xl border bg-white ${isExpected ? "border-emerald-200 shadow-sm" : "border-slate-200"}`}
                            >
                                <CatalogAssetImage
                                    src={identity.preview_url}
                                    alt={identity.display_label}
                                    fallbackLabel={identity.display_label}
                                    className="h-36 rounded-none border-0 border-b border-slate-200"
                                />

                                <div className="space-y-3 p-4">
                                    <div className="flex items-start justify-between gap-3">
                                        <div>
                                            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{identity.dataset_label}</p>
                                            <h4 className="mt-1 text-base font-semibold text-slate-900">{identity.display_label}</h4>
                                        </div>
                                        {isExpected ? (
                                            <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">
                                                Expected top identity
                                            </span>
                                        ) : null}
                                    </div>

                                    <div className="flex flex-wrap gap-2 text-xs font-medium text-slate-600">
                                        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">{identity.capture ?? "plain"}</span>
                                        {identity.subject_id ? (
                                            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">Subject {identity.subject_id}</span>
                                        ) : null}
                                        {identity.gallery_role ? (
                                            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">{identity.gallery_role}</span>
                                        ) : null}
                                    </div>
                                </div>
                            </article>
                        );
                    })}
                </div>
            ) : (
                <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-5 py-8 text-sm text-slate-600">
                    No demo identities are currently available in the server-backed gallery.
                </div>
            )}
        </div>
    );
}
