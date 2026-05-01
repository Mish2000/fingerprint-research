import { Play, SlidersHorizontal } from "lucide-react";
import InlineBanner from "../../../shared/ui/InlineBanner.tsx";
import FormField from "../../../shared/ui/FormField.tsx";
import { INPUT_CLASS_NAME } from "../../../shared/ui/inputClasses.ts";
import { formatMethodLabel } from "../../../shared/storytelling.ts";
import type { CatalogBrowserItem, CatalogDatasetSummary, CatalogIdentityItem } from "../../../types/index.ts";
import type { BrowserSearchFormState } from "../hooks/useIdentification.ts";
import { IDENTIFICATION_RETRIEVAL_OPTIONS, IDENTIFICATION_RERANK_OPTIONS } from "../methodOptions.ts";

interface IdentificationBrowserSummaryPanelProps {
    dataset: CatalogDatasetSummary | null;
    selectedIdentities: CatalogIdentityItem[];
    selectedProbeAsset: CatalogBrowserItem | null;
    browserSearchForm: BrowserSearchFormState;
    warnings: string[];
    busy: boolean;
    browserSeedNotice: string | null;
    onUpdate: (patch: Partial<BrowserSearchFormState>) => void;
    onRun: () => void | Promise<void>;
    onResetStore: () => void | Promise<void>;
}

function uniqueValues(values: Array<string | null | undefined>): string[] {
    return Array.from(new Set(values.filter((value): value is string => Boolean(value))));
}

export default function IdentificationBrowserSummaryPanel({
    dataset,
    selectedIdentities,
    selectedProbeAsset,
    browserSearchForm,
    warnings,
    busy,
    browserSeedNotice,
    onUpdate,
    onRun,
    onResetStore,
}: IdentificationBrowserSummaryPanelProps) {
    const enrollmentCaptures = uniqueValues(selectedIdentities.map((identity) => identity.recommended_enrollment_capture));
    const probeCaptures = uniqueValues([
        ...selectedIdentities.map((identity) => identity.recommended_probe_capture),
        selectedProbeAsset?.capture,
    ]);
    const datasetMismatch = Boolean(
        dataset
        && selectedProbeAsset
        && selectedProbeAsset.dataset !== dataset.dataset,
    );
    const canRun = Boolean(dataset) && selectedIdentities.length > 0 && Boolean(selectedProbeAsset) && !datasetMismatch && !busy;

    return (
        <div className="space-y-5">
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Dataset</p>
                    <p className="mt-2 text-sm font-semibold text-slate-900">{dataset?.dataset_label ?? "Not selected"}</p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Gallery identities</p>
                    <p className="mt-2 text-sm font-semibold text-slate-900">{selectedIdentities.length}</p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Probe asset</p>
                    <p className="mt-2 text-sm font-semibold text-slate-900">{selectedProbeAsset?.asset_id ?? "Not selected"}</p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Methods</p>
                    <p className="mt-2 text-sm font-semibold text-slate-900">
                        {formatMethodLabel(browserSearchForm.retrievalMethod)} / {formatMethodLabel(browserSearchForm.rerankMethod)}
                    </p>
                </div>
            </div>

            <div className="rounded-2xl border border-brand-100 bg-brand-50 px-4 py-4 text-sm leading-6 text-brand-900">
                Browser Mode still reaches <code>/api/identify/search</code>, but it first seeds the selected catalog identities into
                the isolated browser store so the 1:N run uses a real seeded gallery instead of UI-only state.
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
                <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <p className="text-sm font-semibold text-slate-900">Selection summary</p>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs font-medium text-slate-600">
                        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
                            Shortlist {browserSearchForm.shortlistSizeText}
                        </span>
                        {enrollmentCaptures.map((capture) => (
                            <span key={`enroll-${capture}`} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
                                Enroll {capture}
                            </span>
                        ))}
                        {probeCaptures.map((capture) => (
                            <span key={`probe-${capture}`} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
                                Probe {capture}
                            </span>
                        ))}
                        {selectedProbeAsset?.modality ? (
                            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
                                {selectedProbeAsset.modality}
                            </span>
                        ) : null}
                    </div>
                </div>

                <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <p className="text-sm font-semibold text-slate-900">Search controls</p>
                    <p className="mt-1 text-sm leading-6 text-slate-600">
                        Tune the browser-backed search context before seeding the gallery and running the official 1:N endpoint.
                    </p>
                </div>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
                <FormField label="Retrieval method">
                    <select
                        className={INPUT_CLASS_NAME}
                        value={browserSearchForm.retrievalMethod}
                        disabled={busy}
                        onChange={(event) => {
                            onUpdate({ retrievalMethod: event.target.value as BrowserSearchFormState["retrievalMethod"] });
                        }}
                    >
                        {IDENTIFICATION_RETRIEVAL_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                    </select>
                </FormField>

                <FormField label="Re-rank method">
                    <select
                        className={INPUT_CLASS_NAME}
                        value={browserSearchForm.rerankMethod}
                        disabled={busy}
                        onChange={(event) => {
                            onUpdate({ rerankMethod: event.target.value as BrowserSearchFormState["rerankMethod"] });
                        }}
                    >
                        {IDENTIFICATION_RERANK_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                    </select>
                </FormField>

                <FormField label="Shortlist size">
                    <input
                        className={INPUT_CLASS_NAME}
                        value={browserSearchForm.shortlistSizeText}
                        disabled={busy}
                        onChange={(event) => {
                            onUpdate({ shortlistSizeText: event.target.value });
                        }}
                    />
                </FormField>
            </div>

            <div className="flex flex-wrap gap-3">
                <button
                    type="button"
                    onClick={() => {
                        onUpdate({ advancedVisible: !browserSearchForm.advancedVisible });
                    }}
                    className="inline-flex items-center rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                >
                    <SlidersHorizontal className="mr-2 h-4 w-4" />
                    {browserSearchForm.advancedVisible ? "Hide advanced filters" : "Show advanced filters"}
                </button>

                <div className="flex flex-wrap gap-3">
                    <button
                        type="button"
                        onClick={() => void onRun()}
                        disabled={!canRun}
                        className="inline-flex items-center rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        <Play className="mr-2 h-4 w-4" />
                        {busy ? "Running..." : "Seed gallery and run"}
                    </button>
                    <button
                        type="button"
                        onClick={() => void onResetStore()}
                        disabled={busy}
                        className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        Reset browser store
                    </button>
                </div>
            </div>

            {browserSearchForm.advancedVisible ? (
                <div className="grid gap-4 md:grid-cols-2">
                    <FormField label="Threshold" hint="Leave empty to use the backend default.">
                        <input
                            className={INPUT_CLASS_NAME}
                            value={browserSearchForm.thresholdText}
                            disabled={busy}
                            onChange={(event) => {
                                onUpdate({ thresholdText: event.target.value });
                            }}
                        />
                    </FormField>

                    <FormField label="Name pattern">
                        <input
                            className={INPUT_CLASS_NAME}
                            value={browserSearchForm.namePattern}
                            disabled={busy}
                            onChange={(event) => {
                                onUpdate({ namePattern: event.target.value });
                            }}
                        />
                    </FormField>

                    <FormField label="National ID pattern">
                        <input
                            className={INPUT_CLASS_NAME}
                            value={browserSearchForm.nationalIdPattern}
                            disabled={busy}
                            onChange={(event) => {
                                onUpdate({ nationalIdPattern: event.target.value });
                            }}
                        />
                    </FormField>

                    <FormField label="Created from">
                        <input
                            type="date"
                            className={INPUT_CLASS_NAME}
                            value={browserSearchForm.createdFrom}
                            disabled={busy}
                            onChange={(event) => {
                                onUpdate({ createdFrom: event.target.value });
                            }}
                        />
                    </FormField>

                    <FormField label="Created to">
                        <input
                            type="date"
                            className={INPUT_CLASS_NAME}
                            value={browserSearchForm.createdTo}
                            disabled={busy}
                            onChange={(event) => {
                                onUpdate({ createdTo: event.target.value });
                            }}
                        />
                    </FormField>
                </div>
            ) : null}

            {browserSeedNotice ? (
                <InlineBanner variant="info">{browserSeedNotice}</InlineBanner>
            ) : null}

            {warnings.map((warning) => (
                <InlineBanner key={warning} variant="warning">
                    {warning}
                </InlineBanner>
            ))}
        </div>
    );
}
