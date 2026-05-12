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
                <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Dataset</p>
                    <p className="mt-2 text-sm font-semibold text-[var(--app-text)]">{dataset?.dataset_label ?? "Not selected"}</p>
                </div>
                <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Gallery identities</p>
                    <p className="mt-2 text-sm font-semibold text-[var(--app-text)]">{selectedIdentities.length}</p>
                </div>
                <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Probe asset</p>
                    <p className="mt-2 text-sm font-semibold text-[var(--app-text)]">{selectedProbeAsset?.asset_id ?? "Not selected"}</p>
                </div>
                <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Methods</p>
                    <p className="mt-2 text-sm font-semibold text-[var(--app-text)]">
                        {formatMethodLabel(browserSearchForm.retrievalMethod)} / {formatMethodLabel(browserSearchForm.rerankMethod)}
                    </p>
                </div>
            </div>

            <InlineBanner variant="info">
                Browser Mode still reaches <code>/api/identify/search</code>, but it first seeds the selected catalog identities into
                the isolated browser store so the 1:N run uses a real seeded gallery instead of UI-only state.
            </InlineBanner>

            <div className="grid gap-4 lg:grid-cols-2">
                <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <p className="text-sm font-semibold text-[var(--app-text)]">Selection summary</p>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs font-medium text-[var(--app-text-soft)]">
                        <span className="status-pill">
                            Shortlist {browserSearchForm.shortlistSizeText}
                        </span>
                        {enrollmentCaptures.map((capture) => (
                            <span key={`enroll-${capture}`} className="status-pill">
                                Enroll {capture}
                            </span>
                        ))}
                        {probeCaptures.map((capture) => (
                            <span key={`probe-${capture}`} className="status-pill">
                                Probe {capture}
                            </span>
                        ))}
                        {selectedProbeAsset?.modality ? (
                            <span className="status-pill">
                                {selectedProbeAsset.modality}
                            </span>
                        ) : null}
                    </div>
                </div>

                <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                    <p className="text-sm font-semibold text-[var(--app-text)]">Search controls</p>
                    <p className="mt-1 text-sm leading-6 text-[var(--app-text-soft)]">
                        Tune the browser-backed search context before seeding the gallery and running the official 1:N endpoint.
                    </p>
                </div>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
                <div className="md:col-span-3">
                    <InlineBanner variant="info">
                        Vector retrieval uses embedding-based methods for fast shortlist generation; classic methods are applied during reranking.
                    </InlineBanner>
                </div>

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
                    className="app-button app-button--secondary"
                >
                    <SlidersHorizontal className="mr-2 h-4 w-4" />
                    {browserSearchForm.advancedVisible ? "Hide advanced filters" : "Show advanced filters"}
                </button>

                <div className="flex flex-wrap gap-3">
                    <button
                        type="button"
                        onClick={() => void onRun()}
                        disabled={!canRun}
                        className="app-button app-button--primary"
                    >
                        <Play className="mr-2 h-4 w-4" />
                        {busy ? "Running..." : "Seed gallery and run"}
                    </button>
                    <button
                        type="button"
                        onClick={() => void onResetStore()}
                        disabled={busy}
                        className="app-button app-button--secondary"
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
