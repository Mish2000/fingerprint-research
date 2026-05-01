import { Play, SlidersHorizontal } from "lucide-react";
import FormField from "../../../shared/ui/FormField.tsx";
import { INPUT_CLASS_NAME } from "../../../shared/ui/inputClasses.ts";
import type { CatalogIdentifyProbeCase } from "../../../types/index.ts";
import type { DemoSearchFormState } from "../hooks/useIdentification.ts";
import { IDENTIFICATION_RETRIEVAL_OPTIONS, IDENTIFICATION_RERANK_OPTIONS } from "../methodOptions.ts";

interface RunDemoIdentificationActionProps {
    demoSearchForm: DemoSearchFormState;
    selectedProbeCase: CatalogIdentifyProbeCase | null;
    demoStoreReady: boolean;
    busy: boolean;
    onUpdate: (patch: Partial<DemoSearchFormState>) => void;
    onRun: () => void | Promise<void>;
}

export default function RunDemoIdentificationAction({
    demoSearchForm,
    selectedProbeCase,
    demoStoreReady,
    busy,
    onUpdate,
    onRun,
}: RunDemoIdentificationActionProps) {
    const canRun = Boolean(selectedProbeCase) && demoStoreReady && !busy;

    return (
        <div className="space-y-5">
            <div className="rounded-2xl border border-brand-100 bg-brand-50 px-4 py-4 text-sm leading-6 text-brand-900">
                The probe stays server-backed, the search still finishes through <code>/api/identify/search</code>, and you can
                override retrieval, rerank, or shortlist without losing the guided flow.
            </div>

            <div className="grid gap-4 md:grid-cols-3">
                <FormField label="Retrieval method">
                    <select
                        className={INPUT_CLASS_NAME}
                        value={demoSearchForm.retrievalMethod}
                        disabled={busy}
                        onChange={(event) => {
                            onUpdate({ retrievalMethod: event.target.value as DemoSearchFormState["retrievalMethod"] });
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
                        value={demoSearchForm.rerankMethod}
                        disabled={busy}
                        onChange={(event) => {
                            onUpdate({ rerankMethod: event.target.value as DemoSearchFormState["rerankMethod"] });
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
                        value={demoSearchForm.shortlistSizeText}
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
                        onUpdate({ advancedVisible: !demoSearchForm.advancedVisible });
                    }}
                    className="inline-flex items-center rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                >
                    <SlidersHorizontal className="mr-2 h-4 w-4" />
                    {demoSearchForm.advancedVisible ? "Hide advanced filters" : "Show advanced filters"}
                </button>

                <button
                    type="button"
                    onClick={() => {
                        void onRun();
                    }}
                    disabled={!canRun}
                    className="inline-flex items-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-brand-300"
                >
                    <Play className="mr-2 h-4 w-4" />
                    {busy ? "Running identification..." : "Run identification"}
                </button>
            </div>

            {demoSearchForm.advancedVisible ? (
                <div className="grid gap-4 md:grid-cols-2">
                    <FormField label="Threshold" hint="Leave empty to use the backend default.">
                        <input
                            className={INPUT_CLASS_NAME}
                            value={demoSearchForm.thresholdText}
                            disabled={busy}
                            onChange={(event) => {
                                onUpdate({ thresholdText: event.target.value });
                            }}
                        />
                    </FormField>

                    <FormField label="Name pattern">
                        <input
                            className={INPUT_CLASS_NAME}
                            value={demoSearchForm.namePattern}
                            disabled={busy}
                            onChange={(event) => {
                                onUpdate({ namePattern: event.target.value });
                            }}
                        />
                    </FormField>

                    <FormField label="National ID pattern">
                        <input
                            className={INPUT_CLASS_NAME}
                            value={demoSearchForm.nationalIdPattern}
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
                            value={demoSearchForm.createdFrom}
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
                            value={demoSearchForm.createdTo}
                            disabled={busy}
                            onChange={(event) => {
                                onUpdate({ createdTo: event.target.value });
                            }}
                        />
                    </FormField>
                </div>
            ) : null}
        </div>
    );
}
