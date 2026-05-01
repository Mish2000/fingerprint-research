import FormField from "../../../shared/ui/FormField.tsx";
import { INPUT_CLASS_NAME } from "../../../shared/ui/inputClasses.ts";
import type { BrowserFilterOptions, BrowserFilters as BrowserFiltersState, BrowserUiEligibleFilter } from "../browserModel.ts";

interface BrowserFiltersProps {
    filters: BrowserFiltersState;
    options: BrowserFilterOptions;
    activeFilterCount: number;
    disabled?: boolean;
    onChange: (patch: Partial<BrowserFiltersState>) => void;
    onReset: () => void;
}

const LIMIT_OPTIONS = [24, 48, 96];

export default function BrowserFilters({
    filters,
    options,
    activeFilterCount,
    disabled = false,
    onChange,
    onReset,
}: BrowserFiltersProps) {
    return (
        <div className="space-y-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <p className="text-sm font-semibold text-slate-900">Browser Filters</p>
                    <p className="mt-1 text-sm text-slate-600">
                        Narrow by metadata from the catalog browser endpoint. Active filters: {activeFilterCount}.
                    </p>
                </div>

                <button
                    type="button"
                    onClick={onReset}
                    disabled={disabled}
                    className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-60"
                >
                    Reset filters
                </button>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <FormField label="Split" hint="Shown when split metadata exists for the dataset.">
                    <select
                        className={INPUT_CLASS_NAME}
                        value={filters.split}
                        disabled={disabled}
                        onChange={(event) => {
                            onChange({ split: event.target.value });
                        }}
                    >
                        <option value="">All splits</option>
                        {options.splits.map((option) => (
                            <option key={option} value={option}>
                                {option}
                            </option>
                        ))}
                    </select>
                </FormField>

                <FormField label="Capture" hint="Capture metadata stays dataset-aware and optional.">
                    <select
                        className={INPUT_CLASS_NAME}
                        value={filters.capture}
                        disabled={disabled}
                        onChange={(event) => {
                            onChange({ capture: event.target.value });
                        }}
                    >
                        <option value="">All captures</option>
                        {options.captures.map((option) => (
                            <option key={option} value={option}>
                                {option}
                            </option>
                        ))}
                    </select>
                </FormField>

                <FormField label="Modality" hint="Only values exposed by the current dataset appear here.">
                    <select
                        className={INPUT_CLASS_NAME}
                        value={filters.modality}
                        disabled={disabled}
                        onChange={(event) => {
                            onChange({ modality: event.target.value });
                        }}
                    >
                        <option value="">All modalities</option>
                        {options.modalities.map((option) => (
                            <option key={option} value={option}>
                                {option}
                            </option>
                        ))}
                    </select>
                </FormField>

                <FormField label="UI Eligible" hint="Maps directly to the ui_eligible query parameter.">
                    <select
                        className={INPUT_CLASS_NAME}
                        value={filters.uiEligible}
                        disabled={disabled}
                        onChange={(event) => {
                            onChange({ uiEligible: event.target.value as BrowserUiEligibleFilter });
                        }}
                    >
                        <option value="all">All items</option>
                        <option value="eligible">Eligible only</option>
                        <option value="ineligible">Ineligible only</option>
                    </select>
                </FormField>

                <FormField label="Subject ID" hint="Free text because not every dataset exposes a fixed list.">
                    <input
                        className={INPUT_CLASS_NAME}
                        value={filters.subjectId}
                        disabled={disabled}
                        placeholder="e.g. 100001"
                        onChange={(event) => {
                            onChange({ subjectId: event.target.value });
                        }}
                    />
                </FormField>

                <FormField label="Finger" hint="Use the catalog finger identifier when available.">
                    <input
                        className={INPUT_CLASS_NAME}
                        value={filters.finger}
                        disabled={disabled}
                        placeholder="e.g. 1"
                        onChange={(event) => {
                            onChange({ finger: event.target.value });
                        }}
                    />
                </FormField>

                <FormField label="Sort" hint="Uses the backend-supported browser sort values only.">
                    <select
                        className={INPUT_CLASS_NAME}
                        value={filters.sort}
                        disabled={disabled}
                        onChange={(event) => {
                            onChange({ sort: event.target.value as BrowserFiltersState["sort"] });
                        }}
                    >
                        <option value="default">Default order</option>
                        <option value="split_subject_asset">Split / subject / asset</option>
                    </select>
                </FormField>

                <FormField label="Page Size" hint="Keeps the browser paginated instead of loading whole datasets.">
                    <select
                        className={INPUT_CLASS_NAME}
                        value={String(filters.limit)}
                        disabled={disabled}
                        onChange={(event) => {
                            onChange({ limit: Number(event.target.value) });
                        }}
                    >
                        {LIMIT_OPTIONS.map((option) => (
                            <option key={option} value={option}>
                                {option} items
                            </option>
                        ))}
                    </select>
                </FormField>
            </div>
        </div>
    );
}
