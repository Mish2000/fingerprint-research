import type { CatalogDatasetSummary } from "../../../types/index.ts";
import RequestState from "../../../components/RequestState.tsx";

interface DatasetPickerProps {
    datasets: CatalogDatasetSummary[];
    selectedDataset: CatalogDatasetSummary | null;
    loading: boolean;
    error: string | null;
    onSelectDataset: (dataset: CatalogDatasetSummary) => void;
    onRetry: () => void;
    readyDescription?: string;
    unavailableDescription?: string;
    emptyTitle?: string;
    emptyDescription?: string;
}

function featureLabel(value: string): string {
    if (value === "dataset_browser") {
        return "Browser";
    }
    if (value === "verify_cases") {
        return "Verify";
    }
    if (value === "identify_gallery") {
        return "Identify";
    }
    return value.replace(/[_-]/g, " ");
}

function demoHealthLabel(status: NonNullable<CatalogDatasetSummary["demo_health"]>["status"]): string {
    if (status === "healthy") {
        return "Healthy demo evidence";
    }
    if (status === "incomplete") {
        return "Incomplete demo setup";
    }
    return "Degraded demo evidence";
}

function demoHealthBadgeClass(status: NonNullable<CatalogDatasetSummary["demo_health"]>["status"]): string {
    if (status === "healthy") {
        return "border border-emerald-200 bg-emerald-50 text-emerald-700";
    }
    if (status === "incomplete") {
        return "border border-amber-200 bg-amber-50 text-amber-700";
    }
    return "border border-orange-200 bg-orange-50 text-orange-700";
}

export default function DatasetPicker({
    datasets,
    selectedDataset,
    loading,
    error,
    onSelectDataset,
    onRetry,
    readyDescription = "Choose this dataset to browse real assets, build a pair, and send it directly to Verify.",
    unavailableDescription = "Browser assets are not available for this dataset yet, so the full browser flow stays disabled.",
    emptyTitle = "No browser-ready datasets",
    emptyDescription = "The catalog does not currently expose any datasets with server-backed browser assets.",
}: DatasetPickerProps) {
    if (loading && datasets.length === 0) {
        return (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {Array.from({ length: 3 }).map((_, index) => (
                    <div
                        key={index}
                        className="h-36 animate-pulse rounded-2xl border border-slate-200 bg-slate-100"
                    />
                ))}
            </div>
        );
    }

    if (error && datasets.length === 0) {
        return (
            <RequestState
                variant="error"
                title="Failed to load browser-ready datasets"
                description={error}
                actionLabel="Retry"
                onAction={onRetry}
            />
        );
    }

    if (datasets.length === 0) {
        return (
            <RequestState
                variant="empty"
                title={emptyTitle}
                description={emptyDescription}
            />
        );
    }

    return (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {datasets.map((dataset) => {
                const isSelected = selectedDataset?.dataset === dataset.dataset;
                const isDisabled = !dataset.has_browser_assets;

                return (
                    <button
                        key={dataset.dataset}
                        type="button"
                        onClick={() => {
                            if (!isDisabled) {
                                onSelectDataset(dataset);
                            }
                        }}
                        disabled={isDisabled}
                        aria-pressed={isSelected}
                        className={[
                            "rounded-2xl border p-4 text-left transition",
                            isSelected
                                ? "border-brand-300 bg-brand-50 ring-2 ring-brand-100"
                                : "border-slate-200 bg-white hover:border-slate-300",
                            isDisabled ? "cursor-not-allowed opacity-60 hover:border-slate-200" : "",
                        ].join(" ")}
                    >
                        <div className="flex items-start justify-between gap-3">
                            <div>
                                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Dataset</p>
                                <h3 className="mt-2 text-lg font-semibold text-slate-900">{dataset.dataset_label}</h3>
                                <p className="mt-1 text-sm text-slate-500">{dataset.dataset}</p>
                            </div>
                            <div className="flex flex-col items-end gap-2">
                                <span
                                    className={[
                                        "rounded-full px-2.5 py-1 text-xs font-semibold",
                                        dataset.has_browser_assets
                                            ? "border border-emerald-200 bg-emerald-50 text-emerald-700"
                                            : "border border-slate-200 bg-slate-100 text-slate-500",
                                    ].join(" ")}
                                >
                                    {dataset.has_browser_assets ? "Ready" : "Unavailable"}
                                </span>
                                {dataset.demo_health ? (
                                    <span
                                        className={[
                                            "rounded-full px-2.5 py-1 text-xs font-semibold",
                                            demoHealthBadgeClass(dataset.demo_health.status),
                                        ].join(" ")}
                                    >
                                        {demoHealthLabel(dataset.demo_health.status)}
                                    </span>
                                ) : null}
                            </div>
                        </div>

                        <div className="mt-4 grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
                            <div>
                                <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Browser Items</p>
                                <p className="mt-1 font-semibold text-slate-900">{dataset.browser_item_count}</p>
                            </div>
                            <div>
                                <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">Validation</p>
                                <p className="mt-1 font-semibold text-slate-900">{dataset.browser_validation_status ?? "n/a"}</p>
                            </div>
                        </div>

                        {dataset.available_features.length > 0 ? (
                            <div className="mt-4 flex flex-wrap gap-2">
                                {dataset.available_features.map((feature) => (
                                    <span
                                        key={feature}
                                        className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600"
                                    >
                                        {featureLabel(feature)}
                                    </span>
                                ))}
                            </div>
                        ) : null}

                        {dataset.demo_health && dataset.demo_health.status !== "healthy" ? (
                            <div className="mt-4 rounded-2xl border border-orange-100 bg-orange-50/70 px-3 py-3 text-sm text-slate-700">
                                <p className="font-medium text-slate-900">{dataset.demo_health.note}</p>
                                <p className="mt-2 text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
                                    {dataset.demo_health.built_verify_cases} built
                                    {" · "}
                                    {dataset.demo_health.benchmark_backed_cases} benchmark-backed
                                    {" · "}
                                    {dataset.demo_health.heuristic_fallback_cases} fallback
                                </p>
                            </div>
                        ) : null}

                        <p className="mt-4 text-sm leading-6 text-slate-600">
                            {dataset.has_browser_assets
                                ? readyDescription
                                : unavailableDescription}
                        </p>
                    </button>
                );
            })}
        </div>
    );
}
