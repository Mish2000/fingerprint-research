import { Database, RefreshCcw, RotateCcw } from "lucide-react";
import InlineBanner from "../../../shared/ui/InlineBanner.tsx";
import StatCard from "../../../shared/ui/StatCard.tsx";
import type { AsyncState } from "../../../shared/request-state";
import type { IdentificationStatsResponse, IdentifyDemoResetResponse, IdentifyDemoSeedResponse } from "../../../types/index.ts";

interface DemoStoreControlsProps {
    statsState: AsyncState<IdentificationStatsResponse>;
    demoSeedState: AsyncState<IdentifyDemoSeedResponse>;
    demoResetState: AsyncState<IdentifyDemoResetResponse>;
    demoStoreReady: boolean;
    onSeed: () => void | Promise<void>;
    onReset: () => void | Promise<void>;
    onRefresh: () => void | Promise<void>;
}

export default function DemoStoreControls({
    statsState,
    demoSeedState,
    demoResetState,
    demoStoreReady,
    onSeed,
    onReset,
    onRefresh,
}: DemoStoreControlsProps) {
    const isBusy = demoSeedState.status === "loading" || demoResetState.status === "loading" || statsState.status === "loading";

    return (
        <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
                <StatCard
                    icon={Database}
                    label="Demo Store"
                    value={demoStoreReady ? "Seeded" : "Not seeded"}
                    tone={demoStoreReady ? "emerald" : "amber"}
                />
                <StatCard
                    icon={Database}
                    label="Demo seeded identities"
                    value={String(statsState.data?.demo_seeded_count ?? 0)}
                    tone="brand"
                />
                <StatCard
                    icon={Database}
                    label="Total enrolled"
                    value={String(statsState.data?.total_enrolled ?? 0)}
                    tone="slate"
                />
            </div>

            <div className="flex flex-wrap gap-3">
                <button
                    type="button"
                    onClick={() => {
                        void onSeed();
                    }}
                    disabled={isBusy}
                    className="inline-flex items-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-brand-300"
                >
                    <RefreshCcw className="mr-2 h-4 w-4" />
                    {demoSeedState.status === "loading" ? "Seeding..." : "Seed demo identities"}
                </button>

                <button
                    type="button"
                    onClick={() => {
                        void onReset();
                    }}
                    disabled={isBusy}
                    className="inline-flex items-center rounded-xl border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm font-medium text-amber-900 transition hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-60"
                >
                    <RotateCcw className="mr-2 h-4 w-4" />
                    {demoResetState.status === "loading" ? "Resetting..." : "Reset demo store"}
                </button>

                <button
                    type="button"
                    onClick={() => {
                        void onRefresh();
                    }}
                    disabled={statsState.status === "loading"}
                    className="inline-flex items-center rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                >
                    <RefreshCcw className="mr-2 h-4 w-4" />
                    Refresh stats
                </button>
            </div>

            {demoSeedState.status === "success" && demoSeedState.data ? (
                <InlineBanner variant="success" title="Demo seeding completed">
                    Seeded {demoSeedState.data.seeded_count}, refreshed {demoSeedState.data.updated_count}, skipped {demoSeedState.data.skipped_count}.
                </InlineBanner>
            ) : null}

            {demoResetState.status === "success" && demoResetState.data ? (
                <InlineBanner variant="success" title="Demo reset completed">
                    Removed {demoResetState.data.removed_count} demo-seeded identit{demoResetState.data.removed_count === 1 ? "y" : "ies"}.
                </InlineBanner>
            ) : null}

            {demoSeedState.status === "error" && demoSeedState.error ? (
                <InlineBanner variant="error" title="Demo seeding failed">
                    {demoSeedState.error}
                </InlineBanner>
            ) : null}

            {demoResetState.status === "error" && demoResetState.error ? (
                <InlineBanner variant="error" title="Demo reset failed">
                    {demoResetState.error}
                </InlineBanner>
            ) : null}
        </div>
    );
}
