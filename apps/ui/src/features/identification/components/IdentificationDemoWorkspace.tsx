import RequestState from "../../../components/RequestState.tsx";
import SurfaceCard from "../../../shared/ui/SurfaceCard.tsx";
import type { AsyncState } from "../../../shared/request-state";
import type {
    CatalogIdentifyGalleryResponse,
    CatalogIdentifyProbeCase,
    IdentificationStatsResponse,
    IdentifyDemoResetResponse,
    IdentifyDemoSeedResponse,
} from "../../../types/index.ts";
import type { DemoSearchFormState } from "../hooks/useIdentification.ts";
import DemoIdentityGalleryPanel from "./DemoIdentityGalleryPanel.tsx";
import DemoStoreControls from "./DemoStoreControls.tsx";
import ProbeCasesPanel from "./ProbeCasesPanel.tsx";
import RunDemoIdentificationAction from "./RunDemoIdentificationAction.tsx";
import SelectedProbePreview from "./SelectedProbePreview.tsx";

interface IdentificationDemoWorkspaceProps {
    demoGalleryState: AsyncState<CatalogIdentifyGalleryResponse>;
    statsState: AsyncState<IdentificationStatsResponse>;
    demoSeedState: AsyncState<IdentifyDemoSeedResponse>;
    demoResetState: AsyncState<IdentifyDemoResetResponse>;
    demoStoreReady: boolean;
    demoSearchForm: DemoSearchFormState;
    probeCases: CatalogIdentifyProbeCase[];
    recentProbeCases: CatalogIdentifyProbeCase[];
    pinnedProbeCases: CatalogIdentifyProbeCase[];
    selectedProbeCase: CatalogIdentifyProbeCase | null;
    selectedProbeCaseId: string | null;
    pinnedProbeCaseIds: string[];
    onRefreshStats: () => void | Promise<void>;
    onRetryGallery: () => void | Promise<void>;
    onSeed: () => void | Promise<void>;
    onReset: () => void | Promise<void>;
    onSelectProbeCase: (probeCase: CatalogIdentifyProbeCase) => void;
    onTogglePinnedProbeCase: (probeCase: CatalogIdentifyProbeCase) => void;
    onUpdateDemoSearchForm: (patch: Partial<DemoSearchFormState>) => void;
    onRun: () => void | Promise<void>;
    busy: boolean;
}

export default function IdentificationDemoWorkspace({
    demoGalleryState,
    statsState,
    demoSeedState,
    demoResetState,
    demoStoreReady,
    demoSearchForm,
    probeCases,
    recentProbeCases,
    pinnedProbeCases,
    selectedProbeCase,
    selectedProbeCaseId,
    pinnedProbeCaseIds,
    onRefreshStats,
    onRetryGallery,
    onSeed,
    onReset,
    onSelectProbeCase,
    onTogglePinnedProbeCase,
    onUpdateDemoSearchForm,
    onRun,
    busy,
}: IdentificationDemoWorkspaceProps) {
    if (demoGalleryState.status === "loading" && !demoGalleryState.data) {
        return (
            <RequestState
                variant="loading"
                title="Loading identify demo gallery"
                description="Fetching demo identities, probe cases, and preview URLs from the server-backed catalog."
            />
        );
    }

    if (demoGalleryState.status === "error" && demoGalleryState.error && !demoGalleryState.data) {
        return (
            <RequestState
                variant="error"
                title="Failed to load identify demo gallery"
                description={demoGalleryState.error}
                actionLabel="Retry"
                onAction={() => {
                    void onRetryGallery();
                }}
            />
        );
    }

    const demoGallery = demoGalleryState.data;
    const demoIdentities = demoGallery?.demo_identities ?? [];

    return (
        <div className="space-y-6">
            <SurfaceCard
                title="Prepare demo store"
                description="Seed or reset only the demo-managed identities, then confirm the store is ready before running 1:N."
            >
                <DemoStoreControls
                    statsState={statsState}
                    demoSeedState={demoSeedState}
                    demoResetState={demoResetState}
                    demoStoreReady={demoStoreReady}
                    onSeed={onSeed}
                    onReset={onReset}
                    onRefresh={onRefreshStats}
                />
            </SurfaceCard>

            <SurfaceCard
                title="Choose probe"
                description="Browse the gallery-backed probe cases, inspect the selected preview, and keep the expected outcome visible."
            >
                <div className="space-y-5">
                    <SelectedProbePreview
                        probeCase={selectedProbeCase}
                        demoStoreReady={demoStoreReady}
                        pinned={Boolean(selectedProbeCase && pinnedProbeCaseIds.includes(selectedProbeCase.id))}
                        onTogglePinned={() => {
                            if (selectedProbeCase) {
                                onTogglePinnedProbeCase(selectedProbeCase);
                            }
                        }}
                    />

                    {pinnedProbeCases.length > 0 ? (
                        <div className="rounded-2xl border border-amber-200 bg-amber-50/60 p-4">
                            <div className="flex flex-wrap items-center justify-between gap-3">
                                <div>
                                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-amber-700">Pinned probes</p>
                                    <p className="mt-1 text-sm text-amber-900">Keep your key demo entries one click away across reloads.</p>
                                </div>
                                <span className="rounded-full border border-amber-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-amber-700">
                                    {pinnedProbeCases.length} pinned
                                </span>
                            </div>

                            <div className="mt-4 grid gap-3 md:grid-cols-2">
                                {pinnedProbeCases.map((probeCase) => (
                                    <div key={probeCase.id} className="rounded-2xl border border-amber-200 bg-white p-4">
                                        <p className="text-sm font-semibold text-slate-900">{probeCase.title}</p>
                                        <p className="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">
                                            {probeCase.dataset_label} / {probeCase.capture ?? "plain"}
                                        </p>

                                        <div className="mt-3 flex flex-wrap gap-2">
                                            <button
                                                type="button"
                                                onClick={() => onSelectProbeCase(probeCase)}
                                                className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                                            >
                                                Select
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => onTogglePinnedProbeCase(probeCase)}
                                                className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm font-medium text-amber-800 transition hover:bg-amber-100"
                                            >
                                                Unpin
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : null}

                    {recentProbeCases.length > 0 ? (
                        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                            <div className="flex flex-wrap items-center justify-between gap-3">
                                <div>
                                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Recent probes</p>
                                    <p className="mt-1 text-sm text-slate-700">Recently selected or executed probes stay available as lightweight continuity only.</p>
                                </div>
                                <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-slate-600">
                                    {recentProbeCases.length} recent
                                </span>
                            </div>

                            <div className="mt-4 flex flex-wrap gap-2">
                                {recentProbeCases.map((probeCase) => (
                                    <button
                                        key={probeCase.id}
                                        type="button"
                                        onClick={() => onSelectProbeCase(probeCase)}
                                        className="rounded-full border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                                    >
                                        {probeCase.title}
                                    </button>
                                ))}
                            </div>
                        </div>
                    ) : null}

                    <ProbeCasesPanel
                        probeCases={probeCases}
                        selectedProbeCaseId={selectedProbeCaseId}
                        onSelect={onSelectProbeCase}
                    />
                </div>
            </SurfaceCard>

            <SurfaceCard
                title="Run guided identification"
                description="The demo still uses the official identify endpoint. Probe selection stays explicit and the run action remains separate."
            >
                <RunDemoIdentificationAction
                    demoSearchForm={demoSearchForm}
                    selectedProbeCase={selectedProbeCase}
                    demoStoreReady={demoStoreReady}
                    busy={busy}
                    onUpdate={onUpdateDemoSearchForm}
                    onRun={onRun}
                />
            </SurfaceCard>

            <SurfaceCard
                title="Seeded gallery"
                description="A clear view of the identities that belong to the demo flow and can be reset safely without touching operational data."
            >
                <DemoIdentityGalleryPanel identities={demoIdentities} selectedProbeCase={selectedProbeCase} />
            </SurfaceCard>
        </div>
    );
}
