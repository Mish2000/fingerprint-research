import type { AsyncState } from "../../../shared/request-state/index.ts";
import type { CatalogBrowserItem } from "../../../types/index.ts";
import InlineBanner from "../../../shared/ui/InlineBanner.tsx";
import type { BrowserSelectionSide, PairPreviewState } from "../browserModel.ts";
import SelectedPairPreview from "./SelectedPairPreview.tsx";
import UseAsVerifyPairAction from "./UseAsVerifyPairAction.tsx";

interface PairBuilderPanelProps {
    datasetLabel: string | null;
    selectedAssetA: CatalogBrowserItem | null;
    selectedAssetB: CatalogBrowserItem | null;
    pairPreviewState: PairPreviewState;
    replacementTarget: BrowserSelectionSide | null;
    applyPairState: AsyncState<{ pairKey: string }>;
    isCurrentPairApplied: boolean;
    onClearAsset: (side: BrowserSelectionSide) => void;
    onStartReplacing: (side: BrowserSelectionSide) => void;
    onCancelReplacing: () => void;
    onSwap: () => void;
    onApply: () => void;
}

export default function PairBuilderPanel({
    datasetLabel,
    selectedAssetA,
    selectedAssetB,
    pairPreviewState,
    replacementTarget,
    applyPairState,
    isCurrentPairApplied,
    onClearAsset,
    onStartReplacing,
    onCancelReplacing,
    onSwap,
    onApply,
}: PairBuilderPanelProps) {
    const isPairReady = Boolean(selectedAssetA && selectedAssetB);

    return (
        <div className="space-y-5">
            <div className="rounded-2xl border border-brand-100 bg-brand-50 px-4 py-4 text-sm leading-6 text-brand-900">
                Build a pair in two steps: choose side A, choose side B, then send both server-backed previews into Verify.
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Pair Status</p>
                <p className="mt-2 text-base font-semibold text-slate-900">{pairPreviewState.message}</p>
                <p className="mt-2 text-sm text-slate-600">
                    Current dataset: {datasetLabel ?? "Select a browser-ready dataset"}.
                </p>
            </div>

            {applyPairState.status === "error" && applyPairState.error ? (
                <InlineBanner variant="error" title="Failed to load the selected pair into Verify">
                    {applyPairState.error}
                </InlineBanner>
            ) : null}

            {isCurrentPairApplied ? (
                <InlineBanner variant="success" title="Verify inputs are synced with the selected pair">
                    The probe and reference files now come from the dataset browser. Run Verification stays a separate action.
                </InlineBanner>
            ) : null}

            {isPairReady && !isCurrentPairApplied && applyPairState.status !== "loading" ? (
                <InlineBanner variant="info" title="One more step before running Verify">
                    Use the selected pair as the verify pair to populate the probe/reference files from the server.
                </InlineBanner>
            ) : null}

            <div className="grid gap-4 lg:grid-cols-2">
                <SelectedPairPreview
                    side="A"
                    asset={selectedAssetA}
                    datasetLabel={datasetLabel}
                    isReplacing={replacementTarget === "A"}
                    onClear={() => {
                        onClearAsset("A");
                    }}
                    onReplace={() => {
                        onStartReplacing("A");
                    }}
                    onCancelReplace={onCancelReplacing}
                />

                <SelectedPairPreview
                    side="B"
                    asset={selectedAssetB}
                    datasetLabel={datasetLabel}
                    isReplacing={replacementTarget === "B"}
                    onClear={() => {
                        onClearAsset("B");
                    }}
                    onReplace={() => {
                        onStartReplacing("B");
                    }}
                    onCancelReplace={onCancelReplacing}
                />
            </div>

            <div className="flex flex-wrap items-center gap-3">
                <button
                    type="button"
                    onClick={onSwap}
                    disabled={!isPairReady}
                    className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-60"
                >
                    Swap A / B
                </button>

                <UseAsVerifyPairAction
                    disabled={!isPairReady}
                    applyState={applyPairState}
                    isCurrentPairApplied={isCurrentPairApplied}
                    onApply={onApply}
                />
            </div>
        </div>
    );
}
