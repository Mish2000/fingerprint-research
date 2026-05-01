import { LoaderCircle } from "lucide-react";
import type { AsyncState } from "../../../shared/request-state/index.ts";

interface UseAsVerifyPairActionProps {
    disabled: boolean;
    applyState: AsyncState<{ pairKey: string }>;
    isCurrentPairApplied: boolean;
    onApply: () => void;
}

export default function UseAsVerifyPairAction({
    disabled,
    applyState,
    isCurrentPairApplied,
    onApply,
}: UseAsVerifyPairActionProps) {
    const isLoading = applyState.status === "loading";
    const label = isLoading
        ? "Loading pair..."
        : isCurrentPairApplied
            ? "Pair loaded into Verify"
            : "Use as verify pair";

    return (
        <button
            type="button"
            onClick={onApply}
            disabled={disabled || isLoading || isCurrentPairApplied}
            className="inline-flex items-center rounded-xl bg-brand-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-brand-300"
        >
            {isLoading ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : null}
            {label}
        </button>
    );
}
