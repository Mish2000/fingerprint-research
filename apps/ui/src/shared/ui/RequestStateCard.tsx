import type { ReactNode } from "react";
import { AlertCircle, DatabaseZap, LoaderCircle } from "lucide-react";

export type RequestStateVariant = "loading" | "error" | "empty";

interface RequestStateCardProps {
    variant: RequestStateVariant;
    title: string;
    description: string;
    actionLabel?: string;
    onAction?: () => void;
    children?: ReactNode;
}

function variantIcon(variant: RequestStateVariant): ReactNode {
    switch (variant) {
        case "loading":
            return <LoaderCircle className="h-10 w-10 animate-spin" />;
        case "error":
            return <AlertCircle className="h-10 w-10" />;
        case "empty":
            return <DatabaseZap className="h-10 w-10" />;
        default:
            return null;
    }
}

function variantPalette(variant: RequestStateVariant): string {
    return `request-state-card--${variant}`;
}

function liveRoleFor(variant: RequestStateVariant): "alert" | "status" {
    return variant === "error" ? "alert" : "status";
}

function liveModeFor(variant: RequestStateVariant): "assertive" | "polite" {
    return variant === "error" ? "assertive" : "polite";
}

export default function RequestStateCard({
                                             variant,
                                             title,
                                             description,
                                             actionLabel,
                                             onAction,
                                             children,
                                         }: RequestStateCardProps) {
    return (
        <div
            className={`request-state-card ${variantPalette(variant)}`}
            role={liveRoleFor(variant)}
            aria-live={liveModeFor(variant)}
            aria-atomic="true"
        >
            <div className="request-state-card__inner">
                <div className="request-state-card__icon-shell">{variantIcon(variant)}</div>
                <div className="space-y-2">
                    <h3 className="request-state-card__title">{title}</h3>
                    <p className="request-state-card__description">{description}</p>
                </div>
                {children}
                {actionLabel && onAction ? (
                    <button
                        type="button"
                        onClick={onAction}
                        className="request-state-card__button"
                    >
                        {actionLabel}
                    </button>
                ) : null}
            </div>
        </div>
    );
}
