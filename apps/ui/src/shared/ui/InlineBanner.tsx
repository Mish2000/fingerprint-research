import type { ReactNode } from "react";
import { AlertCircle, CheckCircle2, Info, TriangleAlert } from "lucide-react";

export type InlineBannerVariant = "info" | "success" | "warning" | "error";

interface InlineBannerProps {
    variant: InlineBannerVariant;
    title?: string;
    children: ReactNode;
}

function paletteFor(variant: InlineBannerVariant): string {
    return `inline-banner--${variant}`;
}

function iconFor(variant: InlineBannerVariant): ReactNode {
    switch (variant) {
        case "success":
            return <CheckCircle2 className="h-5 w-5" />;
        case "warning":
            return <TriangleAlert className="h-5 w-5" />;
        case "error":
            return <AlertCircle className="h-5 w-5" />;
        case "info":
        default:
            return <Info className="h-5 w-5" />;
    }
}

function liveRoleFor(variant: InlineBannerVariant): "alert" | "status" {
    return variant === "error" || variant === "warning" ? "alert" : "status";
}

function liveModeFor(variant: InlineBannerVariant): "assertive" | "polite" {
    return variant === "error" ? "assertive" : "polite";
}

export default function InlineBanner({ variant, title, children }: InlineBannerProps) {
    return (
        <div
            className={`inline-banner ${paletteFor(variant)}`}
            role={liveRoleFor(variant)}
            aria-live={liveModeFor(variant)}
            aria-atomic="true"
        >
            <div className="inline-banner__icon">{iconFor(variant)}</div>
            <div className="inline-banner__body">
                {title ? <p className="inline-banner__title">{title}</p> : null}
                <div>{children}</div>
            </div>
        </div>
    );
}
