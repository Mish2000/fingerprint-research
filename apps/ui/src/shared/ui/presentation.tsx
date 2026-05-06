import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { DatabaseZap } from "lucide-react";

export type PresentationTone = "brand" | "success" | "warning" | "error" | "info" | "neutral";

function toneClassName(base: string, tone: PresentationTone): string {
    return `${base} ${base}--${tone}`;
}

export interface WorkspaceHeroProps {
    eyebrow: string;
    title: string;
    description: string;
    icon?: LucideIcon;
    actions?: ReactNode;
    children?: ReactNode;
    className?: string;
}

export function WorkspaceHero({
    eyebrow,
    title,
    description,
    icon: Icon,
    actions,
    children,
    className = "",
}: WorkspaceHeroProps) {
    return (
        <section className={`workspace-hero ${className}`.trim()}>
            <div className="workspace-hero__main">
                <div className="min-w-0">
                    <div className="workspace-hero__eyebrow">
                        {Icon ? <Icon className="h-3.5 w-3.5" /> : null}
                        <span>{eyebrow}</span>
                    </div>
                    <h2 className="workspace-hero__title">{title}</h2>
                    <p className="workspace-hero__description">{description}</p>
                </div>
                {actions ? <div className="workspace-hero__actions">{actions}</div> : null}
            </div>
            {children ? <div className="workspace-hero__body">{children}</div> : null}
        </section>
    );
}

export interface SectionHeaderProps {
    eyebrow?: string;
    title: string;
    description?: string;
    actions?: ReactNode;
    className?: string;
}

export function SectionHeader({ eyebrow, title, description, actions, className = "" }: SectionHeaderProps) {
    return (
        <div className={`section-header ${className}`.trim()}>
            <div className="min-w-0">
                {eyebrow ? <p className="section-header__eyebrow">{eyebrow}</p> : null}
                <h3 className="section-header__title">{title}</h3>
                {description ? <p className="section-header__description">{description}</p> : null}
            </div>
            {actions ? <div className="section-header__actions">{actions}</div> : null}
        </div>
    );
}

export interface StatusPillProps {
    children: ReactNode;
    tone?: PresentationTone;
    icon?: LucideIcon;
    title?: string;
    className?: string;
}

export function StatusPill({ children, tone = "neutral", icon: Icon, title, className = "" }: StatusPillProps) {
    return (
        <span className={`${toneClassName("status-pill", tone)} ${className}`.trim()} title={title}>
            {Icon ? <Icon className="h-3.5 w-3.5 shrink-0" /> : null}
            <span className="safe-truncate">{children}</span>
        </span>
    );
}

export interface MetricTileProps {
    label: string;
    value: ReactNode;
    detail?: ReactNode;
    icon?: LucideIcon;
    tone?: PresentationTone;
    className?: string;
    title?: string;
}

export function MetricTile({
    label,
    value,
    detail,
    icon: Icon,
    tone = "neutral",
    className = "",
    title,
}: MetricTileProps) {
    return (
        <div className={`${toneClassName("metric-tile", tone)} ${className}`.trim()} title={title}>
            <div className="metric-tile__label">
                {Icon ? <Icon className="h-4 w-4 shrink-0" /> : null}
                <span className="safe-truncate">{label}</span>
            </div>
            <div className="metric-tile__value">{value}</div>
            {detail ? <div className="metric-tile__detail">{detail}</div> : null}
        </div>
    );
}

export interface CompactEmptyStateProps {
    title: string;
    description: string;
    icon?: LucideIcon;
    action?: ReactNode;
    className?: string;
}

export function CompactEmptyState({
    title,
    description,
    icon: Icon = DatabaseZap,
    action,
    className = "",
}: CompactEmptyStateProps) {
    return (
        <div className={`compact-empty-state ${className}`.trim()}>
            <div className="compact-empty-state__icon">
                <Icon className="h-5 w-5" />
            </div>
            <div className="min-w-0">
                <p className="compact-empty-state__title">{title}</p>
                <p className="compact-empty-state__description">{description}</p>
                {action ? <div className="mt-3">{action}</div> : null}
            </div>
        </div>
    );
}
