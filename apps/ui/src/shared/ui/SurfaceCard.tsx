import type { ReactNode } from "react";

interface SurfaceCardProps {
    title?: string;
    description?: string;
    actions?: ReactNode;
    children: ReactNode;
    className?: string;
}

export default function SurfaceCard({ title, description, actions, children, className = "" }: SurfaceCardProps) {
    return (
        <section className={`surface-card ${className}`.trim()}>
            {title || description || actions ? (
                <div className="surface-card__header">
                    <div>
                        {title ? <h3 className="surface-card__title">{title}</h3> : null}
                        {description ? <p className="surface-card__description">{description}</p> : null}
                    </div>
                    {actions ? <div className="shrink-0">{actions}</div> : null}
                </div>
            ) : null}
            <div className="surface-card__body">{children}</div>
        </section>
    );
}
