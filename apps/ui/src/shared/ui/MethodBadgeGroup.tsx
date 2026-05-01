import { Cpu, Search } from "lucide-react";

export interface MethodBadgeItem {
    label: string;
    value: string;
    emphasis?: "active" | "recommended" | "override" | "inactive";
    detail?: string | null;
}

interface MethodBadgeGroupProps {
    title?: string;
    items: MethodBadgeItem[];
}

function emphasisClassName(value: NonNullable<MethodBadgeItem["emphasis"]>): string {
    return `method-badge--${value}`;
}

export default function MethodBadgeGroup({ title = "Methods", items }: MethodBadgeGroupProps) {
    if (items.length === 0) {
        return null;
    }

    return (
        <section className="space-y-3">
            <div className="method-badge-group-title">
                <Cpu className="h-4 w-4 text-brand-600" />
                {title}
            </div>
            <div className="flex flex-wrap gap-3">
                {items.map((item) => (
                    <div
                        key={`${item.label}_${item.value}`}
                        className={`method-badge ${emphasisClassName(item.emphasis ?? "active")}`}
                    >
                        <div className="method-badge__label">
                            <Search className="h-3.5 w-3.5" />
                            {item.label}
                        </div>
                        <div className="method-badge__value">{item.value}</div>
                        {item.detail ? <div className="method-badge__detail">{item.detail}</div> : null}
                    </div>
                ))}
            </div>
        </section>
    );
}
