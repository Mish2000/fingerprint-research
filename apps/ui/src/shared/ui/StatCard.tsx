import type { LucideIcon } from "lucide-react";

interface StatCardProps {
    icon: LucideIcon;
    label: string;
    value: string;
    tone?: "brand" | "emerald" | "slate" | "amber";
}

function toneClassName(tone: NonNullable<StatCardProps["tone"]>): string {
    return `stat-card--${tone}`;
}

export default function StatCard({ icon: Icon, label, value, tone = "brand" }: StatCardProps) {
    return (
        <div className={`stat-card ${toneClassName(tone)}`}>
            <div className="stat-card__content">
                <div className="stat-card__icon">
                    <Icon className="h-4 w-4" />
                </div>
                <div>
                    <p className="stat-card__label">{label}</p>
                    <p className="stat-card__value">{value}</p>
                </div>
            </div>
        </div>
    );
}
