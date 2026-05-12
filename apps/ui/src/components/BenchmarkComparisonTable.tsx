import { Fragment } from "react";
import type { BenchmarkSortMode, ComparisonRow, ResearchRunGroupInfo } from "../types";
import {
    formatLatency,
    formatMethodLabel,
    formatMetric,
    formatOperatingPoint,
    formatFarLabel,
    highlightClassName,
    isChampionCandidateRow,
    isResearchRow,
    methodStatusBadges,
    researchRunSourceLabel,
    statusLabel,
    statusToneClassName,
} from "../features/benchmark/benchmarkPresentation.ts";

type Props = {
    rows: ComparisonRow[];
    selectedRowKey: string;
    onSelectRow: (rowKey: string) => void;
    rowKey: (row: ComparisonRow) => string;
    sortMode: BenchmarkSortMode;
    researchGroupInfoByRowKey?: Record<string, ResearchRunGroupInfo>;
};

function RankBadge({ children }: { children: string }) {
    return (
        <span className="status-pill status-pill--brand text-[10px]">
            {children}
        </span>
    );
}

function MethodStatusBadges({ row }: { row: ComparisonRow }) {
    const badges = methodStatusBadges(row);
    if (badges.length === 0) {
        return null;
    }
    return (
        <div className="flex flex-wrap gap-1.5">
            {badges.map((badge) => (
                <span
                    key={badge}
                    className={`status-pill text-[10px] ${
                        badge === "Not showcase eligible" ? "status-pill--warning" : "status-pill--info"
                    }`}
                >
                    {badge}
                </span>
            ))}
        </div>
    );
}

function groupedRows(rows: ComparisonRow[]) {
    const canonicalRows = rows.filter((row) => !isResearchRow(row));
    const researchRows = rows.filter((row) => isResearchRow(row));
    if (canonicalRows.length === 0 || researchRows.length === 0) {
        return [{ title: researchRows.length > 0 ? "Research / Experimental" : "Canonical / Showcase", rows }];
    }
    return [
        { title: "Canonical / Showcase", rows: canonicalRows },
        { title: "Research / Experimental", rows: researchRows },
    ];
}

export function BenchmarkComparisonTable({
    rows,
    selectedRowKey,
    onSelectRow,
    rowKey,
    sortMode,
    researchGroupInfoByRowKey = {},
}: Props) {
    return (
        <div className="overflow-hidden rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] shadow-sm">
            <div className="overflow-x-auto">
                <table className="min-w-full border-collapse text-left text-sm text-[var(--app-text-soft)]">
                    <thead>
                        <tr className="border-b border-[var(--app-border)] bg-[var(--app-surface-subtle)] text-[11px] uppercase text-[var(--app-text-muted)]">
                            <th className="px-5 py-4 font-medium">Method</th>
                            <th className="px-5 py-4 text-right font-medium">Accuracy / AUC</th>
                            <th className="px-5 py-4 text-right font-medium">Error / EER</th>
                            <th className="px-5 py-4 text-right font-medium">Operating points</th>
                            <th className="px-5 py-4 text-right font-medium">Speed</th>
                            <th className="px-5 py-4 font-medium">Evidence</th>
                            <th className="px-5 py-4 font-medium">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--app-border-muted)]">
                        {groupedRows(rows).map((group) => (
                            <Fragment key={group.title}>
                                <tr key={`${group.title}_header`} className="bg-[var(--app-surface-muted)]">
                                    <td colSpan={7} className="px-5 py-2 text-[11px] font-semibold uppercase text-[var(--app-text-muted)]">
                                        {group.title}
                                    </td>
                                </tr>
                                {group.rows.map((row, index) => {
                                    const key = rowKey(row);
                                    const isSelected = key === selectedRowKey;
                                    const isHighlighted = index === 0 && isChampionCandidateRow(row);
                                    const researchGroupInfo = researchGroupInfoByRowKey[key];
                                    const isResearch = isResearchRow(row);

                                    return (
                                        <tr
                                            key={key}
                                            className={[
                                                "cursor-pointer align-top transition",
                                                isSelected ? "bg-[var(--app-brand-surface)]" : "hover:bg-[var(--app-surface-subtle)]",
                                                isHighlighted ? highlightClassName(sortMode) : "",
                                            ].join(" ").trim()}
                                            onClick={() => onSelectRow(key)}
                                        >
                                            <td className="px-5 py-4">
                                                <div className="space-y-1">
                                                    <div className="safe-text font-semibold text-[var(--app-text)]">{formatMethodLabel(row.method, row.method_label)}</div>
                                                    <MethodStatusBadges row={row} />
                                                    <div className="safe-text text-xs text-[var(--app-text-muted)]">{row.run_family ?? row.run}</div>
                                                    {isResearch ? (
                                                        <div className="safe-text text-xs font-medium text-[var(--app-text-soft)]">{researchRunSourceLabel(row)}</div>
                                                    ) : null}
                                                    {researchGroupInfo?.hiddenCount > 0 ? (
                                                        <div className="text-xs text-[var(--app-text-muted)]">
                                                            <p>{researchGroupInfo.totalCount} research runs available.</p>
                                                            <p>Showing representative research run; {researchGroupInfo.hiddenCount} archived runs hidden.</p>
                                                        </div>
                                                    ) : null}
                                                    <details className="text-xs text-[var(--app-text-muted)]">
                                                        <summary className="cursor-pointer text-[var(--app-brand-text)]">details</summary>
                                                        <p className="mt-1 text-clamp-2">{row.summary_text}</p>
                                                    </details>
                                                </div>
                                            </td>
                                            <td className="px-5 py-4 text-right">
                                                <div className="space-y-2">
                                                    <div className="font-semibold text-[var(--app-text)]">{formatMetric(row.auc)}</div>
                                                    {isChampionCandidateRow(row) && row.auc_rank === 1 ? <RankBadge>#1 Accuracy</RankBadge> : null}
                                                </div>
                                            </td>
                                            <td className="px-5 py-4 text-right">
                                                <div className="space-y-2">
                                                    <div className="font-semibold text-[var(--app-text)]">{formatMetric(row.eer)}</div>
                                                    {isChampionCandidateRow(row) && row.eer_rank === 1 ? <RankBadge>#1 EER</RankBadge> : null}
                                                </div>
                                            </td>
                                            <td className="px-5 py-4 text-right">
                                                <div className="space-y-1 text-xs text-[var(--app-text-soft)]">
                                                    <div>
                                                        <span className="text-[var(--app-text-muted)]">{formatFarLabel("1e-2", true)}</span>
                                                        <span className="ml-2 font-semibold text-[var(--app-text)]">{formatOperatingPoint(row.tar_at_far_1e_2)}</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-[var(--app-text-muted)]">{formatFarLabel("1e-3", true)}</span>
                                                        <span className="ml-2 font-semibold text-[var(--app-text)]">{formatOperatingPoint(row.tar_at_far_1e_3)}</span>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="px-5 py-4 text-right">
                                                <div className="space-y-2">
                                                    <div className="text-[var(--app-text-soft)]">{formatLatency(row.latency_ms)}</div>
                                                    {isChampionCandidateRow(row) && row.latency_rank === 1 ? <RankBadge>Fastest</RankBadge> : null}
                                                </div>
                                            </td>
                                            <td className="px-5 py-4">
                                                <div className="space-y-1">
                                                    <div className="font-medium text-[var(--app-text-soft)]">{row.artifact_count} artifacts</div>
                                                    <div className="safe-text text-xs text-[var(--app-text-muted)]">{row.run_label ?? "Run evidence"}</div>
                                                </div>
                                            </td>
                                            <td className="px-5 py-4">
                                                <div className="flex flex-col items-start gap-2">
                                                    <span className={`status-pill ${statusToneClassName(row.status)}`}>
                                                        {statusLabel(row.status)}
                                                    </span>
                                                    {row.validation_state === "validated" ? <RankBadge>Validated</RankBadge> : null}
                                                </div>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </Fragment>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
