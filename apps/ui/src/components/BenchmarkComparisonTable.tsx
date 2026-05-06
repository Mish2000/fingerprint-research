import type { BenchmarkSortMode, ComparisonRow } from "../types";
import {
    formatLatency,
    formatMethodLabel,
    formatMetric,
    highlightClassName,
    statusLabel,
    statusToneClassName,
} from "../features/benchmark/benchmarkPresentation.ts";

type Props = {
    rows: ComparisonRow[];
    selectedRowKey: string;
    onSelectRow: (rowKey: string) => void;
    rowKey: (row: ComparisonRow) => string;
    sortMode: BenchmarkSortMode;
};

function RankBadge({ children }: { children: string }) {
    return (
        <span className="status-pill status-pill--brand text-[10px]">
            {children}
        </span>
    );
}

export function BenchmarkComparisonTable({
    rows,
    selectedRowKey,
    onSelectRow,
    rowKey,
    sortMode,
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
                            <th className="px-5 py-4 text-right font-medium">Speed</th>
                            <th className="px-5 py-4 font-medium">Evidence</th>
                            <th className="px-5 py-4 font-medium">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--app-border-muted)]">
                        {rows.map((row, index) => {
                            const key = rowKey(row);
                            const isSelected = key === selectedRowKey;
                            const isHighlighted = index === 0;

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
                                            <div className="safe-text text-xs text-[var(--app-text-muted)]">{row.run_family ?? row.run}</div>
                                            <details className="text-xs text-[var(--app-text-muted)]">
                                                <summary className="cursor-pointer text-[var(--app-brand-text)]">details</summary>
                                                <p className="mt-1 text-clamp-2">{row.summary_text}</p>
                                            </details>
                                        </div>
                                    </td>
                                    <td className="px-5 py-4 text-right">
                                        <div className="space-y-2">
                                            <div className="font-semibold text-[var(--app-text)]">{formatMetric(row.auc)}</div>
                                            {row.auc_rank === 1 ? <RankBadge>#1 Accuracy</RankBadge> : null}
                                        </div>
                                    </td>
                                    <td className="px-5 py-4 text-right">
                                        <div className="space-y-2">
                                            <div className="font-semibold text-[var(--app-text)]">{formatMetric(row.eer)}</div>
                                            {row.eer_rank === 1 ? <RankBadge>#1 EER</RankBadge> : null}
                                        </div>
                                    </td>
                                    <td className="px-5 py-4 text-right">
                                        <div className="space-y-2">
                                            <div className="text-[var(--app-text-soft)]">{formatLatency(row.latency_ms)}</div>
                                            {row.latency_rank === 1 ? <RankBadge>Fastest</RankBadge> : null}
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
                    </tbody>
                </table>
            </div>
        </div>
    );
}
