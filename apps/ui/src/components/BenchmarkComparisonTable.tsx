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

export function BenchmarkComparisonTable({
    rows,
    selectedRowKey,
    onSelectRow,
    rowKey,
    sortMode,
}: Props) {
    return (
        <div className="overflow-hidden rounded-[1.75rem] border border-slate-800 bg-slate-950/70 shadow-[0_20px_80px_rgba(2,6,23,0.35)]">
            <div className="overflow-x-auto">
                <table className="min-w-full border-collapse text-left text-sm text-slate-200">
                    <thead>
                        <tr className="border-b border-slate-800 bg-slate-950/80 text-[11px] uppercase tracking-[0.22em] text-slate-500">
                            <th className="px-5 py-4 font-medium">Method</th>
                            <th className="px-5 py-4 text-right font-medium">AUC</th>
                            <th className="px-5 py-4 text-right font-medium">EER</th>
                            <th className="px-5 py-4 text-right font-medium">Latency</th>
                            <th className="px-5 py-4 font-medium">Run family</th>
                            <th className="px-5 py-4 font-medium">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-900/90">
                        {rows.map((row, index) => {
                            const key = rowKey(row);
                            const isSelected = key === selectedRowKey;
                            const isHighlighted = index === 0;

                            return (
                                <tr
                                    key={key}
                                    className={[
                                        "cursor-pointer align-top transition",
                                        isSelected ? "bg-slate-900/90" : "hover:bg-slate-900/55",
                                        isHighlighted ? highlightClassName(sortMode) : "",
                                    ].join(" ").trim()}
                                    onClick={() => onSelectRow(key)}
                                >
                                    <td className="px-5 py-4">
                                        <div className="space-y-1">
                                            <div className="font-semibold text-slate-100">{formatMethodLabel(row.method, row.method_label)}</div>
                                            <div className="text-xs text-slate-500">{row.summary_text}</div>
                                        </div>
                                    </td>
                                    <td className="px-5 py-4 text-right font-semibold text-slate-100">{formatMetric(row.auc)}</td>
                                    <td className="px-5 py-4 text-right font-semibold text-slate-100">{formatMetric(row.eer)}</td>
                                    <td className="px-5 py-4 text-right text-slate-300">{formatLatency(row.latency_ms)}</td>
                                    <td className="px-5 py-4">
                                        <div className="space-y-1">
                                            <div className="font-medium text-slate-200">{row.run_family ?? row.run}</div>
                                            <div className="text-xs text-slate-500">{row.run_label ?? "Run"}</div>
                                        </div>
                                    </td>
                                    <td className="px-5 py-4">
                                        <span className={`inline-flex rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] ${statusToneClassName(row.status)}`}>
                                            {statusLabel(row.status)}
                                        </span>
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
