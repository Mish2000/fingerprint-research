import {
    Activity,
    Database,
    RefreshCcw,
    Search,
    ShieldAlert,
    ShieldCheck,
    Trash2,
    UserPlus,
} from "lucide-react";
import FileDropBox from "../../components/FileDropBox.tsx";
import RequestState from "../../components/RequestState.tsx";
import SurfaceCard from "../../shared/ui/SurfaceCard.tsx";
import StatCard from "../../shared/ui/StatCard.tsx";
import InlineBanner from "../../shared/ui/InlineBanner.tsx";
import FormField from "../../shared/ui/FormField.tsx";
import { CHECKBOX_CLASS_NAME, INPUT_CLASS_NAME } from "../../shared/ui/inputClasses.ts";
import IdentificationOutcomeStoryPanel from "./components/IdentificationOutcomeStoryPanel.tsx";
import { useIdentification } from "./hooks/useIdentification.ts";
import { IDENTIFICATION_RETRIEVAL_OPTIONS, IDENTIFICATION_RERANK_OPTIONS } from "./methodOptions.ts";

function formatLatency(value: number): string {
    return `${value.toFixed(1)} ms`;
}

function includesAnyToken(message: string | null | undefined, tokens: string[]): boolean {
    const normalized = (message ?? "").toLowerCase();
    return tokens.some((token) => normalized.includes(token));
}

function isServiceInitializationError(message: string | null | undefined): boolean {
    return includesAnyToken(message, ["startup", "init", "ctor", "constructor", "not initialized"]);
}

function isDuplicateEnrollmentError(message: string | null | undefined): boolean {
    return includesAnyToken(message, ["duplicate", "already exists", "existing identity", "replace_existing"]);
}

function isInvalidRetrievalError(message: string | null | undefined): boolean {
    return includesAnyToken(message, ["retrieval", "invalid", "unsupported"]);
}

function formatDisplayLabel(value: string | null | undefined, fallback = "Not reported"): string {
    const normalized = (value ?? "").replace(/_/g, " ").trim();
    if (!normalized) {
        return fallback;
    }
    return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function formatCount(value: number | null | undefined): string {
    return typeof value === "number" ? value.toLocaleString() : "Not reported";
}

function formatBooleanSummary(
    value: boolean | null | undefined,
    positiveLabel: string,
    negativeLabel: string,
    fallback = "Not reported",
): string {
    if (value === true) {
        return positiveLabel;
    }
    if (value === false) {
        return negativeLabel;
    }
    return fallback;
}

function readBooleanFlag(record: Record<string, unknown>, key: string): boolean | null {
    const value = record[key];
    return typeof value === "boolean" ? value : null;
}

function extractIssueMetadata(issue: Record<string, unknown>): Record<string, unknown> {
    return Object.fromEntries(
        Object.entries(issue).filter(([key]) => !["code", "severity", "database_role", "message"].includes(key)),
    );
}

interface IdentificationOperationalWorkspaceProps {
    identification: ReturnType<typeof useIdentification>;
}

export default function IdentificationOperationalWorkspace({ identification }: IdentificationOperationalWorkspaceProps) {
    const topCandidate = identification.searchState.data?.top_candidate ?? null;
    const candidates = identification.searchState.data?.candidates ?? [];
    const latencyEntries = Object.entries(identification.searchState.data?.latency_ms ?? {});
    const hintEntries = Object.entries(identification.searchState.data?.hints_applied ?? {});
    const isEnrollLoading = identification.enrollState.status === "loading";
    const isSearchLoading = identification.searchState.status === "loading";
    const isDeleteLoading = identification.deleteState.status === "loading";
    const isStatsLoading = identification.statsState.status === "loading";
    const isRuntimeInspectionLoading =
        identification.healthState.status === "loading"
        || identification.adminLayoutState.status === "loading";
    const isReconciliationLoading = identification.adminReconciliationState.status === "loading";
    const health = identification.healthState.data;
    const adminLayout = identification.adminLayoutState.data;
    const adminReadiness = adminLayout?.readiness ?? null;
    const reconciliationReport = identification.adminReconciliationState.data;
    const layoutReconciliationRequired = adminLayout
        ? readBooleanFlag(adminLayout.reconciliation, "manual_reconciliation_required")
        : null;
    const runtimeInspectionErrors = [
        { key: "health", label: "Runtime health", message: identification.healthState.error },
        { key: "layout", label: "Layout inspection", message: identification.adminLayoutState.error },
    ].filter((item): item is { key: string; label: string; message: string } => Boolean(item.message));
    const readinessIssues = adminLayout
        ? (adminLayout.issues.length > 0 ? adminLayout.issues : [...adminLayout.errors, ...adminLayout.warnings])
        : [];
    const integrityWarnings = adminLayout?.integrity_warnings ?? [];
    const vectorRowEntries = Object.entries(adminLayout?.row_counts.vectors_by_method ?? {});
    const unexpectedVectorEntries = Object.entries(adminLayout?.unexpected_vector_methods ?? {});
    const reconciliationSeverityEntries = Object.entries(reconciliationReport?.summary.severity ?? {});
    const reconciliationRepairabilityEntries = Object.entries(reconciliationReport?.summary.repairability ?? {});
    const reconciliationCommandEntries = Object.entries(reconciliationReport?.commands ?? {});
    const runtimeReady = Boolean(
        adminReadiness?.ready
        && (health ? health.identify_status === "ready" : true),
    );
    const runtimeStatusLabel = !adminReadiness
        ? "Inspection pending"
        : runtimeReady
            ? "Runtime ready"
            : health && health.identify_status !== "ready"
                ? formatDisplayLabel(health.identify_status, "Runtime unavailable")
                : adminReadiness.error_count > 0
                    ? "Runtime not ready"
                    : adminReadiness.warning_count > 0
                        ? "Warnings detected"
                        : formatDisplayLabel(adminReadiness.status, "Readiness pending");
    const databaseSplitLabel = !adminLayout
        ? "Inspection pending"
        : adminLayout.dual_database_enabled
            ? "Dual-database split active"
            : "Single database path";
    const resolvedSplitLabel = !adminLayout
        ? "Inspection pending"
        : adminLayout.dual_database_enabled
            ? (
                adminLayout.table_presence.identity_db.person
                && adminLayout.table_presence.identity_db.identity
                && adminLayout.table_presence.biometric_db.raw
                && adminLayout.table_presence.biometric_db.vectors
                    ? "Biometric and identity records resolve cleanly"
                    : "Database split needs attention"
            )
            : "Biometric and identity records share one database";
    const vectorExtensionLabel = !adminLayout
        ? "Inspection pending"
        : formatBooleanSummary(
            adminLayout.vector_extension_present_in_biometric_db,
            "Vector extension present",
            "Vector extension missing",
        );
    const issueCountLabel = !adminReadiness
        ? "Inspection pending"
        : adminReadiness.error_count + adminReadiness.warning_count === 0
            ? "No active issues"
            : `${adminReadiness.error_count} errors / ${adminReadiness.warning_count} warnings`;
    const manualReconciliationLabel = reconciliationReport
        ? (
            reconciliationReport.summary.manual_reconciliation_required
                ? "Manual reconciliation required"
                : "No manual reconciliation required"
        )
        : layoutReconciliationRequired === null
            ? "Report available on demand"
            : layoutReconciliationRequired
                ? "Manual reconciliation required"
                : "No manual reconciliation required";

    const showServiceInitHint =
        isServiceInitializationError(identification.statsState.error)
        || isServiceInitializationError(identification.healthState.error)
        || isServiceInitializationError(identification.adminLayoutState.error)
        || isServiceInitializationError(identification.enrollState.error)
        || isServiceInitializationError(identification.searchState.error)
        || isServiceInitializationError(identification.deleteState.error);

    const showDuplicateEnrollHint = isDuplicateEnrollmentError(identification.enrollState.error);
    const showInvalidRetrievalHint = isInvalidRetrievalError(identification.searchState.error);
    const showShortlistZeroHint =
        identification.searchState.status === "success"
        && identification.searchState.data !== null
        && identification.searchState.data.shortlist_size === 0;

    return (
        <div className="space-y-6">
            <InlineBanner variant="info" title="Identification workspace">
                Identification keeps the operational structure while adding clearer negative-path messaging, keyboard submit flow,
                and stronger empty-state behavior for shortlist-zero and no-candidate responses.
            </InlineBanner>

            {identification.notice ? <InlineBanner variant="success">{identification.notice}</InlineBanner> : null}

            {showServiceInitHint ? (
                <InlineBanner variant="warning" title="Backend initialization issue detected">
                    One of the identification endpoints appears to have failed during startup or lazy initialization.
                    Keep the original backend error visible and use this as a release-readiness stop signal.
                </InlineBanner>
            ) : null}

            {showDuplicateEnrollHint ? (
                <InlineBanner variant="warning" title="Duplicate enrollment rejected">
                    The backend rejected a duplicate identity. Verify this once with
                    <code> replace_existing=false </code> and once with <code> replace_existing=true </code>.
                </InlineBanner>
            ) : null}

            {showInvalidRetrievalHint ? (
                <InlineBanner variant="warning" title="Invalid retrieval method rejected">
                    The backend rejected the current retrieval method. The production-safe UI path should stay within the supported
                    <code> dl </code> / <code> vit </code> selector values.
                </InlineBanner>
            ) : null}

            {showShortlistZeroHint ? (
                <InlineBanner variant="warning" title="Shortlist is empty">
                    The request completed successfully, but the backend returned zero shortlisted candidates.
                    This is a valid negative-path regression case and should remain readable in the UI.
                </InlineBanner>
            ) : null}

            <div className="grid gap-6 xl:grid-cols-[0.82fr_1.18fr]">
                <SurfaceCard
                    title="Store statistics"
                    description="Operational totals and storage layout are loaded once and refreshed after destructive or mutating actions."
                    actions={
                        <button
                            type="button"
                            onClick={() => {
                                void identification.refreshStats();
                            }}
                            disabled={isStatsLoading}
                            className="inline-flex items-center rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            <RefreshCcw className="mr-2 h-4 w-4" />
                            Refresh stats
                        </button>
                    }
                >
                    {identification.statsState.status === "loading" && !identification.statsState.data ? (
                        <RequestState
                            variant="loading"
                            title="Loading identification stats"
                            description="Reading the current store layout and total enrollment count."
                        />
                    ) : null}

                    {identification.statsState.status === "error" && identification.statsState.error ? (
                        <RequestState
                            variant="error"
                            title="Failed to load identification stats"
                            description={identification.statsState.error}
                            actionLabel="Retry"
                            onAction={() => {
                                void identification.refreshStats();
                            }}
                        />
                    ) : null}

                    {identification.statsState.data ? (
                        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                            <StatCard
                                icon={Database}
                                label="Total enrolled"
                                value={String(identification.statsState.data.total_enrolled)}
                                tone="brand"
                            />
                            {Object.entries(identification.statsState.data.storage_layout).map(([key, value]) => (
                                <StatCard key={key} icon={Activity} label={key} value={value} tone="slate" />
                            ))}
                        </div>
                    ) : null}
                </SurfaceCard>

                <SurfaceCard
                    title="Runtime readiness"
                    description="Live inspection of backend readiness, dual-database topology, and reconciliation posture for the operational identification runtime."
                    actions={(
                        <div className="flex flex-wrap gap-2">
                            <button
                                type="button"
                                onClick={() => {
                                    void identification.refreshRuntimeReadiness();
                                }}
                                disabled={isRuntimeInspectionLoading}
                                className="inline-flex items-center rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                                <RefreshCcw className="mr-2 h-4 w-4" />
                                Refresh inspection
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    void identification.refreshAdminReconciliationReport();
                                }}
                                disabled={isReconciliationLoading}
                                className="inline-flex items-center rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                                <ShieldCheck className="mr-2 h-4 w-4" />
                                {reconciliationReport ? "Refresh reconciliation" : "Fetch reconciliation report"}
                            </button>
                        </div>
                    )}
                >
                    {!adminLayout && isRuntimeInspectionLoading ? (
                        <RequestState
                            variant="loading"
                            title="Inspecting runtime readiness"
                            description="Reading health status, database layout, and readiness signals from the backend."
                        />
                    ) : null}

                    {!adminLayout && runtimeInspectionErrors.length > 0 ? (
                        <RequestState
                            variant="error"
                            title="Runtime inspection unavailable"
                            description="The backend readiness surface could not be loaded."
                            actionLabel="Retry"
                            onAction={() => {
                                void identification.refreshRuntimeReadiness();
                            }}
                        />
                    ) : null}

                    {runtimeInspectionErrors.map((error) => (
                        <div key={error.key} className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-900">
                            <p className="font-semibold">{error.label} request failed</p>
                            <p className="mt-1 text-red-800">
                                Keeping the exact backend error visible so readiness failures remain presentation-safe.
                            </p>
                            <code className="mt-3 block whitespace-pre-wrap break-words rounded-xl bg-white/80 px-3 py-2 text-xs text-red-900">
                                {error.message}
                            </code>
                        </div>
                    ))}

                    {adminLayout ? (
                        <div className="space-y-6">
                            {isRuntimeInspectionLoading ? (
                                <InlineBanner variant="info" title="Refreshing readiness inspection">
                                    Showing the last confirmed runtime inspection while the backend is re-checked.
                                </InlineBanner>
                            ) : null}

                            <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
                                <StatCard
                                    icon={runtimeReady ? ShieldCheck : ShieldAlert}
                                    label="Runtime readiness"
                                    value={runtimeStatusLabel}
                                    tone={runtimeReady ? "emerald" : adminReadiness?.warning_count ? "amber" : "slate"}
                                />
                                <StatCard
                                    icon={Database}
                                    label="Database topology"
                                    value={databaseSplitLabel}
                                    tone={adminLayout.dual_database_enabled ? "brand" : "slate"}
                                />
                                <StatCard
                                    icon={Database}
                                    label="Resolved data split"
                                    value={resolvedSplitLabel}
                                    tone={
                                        adminLayout.dual_database_enabled
                                        && adminLayout.table_presence.identity_db.person
                                        && adminLayout.table_presence.identity_db.identity
                                        && adminLayout.table_presence.biometric_db.raw
                                        && adminLayout.table_presence.biometric_db.vectors
                                            ? "emerald"
                                            : adminLayout.dual_database_enabled
                                                ? "amber"
                                                : "slate"
                                    }
                                />
                                <StatCard
                                    icon={Activity}
                                    label="Vector extension"
                                    value={vectorExtensionLabel}
                                    tone={
                                        adminLayout.vector_extension_present_in_biometric_db === true
                                            ? "emerald"
                                            : adminLayout.vector_extension_present_in_biometric_db === false
                                                ? "amber"
                                                : "slate"
                                    }
                                />
                                <StatCard
                                    icon={ShieldAlert}
                                    label="Issue counts"
                                    value={issueCountLabel}
                                    tone={adminReadiness && (adminReadiness.error_count + adminReadiness.warning_count) > 0 ? "amber" : "emerald"}
                                />
                                <StatCard
                                    icon={reconciliationReport?.summary.manual_reconciliation_required ? ShieldAlert : ShieldCheck}
                                    label="Manual reconciliation"
                                    value={manualReconciliationLabel}
                                    tone={
                                        reconciliationReport
                                            ? (reconciliationReport.summary.manual_reconciliation_required ? "amber" : "emerald")
                                            : layoutReconciliationRequired === null
                                                ? "slate"
                                                : layoutReconciliationRequired
                                                    ? "amber"
                                                    : "emerald"
                                    }
                                />
                            </div>

                            <div className="grid gap-4 xl:grid-cols-2">
                                <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                                    <h4 className="text-sm font-semibold text-slate-900">Runtime topology</h4>
                                    <dl className="mt-4 space-y-3 text-sm">
                                        <div className="flex items-start justify-between gap-4">
                                            <dt className="text-slate-600">Operational runtime</dt>
                                            <dd className="text-right font-medium text-slate-900">
                                                {formatDisplayLabel(health?.identify_status, "Not reported")}
                                            </dd>
                                        </div>
                                        <div className="flex items-start justify-between gap-4">
                                            <dt className="text-slate-600">Browser runtime</dt>
                                            <dd className="text-right font-medium text-slate-900">
                                                {formatDisplayLabel(health?.identify_browser_status, "Not reported")}
                                            </dd>
                                        </div>
                                        <div className="flex items-start justify-between gap-4">
                                            <dt className="text-slate-600">Backend</dt>
                                            <dd className="text-right font-medium text-slate-900">{adminLayout.backend}</dd>
                                        </div>
                                        <div className="flex items-start justify-between gap-4">
                                            <dt className="text-slate-600">Layout version</dt>
                                            <dd className="text-right font-medium text-slate-900">{adminLayout.layout_version}</dd>
                                        </div>
                                        <div className="flex items-start justify-between gap-4">
                                            <dt className="text-slate-600">Biometric database</dt>
                                            <dd className="text-right font-medium text-slate-900">{adminLayout.redacted_database_urls.biometric_db}</dd>
                                        </div>
                                        <div className="flex items-start justify-between gap-4">
                                            <dt className="text-slate-600">Identity database</dt>
                                            <dd className="text-right font-medium text-slate-900">{adminLayout.redacted_database_urls.identity_db}</dd>
                                        </div>
                                    </dl>

                                    <div className="mt-4 grid gap-3 sm:grid-cols-2">
                                        <div className="rounded-xl border border-white bg-white p-3">
                                            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Identity role tables</p>
                                            <dl className="mt-3 space-y-2 text-sm">
                                                {Object.entries(adminLayout.table_presence.identity_db).map(([key, value]) => (
                                                    <div key={key} className="flex items-center justify-between gap-4">
                                                        <dt className="text-slate-600">{formatDisplayLabel(key)}</dt>
                                                        <dd className={value ? "font-medium text-emerald-700" : "font-medium text-amber-700"}>
                                                            {value ? "Present" : "Missing"}
                                                        </dd>
                                                    </div>
                                                ))}
                                            </dl>
                                        </div>
                                        <div className="rounded-xl border border-white bg-white p-3">
                                            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Biometric role tables</p>
                                            <dl className="mt-3 space-y-2 text-sm">
                                                {Object.entries(adminLayout.table_presence.biometric_db).map(([key, value]) => (
                                                    <div key={key} className="flex items-center justify-between gap-4">
                                                        <dt className="text-slate-600">{formatDisplayLabel(key)}</dt>
                                                        <dd className={value ? "font-medium text-emerald-700" : "font-medium text-amber-700"}>
                                                            {value ? "Present" : "Missing"}
                                                        </dd>
                                                    </div>
                                                ))}
                                            </dl>
                                        </div>
                                    </div>
                                </div>

                                <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                                    <h4 className="text-sm font-semibold text-slate-900">Data readiness</h4>
                                    <dl className="mt-4 grid gap-3 sm:grid-cols-2 text-sm">
                                        <div className="rounded-xl border border-white bg-white p-3">
                                            <dt className="text-slate-500">People rows</dt>
                                            <dd className="mt-1 text-lg font-semibold text-slate-900">
                                                {formatCount(adminLayout.row_counts.people)}
                                            </dd>
                                        </div>
                                        <div className="rounded-xl border border-white bg-white p-3">
                                            <dt className="text-slate-500">Identity rows</dt>
                                            <dd className="mt-1 text-lg font-semibold text-slate-900">
                                                {formatCount(adminLayout.row_counts.identity)}
                                            </dd>
                                        </div>
                                        <div className="rounded-xl border border-white bg-white p-3">
                                            <dt className="text-slate-500">Raw biometric rows</dt>
                                            <dd className="mt-1 text-lg font-semibold text-slate-900">
                                                {formatCount(adminLayout.row_counts.raw)}
                                            </dd>
                                        </div>
                                        <div className="rounded-xl border border-white bg-white p-3">
                                            <dt className="text-slate-500">Resolved tables</dt>
                                            <dd className="mt-2 space-y-1 text-sm text-slate-800">
                                                {Object.entries(adminLayout.resolved_table_names).map(([key, value]) => (
                                                    <div key={key} className="flex items-center justify-between gap-4">
                                                        <span className="text-slate-600">{formatDisplayLabel(key)}</span>
                                                        <code className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-800">{value}</code>
                                                    </div>
                                                ))}
                                            </dd>
                                        </div>
                                    </dl>

                                    <div className="mt-4 space-y-3">
                                        <div className="rounded-xl border border-white bg-white p-3">
                                            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Vector rows by method</p>
                                            {vectorRowEntries.length > 0 ? (
                                                <dl className="mt-3 space-y-2 text-sm">
                                                    {vectorRowEntries.map(([method, count]) => (
                                                        <div key={method} className="flex items-center justify-between gap-4">
                                                            <dt className="text-slate-600">{formatDisplayLabel(method)}</dt>
                                                            <dd className="font-medium text-slate-900">{formatCount(count)}</dd>
                                                        </div>
                                                    ))}
                                                </dl>
                                            ) : (
                                                <p className="mt-3 text-sm text-slate-600">No vector row counts were reported.</p>
                                            )}
                                        </div>

                                        {unexpectedVectorEntries.length > 0 ? (
                                            <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
                                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-amber-800">Unexpected vector methods</p>
                                                <dl className="mt-3 space-y-2 text-sm text-amber-900">
                                                    {unexpectedVectorEntries.map(([method, count]) => (
                                                        <div key={method} className="flex items-center justify-between gap-4">
                                                            <dt>{formatDisplayLabel(method)}</dt>
                                                            <dd className="font-medium">{formatCount(count)}</dd>
                                                        </div>
                                                    ))}
                                                </dl>
                                            </div>
                                        ) : null}

                                        {integrityWarnings.length > 0 ? (
                                            <InlineBanner variant="warning" title="Integrity warnings detected">
                                                <ul className="mt-2 space-y-1 text-sm">
                                                    {integrityWarnings.map((warning) => (
                                                        <li key={warning}>{warning}</li>
                                                    ))}
                                                </ul>
                                            </InlineBanner>
                                        ) : (
                                            <InlineBanner variant="success" title="Integrity checks look clean">
                                                Biometric and identity records are reporting without additional integrity warnings.
                                            </InlineBanner>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {readinessIssues.length > 0 ? (
                                <div className="space-y-3">
                                    <div className="flex items-center justify-between gap-4">
                                        <h4 className="text-sm font-semibold text-slate-900">Readiness issues</h4>
                                        <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
                                            {readinessIssues.length} reported
                                        </p>
                                    </div>
                                    <div className="space-y-3">
                                        {readinessIssues.map((issue) => {
                                            const metadata = extractIssueMetadata(issue);
                                            const hasMetadata = Object.keys(metadata).length > 0;
                                            const isError = issue.severity.toLowerCase() === "error";

                                            return (
                                                <div
                                                    key={`${issue.code}-${issue.database_role}-${issue.message}`}
                                                    className={`rounded-2xl border p-4 ${
                                                        isError
                                                            ? "border-red-200 bg-red-50"
                                                            : "border-amber-200 bg-amber-50"
                                                    }`}
                                                >
                                                    <div className="flex flex-wrap items-center gap-2">
                                                        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.12em] ${
                                                            isError
                                                                ? "bg-red-100 text-red-800"
                                                                : "bg-amber-100 text-amber-800"
                                                        }`}>
                                                            {formatDisplayLabel(issue.severity)}
                                                        </span>
                                                        <span className="rounded-full bg-white/80 px-2.5 py-1 text-xs font-semibold text-slate-700">
                                                            {issue.code}
                                                        </span>
                                                        <span className="rounded-full bg-white/80 px-2.5 py-1 text-xs font-semibold text-slate-700">
                                                            {formatDisplayLabel(issue.database_role)}
                                                        </span>
                                                    </div>
                                                    <p className="mt-3 text-sm font-medium text-slate-900">{issue.message}</p>
                                                    {hasMetadata ? (
                                                        <details className="mt-3">
                                                            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-[0.12em] text-slate-600">
                                                                Extra issue metadata
                                                            </summary>
                                                            <code className="mt-3 block whitespace-pre-wrap break-words rounded-xl bg-slate-950 px-3 py-2 text-xs text-slate-100">
                                                                {JSON.stringify(metadata, null, 2)}
                                                            </code>
                                                        </details>
                                                    ) : null}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            ) : (
                                <InlineBanner variant="success" title="No active readiness issues">
                                    The backend inspection did not report database errors or warnings for the operational runtime.
                                </InlineBanner>
                            )}

                            <div className="rounded-2xl border border-slate-200 bg-white p-4">
                                <div className="flex flex-wrap items-start justify-between gap-4">
                                    <div>
                                        <h4 className="text-sm font-semibold text-slate-900">Reconciliation report</h4>
                                        <p className="mt-1 text-sm text-slate-600">
                                            Report-only inspection for manual cleanup risk. No repairs are triggered from the UI.
                                        </p>
                                    </div>
                                    {reconciliationReport ? (
                                        <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
                                            Generated {reconciliationReport.generated_at}
                                        </p>
                                    ) : null}
                                </div>

                                {reconciliationReport == null && identification.adminReconciliationState.status === "idle" ? (
                                    <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-600">
                                        Fetch the reconciliation report when you want a live summary of repairability, severity mix,
                                        and whether manual reconciliation is required.
                                    </div>
                                ) : null}

                                {reconciliationReport == null && isReconciliationLoading ? (
                                    <div className="mt-4">
                                        <RequestState
                                            variant="loading"
                                            title="Fetching reconciliation report"
                                            description="Reading the latest report-only reconciliation summary from the backend."
                                        />
                                    </div>
                                ) : null}

                                {identification.adminReconciliationState.error ? (
                                    <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-900">
                                        <p className="font-semibold">Reconciliation report request failed</p>
                                        <code className="mt-3 block whitespace-pre-wrap break-words rounded-xl bg-white/80 px-3 py-2 text-xs text-red-900">
                                            {identification.adminReconciliationState.error}
                                        </code>
                                    </div>
                                ) : null}

                                {reconciliationReport ? (
                                    <div className="mt-4 space-y-4">
                                        {isReconciliationLoading ? (
                                            <InlineBanner variant="info" title="Refreshing reconciliation report">
                                                Showing the last report while a fresh reconciliation summary is loaded.
                                            </InlineBanner>
                                        ) : null}

                                        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Report mode</p>
                                                <p className="mt-2 text-sm font-semibold text-slate-900">
                                                    {formatDisplayLabel(reconciliationReport.report_mode)}
                                                </p>
                                            </div>
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Manual reconciliation</p>
                                                <p className="mt-2 text-sm font-semibold text-slate-900">{manualReconciliationLabel}</p>
                                            </div>
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Available repairs</p>
                                                <p className="mt-2 text-sm font-semibold text-slate-900">
                                                    {reconciliationReport.available_repairs.length.toLocaleString()}
                                                </p>
                                            </div>
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Applied repairs</p>
                                                <p className="mt-2 text-sm font-semibold text-slate-900">
                                                    {reconciliationReport.applied_repairs.length.toLocaleString()}
                                                </p>
                                            </div>
                                        </div>

                                        <div className="grid gap-4 xl:grid-cols-3">
                                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Severity mix</p>
                                                {reconciliationSeverityEntries.length > 0 ? (
                                                    <dl className="mt-3 space-y-2 text-sm">
                                                        {reconciliationSeverityEntries.map(([key, value]) => (
                                                            <div key={key} className="flex items-center justify-between gap-4">
                                                                <dt className="text-slate-600">{formatDisplayLabel(key)}</dt>
                                                                <dd className="font-medium text-slate-900">{formatCount(value)}</dd>
                                                            </div>
                                                        ))}
                                                    </dl>
                                                ) : (
                                                    <p className="mt-3 text-sm text-slate-600">No severity breakdown was reported.</p>
                                                )}
                                            </div>

                                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Repairability</p>
                                                {reconciliationRepairabilityEntries.length > 0 ? (
                                                    <dl className="mt-3 space-y-2 text-sm">
                                                        {reconciliationRepairabilityEntries.map(([key, value]) => (
                                                            <div key={key} className="flex items-center justify-between gap-4">
                                                                <dt className="text-slate-600">{formatDisplayLabel(key)}</dt>
                                                                <dd className="font-medium text-slate-900">{formatCount(value)}</dd>
                                                            </div>
                                                        ))}
                                                    </dl>
                                                ) : (
                                                    <p className="mt-3 text-sm text-slate-600">No repairability breakdown was reported.</p>
                                                )}
                                            </div>

                                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Suggested commands</p>
                                                {reconciliationCommandEntries.length > 0 ? (
                                                    <div className="mt-3 space-y-2">
                                                        {reconciliationCommandEntries.map(([key, value]) => (
                                                            <div key={key} className="rounded-lg bg-white p-3">
                                                                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
                                                                    {formatDisplayLabel(key)}
                                                                </p>
                                                                <code className="mt-2 block whitespace-pre-wrap break-words text-xs text-slate-900">
                                                                    {value}
                                                                </code>
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <p className="mt-3 text-sm text-slate-600">No command hints were returned.</p>
                                                )}
                                            </div>
                                        </div>

                                        {reconciliationReport.issues.length > 0 ? (
                                            <InlineBanner
                                                variant={reconciliationReport.summary.manual_reconciliation_required ? "warning" : "info"}
                                                title={manualReconciliationLabel}
                                            >
                                                {reconciliationReport.issues.length} reconciliation issues were reported for review.
                                            </InlineBanner>
                                        ) : (
                                            <InlineBanner variant="success" title="Reconciliation report is clear">
                                                The latest report did not flag additional reconciliation issues.
                                            </InlineBanner>
                                        )}
                                    </div>
                                ) : null}
                            </div>
                        </div>
                    ) : null}
                </SurfaceCard>
            </div>

            <div className="grid gap-6 xl:grid-cols-2">
                <SurfaceCard title="Enroll identity" description="Upload a fingerprint, attach metadata, and control replacement semantics.">
                    <form
                        className="space-y-4"
                        onSubmit={(event) => {
                            event.preventDefault();
                            void identification.submitEnroll();
                        }}
                        aria-busy={isEnrollLoading}
                    >
                        <FileDropBox
                            file={identification.enrollForm.file}
                            onChange={(file) => {
                                identification.updateEnrollForm({ file });
                            }}
                            disabled={isEnrollLoading}
                            title="Enrollment image"
                            description="Choose the fingerprint image that will be stored for this person."
                        />

                        <div className="grid gap-4 md:grid-cols-2">
                            <FormField label="Full name">
                                <input
                                    className={INPUT_CLASS_NAME}
                                    value={identification.enrollForm.fullName}
                                    disabled={isEnrollLoading}
                                    onChange={(event) => {
                                        identification.updateEnrollForm({ fullName: event.target.value });
                                    }}
                                />
                            </FormField>

                            <FormField label="National ID">
                                <input
                                    className={INPUT_CLASS_NAME}
                                    value={identification.enrollForm.nationalId}
                                    disabled={isEnrollLoading}
                                    onChange={(event) => {
                                        identification.updateEnrollForm({ nationalId: event.target.value });
                                    }}
                                />
                            </FormField>

                            <FormField label="Capture">
                                <select
                                    className={INPUT_CLASS_NAME}
                                    value={identification.enrollForm.capture}
                                    disabled={isEnrollLoading}
                                    onChange={(event) => {
                                        identification.updateEnrollForm({ capture: event.target.value as typeof identification.enrollForm.capture });
                                    }}
                                >
                                    <option value="plain">Plain</option>
                                    <option value="roll">Roll</option>
                                    <option value="contactless">Contactless</option>
                                    <option value="contact_based">Contact-based</option>
                                </select>
                            </FormField>
                        </div>

                        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                            <p className="text-sm font-medium text-slate-700">Vector methods</p>

                            <div className="mt-3 flex flex-wrap gap-4">
                                <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                    <input
                                        type="checkbox"
                                        className={CHECKBOX_CLASS_NAME}
                                        checked={identification.enrollForm.includeDl}
                                        disabled={isEnrollLoading}
                                        onChange={(event) => {
                                            identification.updateEnrollForm({ includeDl: event.target.checked });
                                        }}
                                    />
                                    DL
                                </label>

                                <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                    <input
                                        type="checkbox"
                                        className={CHECKBOX_CLASS_NAME}
                                        checked={identification.enrollForm.includeVit}
                                        disabled={isEnrollLoading}
                                        onChange={(event) => {
                                            identification.updateEnrollForm({ includeVit: event.target.checked });
                                        }}
                                    />
                                    ViT
                                </label>

                                <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                    <input
                                        type="checkbox"
                                        className={CHECKBOX_CLASS_NAME}
                                        checked={identification.enrollForm.replaceExisting}
                                        disabled={isEnrollLoading}
                                        onChange={(event) => {
                                            identification.updateEnrollForm({ replaceExisting: event.target.checked });
                                        }}
                                    />
                                    Replace existing
                                </label>
                            </div>
                        </div>

                        {identification.enrollState.status === "error" && identification.enrollState.error ? (
                            <InlineBanner variant="error">{identification.enrollState.error}</InlineBanner>
                        ) : null}

                        {identification.enrollState.status === "success" && identification.enrollState.data ? (
                            <InlineBanner variant="success" title="Enrollment completed">
                                Created <code>{identification.enrollState.data.random_id}</code> with vector methods
                                {` ${identification.enrollState.data.vector_methods.join(", ")}.`}
                            </InlineBanner>
                        ) : null}

                        <button
                            type="submit"
                            disabled={isEnrollLoading}
                            className="inline-flex items-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            <UserPlus className="mr-2 h-4 w-4" />
                            {isEnrollLoading ? "Enrolling..." : "Enroll identity"}
                        </button>
                    </form>
                </SurfaceCard>

                <SurfaceCard title="Delete identity" description="Destructive actions require an explicit confirmation before the request is sent.">
                    <form
                        className="space-y-4"
                        onSubmit={(event) => {
                            event.preventDefault();
                            void identification.submitDelete();
                        }}
                        aria-busy={isDeleteLoading}
                    >
                        <FormField label="Random ID" hint="Use the identifier returned by enrollment or surfaced by search results.">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.deleteForm.randomId}
                                disabled={isDeleteLoading}
                                onChange={(event) => {
                                    identification.updateDeleteForm({ randomId: event.target.value });
                                }}
                            />
                        </FormField>

                        <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                            <input
                                type="checkbox"
                                className={CHECKBOX_CLASS_NAME}
                                checked={identification.deleteForm.confirmChecked}
                                disabled={isDeleteLoading}
                                onChange={(event) => {
                                    identification.updateDeleteForm({ confirmChecked: event.target.checked });
                                }}
                            />
                            I understand this will permanently purge the selected identity.
                        </label>

                        {identification.deleteState.status === "error" && identification.deleteState.error ? (
                            <InlineBanner variant="error">{identification.deleteState.error}</InlineBanner>
                        ) : null}

                        {identification.deleteState.status === "success" && identification.deleteState.data ? (
                            <InlineBanner
                                variant={identification.deleteState.data.removed ? "success" : "warning"}
                                title={identification.deleteState.data.removed ? "Identity removed" : "Identity not found"}
                            >
                                <code>{identification.deleteState.data.random_id}</code>
                            </InlineBanner>
                        ) : null}

                        <button
                            type="submit"
                            disabled={isDeleteLoading}
                            className="inline-flex items-center rounded-xl bg-red-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            <Trash2 className="mr-2 h-4 w-4" />
                            {isDeleteLoading ? "Deleting..." : "Delete identity"}
                        </button>
                    </form>
                </SurfaceCard>
            </div>

            <SurfaceCard
                title="Search identity"
                description="Run retrieval + re-rank over the enrolled store and inspect decision traces, hints, and latency breakdown."
            >
                <form
                    className="space-y-6"
                    onSubmit={(event) => {
                        event.preventDefault();
                        void identification.submitSearch();
                    }}
                    aria-busy={isSearchLoading}
                >
                    <FileDropBox
                        file={identification.searchForm.file}
                        onChange={(file) => {
                            identification.updateSearchForm({ file });
                        }}
                        disabled={isSearchLoading}
                        title="Probe image"
                        description="Upload the fingerprint that should be identified against the store."
                    />

                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                        <FormField label="Capture">
                            <select
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.capture}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ capture: event.target.value as typeof identification.searchForm.capture });
                                }}
                            >
                                <option value="plain">Plain</option>
                                <option value="roll">Roll</option>
                                <option value="contactless">Contactless</option>
                                <option value="contact_based">Contact-based</option>
                            </select>
                        </FormField>

                        <FormField label="Retrieval method">
                            <select
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.retrievalMethod}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ retrievalMethod: event.target.value as typeof identification.searchForm.retrievalMethod });
                                }}
                            >
                                {IDENTIFICATION_RETRIEVAL_OPTIONS.map((option) => (
                                    <option key={option.value} value={option.value}>{option.label}</option>
                                ))}
                            </select>
                        </FormField>

                        <FormField label="Re-rank method">
                            <select
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.rerankMethod}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ rerankMethod: event.target.value as typeof identification.searchForm.rerankMethod });
                                }}
                            >
                                {IDENTIFICATION_RERANK_OPTIONS.map((option) => (
                                    <option key={option.value} value={option.value}>{option.label}</option>
                                ))}
                            </select>
                        </FormField>

                        <FormField label="Shortlist size">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.shortlistSizeText}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ shortlistSizeText: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="Threshold" hint="Leave empty to use the backend default.">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.thresholdText}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ thresholdText: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="Name pattern">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.namePattern}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ namePattern: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="National ID pattern">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.nationalIdPattern}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ nationalIdPattern: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="Created from">
                            <input
                                type="date"
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.createdFrom}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ createdFrom: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="Created to">
                            <input
                                type="date"
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.createdTo}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ createdTo: event.target.value });
                                }}
                            />
                        </FormField>
                    </div>

                    {identification.searchState.status === "error" && identification.searchState.error ? (
                        <InlineBanner variant="error">{identification.searchState.error}</InlineBanner>
                    ) : null}

                    <button
                        type="submit"
                        disabled={isSearchLoading}
                        className="inline-flex items-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        <Search className="mr-2 h-4 w-4" />
                        {isSearchLoading ? "Searching..." : "Search identities"}
                    </button>
                </form>

                {identification.searchState.status === "loading" ? (
                    <RequestState
                        variant="loading"
                        title="Running identification"
                        description="Sending the probe image to the backend and waiting for retrieval / rerank results."
                    />
                ) : null}

                {identification.searchState.status === "success" && identification.searchState.data ? (
                    <div className="mt-6 space-y-6">
                        <IdentificationOutcomeStoryPanel
                            story={identification.searchStoryState}
                            candidates={identification.searchState.data.candidates}
                        />

                        {topCandidate ? (
                            <InlineBanner
                                variant={identification.searchState.data.decision ? "success" : "warning"}
                                title={identification.searchState.data.decision ? "Top candidate accepted" : "Top candidate rejected"}
                            >
                                <span className="font-medium">{topCandidate.full_name}</span>
                                {` · ${topCandidate.random_id} · ${topCandidate.national_id_masked} · ${topCandidate.capture}`}
                            </InlineBanner>
                        ) : (
                            <RequestState
                                variant="empty"
                                title="No top candidate"
                                description="The backend returned a valid IdentifyResponse without a best candidate in scope."
                            />
                        )}

                        <div className="grid gap-6 xl:grid-cols-2">
                            <SurfaceCard title="Hints applied" description="Backend-side filtering hints that affected candidate retrieval.">
                                {hintEntries.length > 0 ? (
                                    <dl className="space-y-2 text-sm">
                                        {hintEntries.map(([key, value]) => (
                                            <div key={key} className="flex items-start justify-between gap-4 rounded-xl bg-slate-50 px-4 py-3">
                                                <dt className="font-medium text-slate-600">{key}</dt>
                                                <dd className="text-right text-slate-800">{value}</dd>
                                            </div>
                                        ))}
                                    </dl>
                                ) : (
                                    <RequestState
                                        variant="empty"
                                        title="No hints were applied"
                                        description="This search ran without name, national ID, or date-range narrowing."
                                    />
                                )}
                            </SurfaceCard>

                            <SurfaceCard title="Latency breakdown" description="Observability fields normalized from the backend response.">
                                {latencyEntries.length > 0 ? (
                                    <dl className="space-y-2 text-sm">
                                        {latencyEntries.map(([key, value]) => (
                                            <div key={key} className="flex items-start justify-between gap-4 rounded-xl bg-slate-50 px-4 py-3">
                                                <dt className="font-medium text-slate-600">{key}</dt>
                                                <dd className="text-right text-slate-800">{formatLatency(value)}</dd>
                                            </div>
                                        ))}
                                    </dl>
                                ) : (
                                    <RequestState
                                        variant="empty"
                                        title="No latency details returned"
                                        description="The response did not include a latency breakdown map."
                                    />
                                )}
                            </SurfaceCard>
                        </div>

                        {candidates.length > 0 ? (
                            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                                <div className="border-b border-slate-100 px-6 py-5">
                                    <h3 className="text-lg font-semibold text-slate-800">Candidate ranking</h3>
                                    <p className="mt-1 text-sm text-slate-500">
                                        Retrieval score, re-rank score, decision, and masked national ID are rendered directly from the normalized IdentifyResponse.
                                    </p>
                                </div>

                                <div className="overflow-x-auto">
                                    <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                                        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                                        <tr>
                                            <th className="px-4 py-3">Rank</th>
                                            <th className="px-4 py-3">Person</th>
                                            <th className="px-4 py-3">Random ID</th>
                                            <th className="px-4 py-3">National ID</th>
                                            <th className="px-4 py-3">Capture</th>
                                            <th className="px-4 py-3 text-right">Retrieval</th>
                                            <th className="px-4 py-3 text-right">Re-rank</th>
                                            <th className="px-4 py-3 text-center">Decision</th>
                                        </tr>
                                        </thead>

                                        <tbody className="divide-y divide-slate-100 bg-white">
                                        {candidates.map((candidate) => (
                                            <tr key={`${candidate.random_id}_${candidate.rank}`} className="hover:bg-slate-50">
                                                <td className="px-4 py-3 font-medium text-slate-800">{candidate.rank}</td>
                                                <td className="px-4 py-3 text-slate-800">{candidate.full_name}</td>
                                                <td className="px-4 py-3 text-slate-600">{candidate.random_id}</td>
                                                <td className="px-4 py-3 text-slate-600">{candidate.national_id_masked}</td>
                                                <td className="px-4 py-3 text-slate-600">{candidate.capture}</td>
                                                <td className="px-4 py-3 text-right text-slate-800">{candidate.retrieval_score.toFixed(4)}</td>
                                                <td className="px-4 py-3 text-right text-slate-800">
                                                    {typeof candidate.rerank_score === "number" ? candidate.rerank_score.toFixed(4) : "-"}
                                                </td>
                                                <td className="px-4 py-3 text-center">
                                                    {candidate.decision === true ? (
                                                        <ShieldCheck className="mx-auto h-4 w-4 text-emerald-600" />
                                                    ) : candidate.decision === false ? (
                                                        <ShieldAlert className="mx-auto h-4 w-4 text-amber-600" />
                                                    ) : (
                                                        <span className="text-slate-400">-</span>
                                                    )}
                                                </td>
                                            </tr>
                                        ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ) : (
                            <RequestState
                                variant="empty"
                                title="No candidates returned"
                                description="The request completed, but the backend did not return any candidate rows for ranking."
                            />
                        )}
                    </div>
                ) : null}
            </SurfaceCard>
        </div>
    );
}
