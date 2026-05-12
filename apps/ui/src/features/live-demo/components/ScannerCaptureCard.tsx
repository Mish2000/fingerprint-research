import {
    CheckCircle2,
    CircleDashed,
    Fingerprint,
    FolderInput,
    Loader2,
    MonitorUp,
    ScanLine,
} from "lucide-react";
import FileDropBox from "../../../components/FileDropBox.tsx";
import type { ScannerStatusResponse } from "../../../api/scannerCaptureService.ts";
import type { AsyncState } from "../../../shared/request-state/index.ts";
import { formatCaptureLabel } from "../../../shared/storytelling.ts";
import InlineBanner from "../../../shared/ui/InlineBanner.tsx";
import type { Capture } from "../../../types/index.ts";

export type ScannerCaptureCardActionKind = "direct" | "scanner_ui" | "import_latest";

export interface ScannerCaptureCardResult {
    sourceLabel: string;
    modeUsed: string;
    directCapture: boolean;
    durationMs: number | null;
    deviceName: string | null;
    normalizedUrl: string;
    fileName: string;
    originalFilename: string | null;
    warning: string | null;
}

export interface ScannerCaptureCardActionState {
    status: "idle" | "loading" | "success" | "error";
    data: ScannerCaptureCardResult | null;
    error: string | null;
    action: ScannerCaptureCardActionKind | null;
    countdownSeconds?: number | null;
}

interface ScannerCaptureCardProps {
    file: File | null;
    capture: Capture;
    disabled: boolean;
    scannerStatusState: AsyncState<ScannerStatusResponse>;
    scannerActionState: ScannerCaptureCardActionState;
    eyebrow?: string;
    title?: string;
    description?: string;
    uploadTitle?: string;
    uploadDescription?: string;
    onFileChange: (file: File | null) => void;
    onCaptureFromScanner: () => void | Promise<void>;
    onCaptureWithScannerUi: () => void | Promise<void>;
    onImportLatestSavedScan: () => void | Promise<void>;
}

function StatusRow({
    active,
    label,
    value,
}: {
    active: boolean;
    label: string;
    value: string;
}) {
    const Icon = active ? CheckCircle2 : CircleDashed;

    return (
        <div className="flex items-center gap-3 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] px-3 py-2">
            <div
                className={`rounded-full p-1.5 ${
                    active
                        ? "bg-[var(--app-success-surface)] text-[var(--app-success-text)]"
                        : "bg-[var(--app-surface-muted)] text-[var(--app-text-muted)]"
                }`}
            >
                <Icon className="h-4 w-4" />
            </div>
            <div className="min-w-0">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">{label}</p>
                <p className="truncate text-sm font-semibold text-[var(--app-text)]">{value}</p>
            </div>
        </div>
    );
}

function StatusBadge({
    label,
    tone,
}: {
    label: string;
    tone: "direct" | "fallback" | "unavailable" | "capturing" | "failed";
}) {
    const classes = {
        direct: "status-pill--success",
        fallback: "status-pill--warning",
        unavailable: "",
        capturing: "status-pill--brand",
        failed: "status-pill--error",
    }[tone];

    return (
        <span className={`status-pill ${classes}`}>
            {label}
        </span>
    );
}

function MetadataItem({ label, value }: { label: string; value: string }) {
    return (
        <div className="rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] px-3 py-2">
            <p className="text-xs font-semibold text-[var(--app-text-muted)]">{label}</p>
            <p className="mt-1 truncate text-sm font-semibold text-[var(--app-text)]" title={value}>{value}</p>
        </div>
    );
}

function scannerAvailabilityMessage(statusState: AsyncState<ScannerStatusResponse>): string {
    if (statusState.status === "loading") {
        return "Checking scanner capture availability...";
    }
    if (statusState.status === "error") {
        return "Scanner capture unavailable";
    }

    const status = statusState.data;
    if (status?.direct_capture_available) {
        return "Direct scanner capture available";
    }
    if (status?.saved_file_bridge_available) {
        return "Direct capture unavailable. Saved-file import fallback available.";
    }
    return "Scanner capture unavailable";
}

function scannerStatusBadge(
    statusState: AsyncState<ScannerStatusResponse>,
    actionState: ScannerCaptureCardActionState,
): { label: string; tone: "direct" | "fallback" | "unavailable" | "capturing" | "failed" } {
    if (actionState.status === "loading") {
        if (actionState.action === "direct" && actionState.countdownSeconds != null) {
            return { label: "Preparing...", tone: "capturing" };
        }
        return { label: "Capturing...", tone: "capturing" };
    }
    if (actionState.status === "error") {
        return { label: "Capture failed", tone: "failed" };
    }

    const status = statusState.data;
    if (status?.direct_capture_available) {
        return { label: "Direct", tone: "direct" };
    }
    if (status?.saved_file_bridge_available) {
        return { label: "Fallback", tone: "fallback" };
    }
    return { label: "Unavailable", tone: "unavailable" };
}

function loadingBanner(actionState: ScannerCaptureCardActionState): { title?: string; message: string } {
    if (actionState.action === "direct" && actionState.countdownSeconds != null) {
        return {
            title: "Place finger on scanner and keep holding still during capture.",
            message: `Capturing in ${actionState.countdownSeconds}...`,
        };
    }
    if (actionState.action === "scanner_ui") {
        return { message: "Capturing with scanner UI... A scanner dialog may appear." };
    }
    if (actionState.action === "import_latest") {
        return { message: "Importing latest saved scan..." };
    }
    return { message: "Scanner is active — keep finger still." };
}

export default function ScannerCaptureCard({
    file,
    capture,
    disabled,
    scannerStatusState,
    scannerActionState,
    eyebrow = "Fingerprint source",
    title = "Fingerprint image source",
    description = "Capture directly from the TWAIN scanner bridge, import the latest saved scan, or upload an image manually.",
    uploadTitle = "Manual fingerprint image",
    uploadDescription = "Choose a saved fingerprint image from disk.",
    onFileChange,
    onCaptureFromScanner,
    onCaptureWithScannerUi,
    onImportLatestSavedScan,
}: ScannerCaptureCardProps) {
    const scannerStatus = scannerStatusState.data;
    const directAvailable = scannerStatus?.direct_capture_available === true;
    const fallbackAvailable = scannerStatus?.saved_file_bridge_available === true;
    const scannerStatusChecking = scannerStatusState.status === "loading";
    const scannerActionBusy = scannerActionState.status === "loading";
    const directButtonDisabled = disabled || scannerStatusChecking || scannerActionBusy || !directAvailable;
    const fallbackButtonDisabled = disabled || scannerStatusChecking || scannerActionBusy || !fallbackAvailable;
    const scannerUiButtonDisabled = disabled || scannerStatusChecking || scannerActionBusy || !directAvailable;
    const statusBadge = scannerStatusBadge(scannerStatusState, scannerActionState);
    const result = scannerActionState.data;

    return (
        <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                    <div className="status-pill status-pill--brand uppercase tracking-[0.14em]">
                        <Fingerprint className="h-3.5 w-3.5" />
                        {eyebrow}
                    </div>
                    <h3 className="mt-3 text-xl font-semibold text-[var(--app-text)]">{title}</h3>
                    <p className="mt-1 text-sm leading-6 text-[var(--app-text-soft)]">
                        {description}
                    </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                    <StatusBadge label={statusBadge.label} tone={statusBadge.tone} />
                    <div className="status-pill">
                        {formatCaptureLabel(capture)}
                    </div>
                </div>
            </div>

            <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_18rem]">
                <FileDropBox
                    file={file}
                    onChange={onFileChange}
                    title={uploadTitle}
                    description={uploadDescription}
                    className="min-h-[320px]"
                    disabled={disabled}
                />

                <div className="space-y-4">
                    <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-4">
                        <div className="flex items-center gap-3">
                            <div className="rounded-lg bg-[var(--app-brand-surface)] p-2 text-[var(--app-brand-text)]">
                                <ScanLine className="h-5 w-5" />
                            </div>
                            <div>
                                <p className="text-sm font-semibold text-[var(--app-text)]">Capture source</p>
                                <p className="text-xs text-[var(--app-text-muted)]">Scanner capture, saved-file import, and manual upload share this image slot.</p>
                            </div>
                        </div>
                        <div className="mt-4 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] px-3 py-3">
                            <p className="text-sm font-semibold text-[var(--app-text)]">{scannerAvailabilityMessage(scannerStatusState)}</p>
                            {directAvailable && scannerStatus?.device_name ? (
                                <p className="mt-1 text-xs font-semibold text-[var(--app-text-soft)]">Source: {scannerStatus.device_name}</p>
                            ) : null}
                            {scannerStatusState.status === "error" && scannerStatusState.error ? (
                                <p className="mt-1 text-xs text-[var(--app-error-text)]">{scannerStatusState.error}</p>
                            ) : null}
                            {!directAvailable && scannerStatus?.last_error ? (
                                <p className="mt-1 text-xs text-[var(--app-text-muted)]">{scannerStatus.last_error}</p>
                            ) : null}
                        </div>
                        <div className="mt-4 space-y-2">
                            <button
                                type="button"
                                disabled={directButtonDisabled}
                                onClick={() => {
                                    void onCaptureFromScanner();
                                }}
                                className="app-button app-button--primary w-full justify-center"
                            >
                                {scannerActionBusy && scannerActionState.action === "direct" ? (
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                ) : (
                                    <ScanLine className="mr-2 h-4 w-4" />
                                )}
                                Capture from scanner
                            </button>
                            <button
                                type="button"
                                disabled={fallbackButtonDisabled}
                                onClick={() => {
                                    void onImportLatestSavedScan();
                                }}
                                className="app-button app-button--secondary w-full justify-center"
                            >
                                {scannerActionBusy && scannerActionState.action === "import_latest" ? (
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                ) : (
                                    <FolderInput className="mr-2 h-4 w-4" />
                                )}
                                Import latest saved scan
                            </button>
                            <button
                                type="button"
                                disabled={scannerUiButtonDisabled}
                                onClick={() => {
                                    void onCaptureWithScannerUi();
                                }}
                                className="app-button app-button--secondary w-full justify-center"
                            >
                                {scannerActionBusy && scannerActionState.action === "scanner_ui" ? (
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                ) : (
                                    <MonitorUp className="mr-2 h-4 w-4" />
                                )}
                                Capture with scanner UI
                            </button>
                        </div>
                        <div className="mt-4 space-y-3">
                            {scannerActionState.status === "loading" ? (
                                <InlineBanner variant="info" title={loadingBanner(scannerActionState).title}>
                                    {loadingBanner(scannerActionState).message}
                                </InlineBanner>
                            ) : null}
                            {scannerActionState.status === "error" && scannerActionState.error ? (
                                <InlineBanner variant="error">{scannerActionState.error}</InlineBanner>
                            ) : null}
                            {scannerActionState.status === "success" && result?.warning ? (
                                <InlineBanner variant="warning">{result.warning}</InlineBanner>
                            ) : null}
                        </div>
                        <div className="mt-4 space-y-3">
                            <StatusRow
                                active={Boolean(file)}
                                label="Capture"
                                value={file ? "Image source ready" : "Waiting for fingerprint"}
                            />
                            <StatusRow
                                active={Boolean(file)}
                                label="Preprocess"
                                value={file ? "Backend preprocessing runs on submit" : "Pending capture"}
                            />
                            <StatusRow active={Boolean(file)} label="Submit" value={file ? "Ready for selected action" : "Waiting for image"} />
                        </div>
                        {scannerActionState.status === "success" && result ? (
                            <div className="mt-4 rounded-xl border border-[var(--app-success-border)] bg-[var(--app-success-surface)] p-3 text-[var(--app-success-text)]">
                                <div className="flex items-start justify-between gap-3">
                                    <div>
                                        <p className="text-sm font-semibold">{result.sourceLabel}</p>
                                        <p className="mt-1 truncate text-xs" title={result.fileName}>
                                            {result.fileName}
                                        </p>
                                    </div>
                                    <CheckCircle2 className="h-5 w-5 text-[var(--app-success-text)]" />
                                </div>
                                <div className="mt-3 grid gap-2">
                                    <MetadataItem label="mode_used" value={result.modeUsed} />
                                    <MetadataItem label="direct_capture" value={result.directCapture ? "true" : "false"} />
                                    <MetadataItem
                                        label="duration_ms"
                                        value={result.durationMs == null ? "Not reported" : String(result.durationMs)}
                                    />
                                    <MetadataItem label="device.name" value={result.deviceName ?? "Saved-file fallback"} />
                                </div>
                            </div>
                        ) : null}
                    </div>
                </div>
            </div>
        </section>
    );
}
