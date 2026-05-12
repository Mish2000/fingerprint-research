import { useCallback, useEffect, useRef, useState } from "react";
import type { LucideIcon } from "lucide-react";
import { Fingerprint, Play, ShieldCheck, UserPlus, UserRoundSearch } from "lucide-react";
import { enrollFingerprint, identifyFingerprint } from "../../api/identificationService.ts";
import {
    captureScanner,
    getScannerStatus,
    importLatestSavedScannerCapture,
    loadScannerCaptureFile,
    type ScannerCaptureFailureResponse,
    type ScannerCaptureSuccessResponse,
    type ScannerImportResponse,
    type ScannerStatusResponse,
} from "../../api/scannerCaptureService.ts";
import {
    createErrorState,
    createIdleState,
    createLoadingState,
    createSuccessState,
    type AsyncState,
} from "../../shared/request-state/index.ts";
import { formatCaptureLabel, formatMethodLabel } from "../../shared/storytelling.ts";
import InlineBanner from "../../shared/ui/InlineBanner.tsx";
import FormField from "../../shared/ui/FormField.tsx";
import { CHECKBOX_CLASS_NAME, INPUT_CLASS_NAME } from "../../shared/ui/inputClasses.ts";
import { WorkspaceHero } from "../../shared/ui/presentation.tsx";
import type {
    Capture,
    EnrollFingerprintResponse,
    IdentificationRetrievalMethod,
    IdentifyResponse,
    Method,
} from "../../types/index.ts";
import { IDENTIFICATION_RETRIEVAL_METHOD_VALUES } from "../../types/index.ts";
import { toErrorMessage } from "../../utils/error.ts";
import LiveEvidenceStrip from "./components/LiveEvidenceStrip.tsx";
import LiveResultHero from "./components/LiveResultHero.tsx";
import ScannerCaptureCard, {
    type ScannerCaptureCardActionKind,
    type ScannerCaptureCardActionState,
    type ScannerCaptureCardResult,
} from "./components/ScannerCaptureCard.tsx";

const DEFAULT_RETRIEVAL_METHOD: IdentificationRetrievalMethod = "dl";
const DEFAULT_RERANK_METHOD: Method = "sift";
const DEFAULT_SHORTLIST_SIZE = 10;
const DEFAULT_ENROLL_VECTOR_METHODS = [...IDENTIFICATION_RETRIEVAL_METHOD_VALUES];
const DIRECT_CAPTURE_SETTLE_SECONDS = 3;
const DIRECT_CAPTURE_COUNTDOWN_TICK_MS = 1000;
const DIRECT_CAPTURE_SETTLE_AFTER_ENABLE_MS = 1500;
const ENROLLMENT_CHANGED_MESSAGE = "Enrollment capture changed — enroll again to update this identity.";
const MISSING_PROBE_MESSAGE = "Upload a probe fingerprint to search the enrolled gallery.";
const SEEDED_GALLERY_HINT = "You can search an existing gallery, but for a clean demo enroll an identity first.";
const CAPTURE_OPTIONS: Array<{ value: Capture; label: string }> = [
    { value: "plain", label: "Plain" },
    { value: "roll", label: "Rolled" },
    { value: "contactless", label: "Contactless" },
    { value: "contact_based", label: "Contact-based" },
];

interface LiveEnrollForm {
    fullName: string;
    nationalId: string;
    replaceExisting: boolean;
}

interface LiveEnrollResult {
    fullName: string;
    capture: Capture;
    sourceFileName: string;
    vectorMethods: IdentificationRetrievalMethod[];
    response: EnrollFingerprintResponse;
}

type ScannerImportTarget = "enrollment" | "probe";

type PendingDirectCaptureTimer = {
    timeoutId: ReturnType<typeof window.setTimeout>;
    resolve: (completed: boolean) => void;
};

interface ActionCardProps {
    title: string;
    detail: string;
    status: string;
    icon: LucideIcon;
    disabled: boolean;
    highlighted?: boolean;
    onClick?: () => void | Promise<void>;
}

function ActionCard({
    title,
    detail,
    status,
    icon: Icon,
    disabled,
    highlighted = false,
    onClick,
}: ActionCardProps) {
    return (
        <button
            type="button"
            disabled={disabled}
            onClick={() => {
                void onClick?.();
            }}
            className={`rounded-2xl border p-4 text-left shadow-sm transition ${
                disabled
                    ? "cursor-not-allowed border-[var(--app-border)] bg-[var(--app-surface-subtle)] text-[var(--app-text-muted)]"
                    : highlighted
                        ? "border-[var(--app-success-border)] bg-[var(--app-success-surface)] text-[var(--app-success-text)] ring-2 ring-[var(--app-success-border)]"
                        : "border-[var(--app-brand-border)] bg-[var(--app-brand-surface)] text-[var(--app-brand-text)] ring-2 ring-[var(--app-brand-border)]"
            }`}
        >
            <div className="flex items-start justify-between gap-3">
                <div className={`rounded-lg border border-current/10 bg-[var(--app-surface)] p-2 ${disabled ? "text-[var(--app-text-muted)]" : ""}`}>
                    <Icon className="h-5 w-5" />
                </div>
                <span
                    className={`status-pill ${
                        disabled
                            ? ""
                            : highlighted
                                ? "status-pill--success"
                                : "status-pill--brand"
                    }`}
                >
                    {status}
                </span>
            </div>
            <h3 className="mt-4 text-base font-semibold">{title}</h3>
            <p className="mt-1 text-sm leading-6 opacity-80">{detail}</p>
        </button>
    );
}

function StepSummaryCard({
    step,
    title,
    detail,
    status,
    icon: Icon,
    highlighted = false,
}: {
    step: string;
    title: string;
    detail: string;
    status: string;
    icon: LucideIcon;
    highlighted?: boolean;
}) {
    return (
        <article
            className={`rounded-2xl border p-4 shadow-sm ${
                highlighted
                    ? "border-[var(--app-success-border)] bg-[var(--app-success-surface)]"
                    : "border-[var(--app-border)] bg-[var(--app-surface)]"
            }`}
        >
            <div className="flex items-start justify-between gap-3">
                <div className="rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] p-2 text-[var(--app-text-muted)]">
                    <Icon className="h-5 w-5" />
                </div>
                <span className="status-pill">
                    {status}
                </span>
            </div>
            <p className="mt-4 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">{step}</p>
            <h3 className="mt-1 text-base font-semibold text-[var(--app-text)]">{title}</h3>
            <p className="mt-1 text-sm leading-6 text-[var(--app-text-soft)]">{detail}</p>
        </article>
    );
}

function QualityStatusPanel({
    enrollmentFile,
    probeFile,
    busy,
}: {
    enrollmentFile: File | null;
    probeFile: File | null;
    busy: boolean;
}) {
    const statusItems = [
        {
            label: "Enrollment source",
            value: enrollmentFile ? "Image source ready" : "Waiting",
        },
        {
            label: "Probe source",
            value: probeFile ? "Image source ready" : "Waiting",
        },
        {
            label: "Preprocessing",
            value: busy
                ? "Running"
                : enrollmentFile || probeFile
                    ? "Backend preprocessing runs on submit"
                    : "Pending",
        },
    ];

    return (
        <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5 shadow-sm">
            <div className="flex items-center gap-3">
                <div className="rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-subtle)] p-2 text-[var(--app-text-muted)]">
                    <Fingerprint className="h-5 w-5" />
                </div>
                <div>
                    <h3 className="text-base font-semibold text-[var(--app-text)]">Capture / preprocessing status</h3>
                    <p className="text-sm text-[var(--app-text-muted)]">Preprocessing runs when you submit enrollment or identify requests.</p>
                </div>
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {statusItems.map((item) => (
                    <div key={item.label} className="rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-subtle)] px-3 py-3">
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">{item.label}</p>
                        <p className="mt-1 text-sm font-semibold text-[var(--app-text)]">{item.value}</p>
                    </div>
                ))}
            </div>
        </section>
    );
}

function formatVectorMethods(methods: string[]): string {
    if (methods.length === 0) {
        return DEFAULT_ENROLL_VECTOR_METHODS.map(formatMethodLabel).join(", ");
    }

    return methods.map(formatMethodLabel).join(", ");
}

function scannerImportErrorMessage(error: unknown): string {
    const message = toErrorMessage(error);
    const normalized = message.toLowerCase();

    if (normalized.includes("no saved scanner capture found")) {
        return "No saved scanner capture found in the configured folder";
    }
    if (normalized.includes("latest saved umpi capture is too old") || normalized.includes("too old")) {
        return "Latest saved scan is too old; save a new scanner capture and try again";
    }

    return message;
}

function scannerCaptureFailureMessage(payload: ScannerCaptureFailureResponse): string {
    const details = [`${payload.message} Error code: ${payload.error_code}.`];
    if (payload.fallback_available) {
        details.push("Use Import latest saved scan.");
    }
    return details.join(" ");
}

function emptyScannerActionState(): ScannerCaptureCardActionState {
    return {
        status: "idle",
        data: null,
        error: null,
        action: null,
        countdownSeconds: null,
    };
}

function loadingScannerActionState(
    action: ScannerCaptureCardActionKind,
    countdownSeconds: number | null = null,
): ScannerCaptureCardActionState {
    return {
        status: "loading",
        data: null,
        error: null,
        action,
        countdownSeconds,
    };
}

function scannerResultFromCapture(payload: ScannerCaptureSuccessResponse, fileName: string): ScannerCaptureCardResult {
    return {
        sourceLabel: payload.direct_capture ? "Direct TWAIN capture" : "Saved-file fallback",
        modeUsed: payload.mode_used,
        directCapture: payload.direct_capture,
        durationMs: payload.duration_ms,
        deviceName: payload.device.name,
        normalizedUrl: payload.normalized_url,
        fileName,
        originalFilename: null,
        warning: payload.warning,
    };
}

function scannerResultFromImport(payload: ScannerImportResponse): ScannerCaptureCardResult {
    return {
        sourceLabel: "Saved-file fallback",
        modeUsed: "saved_file_bridge",
        directCapture: false,
        durationMs: null,
        deviceName: null,
        normalizedUrl: payload.normalized_url,
        fileName: payload.normalized_filename,
        originalFilename: payload.original_filename,
        warning: null,
    };
}

function EnrollmentFormPanel({
    form,
    capture,
    fileReady,
    enrollmentChangedMessage,
    disabled,
    enrollState,
    onUpdate,
    onSubmit,
}: {
    form: LiveEnrollForm;
    capture: Capture;
    fileReady: boolean;
    enrollmentChangedMessage: string | null;
    disabled: boolean;
    enrollState: AsyncState<LiveEnrollResult>;
    onUpdate: (patch: Partial<LiveEnrollForm>) => void;
    onSubmit: () => void | Promise<void>;
}) {
    const isLoading = enrollState.status === "loading";
    const result = enrollState.data;

    return (
        <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                    <div className="status-pill status-pill--brand uppercase tracking-[0.14em]">
                        <UserPlus className="h-3.5 w-3.5" />
                        Enroll identity
                    </div>
                    <h3 className="mt-3 text-xl font-semibold text-[var(--app-text)]">Identity details</h3>
                    <p className="mt-1 text-sm leading-6 text-[var(--app-text-soft)]">
                        Uses the enrollment fingerprint image and the shared {formatCaptureLabel(capture)} capture profile.
                    </p>
                </div>
                <div className="status-pill">
                    {fileReady ? "Fingerprint ready" : "Upload required"}
                </div>
            </div>

            <form
                className="mt-5 space-y-4"
                onSubmit={(event) => {
                    event.preventDefault();
                    void onSubmit();
                }}
                aria-busy={isLoading}
            >
                <div className="grid gap-4 md:grid-cols-2">
                    <FormField label="Full name">
                        <input
                            className={INPUT_CLASS_NAME}
                            value={form.fullName}
                            disabled={disabled}
                            autoComplete="name"
                            onChange={(event) => {
                                onUpdate({ fullName: event.target.value });
                            }}
                        />
                    </FormField>

                    <FormField label="National ID">
                        <input
                            className={INPUT_CLASS_NAME}
                            value={form.nationalId}
                            disabled={disabled}
                            autoComplete="off"
                            onChange={(event) => {
                                onUpdate({ nationalId: event.target.value });
                            }}
                        />
                    </FormField>
                </div>

                <label className="inline-flex items-center gap-2 text-sm font-medium text-[var(--app-text-soft)]">
                    <input
                        type="checkbox"
                        className={CHECKBOX_CLASS_NAME}
                        checked={form.replaceExisting}
                        disabled={disabled}
                        onChange={(event) => {
                            onUpdate({ replaceExisting: event.target.checked });
                        }}
                    />
                    Replace existing identity with the same national ID
                </label>

                {enrollState.status === "error" && enrollState.error ? (
                    <InlineBanner variant="error">{enrollState.error}</InlineBanner>
                ) : null}

                {enrollmentChangedMessage ? (
                    <InlineBanner variant="warning">{enrollmentChangedMessage}</InlineBanner>
                ) : null}

                {result ? (
                    <div className="rounded-2xl border border-[var(--app-success-border)] bg-[var(--app-success-surface)] p-4 text-[var(--app-success-text)]">
                        <div className="flex flex-wrap items-start justify-between gap-4">
                            <div>
                                <p className="text-sm font-semibold">Enrollment completed</p>
                                <p className="mt-1 text-sm leading-6">
                                    Ready for Identify 1:N against the enrolled operational gallery.
                                </p>
                            </div>
                            <UserRoundSearch className="h-5 w-5 text-[var(--app-success-text)]" />
                        </div>
                        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
                            <div className="rounded-lg border border-[var(--app-success-border)] bg-[var(--app-surface)] px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em]">Name</p>
                                <p className="mt-1 text-sm font-semibold">{result.fullName}</p>
                            </div>
                            <div className="rounded-lg border border-[var(--app-success-border)] bg-[var(--app-surface)] px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em]">random_id</p>
                                <p className="mt-1 break-all text-sm font-semibold">{result.response.random_id}</p>
                            </div>
                            <div className="rounded-lg border border-[var(--app-success-border)] bg-[var(--app-surface)] px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em]">Capture</p>
                                <p className="mt-1 text-sm font-semibold">{formatCaptureLabel(result.capture)}</p>
                            </div>
                            <div className="rounded-lg border border-[var(--app-success-border)] bg-[var(--app-surface)] px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em]">Enrollment source file</p>
                                <p className="mt-1 truncate text-sm font-semibold" title={result.sourceFileName}>
                                    {result.sourceFileName}
                                </p>
                            </div>
                            <div className="rounded-lg border border-[var(--app-success-border)] bg-[var(--app-surface)] px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em]">Vectors</p>
                                <p className="mt-1 text-sm font-semibold">{formatVectorMethods(result.response.vector_methods ?? result.vectorMethods)}</p>
                            </div>
                        </div>
                    </div>
                ) : null}

                <button
                    type="submit"
                    disabled={!fileReady || disabled}
                    className="app-button app-button--primary"
                >
                    <UserPlus className="mr-2 h-4 w-4" />
                    {isLoading ? "Enrolling..." : "Enroll identity"}
                </button>
            </form>
        </section>
    );
}

export default function FingerprintLiveDemoWorkspace() {
    const [enrollmentFile, setEnrollmentFile] = useState<File | null>(null);
    const [probeFile, setProbeFile] = useState<File | null>(null);
    const [capture, setCapture] = useState<Capture>("plain");
    const [resultState, setResultState] = useState<AsyncState<IdentifyResponse>>(createIdleState());
    const [enrollState, setEnrollState] = useState<AsyncState<LiveEnrollResult>>(createIdleState());
    const [enrollForm, setEnrollForm] = useState<LiveEnrollForm>({
        fullName: "",
        nationalId: "",
        replaceExisting: false,
    });
    const [notice, setNotice] = useState<string | null>(null);
    const [enrollmentChangedMessage, setEnrollmentChangedMessage] = useState<string | null>(null);
    const [lastIdentifyProbeFileName, setLastIdentifyProbeFileName] = useState<string | null>(null);
    const [scannerStatusState, setScannerStatusState] = useState<AsyncState<ScannerStatusResponse>>(
        createLoadingState<ScannerStatusResponse>(),
    );
    const [scannerCaptureStates, setScannerCaptureStates] = useState<Record<ScannerImportTarget, ScannerCaptureCardActionState>>({
        enrollment: emptyScannerActionState(),
        probe: emptyScannerActionState(),
    });
    const directCaptureCountdownTimersRef = useRef<PendingDirectCaptureTimer[]>([]);
    const scannerCaptureMountedRef = useRef(true);

    const isIdentifyBusy = resultState.status === "loading";
    const isEnrollBusy = enrollState.status === "loading";
    const isScannerCaptureBusy = scannerCaptureStates.enrollment.status === "loading" || scannerCaptureStates.probe.status === "loading";
    const isBusy = isIdentifyBusy || isEnrollBusy || isScannerCaptureBusy;
    const latestResult = resultState.data;
    const latestEnrollment = enrollState.data;
    const identifyDisabled = !probeFile || isBusy;
    const enrollmentStepStatus = isEnrollBusy
        ? "Enrolling"
        : latestEnrollment
            ? enrollmentChangedMessage
                ? "Update needed"
                : "Enrolled"
            : enrollmentFile
                ? "Ready"
                : "Waiting";
    const probeStepStatus = isIdentifyBusy ? "Searching" : probeFile ? "Ready" : "Waiting";
    const resultStepStatus = resultState.status === "success"
        ? "Result ready"
        : resultState.status === "error"
            ? "Needs review"
            : probeFile
                ? "Ready to run"
                : "Needs probe";

    const clearDirectCaptureCountdownTimers = useCallback(() => {
        for (const timer of directCaptureCountdownTimersRef.current) {
            window.clearTimeout(timer.timeoutId);
            timer.resolve(false);
        }
        directCaptureCountdownTimersRef.current = [];
    }, []);

    useEffect(() => {
        scannerCaptureMountedRef.current = true;
        return () => {
            scannerCaptureMountedRef.current = false;
            clearDirectCaptureCountdownTimers();
        };
    }, [clearDirectCaptureCountdownTimers]);

    useEffect(() => {
        let isCancelled = false;

        async function loadStatus(): Promise<void> {
            setScannerStatusState((current) => createLoadingState(current.data));
            try {
                const status = await getScannerStatus();
                if (!isCancelled) {
                    setScannerStatusState(createSuccessState(status));
                }
            } catch (error) {
                if (!isCancelled) {
                    setScannerStatusState((current) => createErrorState(toErrorMessage(error), current.data));
                }
            }
        }

        void loadStatus();
        return () => {
            isCancelled = true;
        };
    }, []);

    function setScannerCaptureState(target: ScannerImportTarget, state: ScannerCaptureCardActionState): void {
        setScannerCaptureStates((current) => ({
            ...current,
            [target]: state,
        }));
    }

    function setEnrollmentCaptureFile(nextFile: File | null): void {
        setEnrollmentFile(nextFile);
        setNotice(null);
        setEnrollmentChangedMessage(enrollState.data ? ENROLLMENT_CHANGED_MESSAGE : null);
        setEnrollState((current) => (current.data ? createIdleState(current.data) : createIdleState()));
    }

    function handleEnrollmentFileChange(nextFile: File | null): void {
        setEnrollmentCaptureFile(nextFile);
        setScannerCaptureState("enrollment", emptyScannerActionState());
    }

    function handleCaptureChange(nextCapture: Capture): void {
        if (nextCapture === capture) {
            return;
        }

        setCapture(nextCapture);
        setNotice(null);
        setEnrollmentChangedMessage(enrollState.data ? ENROLLMENT_CHANGED_MESSAGE : null);
        setEnrollState((current) => (current.data ? createIdleState(current.data) : createIdleState()));
        setResultState(createIdleState());
        setLastIdentifyProbeFileName(null);
    }

    function setProbeCaptureFile(nextFile: File | null): void {
        setProbeFile(nextFile);
        setNotice(null);
        setResultState(createIdleState());
        setLastIdentifyProbeFileName(null);
    }

    function handleProbeFileChange(nextFile: File | null): void {
        setProbeCaptureFile(nextFile);
        setScannerCaptureState("probe", emptyScannerActionState());
    }

    function useEnrollmentImageAsProbe(): void {
        if (!enrollmentFile) {
            return;
        }

        setProbeCaptureFile(enrollmentFile);
        setScannerCaptureState("probe", emptyScannerActionState());
    }

    function updateEnrollForm(patch: Partial<LiveEnrollForm>): void {
        setEnrollForm((current) => ({ ...current, ...patch }));
        setEnrollState((current) => (current.status === "error" && !current.data ? createIdleState() : current));
        setNotice(null);
    }

    function waitForDirectCaptureCountdownTick(): Promise<boolean> {
        return new Promise((resolve) => {
            const timeoutId = window.setTimeout(() => {
                directCaptureCountdownTimersRef.current = directCaptureCountdownTimersRef.current.filter(
                    (timer) => timer.timeoutId !== timeoutId,
                );
                resolve(scannerCaptureMountedRef.current);
            }, DIRECT_CAPTURE_COUNTDOWN_TICK_MS);

            directCaptureCountdownTimersRef.current.push({ timeoutId, resolve });
        });
    }

    async function runDirectCaptureCountdown(target: ScannerImportTarget): Promise<boolean> {
        clearDirectCaptureCountdownTimers();

        for (let seconds = DIRECT_CAPTURE_SETTLE_SECONDS; seconds > 0; seconds -= 1) {
            if (!scannerCaptureMountedRef.current) {
                return false;
            }

            setScannerCaptureState(target, loadingScannerActionState("direct", seconds));
            const tickCompleted = await waitForDirectCaptureCountdownTick();
            if (!tickCompleted) {
                return false;
            }
        }

        return scannerCaptureMountedRef.current;
    }

    async function runScannerCapture(target: ScannerImportTarget, action: ScannerCaptureCardActionKind): Promise<void> {
        setNotice(null);
        if (action === "direct") {
            const settled = await runDirectCaptureCountdown(target);
            if (!settled) {
                return;
            }
        }

        if (!scannerCaptureMountedRef.current) {
            return;
        }

        setScannerCaptureState(target, loadingScannerActionState(action));

        try {
            const payload = await captureScanner({
                mode: action === "scanner_ui" ? "twain" : "auto",
                timeout_ms: action === "scanner_ui" ? 60000 : 15000,
                fallback_allowed: false,
                normalize: true,
                show_ui: action === "scanner_ui",
                settle_after_enable_ms: action === "scanner_ui" ? 0 : DIRECT_CAPTURE_SETTLE_AFTER_ENABLE_MS,
            });

            if (!scannerCaptureMountedRef.current) {
                return;
            }

            if (!payload.ok) {
                setScannerCaptureState(target, {
                    status: "error",
                    data: null,
                    error: scannerCaptureFailureMessage(payload),
                    action: null,
                });
                return;
            }

            const scannerFile = await loadScannerCaptureFile(payload);

            if (!scannerCaptureMountedRef.current) {
                return;
            }

            if (target === "enrollment") {
                setEnrollmentCaptureFile(scannerFile);
            } else {
                setProbeCaptureFile(scannerFile);
            }

            setScannerCaptureState(target, {
                status: "success",
                data: scannerResultFromCapture(payload, scannerFile.name),
                error: null,
                action: null,
            });
        } catch (error) {
            if (!scannerCaptureMountedRef.current) {
                return;
            }
            setScannerCaptureState(target, {
                status: "error",
                data: null,
                error: toErrorMessage(error),
                action: null,
            });
        }
    }

    async function importLatestScannerCapture(target: ScannerImportTarget): Promise<void> {
        setNotice(null);
        setScannerCaptureState(target, loadingScannerActionState("import_latest"));

        try {
            const importedCapture = await importLatestSavedScannerCapture();
            const scannerFile = await loadScannerCaptureFile(importedCapture);

            if (target === "enrollment") {
                setEnrollmentCaptureFile(scannerFile);
            } else {
                setProbeCaptureFile(scannerFile);
            }

            setScannerCaptureState(target, {
                status: "success",
                data: scannerResultFromImport(importedCapture),
                error: null,
                action: null,
            });
        } catch (error) {
            setScannerCaptureState(target, {
                status: "error",
                data: null,
                error: scannerImportErrorMessage(error),
                action: null,
            });
        }
    }

    async function runEnroll(): Promise<void> {
        setNotice(null);

        if (!enrollmentFile) {
            setEnrollState((current) => createErrorState("Upload an enrollment fingerprint before enrolling an identity.", current.data));
            return;
        }

        const fullName = enrollForm.fullName.trim();
        const nationalId = enrollForm.nationalId.trim();
        if (!fullName || !nationalId) {
            setEnrollState((current) => createErrorState("Full name and national ID are required for enrollment.", current.data));
            return;
        }

        const vectorMethods = [...DEFAULT_ENROLL_VECTOR_METHODS];
        setEnrollState((current) => createLoadingState(current.data));

        try {
            const payload = await enrollFingerprint({
                file: enrollmentFile,
                fullName,
                nationalId,
                capture,
                vectorMethods,
                replaceExisting: enrollForm.replaceExisting,
            });
            setEnrollState(createSuccessState({
                fullName,
                capture,
                sourceFileName: enrollmentFile.name,
                vectorMethods,
                response: payload,
            }));
            setEnrollmentChangedMessage(null);
            setResultState(createIdleState());
            setLastIdentifyProbeFileName(null);
        } catch (error) {
            setEnrollState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }

    async function runIdentify(): Promise<void> {
        setNotice(null);

        if (!probeFile) {
            setResultState((current) => createErrorState(MISSING_PROBE_MESSAGE, current.data));
            return;
        }

        setResultState((current) => createLoadingState(current.data));

        try {
            const payload = await identifyFingerprint({
                file: probeFile,
                capture,
                retrievalMethod: DEFAULT_RETRIEVAL_METHOD,
                rerankMethod: DEFAULT_RERANK_METHOD,
                shortlistSize: DEFAULT_SHORTLIST_SIZE,
            });
            setResultState(createSuccessState(payload));
            setLastIdentifyProbeFileName(probeFile.name);
            setNotice(
                payload.top_candidate
                    ? `Top candidate: ${payload.top_candidate.full_name}.`
                    : "Identification completed without a top candidate.",
            );
        } catch (error) {
            setResultState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }

    return (
        <div className="space-y-6">
            <WorkspaceHero
                eyebrow="Fingerprint biometrics"
                title="Live Demo"
                description="A focused three-step stakeholder flow for enrollment capture, separate probe capture, and an Identify 1:N result."
                icon={Fingerprint}
                actions={(
                    <button
                        type="button"
                        onClick={() => void runIdentify()}
                        disabled={identifyDisabled}
                        className="app-button app-button--primary"
                    >
                        <Play className="mr-2 h-4 w-4" />
                        {isIdentifyBusy ? "Running..." : "Run Identify 1:N"}
                    </button>
                )}
            />

            <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5 shadow-sm" aria-label="Shared capture profile">
                <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_18rem] md:items-end">
                    <div>
                        <h3 className="text-base font-semibold text-[var(--app-text)]">Shared capture profile</h3>
                        <p className="mt-1 text-sm leading-6 text-[var(--app-text-soft)]">
                            Used for both enrollment and probe in this demo.
                        </p>
                    </div>
                    <FormField label="Profile">
                        <select
                            className={INPUT_CLASS_NAME}
                            value={capture}
                            disabled={isBusy}
                            aria-label="Shared capture profile"
                            onChange={(event) => {
                                handleCaptureChange(event.target.value as Capture);
                            }}
                        >
                            {CAPTURE_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                    </FormField>
                </div>
            </section>

            <section className="grid gap-3 md:grid-cols-3" aria-label="Live demo stakeholder flow">
                <StepSummaryCard
                    step="Step 1"
                    title="Enrollment capture"
                    detail="Upload the fingerprint used to enroll the identity into the gallery."
                    status={enrollmentStepStatus}
                    icon={UserPlus}
                    highlighted={Boolean(latestEnrollment)}
                />
                <StepSummaryCard
                    step="Step 2"
                    title="Probe capture"
                    detail="Upload a separate probe fingerprint for the 1:N search."
                    status={probeStepStatus}
                    icon={Fingerprint}
                    highlighted={Boolean(probeFile)}
                />
                <StepSummaryCard
                    step="Step 3"
                    title="Identify 1:N result"
                    detail="Run the existing identify API and show the stakeholder-readable decision."
                    status={resultStepStatus}
                    icon={UserRoundSearch}
                    highlighted={resultState.status === "success"}
                />
            </section>

            {notice ? <InlineBanner variant="success">{notice}</InlineBanner> : null}

            <InlineBanner
                variant={latestEnrollment ? (enrollmentChangedMessage ? "warning" : "success") : "info"}
                title="Gallery readiness"
            >
                {latestEnrollment
                    ? `Keep showing the last enrolled identity as gallery evidence: ${latestEnrollment.fullName} from ${latestEnrollment.sourceFileName}. ${enrollmentChangedMessage ?? "Ready for Identify 1:N against the enrolled operational gallery."}`
                    : SEEDED_GALLERY_HINT}
            </InlineBanner>

            {!probeFile ? <InlineBanner variant="info">{MISSING_PROBE_MESSAGE}</InlineBanner> : null}

            <div className="grid gap-5 xl:grid-cols-2">
                <section className="space-y-5" aria-labelledby="enrollment-capture-heading">
                    <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">Step 1</p>
                        <h3 id="enrollment-capture-heading" className="mt-1 text-xl font-semibold text-[var(--app-text)]">
                            Enrollment capture
                        </h3>
                    </div>
                    <ScannerCaptureCard
                        file={enrollmentFile}
                        capture={capture}
                        disabled={isBusy}
                        scannerStatusState={scannerStatusState}
                        scannerActionState={scannerCaptureStates.enrollment}
                        eyebrow="Enrollment capture"
                        title="Enrollment fingerprint image"
                        description="Capture directly from the TWAIN scanner bridge, import the latest saved scan, or upload an image manually."
                        uploadTitle="Upload enrollment fingerprint"
                        uploadDescription="Recommended first step: choose the identity image that should enter the gallery."
                        onFileChange={handleEnrollmentFileChange}
                        onCaptureFromScanner={() => runScannerCapture("enrollment", "direct")}
                        onCaptureWithScannerUi={() => runScannerCapture("enrollment", "scanner_ui")}
                        onImportLatestSavedScan={() => importLatestScannerCapture("enrollment")}
                    />

                    <EnrollmentFormPanel
                        form={enrollForm}
                        capture={capture}
                        fileReady={Boolean(enrollmentFile)}
                        enrollmentChangedMessage={enrollmentChangedMessage}
                        disabled={isBusy}
                        enrollState={enrollState}
                        onUpdate={updateEnrollForm}
                        onSubmit={runEnroll}
                    />
                </section>

                <section className="space-y-5" aria-labelledby="probe-capture-heading">
                    <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">Step 2</p>
                        <h3 id="probe-capture-heading" className="mt-1 text-xl font-semibold text-[var(--app-text)]">
                            Probe capture
                        </h3>
                    </div>
                    <ScannerCaptureCard
                        file={probeFile}
                        capture={capture}
                        disabled={isBusy}
                        scannerStatusState={scannerStatusState}
                        scannerActionState={scannerCaptureStates.probe}
                        eyebrow="Probe capture"
                        title="Probe fingerprint image"
                        description="Capture directly from the TWAIN scanner bridge, import the latest saved scan, or upload an image manually."
                        uploadTitle="Upload probe fingerprint"
                        uploadDescription="Primary demo path: use a separate probe image to avoid same-image matching."
                        onFileChange={handleProbeFileChange}
                        onCaptureFromScanner={() => runScannerCapture("probe", "direct")}
                        onCaptureWithScannerUi={() => runScannerCapture("probe", "scanner_ui")}
                        onImportLatestSavedScan={() => importLatestScannerCapture("probe")}
                    />

                    <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4 shadow-sm">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                            <div>
                                <p className="text-sm font-semibold text-[var(--app-text)]">Quick smoke test</p>
                                <p className="text-sm leading-6 text-[var(--app-text-soft)]">
                                    Use this only when you need a same-image API smoke check; the recommended demo uses a separate probe image.
                                </p>
                            </div>
                            <button
                                type="button"
                                disabled={!enrollmentFile || isBusy}
                                onClick={useEnrollmentImageAsProbe}
                                className="app-button app-button--secondary"
                            >
                                Use enrollment image as probe
                            </button>
                        </div>
                    </div>

                    <QualityStatusPanel enrollmentFile={enrollmentFile} probeFile={probeFile} busy={isBusy} />
                </section>
            </div>

            <section className="space-y-4" aria-labelledby="identify-result-heading">
                <div className="flex flex-wrap items-end justify-between gap-4">
                    <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--app-text-muted)]">Step 3</p>
                        <h3 id="identify-result-heading" className="mt-1 text-xl font-semibold text-[var(--app-text)]">
                            Identify 1:N result
                        </h3>
                    </div>
                    {!latestEnrollment && probeFile ? (
                        <p className="max-w-xl text-sm leading-6 text-[var(--app-text-soft)]">{SEEDED_GALLERY_HINT}</p>
                    ) : null}
                </div>

                <section className="space-y-3">
                    <h3 className="text-base font-semibold text-[var(--app-text)]">Choose action</h3>
                    <div className="grid gap-3 lg:grid-cols-3">
                        <ActionCard
                            title="Enroll"
                            detail="Create a fingerprint identity in the operational gallery from the enrollment capture."
                            status={isEnrollBusy ? "Enrolling..." : latestEnrollment ? "Enrolled" : enrollmentFile ? "Available" : "Needs enrollment"}
                            icon={UserPlus}
                            disabled={!enrollmentFile || isBusy}
                            highlighted={Boolean(latestEnrollment)}
                            onClick={runEnroll}
                        />
                        <ActionCard
                            title="Verify 1:1"
                            detail="Use the full Verify workspace for one-to-one comparisons."
                            status="Available in Verify tab"
                            icon={ShieldCheck}
                            disabled
                        />
                        <ActionCard
                            title="Identify 1:N"
                            detail="Search the enrolled operational gallery with the probe capture."
                            status={isIdentifyBusy ? "Running..." : probeFile ? "Available" : "Needs probe"}
                            icon={UserRoundSearch}
                            disabled={identifyDisabled}
                            highlighted={resultState.status === "success"}
                            onClick={runIdentify}
                        />
                    </div>
                </section>

                <LiveResultHero
                    resultState={resultState}
                    enrollmentSourceFileName={latestEnrollment?.sourceFileName ?? null}
                    probeSourceFileName={lastIdentifyProbeFileName}
                    onRetry={runIdentify}
                />
            </section>

            <LiveEvidenceStrip
                result={latestResult}
                retrievalMethod={DEFAULT_RETRIEVAL_METHOD}
                rerankMethod={DEFAULT_RERANK_METHOD}
            />
        </div>
    );
}
