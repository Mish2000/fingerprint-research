import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import { Fingerprint, Play, ShieldCheck, UserPlus, UserRoundSearch } from "lucide-react";
import { enrollFingerprint, identifyFingerprint } from "../../api/identificationService.ts";
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
import type {
    Capture,
    EnrollFingerprintResponse,
    IdentificationRetrievalMethod,
    IdentifyResponse,
    Method,
} from "../../types/index.ts";
import { toErrorMessage } from "../../utils/error.ts";
import LiveEvidenceStrip from "./components/LiveEvidenceStrip.tsx";
import LiveResultHero from "./components/LiveResultHero.tsx";
import ScannerCaptureCard from "./components/ScannerCaptureCard.tsx";

const DEFAULT_RETRIEVAL_METHOD: IdentificationRetrievalMethod = "dl";
const DEFAULT_RERANK_METHOD: Method = "sift";
const DEFAULT_SHORTLIST_SIZE = 10;
const DEFAULT_ENROLL_VECTOR_METHODS = ["dl", "vit"];
const ENROLLMENT_CHANGED_MESSAGE = "Enrollment capture changed — enroll again to update this identity.";
const MISSING_PROBE_MESSAGE = "Upload a probe fingerprint to search the enrolled gallery.";
const SEEDED_GALLERY_HINT = "You can search an existing gallery, but for a clean demo enroll an identity first.";

interface LiveEnrollForm {
    fullName: string;
    nationalId: string;
    replaceExisting: boolean;
}

interface LiveEnrollResult {
    fullName: string;
    capture: Capture;
    sourceFileName: string;
    vectorMethods: string[];
    response: EnrollFingerprintResponse;
}

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
                    ? "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-400"
                    : highlighted
                        ? "border-emerald-200 bg-emerald-50 text-emerald-950 ring-2 ring-emerald-100"
                        : "border-brand-200 bg-brand-50 text-brand-950 ring-2 ring-brand-100"
            }`}
        >
            <div className="flex items-start justify-between gap-3">
                <div className={`rounded-lg p-2 ${disabled ? "bg-white text-slate-400" : "bg-white text-brand-700"}`}>
                    <Icon className="h-5 w-5" />
                </div>
                <span
                    className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${
                        disabled
                            ? "border-slate-200 bg-white text-slate-400"
                            : highlighted
                                ? "border-emerald-200 bg-white text-emerald-800"
                                : "border-brand-200 bg-white text-brand-800"
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
                highlighted ? "border-emerald-200 bg-emerald-50" : "border-slate-200 bg-white"
            }`}
        >
            <div className="flex items-start justify-between gap-3">
                <div className="rounded-lg border border-slate-200 bg-white p-2 text-slate-700">
                    <Icon className="h-5 w-5" />
                </div>
                <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-xs font-semibold text-slate-600">
                    {status}
                </span>
            </div>
            <p className="mt-4 text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">{step}</p>
            <h3 className="mt-1 text-base font-semibold text-slate-900">{title}</h3>
            <p className="mt-1 text-sm leading-6 text-slate-600">{detail}</p>
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
            value: enrollmentFile ? "Manual upload ready" : "Waiting",
        },
        {
            label: "Probe source",
            value: probeFile ? "Manual upload ready" : "Waiting",
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
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-2 text-slate-600">
                    <Fingerprint className="h-5 w-5" />
                </div>
                <div>
                    <h3 className="text-base font-semibold text-slate-900">Quality / preprocessing status</h3>
                    <p className="text-sm text-slate-500">Scanner quality score pending; preprocessing runs when you submit.</p>
                </div>
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {statusItems.map((item) => (
                    <div key={item.label} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3">
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">{item.label}</p>
                        <p className="mt-1 text-sm font-semibold text-slate-800">{item.value}</p>
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
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                    <div className="inline-flex items-center gap-2 rounded-full border border-brand-100 bg-brand-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-brand-900">
                        <UserPlus className="h-3.5 w-3.5" />
                        Enroll identity
                    </div>
                    <h3 className="mt-3 text-xl font-semibold text-slate-900">Identity details</h3>
                    <p className="mt-1 text-sm leading-6 text-slate-600">
                        Uses the enrollment fingerprint image and the shared {formatCaptureLabel(capture)} capture profile.
                    </p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-700">
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

                <label className="inline-flex items-center gap-2 text-sm font-medium text-slate-700">
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
                    <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-emerald-950">
                        <div className="flex flex-wrap items-start justify-between gap-4">
                            <div>
                                <p className="text-sm font-semibold">Enrollment completed</p>
                                <p className="mt-1 text-sm leading-6 text-emerald-800">
                                    Ready for Identify 1:N against the enrolled operational gallery.
                                </p>
                            </div>
                            <UserRoundSearch className="h-5 w-5 text-emerald-700" />
                        </div>
                        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
                            <div className="rounded-lg border border-emerald-200 bg-white/75 px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-600">Name</p>
                                <p className="mt-1 text-sm font-semibold">{result.fullName}</p>
                            </div>
                            <div className="rounded-lg border border-emerald-200 bg-white/75 px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-600">random_id</p>
                                <p className="mt-1 break-all text-sm font-semibold">{result.response.random_id}</p>
                            </div>
                            <div className="rounded-lg border border-emerald-200 bg-white/75 px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-600">Capture</p>
                                <p className="mt-1 text-sm font-semibold">{formatCaptureLabel(result.capture)}</p>
                            </div>
                            <div className="rounded-lg border border-emerald-200 bg-white/75 px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-600">Enrollment source file</p>
                                <p className="mt-1 truncate text-sm font-semibold" title={result.sourceFileName}>
                                    {result.sourceFileName}
                                </p>
                            </div>
                            <div className="rounded-lg border border-emerald-200 bg-white/75 px-3 py-3">
                                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-600">Vectors</p>
                                <p className="mt-1 text-sm font-semibold">{formatVectorMethods(result.response.vector_methods ?? result.vectorMethods)}</p>
                            </div>
                        </div>
                    </div>
                ) : null}

                <button
                    type="submit"
                    disabled={!fileReady || disabled}
                    className="inline-flex items-center rounded-lg bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-55"
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

    const isIdentifyBusy = resultState.status === "loading";
    const isEnrollBusy = enrollState.status === "loading";
    const isBusy = isIdentifyBusy || isEnrollBusy;
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

    function handleEnrollmentFileChange(nextFile: File | null): void {
        setEnrollmentFile(nextFile);
        setNotice(null);
        setEnrollmentChangedMessage(enrollState.data ? ENROLLMENT_CHANGED_MESSAGE : null);
        setEnrollState((current) => (current.data ? createIdleState(current.data) : createIdleState()));
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

    function handleProbeFileChange(nextFile: File | null): void {
        setProbeFile(nextFile);
        setNotice(null);
        setResultState(createIdleState());
        setLastIdentifyProbeFileName(null);
    }

    function useEnrollmentImageAsProbe(): void {
        if (!enrollmentFile) {
            return;
        }

        setProbeFile(enrollmentFile);
        setNotice(null);
        setResultState(createIdleState());
        setLastIdentifyProbeFileName(null);
    }

    function updateEnrollForm(patch: Partial<LiveEnrollForm>): void {
        setEnrollForm((current) => ({ ...current, ...patch }));
        setEnrollState((current) => (current.status === "error" && !current.data ? createIdleState() : current));
        setNotice(null);
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
            <section className="rounded-2xl border border-brand-100 bg-[linear-gradient(120deg,var(--app-brand-surface)_0%,var(--app-surface)_52%,var(--app-success-surface)_100%)] p-6 shadow-sm">
                <div className="flex flex-wrap items-start justify-between gap-5">
                    <div className="max-w-3xl">
                        <div className="inline-flex items-center gap-2 rounded-full border border-brand-100 bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-brand-900">
                            <Fingerprint className="h-3.5 w-3.5" />
                            Fingerprint biometrics
                        </div>
                        <h3 className="mt-4 text-3xl font-semibold text-slate-950">Live Demo</h3>
                        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
                            A focused three-step stakeholder flow for enrollment capture, separate probe capture, and an Identify 1:N result.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={() => void runIdentify()}
                        disabled={identifyDisabled}
                        className="inline-flex items-center rounded-lg bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-55"
                    >
                        <Play className="mr-2 h-4 w-4" />
                        {isIdentifyBusy ? "Running..." : "Run Identify 1:N"}
                    </button>
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
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Step 1</p>
                        <h3 id="enrollment-capture-heading" className="mt-1 text-xl font-semibold text-slate-900">
                            Enrollment capture
                        </h3>
                    </div>
                    <ScannerCaptureCard
                        file={enrollmentFile}
                        capture={capture}
                        disabled={isBusy}
                        eyebrow="Enrollment capture"
                        title="Enrollment fingerprint image"
                        description="Manual upload fallback; scanner SDK wiring next. This image is used only for Enroll identity."
                        uploadTitle="Upload enrollment fingerprint"
                        uploadDescription="Recommended first step: choose the identity image that should enter the gallery."
                        onFileChange={handleEnrollmentFileChange}
                        onCaptureChange={handleCaptureChange}
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
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Step 2</p>
                        <h3 id="probe-capture-heading" className="mt-1 text-xl font-semibold text-slate-900">
                            Probe capture
                        </h3>
                    </div>
                    <ScannerCaptureCard
                        file={probeFile}
                        capture={capture}
                        disabled={isBusy}
                        eyebrow="Probe capture"
                        title="Probe fingerprint image"
                        description="Manual upload fallback; scanner SDK wiring next. This image is used only for Identify 1:N."
                        uploadTitle="Upload probe fingerprint"
                        uploadDescription="Primary demo path: use a separate probe image to avoid same-image matching."
                        onFileChange={handleProbeFileChange}
                        onCaptureChange={handleCaptureChange}
                    />

                    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                            <div>
                                <p className="text-sm font-semibold text-slate-900">Quick smoke test</p>
                                <p className="text-sm leading-6 text-slate-600">
                                    Use this only when you need a same-image API smoke check; the recommended demo uses a separate probe image.
                                </p>
                            </div>
                            <button
                                type="button"
                                disabled={!enrollmentFile || isBusy}
                                onClick={useEnrollmentImageAsProbe}
                                className="inline-flex items-center rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-55"
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
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Step 3</p>
                        <h3 id="identify-result-heading" className="mt-1 text-xl font-semibold text-slate-900">
                            Identify 1:N result
                        </h3>
                    </div>
                    {!latestEnrollment && probeFile ? (
                        <p className="max-w-xl text-sm leading-6 text-slate-600">{SEEDED_GALLERY_HINT}</p>
                    ) : null}
                </div>

                <section className="space-y-3">
                    <h3 className="text-base font-semibold text-slate-900">Choose action</h3>
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
