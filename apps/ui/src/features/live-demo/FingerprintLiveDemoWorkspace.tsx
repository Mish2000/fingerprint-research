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

interface LiveEnrollForm {
    fullName: string;
    nationalId: string;
    replaceExisting: boolean;
}

interface LiveEnrollResult {
    fullName: string;
    capture: Capture;
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

function QualityStatusPanel({ file, busy }: { file: File | null; busy: boolean }) {
    const statusItems = [
        {
            label: "Image source",
            value: file ? "Manual upload ready" : "Waiting",
        },
        {
            label: "Quality",
            value: file ? "Not scored yet" : "Pending capture",
        },
        {
            label: "Preprocessing",
            value: busy ? "Running" : file ? "Backend preprocessing runs on submit" : "Pending",
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
    disabled,
    enrollState,
    onUpdate,
    onSubmit,
}: {
    form: LiveEnrollForm;
    capture: Capture;
    fileReady: boolean;
    disabled: boolean;
    enrollState: AsyncState<LiveEnrollResult>;
    onUpdate: (patch: Partial<LiveEnrollForm>) => void;
    onSubmit: () => void | Promise<void>;
}) {
    const isLoading = enrollState.status === "loading";
    const result = enrollState.status === "success" ? enrollState.data : null;

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
                        Uses the current fingerprint upload and {formatCaptureLabel(capture)} capture profile.
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
                        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
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
    const [file, setFile] = useState<File | null>(null);
    const [capture, setCapture] = useState<Capture>("plain");
    const [resultState, setResultState] = useState<AsyncState<IdentifyResponse>>(createIdleState());
    const [enrollState, setEnrollState] = useState<AsyncState<LiveEnrollResult>>(createIdleState());
    const [enrollForm, setEnrollForm] = useState<LiveEnrollForm>({
        fullName: "",
        nationalId: "",
        replaceExisting: false,
    });
    const [notice, setNotice] = useState<string | null>(null);

    const isIdentifyBusy = resultState.status === "loading";
    const isEnrollBusy = enrollState.status === "loading";
    const isBusy = isIdentifyBusy || isEnrollBusy;
    const latestResult = resultState.data;
    const latestEnrollment = enrollState.status === "success" ? enrollState.data : null;

    function handleFileChange(nextFile: File | null): void {
        setFile(nextFile);
        setNotice(null);
        setResultState(createIdleState());
        setEnrollState(createIdleState());
    }

    function handleCaptureChange(nextCapture: Capture): void {
        setCapture(nextCapture);
        setNotice(null);
        setResultState(createIdleState());
        setEnrollState(createIdleState());
    }

    function updateEnrollForm(patch: Partial<LiveEnrollForm>): void {
        setEnrollForm((current) => ({ ...current, ...patch }));
        if (enrollState.status !== "idle") {
            setEnrollState(createIdleState());
        }
        setNotice(null);
    }

    async function runEnroll(): Promise<void> {
        setNotice(null);

        if (!file) {
            setEnrollState(createErrorState("Upload a fingerprint image before enrolling an identity."));
            return;
        }

        const fullName = enrollForm.fullName.trim();
        const nationalId = enrollForm.nationalId.trim();
        if (!fullName || !nationalId) {
            setEnrollState(createErrorState("Full name and national ID are required for enrollment."));
            return;
        }

        const vectorMethods = [...DEFAULT_ENROLL_VECTOR_METHODS];
        setEnrollState((current) => createLoadingState(current.data));

        try {
            const payload = await enrollFingerprint({
                file,
                fullName,
                nationalId,
                capture,
                vectorMethods,
                replaceExisting: enrollForm.replaceExisting,
            });
            setEnrollState(createSuccessState({
                fullName,
                capture,
                vectorMethods,
                response: payload,
            }));
            setResultState(createIdleState());
        } catch (error) {
            setEnrollState((current) => createErrorState(toErrorMessage(error), current.data));
        }
    }

    async function runIdentify(): Promise<void> {
        setNotice(null);

        if (!file) {
            setResultState((current) => createErrorState("Upload a fingerprint image before running Identify 1:N.", current.data));
            return;
        }

        setResultState((current) => createLoadingState(current.data));

        try {
            const payload = await identifyFingerprint({
                file,
                capture,
                retrievalMethod: DEFAULT_RETRIEVAL_METHOD,
                rerankMethod: DEFAULT_RERANK_METHOD,
                shortlistSize: DEFAULT_SHORTLIST_SIZE,
            });
            setResultState(createSuccessState(payload));
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
                            A focused presentation flow for capture, quality status, action selection, and a stakeholder-readable result.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={() => void runIdentify()}
                        disabled={!file || isBusy}
                        className="inline-flex items-center rounded-lg bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-55"
                    >
                        <Play className="mr-2 h-4 w-4" />
                        {isIdentifyBusy ? "Running..." : "Run Identify 1:N"}
                    </button>
                </div>
            </section>

            {notice ? <InlineBanner variant="success">{notice}</InlineBanner> : null}

            <InlineBanner variant={latestEnrollment ? "success" : "info"} title={latestEnrollment ? "Ready for Identify 1:N" : "Gallery readiness"}>
                {latestEnrollment
                    ? `Identify searches the enrolled operational gallery. ${latestEnrollment.fullName} is enrolled for this session.`
                    : "Identify searches the enrolled operational gallery. Enroll an identity first or use an already seeded gallery."}
            </InlineBanner>

            <div className="grid gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(20rem,0.8fr)]">
                <ScannerCaptureCard
                    file={file}
                    capture={capture}
                    disabled={isBusy}
                    onFileChange={handleFileChange}
                    onCaptureChange={handleCaptureChange}
                />

                <div className="space-y-5">
                    <EnrollmentFormPanel
                        form={enrollForm}
                        capture={capture}
                        fileReady={Boolean(file)}
                        disabled={isBusy}
                        enrollState={enrollState}
                        onUpdate={updateEnrollForm}
                        onSubmit={runEnroll}
                    />

                    <QualityStatusPanel file={file} busy={isBusy} />

                    <section className="space-y-3">
                        <h3 className="text-base font-semibold text-slate-900">Choose action</h3>
                        <div className="grid gap-3">
                            <ActionCard
                                title="Enroll"
                                detail="Create a fingerprint identity in the operational gallery."
                                status={isEnrollBusy ? "Enrolling..." : latestEnrollment ? "Enrolled" : file ? "Available" : "Needs fingerprint"}
                                icon={UserPlus}
                                disabled={!file || isBusy}
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
                                detail="Search the enrolled operational gallery."
                                status={isIdentifyBusy ? "Running..." : latestEnrollment ? "Ready now" : file ? "Available" : "Needs fingerprint"}
                                icon={UserRoundSearch}
                                disabled={!file || isBusy}
                                highlighted={Boolean(latestEnrollment)}
                                onClick={runIdentify}
                            />
                        </div>
                    </section>
                </div>
            </div>

            <LiveResultHero
                resultState={resultState}
                sourceFileName={file?.name ?? null}
                onRetry={runIdentify}
            />

            <LiveEvidenceStrip
                result={latestResult}
                retrievalMethod={DEFAULT_RETRIEVAL_METHOD}
                rerankMethod={DEFAULT_RERANK_METHOD}
            />
        </div>
    );
}
