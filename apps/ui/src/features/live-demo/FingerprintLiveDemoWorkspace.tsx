import { useState } from "react";
import { Fingerprint, Play, ShieldCheck, UserPlus, UserRoundSearch } from "lucide-react";
import { identifyFingerprint } from "../../api/identificationService.ts";
import {
    createErrorState,
    createIdleState,
    createLoadingState,
    createSuccessState,
    type AsyncState,
} from "../../shared/request-state/index.ts";
import InlineBanner from "../../shared/ui/InlineBanner.tsx";
import type {
    Capture,
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

const ACTION_CARDS = [
    {
        id: "enroll",
        title: "Enroll",
        detail: "Create a new fingerprint identity.",
        status: "Coming next",
        icon: UserPlus,
        disabled: true,
    },
    {
        id: "verify",
        title: "Verify 1:1",
        detail: "Compare one probe to one reference.",
        status: "Coming next",
        icon: ShieldCheck,
        disabled: true,
    },
    {
        id: "identify",
        title: "Identify 1:N",
        detail: "Search the enrolled gallery.",
        status: "Available",
        icon: UserRoundSearch,
        disabled: false,
    },
] as const;

function ActionCard({
    title,
    detail,
    status,
    icon: Icon,
    disabled,
}: (typeof ACTION_CARDS)[number]) {
    return (
        <button
            type="button"
            disabled={disabled}
            aria-pressed={!disabled}
            className={`rounded-2xl border p-4 text-left shadow-sm transition ${
                disabled
                    ? "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-400"
                    : "border-brand-200 bg-brand-50 text-brand-950 ring-2 ring-brand-100"
            }`}
        >
            <div className="flex items-start justify-between gap-3">
                <div className={`rounded-lg p-2 ${disabled ? "bg-white text-slate-400" : "bg-white text-brand-700"}`}>
                    <Icon className="h-5 w-5" />
                </div>
                <span
                    className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${
                        disabled ? "border-slate-200 bg-white text-slate-400" : "border-brand-200 bg-white text-brand-800"
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
            value: file ? "Placeholder pass" : "Pending capture",
        },
        {
            label: "Preprocessing",
            value: busy ? "Running" : file ? "Prepared on submit" : "Pending",
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
                    <p className="text-sm text-slate-500">Placeholder states for the future scanner pipeline.</p>
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

export default function FingerprintLiveDemoWorkspace() {
    const [file, setFile] = useState<File | null>(null);
    const [capture, setCapture] = useState<Capture>("plain");
    const [resultState, setResultState] = useState<AsyncState<IdentifyResponse>>(createIdleState());
    const [notice, setNotice] = useState<string | null>(null);

    const isBusy = resultState.status === "loading";
    const latestResult = resultState.data;

    function handleFileChange(nextFile: File | null): void {
        setFile(nextFile);
        setNotice(null);
        setResultState(createIdleState());
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
                        {isBusy ? "Running..." : "Run Identify 1:N"}
                    </button>
                </div>
            </section>

            {notice ? <InlineBanner variant="success">{notice}</InlineBanner> : null}

            <div className="grid gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(20rem,0.8fr)]">
                <ScannerCaptureCard
                    file={file}
                    capture={capture}
                    disabled={isBusy}
                    onFileChange={handleFileChange}
                    onCaptureChange={setCapture}
                />

                <div className="space-y-5">
                    <QualityStatusPanel file={file} busy={isBusy} />

                    <section className="space-y-3">
                        <h3 className="text-base font-semibold text-slate-900">Choose action</h3>
                        <div className="grid gap-3">
                            {ACTION_CARDS.map((action) => (
                                <ActionCard key={action.id} {...action} />
                            ))}
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
