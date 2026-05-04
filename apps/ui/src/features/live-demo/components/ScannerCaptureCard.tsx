import { CheckCircle2, CircleDashed, Fingerprint, ScanLine } from "lucide-react";
import FileDropBox from "../../../components/FileDropBox.tsx";
import { formatCaptureLabel } from "../../../shared/storytelling.ts";
import type { Capture } from "../../../types/index.ts";

interface ScannerCaptureCardProps {
    file: File | null;
    capture: Capture;
    disabled: boolean;
    eyebrow?: string;
    title?: string;
    description?: string;
    uploadTitle?: string;
    uploadDescription?: string;
    onFileChange: (file: File | null) => void;
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
        <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2">
            <div
                className={`rounded-full p-1.5 ${
                    active ? "bg-emerald-50 text-emerald-600" : "bg-slate-100 text-slate-400"
                }`}
            >
                <Icon className="h-4 w-4" />
            </div>
            <div className="min-w-0">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">{label}</p>
                <p className="truncate text-sm font-semibold text-slate-800">{value}</p>
            </div>
        </div>
    );
}

export default function ScannerCaptureCard({
    file,
    capture,
    disabled,
    eyebrow = "Fingerprint source",
    title = "Fingerprint image source",
    description = "Manual upload remains available. Direct SDK capture is a future milestone.",
    uploadTitle = "Manual fingerprint image",
    uploadDescription = "Choose a saved fingerprint image from disk.",
    onFileChange,
}: ScannerCaptureCardProps) {
    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                    <div className="inline-flex items-center gap-2 rounded-full border border-brand-100 bg-brand-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-brand-900">
                        <Fingerprint className="h-3.5 w-3.5" />
                        {eyebrow}
                    </div>
                    <h3 className="mt-3 text-xl font-semibold text-slate-900">{title}</h3>
                    <p className="mt-1 text-sm leading-6 text-slate-600">
                        {description}
                    </p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-700">
                    {formatCaptureLabel(capture)}
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
                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                        <div className="flex items-center gap-3">
                            <div className="rounded-lg bg-slate-900 p-2 text-white">
                                <ScanLine className="h-5 w-5" />
                            </div>
                            <div>
                                <p className="text-sm font-semibold text-slate-900">Capture source</p>
                                <p className="text-xs text-slate-500">Manual upload and saved-file import share this image slot.</p>
                            </div>
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
                    </div>
                </div>
            </div>
        </section>
    );
}
