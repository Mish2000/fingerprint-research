import { CheckCircle2, CircleDashed, Fingerprint, ScanLine } from "lucide-react";
import FileDropBox from "../../../components/FileDropBox.tsx";
import { formatCaptureLabel } from "../../../shared/storytelling.ts";
import FormField from "../../../shared/ui/FormField.tsx";
import { INPUT_CLASS_NAME } from "../../../shared/ui/inputClasses.ts";
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
    captureProfileNote?: string;
    onFileChange: (file: File | null) => void;
    onCaptureChange: (capture: Capture) => void;
}

const CAPTURE_OPTIONS: Array<{ value: Capture; label: string }> = [
    { value: "plain", label: "Plain" },
    { value: "roll", label: "Rolled" },
    { value: "contactless", label: "Contactless" },
    { value: "contact_based", label: "Contact-based" },
];

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
    title = "Place finger / capture fingerprint",
    description = "Manual upload fallback; scanner SDK wiring next.",
    uploadTitle = "Manual fingerprint image",
    uploadDescription = "Manual upload fallback until scanner SDK wiring is added.",
    captureProfileNote = "One shared capture profile is used for enrollment and probe in this demo.",
    onFileChange,
    onCaptureChange,
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
                                <p className="text-sm font-semibold text-slate-900">Scanner flow</p>
                                <p className="text-xs text-slate-500">Manual upload fallback; scanner SDK wiring next.</p>
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
                            <StatusRow active={false} label="Quality" value="Scanner quality score pending" />
                        </div>
                    </div>

                    <FormField label="Capture profile">
                        <select
                            className={INPUT_CLASS_NAME}
                            value={capture}
                            disabled={disabled}
                            onChange={(event) => {
                                onCaptureChange(event.target.value as Capture);
                            }}
                        >
                            {CAPTURE_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                        <p className="mt-2 text-xs leading-5 text-slate-500">{captureProfileNote}</p>
                    </FormField>
                </div>
            </div>
        </section>
    );
}
