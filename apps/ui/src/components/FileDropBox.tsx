import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, KeyboardEvent, MouseEvent } from "react";
import { FileImage, Image as ImageIcon, UploadCloud, X } from "lucide-react";

interface FileDropBoxProps {
    file?: File | null;
    onChange?: (file: File | null) => void;
    onFileSelect?: (file: File) => void;
    title?: string;
    description?: string;
    className?: string;
    disabled?: boolean;
    error?: string | null;
}

export default function FileDropBox({
    file,
    onChange,
    onFileSelect,
    title = "Upload fingerprint",
    description = "Drag & drop or click to select an image",
    className = "",
    disabled = false,
    error = null,
}: FileDropBoxProps) {
    const isControlled = file !== undefined;
    const [internalFile, setInternalFile] = useState<File | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [localError, setLocalError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);

    const selectedFile = isControlled ? (file ?? null) : internalFile;
    const activeError = error ?? localError;
    const previewUrl = useMemo(() => (selectedFile ? URL.createObjectURL(selectedFile) : null), [selectedFile]);

    useEffect(() => {
        return () => {
            if (previewUrl) {
                URL.revokeObjectURL(previewUrl);
            }
        };
    }, [previewUrl]);

    function emitFileChange(nextFile: File | null): void {
        if (!isControlled) {
            setInternalFile(nextFile);
        }

        onChange?.(nextFile);
        if (nextFile) {
            onFileSelect?.(nextFile);
        }
    }

    function openPicker(): void {
        if (!disabled) {
            fileInputRef.current?.click();
        }
    }

    function handleFile(nextFile: File): void {
        if (!nextFile.type.startsWith("image/")) {
            setLocalError("Please choose an image file (PNG, JPG, BMP, TIFF, etc.).");
            return;
        }

        setLocalError(null);
        emitFileChange(nextFile);
    }

    function handleDragOver(event: DragEvent<HTMLDivElement>): void {
        event.preventDefault();
        if (!disabled) {
            setIsDragging(true);
        }
    }

    function handleDragLeave(event: DragEvent<HTMLDivElement>): void {
        event.preventDefault();
        setIsDragging(false);
    }

    function handleDrop(event: DragEvent<HTMLDivElement>): void {
        event.preventDefault();
        setIsDragging(false);

        if (disabled) {
            return;
        }

        const droppedFile = event.dataTransfer.files.item(0);
        if (droppedFile) {
            handleFile(droppedFile);
        }
    }

    function handleFileInput(event: ChangeEvent<HTMLInputElement>): void {
        const pickedFile = event.target.files?.item(0) ?? null;
        if (pickedFile) {
            handleFile(pickedFile);
        }
    }

    function clearFile(event: MouseEvent<HTMLButtonElement>): void {
        event.stopPropagation();
        setLocalError(null);
        emitFileChange(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }

    const containerClassName = [
        "relative flex w-full cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 text-center transition-all duration-300 ease-in-out",
        className,
        disabled ? "cursor-not-allowed border-slate-200 bg-slate-100 opacity-70" : "",
        !disabled && isDragging ? "scale-[1.01] border-brand-500 bg-brand-50 ring-2 ring-brand-100 shadow-lg" : "",
        !disabled && !isDragging && selectedFile ? "border-slate-200 bg-white" : "",
        !disabled && !isDragging && !selectedFile ? "border-dashed border-slate-300 bg-slate-50 hover:border-slate-400 hover:bg-slate-100" : "",
        activeError ? "border-red-300 ring-2 ring-red-100" : "",
    ]
        .filter(Boolean)
        .join(" ");

    return (
        <div className="space-y-3">
            <div
                className={containerClassName}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => {
                    if (!selectedFile) {
                        openPicker();
                    }
                }}
                role="button"
                tabIndex={disabled ? -1 : 0}
                onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
                    if ((event.key === "Enter" || event.key === " ") && !selectedFile) {
                        event.preventDefault();
                        openPicker();
                    }
                }}
                aria-disabled={disabled}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept="image/*"
                    disabled={disabled}
                    onChange={handleFileInput}
                />

                {selectedFile && previewUrl ? (
                    <div className="flex h-full w-full flex-col p-3">
                        <button
                            type="button"
                            onClick={clearFile}
                            className="absolute top-4 right-4 z-10 rounded-full border border-slate-100 bg-white/80 p-1.5 text-slate-500 shadow-md backdrop-blur-sm transition-colors hover:bg-red-50 hover:text-red-500"
                            title="Remove file"
                            disabled={disabled}
                        >
                            <X className="h-4 w-4" />
                        </button>

                        <div className="flex flex-1 items-center justify-center overflow-hidden rounded-lg border border-slate-200 bg-slate-100">
                            <img
                                src={previewUrl}
                                alt="Uploaded fingerprint preview"
                                className="max-h-full max-w-full object-contain shadow-inner"
                            />
                        </div>

                        <div className="flex items-center justify-between px-1 pt-3 text-left">
                            <div className="flex items-center gap-2.5 overflow-hidden">
                                <div className="shrink-0 rounded-lg border border-emerald-100 bg-emerald-50 p-1.5 text-emerald-600">
                                    <ImageIcon className="h-4 w-4" />
                                </div>
                                <div className="overflow-hidden">
                                    <p className="truncate text-sm font-medium text-slate-800" title={selectedFile.name}>
                                        {selectedFile.name}
                                    </p>
                                    <p className="text-xs text-slate-500">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                                </div>
                            </div>
                            <span className="rounded-full border border-emerald-100 bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700">
                                Ready
                            </span>
                        </div>
                    </div>
                ) : (
                    <div className="pointer-events-none flex flex-col items-center p-10 py-16">
                        <div
                            className={`mb-5 rounded-full p-5 transition-all ${
                                isDragging ? "scale-110 bg-brand-100 text-brand-600" : "bg-white text-slate-400 shadow-inner"
                            }`}
                        >
                            {isDragging ? <UploadCloud className="h-10 w-10" /> : <FileImage className="h-10 w-10" />}
                        </div>
                        <h3 className={`mb-1.5 text-xl font-semibold ${isDragging ? "text-brand-700" : "text-slate-700"}`}>
                            {title}
                        </h3>
                        <p className="max-w-xs text-sm text-slate-500">{description}</p>
                    </div>
                )}
            </div>

            {activeError ? <p className="text-sm text-red-600">{activeError}</p> : null}
        </div>
    );
}
