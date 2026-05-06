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
        "file-drop-zone relative flex w-full cursor-pointer flex-col items-center justify-center overflow-hidden text-center transition-all duration-300 ease-in-out",
        className,
        disabled ? "file-drop-zone--disabled" : "",
        !disabled && isDragging ? "file-drop-zone--dragging" : "",
        !disabled && !isDragging && selectedFile ? "file-drop-zone--ready" : "",
        activeError ? "file-drop-zone--error" : "",
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
                            className="absolute top-4 right-4 z-10 rounded-full border border-[var(--app-border)] bg-[var(--app-surface)] p-1.5 text-[var(--app-text-muted)] shadow-md backdrop-blur-sm transition-colors hover:bg-[var(--app-error-surface)] hover:text-[var(--app-error-text)]"
                            title="Remove file"
                            disabled={disabled}
                        >
                            <X className="h-4 w-4" />
                        </button>

                        <div className="file-drop-zone__preview flex flex-1 items-center justify-center overflow-hidden rounded-lg">
                            <img
                                src={previewUrl}
                                alt="Uploaded fingerprint preview"
                                className="max-h-full max-w-full object-contain shadow-inner"
                            />
                        </div>

                        <div className="flex min-w-0 items-center justify-between gap-3 px-1 pt-3 text-left">
                            <div className="flex min-w-0 items-center gap-2.5 overflow-hidden">
                                <div className="shrink-0 rounded-lg border border-[var(--app-success-border)] bg-[var(--app-success-surface)] p-1.5 text-[var(--app-success-text)]">
                                    <ImageIcon className="h-4 w-4" />
                                </div>
                                <div className="min-w-0 overflow-hidden">
                                    <p className="safe-truncate text-sm font-medium text-[var(--app-text-soft)]" title={selectedFile.name}>
                                        {selectedFile.name}
                                    </p>
                                    <p className="text-xs text-[var(--app-text-muted)]">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                                </div>
                            </div>
                            <span className="status-pill status-pill--success shrink-0">
                                Ready
                            </span>
                        </div>
                    </div>
                ) : (
                    <div className="pointer-events-none flex flex-col items-center p-10 py-16">
                        <div
                            className={`file-drop-zone__icon mb-5 transition-all ${isDragging ? "scale-110 text-[var(--app-brand-text)]" : "shadow-inner"}`}
                        >
                            {isDragging ? <UploadCloud className="h-10 w-10" /> : <FileImage className="h-10 w-10" />}
                        </div>
                        <h3 className={`mb-1.5 text-xl font-semibold ${isDragging ? "text-[var(--app-brand-text)]" : "text-[var(--app-text-soft)]"}`}>
                            {title}
                        </h3>
                        <p className="max-w-xs text-sm text-[var(--app-text-muted)]">{description}</p>
                    </div>
                )}
            </div>

            {activeError ? <p className="safe-text text-sm text-[var(--app-error-text)]">{activeError}</p> : null}
        </div>
    );
}
