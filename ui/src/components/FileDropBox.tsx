import { useState, useMemo, useEffect, type DragEvent, type ChangeEvent } from "react";
import styles from "../App.module.css";

interface FileDropBoxProps {
    label: string;
    file: File | null;
    setFile: (f: File | null) => void;
    accept?: string;
}

export function FileDropBox({
                                label,
                                file,
                                setFile,
                                accept = "image/*"
                            }: FileDropBoxProps) {
    const [isDragging, setIsDragging] = useState(false);

    const previewUrl = useMemo(() => {
        if (!file) return null;
        return URL.createObjectURL(file);
    }, [file]);

    useEffect(() => {
        return () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
        };
    }, [previewUrl]);

    // Fix: Changed HTMLDivElement to HTMLLabelElement to match the <label> tag usage
    function onDragOver(e: DragEvent<HTMLLabelElement>) {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    }

    function onDragLeave(e: DragEvent<HTMLLabelElement>) {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    }

    function onDrop(e: DragEvent<HTMLLabelElement>) {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        const f = e.dataTransfer.files?.[0] ?? null;
        if (f) setFile(f);
    }

    function onPick(e: ChangeEvent<HTMLInputElement>) {
        setFile(e.target.files?.[0] ?? null);
    }

    return (
        <div className={styles.dropBox}>
            <div className={styles.label} style={{ marginBottom: '12px' }}>{label}</div>

            <label
                className={`${styles.dropZone} ${isDragging ? styles.dragging : ''}`}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
            >
                {previewUrl ? (
                    <>
                        <img src={previewUrl} alt="preview" className={styles.previewImage} />
                        <button
                            type="button"
                            className={styles.clearBtn}
                            onClick={(e) => {
                                e.preventDefault();
                                setFile(null);
                            }}
                        >
                            Remove Image
                        </button>
                    </>
                ) : (
                    <>
                        <svg className={styles.uploadIcon} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        <span className={styles.fileLabel}>Click or Drag to Upload</span>
                        <span className={styles.fileHint}>Supports JPG, PNG</span>
                        <input type="file" accept={accept} onChange={onPick} style={{ display: "none" }} />
                    </>
                )}
            </label>

            <div style={{ marginTop: 10, fontSize: 12, color: "var(--muted, #6b7280)", textAlign: 'center', height: '1.2em' }}>
                {file ? file.name : ""}
            </div>
        </div>
    );
}