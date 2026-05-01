import type { ReactNode } from "react";

interface FormFieldProps {
    label: string;
    hint?: string;
    children: ReactNode;
}

export default function FormField({ label, hint, children }: FormFieldProps) {
    return (
        <label className="form-field">
            <span className="form-field__label">{label}</span>
            <span className="form-field__control">{children}</span>
            {hint ? <span className="form-field__hint">{hint}</span> : null}
        </label>
    );
}
