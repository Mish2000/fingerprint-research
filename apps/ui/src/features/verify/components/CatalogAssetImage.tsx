import { ImageOff } from "lucide-react";
import { useState } from "react";

interface CatalogAssetImageProps {
    src?: string | null;
    alt: string;
    className?: string;
    imageClassName?: string;
    fallbackLabel: string;
}

export default function CatalogAssetImage({
    src,
    alt,
    className = "",
    imageClassName = "",
    fallbackLabel,
}: CatalogAssetImageProps) {
    const [failedSrc, setFailedSrc] = useState<string | null>(null);
    const didFail = !src || failedSrc === src;

    return (
        <div className={`relative overflow-hidden rounded-2xl border border-slate-200 bg-slate-100 ${className}`.trim()}>
            {src && !didFail ? (
                <img
                    src={src}
                    alt={alt}
                    loading="lazy"
                    className={`h-full w-full object-contain ${imageClassName}`.trim()}
                    onError={() => {
                        setFailedSrc(src);
                    }}
                />
            ) : (
                <div className="flex h-full min-h-32 flex-col items-center justify-center gap-3 px-4 py-6 text-center text-slate-500">
                    <div className="rounded-full bg-white p-3 shadow-sm">
                        <ImageOff className="h-5 w-5" />
                    </div>
                    <div>
                        <p className="text-sm font-medium text-slate-700">Preview unavailable</p>
                        <p className="mt-1 text-xs leading-5">{fallbackLabel}</p>
                    </div>
                </div>
            )}
        </div>
    );
}
