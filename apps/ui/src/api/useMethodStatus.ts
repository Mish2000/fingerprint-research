import { useEffect, useState } from "react";

export type MethodStatus = { available: boolean; error: string | null; device: string | null };

export function useMethodStatus() {
    const [methods, setMethods] = useState<Record<string, MethodStatus>>({});
    const [error, setError] = useState<string | null>(null);
    useEffect(() => {
        let active = true;
        void fetch("/api/methods").then(async response => {
            if (!response.ok) throw new Error("Method readiness is unavailable");
            const payload = await response.json() as { methods: Array<{ id: string; availability: MethodStatus }> };
            if (!Array.isArray(payload.methods)) throw new Error("Invalid method readiness response");
            if (active) setMethods(Object.fromEntries(payload.methods.map(method => [method.id, {
                available: method.availability?.available === true,
                error: method.availability?.error ?? null,
                device: method.availability?.device ?? null,
            }])));
        }).catch(failure => { if (active) setError(String(failure)); });
        return () => { active = false; };
    }, []);
    return { methods, error };
}
