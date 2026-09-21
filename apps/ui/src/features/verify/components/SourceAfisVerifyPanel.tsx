import { useState } from "react";
import { readJsonOrThrow } from "../../../api/http.ts";

type EngineResult = { provider: string; score: number; provider_version: string; latency_ms: number | null };

export default function SourceAfisVerifyPanel() {
    const [left, setLeft] = useState<File | null>(null);
    const [right, setRight] = useState<File | null>(null);
    const [dpiA, setDpiA] = useState("");
    const [dpiB, setDpiB] = useState("");
    const [synthetic, setSynthetic] = useState(false);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<EngineResult | null>(null);

    async function loadSynthetic() {
        setError(null);
        try {
            const response = await fetch("/api/demo/synthetic/0.png");
            if (!response.ok) throw new Error("Synthetic fixture is unavailable");
            const blob = await response.blob();
            setLeft(new File([blob], "synthetic-a.png", { type: "image/png" }));
            setRight(new File([blob], "synthetic-b.png", { type: "image/png" }));
            setDpiA("500"); setDpiB("500"); setSynthetic(true); setResult(null);
        } catch (failure) { setError(String(failure)); }
    }

    async function run() {
        if (!left || !right) return;
        setBusy(true); setResult(null); setError(null);
        const form = new FormData();
        form.append("img_a", left); form.append("img_b", right);
        form.append("dpi_a", dpiA); form.append("dpi_b", dpiB);
        form.append("provider", "sourceafis_open");
        try {
            const response = await fetch("/api/fingerprint-engine/verify", { method: "POST", body: form });
            const data = await readJsonOrThrow(response, payload => payload as EngineResult);
            if (data.provider !== "sourceafis_open" || !Number.isFinite(data.score)) throw new Error("Invalid engine result");
            setResult(data);
        } catch (failure) { setError(String(failure)); }
        finally { setBusy(false); }
    }

    return <section className="rounded-2xl border border-slate-200 bg-white p-5 space-y-3" aria-label="SourceAFIS verification">
        <h2 className="text-lg font-semibold">SourceAFIS · 1:1 verification</h2>
        <p>External Java engine on CPU. Enter the source resolution for each image.</p>
        <button className="rounded border px-3 py-2" disabled={busy} onClick={() => void loadSynthetic()}>Use synthetic SourceAFIS pair</button>
        <div className="grid gap-3 md:grid-cols-2">
            <label>SourceAFIS probe <input aria-label="SourceAFIS probe" type="file" accept="image/*" disabled={busy} onChange={event => { setLeft(event.target.files?.[0] ?? null); setDpiA(""); setSynthetic(false); setResult(null); }} /></label>
            <label>SourceAFIS reference <input aria-label="SourceAFIS reference" type="file" accept="image/*" disabled={busy} onChange={event => { setRight(event.target.files?.[0] ?? null); setDpiB(""); setSynthetic(false); setResult(null); }} /></label>
            <label>Probe DPI <input className="rounded border p-2" aria-label="Probe DPI" type="number" min="20" max="20000" value={dpiA} onChange={event => setDpiA(event.target.value)} /></label>
            <label>Reference DPI <input className="rounded border p-2" aria-label="Reference DPI" type="number" min="20" max="20000" value={dpiB} onChange={event => setDpiB(event.target.value)} /></label>
        </div>
        {left && right && <p>{left.name} · {right.name}</p>}
        {synthetic && <p>Synthetic software fixture. 500 DPI is a test input; no human fingerprint or accuracy claim.</p>}
        <button className="rounded bg-slate-900 px-4 py-2 text-white" disabled={busy || !left || !right || !dpiA || !dpiB} onClick={() => void run()}>{busy ? "Comparing…" : "Compare with SourceAFIS"}</button>
        {error && <p role="alert">{error}</p>}
        {result && <div role="status"><strong>Raw score: {result.score.toFixed(4)}</strong><p>SourceAFIS {result.provider_version} · CPU. Raw similarity, not a probability. No acceptance threshold applied.</p></div>}
    </section>;
}
