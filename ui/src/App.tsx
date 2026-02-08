import { useState, useMemo, useRef } from "react";
import { FileDropBox } from "./components/FileDropBox";
import { MatchCanvas } from "./components/MatchCanvas";
import { ResultSummary } from "./components/ResultSummary";
import { matchFingerprints, warmUp } from "./api/matchService";
import type {Capture, MatchResponse, Method, OverlayMatch} from "./types";
import styles from "./App.module.css";

export default function App() {
    // Configuration State
    const [method, setMethod] = useState<Method>("dedicated");
    const [captureA, setCaptureA] = useState<Capture>("plain");
    const [captureB, setCaptureB] = useState<Capture>("plain");
    const [returnOverlay, setReturnOverlay] = useState<boolean>(true);

    // Threshold Override State
    const [thresholdOverride, setThresholdOverride] = useState<string>("");

    // File State
    const [fileA, setFileA] = useState<File | null>(null);
    const [fileB, setFileB] = useState<File | null>(null);

    // Response State
    const [loading, setLoading] = useState(false);
    const [resp, setResp] = useState<MatchResponse | null>(null);
    const [err, setErr] = useState<string | null>(null);

    // Visualization Controls
    const [showOutliers, setShowOutliers] = useState(true);
    const [showTentative, setShowTentative] = useState(true);
    const [maxMatches, setMaxMatches] = useState(200);

    const topRef = useRef<HTMLDivElement | null>(null);

    const canSubmit = useMemo(() => !!fileA && !!fileB && !loading, [fileA, fileB, loading]);

    const overlayMatches: OverlayMatch[] = useMemo(() => {
        const ms = resp?.overlay?.matches;
        return Array.isArray(ms) ? ms : [];
    }, [resp]);

    const overlayAvailable = useMemo(() => {
        if (!resp) return false;
        if (resp.method === "dl") return false;
        return overlayMatches.length > 0;
    }, [resp, overlayMatches.length]);

    async function onMatch() {
        setErr(null);
        setResp(null);

        if (!fileA || !fileB) {
            setErr("Please select both source images.");
            return;
        }

        setLoading(true);
        try {
            const result = await matchFingerprints({
                method,
                fileA,
                fileB,
                captureA,
                captureB,
                returnOverlay,
                threshold: thresholdOverride // Pass the override
            });
            setResp(result);
        } catch (e) {
            if (e instanceof Error) setErr(e.message);
            else setErr(String(e));
        } finally {
            setLoading(false);
        }
    }

    // New: Warm Up Handler
    async function onWarmUp() {
        if (loading) return;
        setLoading(true);
        setErr(null);
        try {
            await warmUp(method);
            alert(`Warm-up complete for method: ${method}`);
        } catch (e) {
            console.error(e);
            setErr("Warm-up failed (check console).");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className={styles.container}>
            <div ref={topRef} />

            <header className={styles.header}>
                <h2>Fingerprint Matcher</h2>
                <p style={{ color: 'var(--muted)', marginTop: '0.5rem' }}>
                    Compare two fingerprint images using Deep Learning or Classic algorithms.
                </p>
            </header>

            {/* Upload Section */}
            <section className={styles.uploadGrid}>
                <FileDropBox label="Reference Image (A)" file={fileA} setFile={setFileA} />
                <FileDropBox label="Probe Image (B)" file={fileB} setFile={setFileB} />
            </section>

            {/* Main Controls Bar */}
            <section className={styles.controlsBar}>
                <div className={styles.controlGroup}>
                    <div className={styles.field}>
                        <label className={styles.label}>Method</label>
                        <select className={styles.select} value={method} onChange={(e) => setMethod(e.target.value as Method)}>
                            <option value="classic">Classic (Minutiae)</option>
                            <option value="dl">Deep Learning</option>
                            <option value="dedicated">Dedicated</option>
                        </select>
                    </div>

                    <div className={styles.field}>
                        <label className={styles.label}>Threshold</label>
                        <input
                            type="number"
                            step="0.01"
                            placeholder="Default"
                            value={thresholdOverride}
                            onChange={(e) => setThresholdOverride(e.target.value)}
                            className={styles.thresholdInput}
                        />
                    </div>

                    <div className={styles.field}>
                        <label className={styles.label}>Capture Type A</label>
                        <select className={styles.select} value={captureA} onChange={(e) => setCaptureA(e.target.value as Capture)}>
                            <option value="plain">Plain</option>
                            <option value="roll">Roll</option>
                        </select>
                    </div>

                    <div className={styles.field}>
                        <label className={styles.label}>Capture Type B</label>
                        <select className={styles.select} value={captureB} onChange={(e) => setCaptureB(e.target.value as Capture)}>
                            <option value="plain">Plain</option>
                            <option value="roll">Roll</option>
                        </select>
                    </div>

                    <div className={styles.field} style={{ justifyContent: 'flex-end' }}>
                        <label className={styles.checkboxLabel}>
                            <input
                                type="checkbox"
                                checked={returnOverlay}
                                onChange={(e) => setReturnOverlay(e.target.checked)}
                            />
                            <span>Generate Overlay</span>
                        </label>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <button
                        type="button"
                        onClick={onWarmUp}
                        disabled={loading}
                        className={styles.secondaryBtn}
                        title="Run a tiny dummy request to warm up the GPU/Model"
                    >
                        Warm Up
                    </button>

                    <button
                        type="button"
                        onClick={onMatch}
                        disabled={!canSubmit}
                        className={styles.matchBtn}
                    >
                        {loading ? "Processing..." : "Run Match"}
                    </button>
                </div>
            </section>

            {/* Visualization Settings */}
            {resp && overlayAvailable && (
                <section className={styles.controlsBar} style={{ padding: '1rem', background: 'var(--panel2)' }}>
                    <div className={styles.controlGroup} style={{ alignItems: 'center' }}>
                        <span className={styles.label} style={{ marginRight: '1rem' }}>Overlay Filters:</span>

                        <label className={styles.checkboxLabel}>
                            <input
                                type="checkbox"
                                checked={showOutliers}
                                onChange={(e) => setShowOutliers(e.target.checked)}
                            />
                            <span>Show Outliers</span>
                        </label>

                        <label className={styles.checkboxLabel}>
                            <input
                                type="checkbox"
                                checked={showTentative}
                                onChange={(e) => setShowTentative(e.target.checked)}
                            />
                            <span>Show Tentative</span>
                        </label>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginLeft: '1rem' }}>
                            <label className={styles.label}>Limit:</label>
                            <input
                                type="number"
                                min={0}
                                max={2000}
                                step={10}
                                value={maxMatches}
                                onChange={(e) => setMaxMatches(Number(e.target.value))}
                                className={styles.select}
                                style={{ width: '80px', paddingRight: '0.5rem' }}
                            />
                        </div>
                    </div>
                </section>
            )}

            {/* Output Section */}
            <section>
                {err && (
                    <div className={styles.errorBox}>
                        <strong>Error:</strong> {err}
                    </div>
                )}

                {resp && (
                    <div style={{ marginTop: '1rem' }}>
                        <ResultSummary resp={resp} />
                    </div>
                )}

                {resp && fileA && fileB && overlayAvailable && (
                    <MatchCanvas
                        fileA={fileA}
                        fileB={fileB}
                        matches={overlayMatches}
                        showOutliers={showOutliers}
                        showTentative={showTentative}
                        maxMatches={maxMatches}
                    />
                )}

                {resp && resp.method === "dl" && (
                    <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--muted)' }}>
                        Overlay is not available for Deep Learning method.
                    </div>
                )}

                {resp && (
                    <div className={styles.visPanel}>
                        <div className={styles.visHeader}>
                            <div className={styles.visTitle}>Raw Response Data</div>
                        </div>
                        <pre className={styles.jsonPre}>
                            {JSON.stringify(resp, null, 2)}
                        </pre>
                    </div>
                )}
            </section>

            <button
                type="button"
                className={styles.scrollTop}
                onClick={() => topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
                title="Back to Top"
            >
                ↑
            </button>
        </div>
    );
}