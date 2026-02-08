import type {MatchResponse} from "../types";
import styles from "../App.module.css";

interface ResultSummaryProps {
    resp: MatchResponse;
}

export function ResultSummary({ resp }: ResultSummaryProps) {
    const { method, score, decision, threshold, latency_ms, meta } = resp;

    // Helper to safely format numbers from meta
    const fmt = (val: unknown, digits = 2) => {
        if (typeof val === "number") return val.toFixed(digits);
        return "-";
    };

    // Helper to get nested DL config properties
    const getDlConfig = (key: string) => {
        const conf = meta.dl_config as Record<string, unknown> | undefined;
        return conf ? String(conf[key] ?? "-") : "-";
    };

    return (
        <div className={styles.summaryCard}>
            {/* Header / Main Verdict */}
            <div className={styles.summaryHeader}>
                <div className={styles.summaryMainMetric}>
                    <span className={styles.summaryLabel}>Score</span>
                    <span className={styles.summaryValue} style={{
                        color: decision ? 'var(--accent-success)' : 'var(--muted)'
                    }}>
                        {Math.min(Math.max(score, 0), 1).toFixed(4)}
                    </span>
                </div>

                <div className={styles.summaryMainMetric}>
                    <span className={styles.summaryLabel}>Decision</span>
                    <span className={`${styles.badge} ${decision ? styles.badgeSuccess : styles.badgeNeutral}`}>
                        {decision ? "MATCH" : "NO MATCH"}
                     </span>
                </div>

                <div className={styles.summaryMainMetric}>
                    <span className={styles.summaryLabel}>Latency</span>
                    <span className={styles.summaryValue}>{latency_ms.toFixed(0)}ms</span>
                </div>
            </div>

            <hr className={styles.summaryDivider} />

            {/* Method Specific Details */}
            <div className={styles.summaryGrid}>
                {/* Common Params */}
                <div className={styles.summaryItem}>
                    <span className={styles.summaryLabel}>Threshold</span>
                    <span className={styles.summaryValue}>{threshold}</span>
                </div>

                {/* Classic Specifics */}
                {method === "classic" && (
                    <>
                        <div className={styles.summaryItem}>
                            <span className={styles.summaryLabel}>Inliers / Matches</span>
                            <span className={styles.summaryValue}>{meta.inliers} / {meta.matches}</span>
                        </div>
                        <div className={styles.summaryItem}>
                            <span className={styles.summaryLabel}>Params (k1/k2)</span>
                            <span className={styles.summaryValue}>{meta.k1} / {meta.k2}</span>
                        </div>
                    </>
                )}

                {/* Dedicated Specifics */}
                {method === "dedicated" && (
                    <>
                        <div className={styles.summaryItem}>
                            <span className={styles.summaryLabel}>Matches (Tent / Inlier)</span>
                            <span className={styles.summaryValue}>{meta.tentative_count} / {meta.inliers_count}</span>
                        </div>
                        <div className={styles.summaryItem}>
                            <span className={styles.summaryLabel}>Mean Sim (Tent / Inlier)</span>
                            <span className={styles.summaryValue}>{fmt(meta.mean_tentative_sim)} / {fmt(meta.mean_inlier_sim)}</span>
                        </div>
                        <div className={styles.summaryItem}>
                            <span className={styles.summaryLabel}>Embeddings</span>
                            <span className={styles.summaryValue}>{meta.embed_a_total} / {meta.embed_b_total}</span>
                        </div>
                        <div className={styles.summaryItem}>
                            <span className={styles.summaryLabel}>Timing (Match/RANSAC)</span>
                            <span className={styles.summaryValue}>{fmt(meta.match_ms, 0)}ms / {fmt(meta.ransac_ms, 0)}ms</span>
                        </div>
                    </>
                )}

                {/* DL Specifics */}
                {method === "dl" && (
                    <>
                        <div className={styles.summaryItem}>
                            <span className={styles.summaryLabel}>Backbone</span>
                            <span className={styles.summaryValue}>{getDlConfig('backbone')}</span>
                        </div>
                        <div className={styles.summaryItem}>
                            <span className={styles.summaryLabel}>Input Size</span>
                            <span className={styles.summaryValue}>{getDlConfig('input_size')}</span>
                        </div>
                        <div className={styles.summaryItem}>
                            <span className={styles.summaryLabel}>Embed Latency (A/B)</span>
                            <span className={styles.summaryValue}>{fmt(meta.embed_ms_a, 0)}ms / {fmt(meta.embed_ms_b, 0)}ms</span>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}