import {
    Activity,
    Database,
    Fingerprint,
    RefreshCcw,
    Search,
    ShieldAlert,
    ShieldCheck,
    Trash2,
    UserPlus,
} from "lucide-react";
import FileDropBox from "../../components/FileDropBox.tsx";
import RequestState from "../../components/RequestState.tsx";
import SurfaceCard from "../../shared/ui/SurfaceCard.tsx";
import StatCard from "../../shared/ui/StatCard.tsx";
import InlineBanner from "../../shared/ui/InlineBanner.tsx";
import FormField from "../../shared/ui/FormField.tsx";
import { CHECKBOX_CLASS_NAME, INPUT_CLASS_NAME } from "../../shared/ui/inputClasses.ts";
import { useIdentification } from "./hooks/useIdentification.ts";

function formatLatency(value: number): string {
    return `${value.toFixed(1)} ms`;
}

function includesAnyToken(message: string | null | undefined, tokens: string[]): boolean {
    const normalized = (message ?? "").toLowerCase();
    return tokens.some((token) => normalized.includes(token));
}

function isServiceInitializationError(message: string | null | undefined): boolean {
    return includesAnyToken(message, ["startup", "init", "ctor", "constructor", "not initialized"]);
}

function isDuplicateEnrollmentError(message: string | null | undefined): boolean {
    return includesAnyToken(message, ["duplicate", "already exists", "existing identity", "replace_existing"]);
}

function isInvalidRetrievalError(message: string | null | undefined): boolean {
    return includesAnyToken(message, ["retrieval", "invalid", "unsupported"]);
}

export default function IdentificationWorkspace() {
    const identification = useIdentification();
    const topCandidate = identification.searchState.data?.top_candidate ?? null;
    const candidates = identification.searchState.data?.candidates ?? [];
    const latencyEntries = Object.entries(identification.searchState.data?.latency_ms ?? {});
    const hintEntries = Object.entries(identification.searchState.data?.hints_applied ?? {});
    const isEnrollLoading = identification.enrollState.status === "loading";
    const isSearchLoading = identification.searchState.status === "loading";
    const isDeleteLoading = identification.deleteState.status === "loading";
    const isStatsLoading = identification.statsState.status === "loading";

    const showServiceInitHint =
        isServiceInitializationError(identification.statsState.error)
        || isServiceInitializationError(identification.enrollState.error)
        || isServiceInitializationError(identification.searchState.error)
        || isServiceInitializationError(identification.deleteState.error);

    const showDuplicateEnrollHint = isDuplicateEnrollmentError(identification.enrollState.error);
    const showInvalidRetrievalHint = isInvalidRetrievalError(identification.searchState.error);
    const showShortlistZeroHint =
        identification.searchState.status === "success"
        && identification.searchState.data !== null
        && identification.searchState.data.shortlist_size === 0;

    return (
        <div className="space-y-6">
            <InlineBanner variant="info" title="Identification workspace">
                Identification keeps the operational structure while adding clearer negative-path messaging, keyboard submit flow,
                and stronger empty-state behavior for shortlist-zero and no-candidate responses.
            </InlineBanner>

            {identification.notice ? <InlineBanner variant="success">{identification.notice}</InlineBanner> : null}

            {showServiceInitHint ? (
                <InlineBanner variant="warning" title="Backend initialization issue detected">
                    One of the identification endpoints appears to have failed during startup or lazy initialization.
                    Keep the original backend error visible and use this as a release-readiness stop signal.
                </InlineBanner>
            ) : null}

            {showDuplicateEnrollHint ? (
                <InlineBanner variant="warning" title="Duplicate enrollment rejected">
                    The backend rejected a duplicate identity. Verify this once with
                    <code> replace_existing=false </code> and once with <code> replace_existing=true </code>.
                </InlineBanner>
            ) : null}

            {showInvalidRetrievalHint ? (
                <InlineBanner variant="warning" title="Invalid retrieval method rejected">
                    The backend rejected the current retrieval method. The production-safe UI path should stay within the supported
                    <code> dl </code> / <code> vit </code> selector values.
                </InlineBanner>
            ) : null}

            {showShortlistZeroHint ? (
                <InlineBanner variant="warning" title="Shortlist is empty">
                    The request completed successfully, but the backend returned zero shortlisted candidates.
                    This is a valid negative-path regression case and should remain readable in the UI.
                </InlineBanner>
            ) : null}

            <SurfaceCard
                title="Store statistics"
                description="Operational totals and storage layout are loaded once and refreshed after destructive or mutating actions."
                actions={
                    <button
                        type="button"
                        onClick={() => {
                            void identification.refreshStats();
                        }}
                        disabled={isStatsLoading}
                        className="inline-flex items-center rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        <RefreshCcw className="mr-2 h-4 w-4" />
                        Refresh stats
                    </button>
                }
            >
                {identification.statsState.status === "loading" && !identification.statsState.data ? (
                    <RequestState
                        variant="loading"
                        title="Loading identification stats"
                        description="Reading the current store layout and total enrollment count."
                    />
                ) : null}

                {identification.statsState.status === "error" && identification.statsState.error ? (
                    <RequestState
                        variant="error"
                        title="Failed to load identification stats"
                        description={identification.statsState.error}
                        actionLabel="Retry"
                        onAction={() => {
                            void identification.refreshStats();
                        }}
                    />
                ) : null}

                {identification.statsState.data ? (
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                        <StatCard
                            icon={Database}
                            label="Total enrolled"
                            value={String(identification.statsState.data.total_enrolled)}
                            tone="brand"
                        />
                        {Object.entries(identification.statsState.data.storage_layout).map(([key, value]) => (
                            <StatCard key={key} icon={Activity} label={key} value={value} tone="slate" />
                        ))}
                    </div>
                ) : null}
            </SurfaceCard>

            <div className="grid gap-6 xl:grid-cols-2">
                <SurfaceCard title="Enroll identity" description="Upload a fingerprint, attach metadata, and control replacement semantics.">
                    <form
                        className="space-y-4"
                        onSubmit={(event) => {
                            event.preventDefault();
                            void identification.submitEnroll();
                        }}
                        aria-busy={isEnrollLoading}
                    >
                        <FileDropBox
                            file={identification.enrollForm.file}
                            onChange={(file) => {
                                identification.updateEnrollForm({ file });
                            }}
                            disabled={isEnrollLoading}
                            title="Enrollment image"
                            description="Choose the fingerprint image that will be stored for this person."
                        />

                        <div className="grid gap-4 md:grid-cols-2">
                            <FormField label="Full name">
                                <input
                                    className={INPUT_CLASS_NAME}
                                    value={identification.enrollForm.fullName}
                                    disabled={isEnrollLoading}
                                    onChange={(event) => {
                                        identification.updateEnrollForm({ fullName: event.target.value });
                                    }}
                                />
                            </FormField>

                            <FormField label="National ID">
                                <input
                                    className={INPUT_CLASS_NAME}
                                    value={identification.enrollForm.nationalId}
                                    disabled={isEnrollLoading}
                                    onChange={(event) => {
                                        identification.updateEnrollForm({ nationalId: event.target.value });
                                    }}
                                />
                            </FormField>

                            <FormField label="Capture">
                                <select
                                    className={INPUT_CLASS_NAME}
                                    value={identification.enrollForm.capture}
                                    disabled={isEnrollLoading}
                                    onChange={(event) => {
                                        identification.updateEnrollForm({ capture: event.target.value as typeof identification.enrollForm.capture });
                                    }}
                                >
                                    <option value="plain">Plain</option>
                                    <option value="roll">Roll</option>
                                    <option value="contactless">Contactless</option>
                                    <option value="contact_based">Contact-based</option>
                                </select>
                            </FormField>
                        </div>

                        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                            <p className="text-sm font-medium text-slate-700">Vector methods</p>

                            <div className="mt-3 flex flex-wrap gap-4">
                                <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                    <input
                                        type="checkbox"
                                        className={CHECKBOX_CLASS_NAME}
                                        checked={identification.enrollForm.includeDl}
                                        disabled={isEnrollLoading}
                                        onChange={(event) => {
                                            identification.updateEnrollForm({ includeDl: event.target.checked });
                                        }}
                                    />
                                    DL
                                </label>

                                <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                    <input
                                        type="checkbox"
                                        className={CHECKBOX_CLASS_NAME}
                                        checked={identification.enrollForm.includeVit}
                                        disabled={isEnrollLoading}
                                        onChange={(event) => {
                                            identification.updateEnrollForm({ includeVit: event.target.checked });
                                        }}
                                    />
                                    ViT
                                </label>

                                <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                                    <input
                                        type="checkbox"
                                        className={CHECKBOX_CLASS_NAME}
                                        checked={identification.enrollForm.replaceExisting}
                                        disabled={isEnrollLoading}
                                        onChange={(event) => {
                                            identification.updateEnrollForm({ replaceExisting: event.target.checked });
                                        }}
                                    />
                                    Replace existing
                                </label>
                            </div>
                        </div>

                        {identification.enrollState.status === "error" && identification.enrollState.error ? (
                            <InlineBanner variant="error">{identification.enrollState.error}</InlineBanner>
                        ) : null}

                        {identification.enrollState.status === "success" && identification.enrollState.data ? (
                            <InlineBanner variant="success" title="Enrollment completed">
                                Created <code>{identification.enrollState.data.random_id}</code> with vector methods
                                {` ${identification.enrollState.data.vector_methods.join(", ")}.`}
                            </InlineBanner>
                        ) : null}

                        <button
                            type="submit"
                            disabled={isEnrollLoading}
                            className="inline-flex items-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            <UserPlus className="mr-2 h-4 w-4" />
                            {isEnrollLoading ? "Enrolling..." : "Enroll identity"}
                        </button>
                    </form>
                </SurfaceCard>

                <SurfaceCard title="Delete identity" description="Destructive actions require an explicit confirmation before the request is sent.">
                    <form
                        className="space-y-4"
                        onSubmit={(event) => {
                            event.preventDefault();
                            void identification.submitDelete();
                        }}
                        aria-busy={isDeleteLoading}
                    >
                        <FormField label="Random ID" hint="Use the identifier returned by enrollment or surfaced by search results.">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.deleteForm.randomId}
                                disabled={isDeleteLoading}
                                onChange={(event) => {
                                    identification.updateDeleteForm({ randomId: event.target.value });
                                }}
                            />
                        </FormField>

                        <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                            <input
                                type="checkbox"
                                className={CHECKBOX_CLASS_NAME}
                                checked={identification.deleteForm.confirmChecked}
                                disabled={isDeleteLoading}
                                onChange={(event) => {
                                    identification.updateDeleteForm({ confirmChecked: event.target.checked });
                                }}
                            />
                            I understand this will permanently purge the selected identity.
                        </label>

                        {identification.deleteState.status === "error" && identification.deleteState.error ? (
                            <InlineBanner variant="error">{identification.deleteState.error}</InlineBanner>
                        ) : null}

                        {identification.deleteState.status === "success" && identification.deleteState.data ? (
                            <InlineBanner
                                variant={identification.deleteState.data.removed ? "success" : "warning"}
                                title={identification.deleteState.data.removed ? "Identity removed" : "Identity not found"}
                            >
                                <code>{identification.deleteState.data.random_id}</code>
                            </InlineBanner>
                        ) : null}

                        <button
                            type="submit"
                            disabled={isDeleteLoading}
                            className="inline-flex items-center rounded-xl bg-red-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            <Trash2 className="mr-2 h-4 w-4" />
                            {isDeleteLoading ? "Deleting..." : "Delete identity"}
                        </button>
                    </form>
                </SurfaceCard>
            </div>

            <SurfaceCard
                title="Search identity"
                description="Run retrieval + re-rank over the enrolled store and inspect decision traces, hints, and latency breakdown."
            >
                <form
                    className="space-y-6"
                    onSubmit={(event) => {
                        event.preventDefault();
                        void identification.submitSearch();
                    }}
                    aria-busy={isSearchLoading}
                >
                    <FileDropBox
                        file={identification.searchForm.file}
                        onChange={(file) => {
                            identification.updateSearchForm({ file });
                        }}
                        disabled={isSearchLoading}
                        title="Probe image"
                        description="Upload the fingerprint that should be identified against the store."
                    />

                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                        <FormField label="Capture">
                            <select
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.capture}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ capture: event.target.value as typeof identification.searchForm.capture });
                                }}
                            >
                                <option value="plain">Plain</option>
                                <option value="roll">Roll</option>
                                <option value="contactless">Contactless</option>
                                <option value="contact_based">Contact-based</option>
                            </select>
                        </FormField>

                        <FormField label="Retrieval method">
                            <select
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.retrievalMethod}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ retrievalMethod: event.target.value as typeof identification.searchForm.retrievalMethod });
                                }}
                            >
                                <option value="dl">DL</option>
                                <option value="vit">ViT</option>
                            </select>
                        </FormField>

                        <FormField label="Re-rank method">
                            <select
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.rerankMethod}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ rerankMethod: event.target.value as typeof identification.searchForm.rerankMethod });
                                }}
                            >
                                <option value="classic">classic</option>
                                <option value="harris">harris</option>
                                <option value="sift">sift</option>
                                <option value="dl">dl</option>
                                <option value="vit">vit</option>
                                <option value="dedicated">dedicated</option>
                            </select>
                        </FormField>

                        <FormField label="Shortlist size">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.shortlistSizeText}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ shortlistSizeText: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="Threshold" hint="Leave empty to use the backend default.">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.thresholdText}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ thresholdText: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="Name pattern">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.namePattern}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ namePattern: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="National ID pattern">
                            <input
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.nationalIdPattern}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ nationalIdPattern: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="Created from">
                            <input
                                type="date"
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.createdFrom}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ createdFrom: event.target.value });
                                }}
                            />
                        </FormField>

                        <FormField label="Created to">
                            <input
                                type="date"
                                className={INPUT_CLASS_NAME}
                                value={identification.searchForm.createdTo}
                                disabled={isSearchLoading}
                                onChange={(event) => {
                                    identification.updateSearchForm({ createdTo: event.target.value });
                                }}
                            />
                        </FormField>
                    </div>

                    {identification.searchState.status === "error" && identification.searchState.error ? (
                        <InlineBanner variant="error">{identification.searchState.error}</InlineBanner>
                    ) : null}

                    <button
                        type="submit"
                        disabled={isSearchLoading}
                        className="inline-flex items-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        <Search className="mr-2 h-4 w-4" />
                        {isSearchLoading ? "Searching..." : "Search identities"}
                    </button>
                </form>

                {identification.searchState.status === "loading" ? (
                    <RequestState
                        variant="loading"
                        title="Running identification"
                        description="Sending the probe image to the backend and waiting for retrieval / rerank results."
                    />
                ) : null}

                {identification.searchState.status === "success" && identification.searchState.data ? (
                    <div className="mt-6 space-y-6">
                        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                            <StatCard
                                icon={Fingerprint}
                                label="Decision"
                                value={identification.searchState.data.decision ? "MATCH" : "NO MATCH"}
                                tone={identification.searchState.data.decision ? "emerald" : "amber"}
                            />
                            <StatCard
                                icon={Database}
                                label="Candidate pool"
                                value={String(identification.searchState.data.candidate_pool_size)}
                                tone="slate"
                            />
                            <StatCard
                                icon={Search}
                                label="Shortlist size"
                                value={String(identification.searchState.data.shortlist_size)}
                                tone="brand"
                            />
                            <StatCard
                                icon={Activity}
                                label="Threshold"
                                value={identification.searchState.data.threshold.toFixed(2)}
                                tone="slate"
                            />
                        </div>

                        {topCandidate ? (
                            <InlineBanner
                                variant={identification.searchState.data.decision ? "success" : "warning"}
                                title={identification.searchState.data.decision ? "Top candidate accepted" : "Top candidate rejected"}
                            >
                                <span className="font-medium">{topCandidate.full_name}</span>
                                {` · ${topCandidate.random_id} · ${topCandidate.national_id_masked} · ${topCandidate.capture}`}
                            </InlineBanner>
                        ) : (
                            <RequestState
                                variant="empty"
                                title="No top candidate"
                                description="The backend returned a valid IdentifyResponse without a best candidate in scope."
                            />
                        )}

                        <div className="grid gap-6 xl:grid-cols-2">
                            <SurfaceCard title="Hints applied" description="Backend-side filtering hints that affected candidate retrieval.">
                                {hintEntries.length > 0 ? (
                                    <dl className="space-y-2 text-sm">
                                        {hintEntries.map(([key, value]) => (
                                            <div key={key} className="flex items-start justify-between gap-4 rounded-xl bg-slate-50 px-4 py-3">
                                                <dt className="font-medium text-slate-600">{key}</dt>
                                                <dd className="text-right text-slate-800">{value}</dd>
                                            </div>
                                        ))}
                                    </dl>
                                ) : (
                                    <RequestState
                                        variant="empty"
                                        title="No hints were applied"
                                        description="This search ran without name, national ID, or date-range narrowing."
                                    />
                                )}
                            </SurfaceCard>

                            <SurfaceCard title="Latency breakdown" description="Observability fields normalized from the backend response.">
                                {latencyEntries.length > 0 ? (
                                    <dl className="space-y-2 text-sm">
                                        {latencyEntries.map(([key, value]) => (
                                            <div key={key} className="flex items-start justify-between gap-4 rounded-xl bg-slate-50 px-4 py-3">
                                                <dt className="font-medium text-slate-600">{key}</dt>
                                                <dd className="text-right text-slate-800">{formatLatency(value)}</dd>
                                            </div>
                                        ))}
                                    </dl>
                                ) : (
                                    <RequestState
                                        variant="empty"
                                        title="No latency details returned"
                                        description="The response did not include a latency breakdown map."
                                    />
                                )}
                            </SurfaceCard>
                        </div>

                        {candidates.length > 0 ? (
                            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                                <div className="border-b border-slate-100 px-6 py-5">
                                    <h3 className="text-lg font-semibold text-slate-800">Candidate ranking</h3>
                                    <p className="mt-1 text-sm text-slate-500">
                                        Retrieval score, re-rank score, decision, and masked national ID are rendered directly from the normalized IdentifyResponse.
                                    </p>
                                </div>

                                <div className="overflow-x-auto">
                                    <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                                        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                                        <tr>
                                            <th className="px-4 py-3">Rank</th>
                                            <th className="px-4 py-3">Person</th>
                                            <th className="px-4 py-3">Random ID</th>
                                            <th className="px-4 py-3">National ID</th>
                                            <th className="px-4 py-3">Capture</th>
                                            <th className="px-4 py-3 text-right">Retrieval</th>
                                            <th className="px-4 py-3 text-right">Re-rank</th>
                                            <th className="px-4 py-3 text-center">Decision</th>
                                        </tr>
                                        </thead>

                                        <tbody className="divide-y divide-slate-100 bg-white">
                                        {candidates.map((candidate) => (
                                            <tr key={`${candidate.random_id}_${candidate.rank}`} className="hover:bg-slate-50">
                                                <td className="px-4 py-3 font-medium text-slate-800">{candidate.rank}</td>
                                                <td className="px-4 py-3 text-slate-800">{candidate.full_name}</td>
                                                <td className="px-4 py-3 text-slate-600">{candidate.random_id}</td>
                                                <td className="px-4 py-3 text-slate-600">{candidate.national_id_masked}</td>
                                                <td className="px-4 py-3 text-slate-600">{candidate.capture}</td>
                                                <td className="px-4 py-3 text-right text-slate-800">{candidate.retrieval_score.toFixed(4)}</td>
                                                <td className="px-4 py-3 text-right text-slate-800">
                                                    {typeof candidate.rerank_score === "number" ? candidate.rerank_score.toFixed(4) : "-"}
                                                </td>
                                                <td className="px-4 py-3 text-center">
                                                    {candidate.decision === true ? (
                                                        <ShieldCheck className="mx-auto h-4 w-4 text-emerald-600" />
                                                    ) : candidate.decision === false ? (
                                                        <ShieldAlert className="mx-auto h-4 w-4 text-amber-600" />
                                                    ) : (
                                                        <span className="text-slate-400">-</span>
                                                    )}
                                                </td>
                                            </tr>
                                        ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ) : (
                            <RequestState
                                variant="empty"
                                title="No candidates returned"
                                description="The request completed, but the backend did not return any candidate rows for ranking."
                            />
                        )}
                    </div>
                ) : null}
            </SurfaceCard>
        </div>
    );
}
