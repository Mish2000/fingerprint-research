import type { LucideIcon } from "lucide-react";
import {
    Bug,
    CheckCircle2,
    FileText,
    Keyboard,
    MonitorSmartphone,
    ShieldAlert,
} from "lucide-react";
import InlineBanner from "../../shared/ui/InlineBanner.tsx";
import SurfaceCard from "../../shared/ui/SurfaceCard.tsx";

type ChecklistItem = {
    id: string;
    title: string;
    description: string;
    expected: string;
    evidence: string[];
};

type ChecklistSectionProps = {
    title: string;
    description: string;
    icon: LucideIcon;
    items: ChecklistItem[];
};

const smokeChecklist: ChecklistItem[] = [
    {
        id: "verify-happy-path",
        title: "Verify happy path",
        description: "Upload two images, switch methods, toggle warm-up and overlay, then run verification.",
        expected: "Summary renders, overlay fallback is clear when unsupported, and retry remains available after failures.",
        evidence: ["POST /api/match", "Demo cases → Run case", "Warm-up toggle", "ResultSummary + MatchCanvas"],
    },
    {
        id: "identify-happy-path",
        title: "Identification happy path",
        description: "Refresh stats, enroll one identity, search for it, then delete it.",
        expected: "Stats refresh after mutations, top candidate / candidates / hints / latency render, and delete confirmation is enforced.",
        evidence: [
            "artifacts/logs/07_stats_before.json",
            "artifacts/logs/08_enroll_easy_a.json",
            "artifacts/logs/12_search_positive_dl_sift.json",
            "artifacts/logs/24_stats_final.json",
        ],
    },
    {
        id: "benchmark-happy-path",
        title: "Benchmark happy path",
        description: "Select a run, inspect summary, filter by split, inspect comparison, and confirm best-method highlights.",
        expected: "URL query state stays shareable and the selected dataset drives comparison + best views consistently.",
        evidence: [
            "reports/benchmark/april_comparison/benchmark_comparison.md",
            "reports/benchmark/april_comparison/best_methods.json",
            "reports/benchmark/full_nist_sd300b_h6/validation.ok",
        ],
    },
];

const failureChecklist: ChecklistItem[] = [
    {
        id: "service-init-failed",
        title: "Service initialization failure",
        description: "Backend service ctor / startup issues should surface as actionable errors, not silent blank states.",
        expected: "The UI shows a warning hint plus the original backend message so QA can triage startup failures quickly.",
        evidence: [
            "artifacts/logs/01_torch_import.txt",
            "artifacts/logs/04_matchservice_ctor_rerun.txt",
            "artifacts/logs/triage_startup_terminal.txt",
        ],
    },
    {
        id: "demo-asset-404",
        title: "404 demo asset",
        description: "One curated case image is missing or unavailable.",
        expected: "The user gets a specific demo-asset message instead of a generic unknown fetch failure.",
        evidence: [
            "apps/api/demo_store.py",
            "GET /api/demo/cases/{case_id}/{slot}",
            "Demo case asset loader in the verify flow",
        ],
    },
    {
        id: "shortlist-zero",
        title: "Shortlist zero",
        description: "Identification returns an empty shortlist / candidate list.",
        expected: "The UI calls out that no candidates were returned and keeps the observability blocks readable.",
        evidence: ["artifacts/logs/16_search_shortlist_zero.json"],
    },
    {
        id: "invalid-retrieval-method",
        title: "Invalid retrieval method",
        description: "Backend rejects a search request because retrieval_method is invalid.",
        expected: "The UI surfaces an error hint that points QA back to the supported dl/vit selector.",
        evidence: ["artifacts/logs/17_search_invalid_retrieval.txt"],
    },
    {
        id: "duplicate-enroll",
        title: "Duplicate enrollment",
        description: "Enroll the same identity once with replace_existing=false and once with replace_existing=true.",
        expected: "The first path fails loudly, while the replace path succeeds and remains understandable in the UI.",
        evidence: [
            "artifacts/logs/18_duplicate_enroll_false.txt",
            "artifacts/logs/19_duplicate_enroll_true.json",
            "artifacts/logs/20_search_after_replace.json",
        ],
    },
    {
        id: "delete-not-found",
        title: "Delete not found",
        description: "Attempt to delete a random_id that is already gone or never existed.",
        expected: "The UI shows a non-destructive ‘not found’ state and still refreshes stats safely.",
        evidence: [
            "artifacts/logs/21_delete_hard.json",
            "artifacts/logs/22_delete_ood.json",
            "artifacts/logs/23_delete_replaced_easy.json",
        ],
    },
];

const accessibilityChecklist: ChecklistItem[] = [
    {
        id: "keyboard-flow",
        title: "Keyboard flow",
        description: "Tab through every major card and submit flows with keyboard only.",
        expected: "Submit buttons, retry buttons, select inputs, and destructive confirmations remain reachable and predictable.",
        evidence: ["Verify form submit", "Enroll form submit", "Search form submit", "Delete form submit"],
    },
    {
        id: "announcements",
        title: "Screen-reader announcements",
        description: "Loading / error / status banners should announce themselves appropriately.",
        expected: "Inline banners and request-state cards use alert/status semantics with live-region behavior.",
        evidence: ["InlineBanner", "RequestStateCard"],
    },
    {
        id: "responsive-shell",
        title: "Responsive layout",
        description: "Check the shell and primary workspaces at narrow widths.",
        expected: "Sidebar becomes a horizontal selector on small screens and tables remain scrollable instead of breaking layout.",
        evidence: ["App shell", "Identification candidate table", "Benchmark tables"],
    },
];

const benchmarkRunFamilies = [
    "full_nist_sd300b_h6",
    "full_nist_sd300c_h6",
    "full_polyu_cross_h5",
    "smoke_nist_sd300b_h6",
    "smoke_nist_sd300c_h6",
    "smoke_polyu_cross_h5",
];

function ChecklistSection({ title, description, icon: Icon, items }: ChecklistSectionProps) {
    return (
        <SurfaceCard
            title={title}
            description={description}
            actions={
                <div className="rounded-full border border-slate-200 bg-slate-50 p-2 text-slate-600">
                    <Icon className="h-4 w-4" />
                </div>
            }
        >
            <div className="grid gap-4 xl:grid-cols-2">
                {items.map((item) => (
                    <article key={item.id} className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                        <h4 className="text-base font-semibold text-slate-800">{item.title}</h4>
                        <p className="mt-2 text-sm leading-6 text-slate-600">{item.description}</p>

                        <div className="mt-4 rounded-xl border border-emerald-100 bg-emerald-50 px-4 py-3">
                            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700">Expected</p>
                            <p className="mt-1 text-sm leading-6 text-emerald-900">{item.expected}</p>
                        </div>

                        <div className="mt-4">
                            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Evidence sources</p>
                            <ul className="mt-2 space-y-2 text-sm text-slate-700">
                                {item.evidence.map((entry) => (
                                    <li key={entry} className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                                        <code>{entry}</code>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </article>
                ))}
            </div>
        </SurfaceCard>
    );
}

export default function ReleaseReadinessWorkspace() {
    return (
        <div className="space-y-6">
            <InlineBanner variant="info" title="Release-readiness scope">
                This workspace is intentionally manual-first: use it to drive smoke checks, regression review, accessibility passes,
                and failure-state verification against the real backend instead of reintroducing mocks.
            </InlineBanner>

            <InlineBanner variant="warning" title="Grounding note">
                The checklist below is grounded in the artifact and log names that are available in the current project tree.
                I do not have the full contents of every log file here, so expected outcomes are written conservatively.
            </InlineBanner>

            <ChecklistSection
                title="Manual smoke checklist"
                description="Run these before every demo, integration handoff, or release candidate build."
                icon={CheckCircle2}
                items={smokeChecklist}
            />

            <ChecklistSection
                title="Failure-state checklist"
                description="These are the negative paths called out explicitly by the release verification plan."
                icon={Bug}
                items={failureChecklist}
            />

            <ChecklistSection
                title="Accessibility and responsive checklist"
                description="Baseline release-readiness checks that should pass even before any formal audit."
                icon={Keyboard}
                items={accessibilityChecklist}
            />

            <SurfaceCard
                title="Benchmark regression sources"
                description="Use the same benchmark families consistently when validating summary / comparison / best-method views."
                actions={
                    <div className="rounded-full border border-slate-200 bg-slate-50 p-2 text-slate-600">
                        <FileText className="h-4 w-4" />
                    </div>
                }
            >
                <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                        <h4 className="text-base font-semibold text-slate-800">Suggested run families</h4>
                        <ul className="mt-3 space-y-2 text-sm text-slate-700">
                            {benchmarkRunFamilies.map((run) => (
                                <li key={run} className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                                    <code>{run}</code>
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                        <h4 className="text-base font-semibold text-slate-800">Regression anchors</h4>
                        <ul className="mt-3 space-y-2 text-sm text-slate-700">
                            <li className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                                <code>reports/benchmark/april_comparison/benchmark_comparison.md</code>
                            </li>
                            <li className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                                <code>reports/benchmark/april_comparison/best_methods.json</code>
                            </li>
                            <li className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                                <code>reports/benchmark/*/validation.ok</code>
                            </li>
                        </ul>
                    </div>
                </div>
            </SurfaceCard>

            <SurfaceCard
                title="Release operator checklist"
                description="Recommended order for manual release verification."
                actions={
                    <div className="rounded-full border border-slate-200 bg-slate-50 p-2 text-slate-600">
                        <MonitorSmartphone className="h-4 w-4" />
                    </div>
                }
            >
                <ol className="space-y-3 text-sm leading-6 text-slate-700">
                    <li className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                        1. Run <code>npm run build</code> and <code>npm run lint</code>.
                    </li>
                    <li className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                        2. Start the UI, open <code>?tab=release</code>, and walk through the smoke checklist top-to-bottom.
                    </li>
                    <li className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                        3. Trigger the failure paths one at a time and paste the exact UI message / terminal output back here.
                    </li>
                    <li className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                        4. Resize to mobile width and verify that the shell, candidate table, and benchmark tables remain usable.
                    </li>
                </ol>

                <div className="mt-5 rounded-2xl border border-amber-100 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                    <div className="flex items-start gap-3">
                        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0" />
                        <p>
                            Keep the negative-path checks manual for now. This avoids hiding backend regressions
                            behind a large one-shot automation script.
                        </p>
                    </div>
                </div>
            </SurfaceCard>
        </div>
    );
}
