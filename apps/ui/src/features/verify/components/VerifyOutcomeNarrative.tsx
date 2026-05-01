interface VerifyOutcomeNarrativeProps {
    methodStory: string;
}

export default function VerifyOutcomeNarrative({ methodStory }: VerifyOutcomeNarrativeProps) {
    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Method story</p>
            <p className="mt-2 text-sm leading-6 text-slate-600">{methodStory}</p>
        </section>
    );
}
