interface VerifyOutcomeNarrativeProps {
    methodStory: string;
}

export default function VerifyOutcomeNarrative({ methodStory }: VerifyOutcomeNarrativeProps) {
    return (
        <section className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--app-text-muted)]">Method story</p>
            <p className="mt-2 text-sm leading-6 text-[var(--app-text-soft)]">{methodStory}</p>
        </section>
    );
}
