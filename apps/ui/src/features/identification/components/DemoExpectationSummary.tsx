import type { DemoExpectationSummary as DemoExpectationSummaryState } from "../model.ts";
import IdentificationExpectationNarrative from "./IdentificationExpectationNarrative.tsx";

interface DemoExpectationSummaryProps {
    summary: DemoExpectationSummaryState;
}

export default function DemoExpectationSummary({ summary }: DemoExpectationSummaryProps) {
    return <IdentificationExpectationNarrative summary={summary} />;
}
