import RequestState from "../../../components/RequestState.tsx";

export default function NoMatchNarrativeState() {
    return (
        <RequestState
            variant="empty"
            title="No shortlist candidates"
            description="The search completed without an accepted match or shortlist rows."
        />
    );
}
