import { formatMethodLabel } from "../../shared/storytelling.ts";
import { IDENTIFICATION_RETRIEVAL_METHOD_VALUES } from "../../types/index.ts";
import type { IdentificationRetrievalMethod, Method } from "../../types/index.ts";

export const IDENTIFICATION_RETRIEVAL_OPTIONS: Array<{
    value: IdentificationRetrievalMethod;
    label: string;
}> = IDENTIFICATION_RETRIEVAL_METHOD_VALUES.map((value) => ({
    value,
    label: formatMethodLabel(value),
}));

export const IDENTIFICATION_RERANK_OPTIONS: Array<{
    value: Method;
    label: string;
}> = [
    { value: "sift", label: formatMethodLabel("sift") },
    { value: "minutiae", label: formatMethodLabel("minutiae") },
    { value: "harris", label: formatMethodLabel("harris") },
    { value: "classic_orb", label: formatMethodLabel("classic_orb") },
    { value: "classic_gftt_orb", label: formatMethodLabel("classic_gftt_orb") },
    { value: "dl", label: formatMethodLabel("dl") },
    { value: "vit", label: formatMethodLabel("vit") },
    { value: "dedicated", label: formatMethodLabel("dedicated") },
];
