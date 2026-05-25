package com.fpbench.sourceafis;

import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class SourceAfisEngine {
    static final String PROVIDER_ID = "sourceafis_open";
    static final String ENGINE = "SourceAFIS";
    static final String ENGINE_VERSION = "3.18.1";
    static final String TEMPLATE_FORMAT = "sourceafis";
    static final String CALIBRATION_WARNING = "Raw SourceAFIS score requires dataset-level calibration.";
    static final double DEFAULT_DPI = 500.0;

    Map<String, Object> health() {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("status", "ok");
        response.put("provider_id", PROVIDER_ID);
        response.put("engine", ENGINE);
        response.put("engine_version", ENGINE_VERSION);
        response.put("sourceafis_version", ENGINE_VERSION);
        response.put("template_format", TEMPLATE_FORMAT);
        response.put("supports_verification", true);
        response.put("supports_identification", true);
        response.put("supports_quality", false);
        return response;
    }

    Map<String, Object> extractTemplate(Map<String, Object> request) {
        byte[] imageBytes = decodeRequiredBase64(
            firstString(request.get("image_base64"), nestedString(request, "image", "image_bytes_b64")),
            "image_base64"
        );
        try {
            DpiSetting dpi = readDpi(request);
            FingerprintImageOptions options = new FingerprintImageOptions();
            if (dpi.supplied()) {
                options = options.dpi(dpi.value());
            }
            FingerprintTemplate template = new FingerprintTemplate(new FingerprintImage(imageBytes, options));
            String encodedTemplate = Base64.getEncoder().encodeToString(template.toByteArray());

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("provider_id", PROVIDER_ID);
            response.put("provider_version", ENGINE_VERSION);
            response.put("engine_version", ENGINE_VERSION);
            response.put("sourceafis_version", ENGINE_VERSION);
            response.put("template_format", TEMPLATE_FORMAT);
            response.put("template_version", ENGINE_VERSION);
            response.put("template_base64", encodedTemplate);
            response.put("metadata", Map.of("dpi", dpi.value(), "dpi_source", dpi.source()));
            response.put("warnings", List.of());
            return response;
        } catch (IllegalArgumentException e) {
            throw new ApiException(422, "invalid_image", "Image bytes are not a valid supported fingerprint image.");
        } catch (RuntimeException e) {
            throw new ApiException(500, "sourceafis_failure", "SourceAFIS template extraction failed.");
        }
    }

    Map<String, Object> verify(Map<String, Object> request) {
        FingerprintTemplate probe = templateFromRequest(
            firstString(request.get("probe_template_base64"), nestedString(request, "probe_template", "template_bytes_b64")),
            "probe_template_base64"
        );
        FingerprintTemplate candidate = templateFromRequest(
            firstString(request.get("candidate_template_base64"), nestedString(request, "candidate_template", "template_bytes_b64")),
            "candidate_template_base64"
        );

        try {
            double score = new FingerprintMatcher(probe).match(candidate);
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("score", score);
            response.put("normalized_score", null);
            response.put("threshold", null);
            response.put("decision", null);
            response.put("warnings", List.of(CALIBRATION_WARNING));
            return response;
        } catch (RuntimeException e) {
            throw new ApiException(500, "sourceafis_failure", "SourceAFIS verification failed.");
        }
    }

    Map<String, Object> identify(Map<String, Object> request) {
        FingerprintTemplate probe = templateFromRequest(
            firstString(request.get("probe_template_base64"), nestedString(request, "probe_template", "template_bytes_b64")),
            "probe_template_base64"
        );
        List<Object> rawGallery = optionalList(request.get("gallery"));
        int topK = Math.max(optionalInt(request.get("top_k"), 10), 0);

        List<ScoredCandidate> scored = new ArrayList<>();
        FingerprintMatcher matcher = new FingerprintMatcher(probe);
        for (Object rawCandidate : rawGallery) {
            Map<String, Object> candidate = requiredObject(rawCandidate, "gallery entries");
            String candidateId = firstString(candidate.get("candidate_id"), candidate.get("gallery_id"));
            if (candidateId == null) {
                throw new ApiException(400, "missing_field", "Gallery entries require candidate_id.");
            }
            FingerprintTemplate template = templateFromRequest(
                firstString(candidate.get("template_base64"), nestedString(candidate, "template", "template_bytes_b64")),
                "template_base64"
            );
            Map<String, Object> metadata = optionalObject(candidate.get("metadata"));
            try {
                scored.add(new ScoredCandidate(candidateId, matcher.match(template), metadata));
            } catch (RuntimeException e) {
                throw new ApiException(500, "sourceafis_failure", "SourceAFIS identification failed.");
            }
        }

        scored.sort(
            Comparator.comparingDouble(ScoredCandidate::score)
                .reversed()
                .thenComparing(ScoredCandidate::candidateId)
        );

        List<Map<String, Object>> candidates = new ArrayList<>();
        int limit = Math.min(topK, scored.size());
        for (int index = 0; index < limit; index++) {
            ScoredCandidate item = scored.get(index);
            Map<String, Object> candidate = new LinkedHashMap<>();
            candidate.put("candidate_id", item.candidateId());
            candidate.put("score", item.score());
            candidate.put("normalized_score", null);
            candidate.put("rank", index + 1);
            candidate.put("metadata", item.metadata());
            candidates.add(candidate);
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("candidates", candidates);
        response.put("warnings", List.of(CALIBRATION_WARNING));
        return response;
    }

    private FingerprintTemplate templateFromRequest(String encoded, String fieldName) {
        byte[] serialized = decodeRequiredBase64(encoded, fieldName);
        try {
            return new FingerprintTemplate(serialized);
        } catch (RuntimeException e) {
            throw new ApiException(422, "invalid_template", "Template bytes are not a valid SourceAFIS template.");
        }
    }

    private byte[] decodeRequiredBase64(String encoded, String fieldName) {
        if (encoded == null) {
            throw new ApiException(400, "missing_field", "Request field " + fieldName + " is required.");
        }
        try {
            return Base64.getDecoder().decode(encoded);
        } catch (IllegalArgumentException e) {
            throw new ApiException(400, "invalid_base64", "Request field " + fieldName + " must be valid base64.");
        }
    }

    private DpiSetting readDpi(Map<String, Object> request) {
        Object metadata = request.get("metadata");
        Object dpi = null;
        String source = "sidecar_default";
        if (metadata instanceof Map<?, ?> && ((Map<?, ?>) metadata).containsKey("dpi")) {
            dpi = ((Map<?, ?>) metadata).get("dpi");
            source = "metadata";
        }
        if (dpi == null) {
            Object legacyImage = request.get("image");
            if (legacyImage instanceof Map<?, ?> && ((Map<?, ?>) legacyImage).containsKey("dpi")) {
                dpi = ((Map<?, ?>) legacyImage).get("dpi");
                source = "legacy_image";
            }
        }
        if (dpi == null) {
            return DpiSetting.sidecarDefault();
        }
        double value = optionalDouble(dpi, Double.NaN);
        if (Double.isFinite(value) && value >= 100.0 && value <= 2000.0) {
            return new DpiSetting(value, source, true);
        }
        return DpiSetting.sidecarDefault();
    }

    private static String nestedString(Map<String, Object> root, String objectField, String leafField) {
        Object nested = root.get(objectField);
        if (!(nested instanceof Map<?, ?>)) {
            return null;
        }
        return stringValue(((Map<?, ?>) nested).get(leafField));
    }

    private static String firstString(Object first, Object second) {
        String firstText = stringValue(first);
        return firstText != null ? firstText : stringValue(second);
    }

    private static String stringValue(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }

    private static int optionalInt(Object value, int fallback) {
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        String text = stringValue(value);
        if (text == null) {
            return fallback;
        }
        try {
            return Integer.parseInt(text);
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static double optionalDouble(Object value, double fallback) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        String text = stringValue(value);
        if (text == null) {
            return fallback;
        }
        try {
            return Double.parseDouble(text);
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static List<Object> optionalList(Object value) {
        if (value == null) {
            return List.of();
        }
        if (value instanceof List<?>) {
            return new ArrayList<>((List<?>) value);
        }
        throw new ApiException(400, "invalid_field", "Request field gallery must be a list.");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> optionalObject(Object value) {
        if (value == null) {
            return Map.of();
        }
        if (value instanceof Map<?, ?>) {
            return (Map<String, Object>) value;
        }
        return Map.of();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> requiredObject(Object value, String label) {
        if (value instanceof Map<?, ?>) {
            return (Map<String, Object>) value;
        }
        throw new ApiException(400, "invalid_field", "Request " + label + " must be JSON objects.");
    }

    private static final class ScoredCandidate {
        private final String candidateId;
        private final double score;
        private final Map<String, Object> metadata;

        private ScoredCandidate(String candidateId, double score, Map<String, Object> metadata) {
            this.candidateId = candidateId;
            this.score = score;
            this.metadata = metadata;
        }

        private String candidateId() {
            return candidateId;
        }

        private double score() {
            return score;
        }

        private Map<String, Object> metadata() {
            return metadata;
        }
    }

    private static final class DpiSetting {
        private final double value;
        private final String source;
        private final boolean supplied;

        private DpiSetting(double value, String source, boolean supplied) {
            this.value = value;
            this.source = source;
            this.supplied = supplied;
        }

        private static DpiSetting sidecarDefault() {
            return new DpiSetting(DEFAULT_DPI, "sidecar_default", false);
        }

        private double value() {
            return value;
        }

        private String source() {
            return source;
        }

        private boolean supplied() {
            return supplied;
        }
    }
}
