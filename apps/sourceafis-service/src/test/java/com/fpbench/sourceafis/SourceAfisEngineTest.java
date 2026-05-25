package com.fpbench.sourceafis;

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SourceAfisEngineTest {
    private final SourceAfisEngine engine = new SourceAfisEngine();

    @Test
    void healthMatchesSidecarContract() {
        Map<String, Object> health = engine.health();

        assertEquals("ok", health.get("status"));
        assertEquals("sourceafis_open", health.get("provider_id"));
        assertEquals("SourceAFIS", health.get("engine"));
        assertEquals("sourceafis", health.get("template_format"));
        assertEquals(true, health.get("supports_verification"));
        assertEquals(true, health.get("supports_identification"));
        assertEquals(false, health.get("supports_quality"));
    }

    @Test
    void extractVerifyAndIdentifyRoundTrip() throws IOException {
        String template = extractTemplateBase64(syntheticFingerprintPng(0));

        Map<String, Object> verification = engine.verify(Map.of(
            "probe_template_base64", template,
            "candidate_template_base64", template
        ));
        assertInstanceOf(Number.class, verification.get("score"));
        assertEquals(null, verification.get("normalized_score"));
        assertEquals(null, verification.get("decision"));

        Map<String, Object> identification = engine.identify(Map.of(
            "probe_template_base64", template,
            "gallery", List.of(
                Map.of("candidate_id", "subject-1", "template_base64", template, "metadata", Map.of("fixture", "synthetic")),
                Map.of("candidate_id", "subject-2", "template_base64", template, "metadata", Map.of())
            ),
            "top_k", 1
        ));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> candidates = (List<Map<String, Object>>) identification.get("candidates");
        assertEquals(1, candidates.size());
        assertEquals("subject-1", candidates.get(0).get("candidate_id"));
        assertEquals(1, candidates.get(0).get("rank"));
        assertFalse(((List<?>) identification.get("warnings")).isEmpty());
    }

    @Test
    void emptyGalleryReturnsNoCandidates() throws IOException {
        String template = extractTemplateBase64(syntheticFingerprintPng(1));

        Map<String, Object> identification = engine.identify(Map.of(
            "probe_template_base64", template,
            "gallery", List.of(),
            "top_k", 10
        ));

        assertTrue(((List<?>) identification.get("candidates")).isEmpty());
    }

    @Test
    void metadataDpiIsAcceptedAndAppliedToExtraction() throws IOException {
        byte[] image = syntheticFingerprintPng(2);
        Map<String, Object> dpi500 = extractTemplateResponse(image, 500);
        Map<String, Object> dpi1000 = extractTemplateResponse(image, 1000);

        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = (Map<String, Object>) dpi1000.get("metadata");

        assertEquals(1000.0, ((Number) metadata.get("dpi")).doubleValue());
        assertEquals("metadata", metadata.get("dpi_source"));
        assertNotEquals(dpi500.get("template_base64"), dpi1000.get("template_base64"));
    }

    @Test
    void invalidBase64ReturnsBadRequest() {
        ApiException error = assertThrows(ApiException.class, () -> engine.extractTemplate(Map.of(
            "image_base64", "not base64"
        )));

        assertEquals(400, error.statusCode);
        assertEquals("invalid_base64", error.code);
    }

    @Test
    void invalidTemplateReturnsUnprocessableEntity() {
        String bogusTemplate = Base64.getEncoder().encodeToString("not a template".getBytes(java.nio.charset.StandardCharsets.UTF_8));

        ApiException error = assertThrows(ApiException.class, () -> engine.verify(Map.of(
            "probe_template_base64", bogusTemplate,
            "candidate_template_base64", bogusTemplate
        )));

        assertEquals(422, error.statusCode);
        assertEquals("invalid_template", error.code);
    }

    private String extractTemplateBase64(byte[] image) {
        Map<String, Object> response = extractTemplateResponse(image, 500);

        String template = (String) response.get("template_base64");
        assertTrue(template.length() > 10);
        return template;
    }

    private Map<String, Object> extractTemplateResponse(byte[] image, int dpi) {
        Map<String, Object> response = engine.extractTemplate(Map.of(
            "image_base64", Base64.getEncoder().encodeToString(image),
            "image_format", "png",
            "metadata", Map.of("dpi", dpi)
        ));

        String template = (String) response.get("template_base64");
        assertTrue(template.length() > 10);
        return response;
    }

    private byte[] syntheticFingerprintPng(int variant) throws IOException {
        int width = 360;
        int height = 460;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics = image.createGraphics();
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, width, height);
        graphics.setColor(Color.BLACK);
        graphics.setStroke(new BasicStroke(3.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        for (int ridge = 0; ridge < 34; ridge++) {
            double y = 55 + ridge * 10.5;
            double phase = variant * 0.35 + ridge * 0.18;
            Path2D path = new Path2D.Double();
            path.moveTo(42, y + Math.sin(phase) * 10);
            for (int x = 62; x <= width - 42; x += 28) {
                double curveY = y + Math.sin(x * 0.035 + phase) * 18;
                path.lineTo(x, curveY);
            }
            graphics.draw(path);
            if (ridge % 7 == 3) {
                graphics.drawLine(width / 2, (int) y, width / 2 + 42, (int) y - 18);
            }
        }
        graphics.dispose();

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        ImageIO.write(image, "png", output);
        return output.toByteArray();
    }
}
