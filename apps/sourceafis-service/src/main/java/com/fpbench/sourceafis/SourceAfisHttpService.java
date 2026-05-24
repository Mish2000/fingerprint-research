package com.fpbench.sourceafis;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.Executors;

public final class SourceAfisHttpService {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final TypeReference<Map<String, Object>> JSON_OBJECT = new TypeReference<>() {};
    private static final int DEFAULT_PORT = 8765;
    private static final int MAX_REQUEST_BYTES = 20 * 1024 * 1024;

    private final SourceAfisEngine engine;

    SourceAfisHttpService(SourceAfisEngine engine) {
        this.engine = engine;
    }

    public static void main(String[] args) throws IOException {
        String host = env("SOURCEAFIS_HOST", "127.0.0.1");
        int port = parsePort(env("SOURCEAFIS_PORT", String.valueOf(DEFAULT_PORT)));
        SourceAfisHttpService service = new SourceAfisHttpService(new SourceAfisEngine());
        HttpServer server = HttpServer.create(new InetSocketAddress(host, port), 0);
        service.register(server);
        server.setExecutor(Executors.newFixedThreadPool(Math.max(2, Runtime.getRuntime().availableProcessors())));
        server.start();
        System.out.printf(
            "SourceAFIS sidecar listening on http://%s:%d at %s%n",
            host,
            port,
            Instant.now()
        );
    }

    void register(HttpServer server) {
        server.createContext("/health", exchange -> handle(exchange, "/health", "GET", request -> engine.health()));
        server.createContext("/extract-template", exchange -> handle(exchange, "/extract-template", "POST", engine::extractTemplate));
        server.createContext("/verify", exchange -> handle(exchange, "/verify", "POST", engine::verify));
        server.createContext("/identify", exchange -> handle(exchange, "/identify", "POST", engine::identify));
    }

    private void handle(HttpExchange exchange, String expectedPath, String expectedMethod, Route route) throws IOException {
        try {
            if (!expectedPath.equals(exchange.getRequestURI().getPath())) {
                writeJson(exchange, 404, error("not_found", "Endpoint not found."));
                return;
            }
            if (!expectedMethod.equalsIgnoreCase(exchange.getRequestMethod())) {
                writeJson(exchange, 405, error("method_not_allowed", "HTTP method is not allowed for this endpoint."));
                return;
            }
            Map<String, Object> request = "GET".equalsIgnoreCase(expectedMethod) ? Map.of() : readJsonObject(exchange);
            writeJson(exchange, 200, route.handle(request));
        } catch (ApiException e) {
            writeJson(exchange, e.statusCode, error(e.code, e.getMessage()));
        } catch (JsonProcessingException e) {
            writeJson(exchange, 400, error("invalid_json", "Request body must be a JSON object."));
        } catch (IOException e) {
            throw e;
        } catch (RuntimeException e) {
            System.err.printf("SourceAFIS sidecar internal error: %s%n", e.getClass().getSimpleName());
            writeJson(exchange, 500, error("internal_error", "SourceAFIS sidecar failed to process the request."));
        } finally {
            exchange.close();
        }
    }

    private Map<String, Object> readJsonObject(HttpExchange exchange) throws IOException {
        byte[] body = readLimited(exchange.getRequestBody(), MAX_REQUEST_BYTES);
        if (body.length == 0) {
            throw new ApiException(400, "invalid_json", "Request body must be a JSON object.");
        }
        Object payload = JSON.readValue(body, JSON_OBJECT);
        if (!(payload instanceof Map<?, ?>)) {
            throw new ApiException(400, "invalid_json", "Request body must be a JSON object.");
        }
        return JSON.readValue(body, JSON_OBJECT);
    }

    private byte[] readLimited(InputStream input, int maxBytes) throws IOException {
        byte[] buffer = new byte[8192];
        int total = 0;
        try (java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream()) {
            while (true) {
                int count = input.read(buffer);
                if (count < 0) {
                    return output.toByteArray();
                }
                total += count;
                if (total > maxBytes) {
                    throw new ApiException(413, "request_too_large", "Request body is too large.");
                }
                output.write(buffer, 0, count);
            }
        }
    }

    private void writeJson(HttpExchange exchange, int statusCode, Map<String, Object> payload) throws IOException {
        byte[] response = JSON.writeValueAsBytes(payload);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/json; charset=utf-8");
        headers.set("Cache-Control", "no-store");
        exchange.sendResponseHeaders(statusCode, response.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(response);
        }
    }

    private static Map<String, Object> error(String code, String detail) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("error", code);
        payload.put("detail", detail);
        return payload;
    }

    private static String env(String name, String fallback) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? fallback : value.trim();
    }

    private static int parsePort(String value) {
        try {
            int port = Integer.parseInt(value);
            if (port > 0 && port <= 65535) {
                return port;
            }
        } catch (NumberFormatException ignored) {
            // Fall through to default.
        }
        return DEFAULT_PORT;
    }

    @FunctionalInterface
    private interface Route {
        Map<String, Object> handle(Map<String, Object> request);
    }
}
