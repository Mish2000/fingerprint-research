# SourceAFIS Sidecar Contract

Phase 3 provides `sourceafis_open` through a local Java HTTP sidecar in
`apps/sourceafis-service`. The Python API does not import SourceAFIS directly,
and the app still starts when the sidecar is absent. Without the sidecar,
`sourceafis_open` remains registered and appears in `/fingerprint-engine/metadata`
with `available=false` and an actionable configuration message.

The sidecar is stateless. It does not persist images or templates, does not make
external network calls, and must not log image bytes or template bytes.

## Build

The sidecar uses Maven and pins the official SourceAFIS Java artifact:

```xml
<groupId>com.machinezoo.sourceafis</groupId>
<artifactId>sourceafis</artifactId>
<version>3.18.1</version>
```

Building the sidecar may download Maven dependencies. Normal Python tests do
not build it and do not require internet access.

```powershell
cd apps/sourceafis-service
mvn test
mvn package
cd ../..
```

## Run Locally

```powershell
.\scripts\dev\run_sourceafis_sidecar.ps1
```

The service defaults to `127.0.0.1:8765`. The runner builds the sidecar when the
packaged jar is missing and avoids changing global system state.

To enable the Python provider in a local shell:

```powershell
$env:SOURCEAFIS_ENABLED = "true"
$env:SOURCEAFIS_SERVICE_URL = "http://127.0.0.1:8765"
```

Health check examples:

```powershell
curl.exe http://127.0.0.1:8765/health
```

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8765/health"
```

Expected response:

```json
{
  "status": "ok",
  "provider_id": "sourceafis_open",
  "engine": "SourceAFIS",
  "engine_version": "3.18.1",
  "template_format": "sourceafis",
  "supports_verification": true,
  "supports_identification": true,
  "supports_quality": false
}
```

## Local Validation Checklist

Maven is required for the Java build. To validate the sidecar build, package,
startup, and `/health` endpoint from the repository root:

```powershell
.\scripts\dev\check_sourceafis_sidecar.ps1
```

Manual build commands:

```powershell
cd apps/sourceafis-service
mvn test
mvn package
cd ../..
```

Manual run command:

```powershell
.\scripts\dev\run_sourceafis_sidecar.ps1
```

Python provider validation:

```powershell
$env:SOURCEAFIS_ENABLED = "true"
$env:SOURCEAFIS_SERVICE_URL = "http://127.0.0.1:8765"
$env:SOURCEAFIS_INTEGRATION_TESTS = "true"
python -m pytest tests/test_sourceafis_sidecar_integration_optional.py
```

## Endpoints

### `GET /health`

Returns provider and engine metadata. No request body.

### `POST /extract-template`

Request:

```json
{
  "image_base64": "...",
  "image_format": "png",
  "metadata": {
    "dpi": 500
  }
}
```

`image_format` may be `png`, `jpg`, `bmp`, `tif`, or `unknown`. DPI is optional;
SourceAFIS defaults to 500 DPI when it is not supplied.

Response:

```json
{
  "provider_id": "sourceafis_open",
  "template_format": "sourceafis",
  "template_version": "3.18.1",
  "template_base64": "...",
  "warnings": []
}
```

The template is the native SourceAFIS serialized template produced by
`FingerprintTemplate.toByteArray()`. It is deserialized with the
`FingerprintTemplate(byte[])` constructor.

### `POST /verify`

Request:

```json
{
  "probe_template_base64": "...",
  "candidate_template_base64": "..."
}
```

Response:

```json
{
  "score": 123.45,
  "normalized_score": null,
  "threshold": null,
  "decision": null,
  "warnings": [
    "Raw SourceAFIS score requires dataset-level calibration."
  ]
}
```

The sidecar returns SourceAFIS raw similarity scores only. It does not invent
normalized scores, thresholds, or accept/reject decisions.

### `POST /identify`

Request:

```json
{
  "probe_template_base64": "...",
  "gallery": [
    {
      "candidate_id": "subject-1",
      "template_base64": "...",
      "metadata": {}
    }
  ],
  "top_k": 10
}
```

Response:

```json
{
  "candidates": [
    {
      "candidate_id": "subject-1",
      "score": 123.45,
      "normalized_score": null,
      "rank": 1,
      "metadata": {}
    }
  ],
  "warnings": [
    "Raw SourceAFIS score requires dataset-level calibration."
  ]
}
```

The sidecar handles empty galleries, sorts candidates by score descending, and
respects `top_k`. Ties are deterministic by `candidate_id`.

## Errors

Errors are JSON objects with `error` and `detail` fields. Normal error responses
must not include local file paths or stack traces.

- Invalid base64: `400`
- Invalid image or template bytes: `422`
- Internal SourceAFIS failure: `500`

## Calibration Note

Raw SourceAFIS scores require dataset-level calibration before production
decisions. Calibrate thresholds on a validation split before comparing scores
with other matcher families or using them for accept/reject decisions.
