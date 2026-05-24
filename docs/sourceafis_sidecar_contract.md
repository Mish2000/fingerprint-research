# SourceAFIS Sidecar Contract

Phase 2 registers `sourceafis_open` as an optional external fingerprint engine.
The Python API does not import or run SourceAFIS directly. A real runtime should
be exposed as an HTTP sidecar and enabled with:

```powershell
$env:SOURCEAFIS_ENABLED = "true"
$env:SOURCEAFIS_SERVICE_URL = "http://127.0.0.1:8765"
```

Expected endpoints:

- `GET /health`
- `POST /extract-template`
- `POST /verify`
- `POST /identify`

Binary images and templates are transported in JSON as base64 fields. The
sidecar should return raw SourceAFIS matching scores. The Python provider leaves
`normalized_score` unset unless the sidecar explicitly returns a calibrated
normalization, because SourceAFIS scores need dataset-level threshold calibration.

The app starts without this sidecar. In that case `sourceafis_open` remains
registered and appears in `/fingerprint-engine/metadata` with `available=false`
and an actionable configuration message.
