# Scanner Capture

The scanner backend supports two capture providers:

- `twain`: direct capture through the production x86 helper at `tools/biometrika_capture/bin/x86/biometrika_twain_capture.exe`.
- `saved_file_bridge`: the existing folder bridge used by `POST /api/scanner/import-latest`.

## Vendor and TWAIN Setup

Direct capture expects the Biometrika Windows TWAIN driver to be installed and registered as the exact source name `TWAIN Biometrika Driver`. The helper uses the legacy Windows TWAIN DSM through `C:\WINDOWS\TWAIN_32.dll`; it does not call Biometrika SDK DLL exports and does not use Python `ctypes` or `cffi`.

The helper is compiled as x86 because the confirmed working capture path is the 32-bit TWAIN stack. FastAPI stays out of the TWAIN process entirely and only starts the helper subprocess.

## Build the Helper

From the repository root:

```powershell
tools\biometrika_capture\build_x86.cmd
```

The output executable is:

```text
tools\biometrika_capture\bin\x86\biometrika_twain_capture.exe
```

## Manual Helper Checks

Status:

```powershell
tools\biometrika_capture\bin\x86\biometrika_twain_capture.exe --status
```

Capture without source UI:

```powershell
New-Item -ItemType Directory -Force -Path reports\diagnostics\biometrika_capture\manual_output
tools\biometrika_capture\bin\x86\biometrika_twain_capture.exe --capture --output-dir reports\diagnostics\biometrika_capture\manual_output --show-ui false --timeout-ms 15000 --settle-after-enable-ms 1500
```

Manual fallback with source UI:

```powershell
tools\biometrika_capture\bin\x86\biometrika_twain_capture.exe --capture --output-dir reports\diagnostics\biometrika_capture\manual_output --show-ui true --timeout-ms 60000 --settle-after-enable-ms 0
```

The helper prints exactly one JSON object to stdout. Diagnostic logs go under `reports/diagnostics/biometrika_capture/`.

## Headless Settle Delay

The Live Demo UI countdown is only preparation time for the user to place a finger. It does not activate the scanner.

The real headless quality delay is `--settle-after-enable-ms`. The helper applies it after `DAT_USERINTERFACE / MSG_ENABLEDS` succeeds, while the TWAIN source is enabled and the scanner sensor/light is active. The helper keeps pumping TWAIN/Windows messages during this interval. If `MSG_XFERREADY` arrives early, the helper records it and waits until the settle interval has elapsed before calling `DAT_IMAGENATIVEXFER / MSG_GET`.

The helper success JSON includes:

```json
{
  "settle_after_enable_ms": 1500,
  "xferready_waited_for_settle": true
}
```

Relevant event markers include `enable_ds_success`, `settle_after_enable_start_1500ms`, `MSG_XFERREADY`, `xferready_seen_before_settle_elapsed`, `settle_after_enable_elapsed`, and `native_transfer_xferdone`.

If headless quality is still low, try `--settle-after-enable-ms 2000` or `--settle-after-enable-ms 2500`. Do not treat this as a scanner-provided quality percentage; the current TWAIN helper exposes image dimensions/format and event timing, not a built-in quality score.

## Backend Selection

`GET /api/scanner/status` reports available modes and only sets `direct_capture_available=true` when the helper status succeeds and confirms the exact TWAIN source. If the helper is missing, fails, or does not detect the source, TWAIN is not advertised as available.

`POST /api/scanner/capture` accepts:

```json
{
  "mode": "auto",
  "timeout_ms": 15000,
  "fallback_allowed": false,
  "normalize": true,
  "show_ui": false,
  "settle_after_enable_ms": 1500
}
```

`auto` prefers TWAIN when available. If TWAIN is unavailable or fails, the backend only uses the saved-file bridge when `fallback_allowed=true`. `twain` does not silently fall back unless fallback is explicitly allowed. `saved_file_bridge` calls the existing folder bridge path.

When `settle_after_enable_ms` is omitted, the backend uses 1500 ms for headless capture and 0 ms for `show_ui=true` capture. Values are clamped to the safe range `0..10000`.

When `normalize=true`, both providers normalize the selected capture to PNG and return a `/api/scanner/captures/{capture_id}` URL.
