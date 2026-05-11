#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "twain.h"

namespace {

constexpr const char* kStatusMode = "status";
constexpr const char* kListMode = "status";
constexpr const char* kSelectMode = "status";
constexpr const char* kDryRunMode = "status";
constexpr const char* kCaptureMode = "capture";
constexpr const char* kBiometrikaSourceName = "TWAIN Biometrika Driver";
constexpr DWORD kDefaultHeadlessSettleAfterEnableMs = 1200;
constexpr DWORD kMaxSettleAfterEnableMs = 10000;

struct SourceRecord {
    TW_IDENTITY identity{};
    std::string product_name;
    std::string manufacturer;
    std::string family;
    std::string version;
};

struct TwainStatus {
    TW_UINT16 return_code = 0xffff;
    TW_UINT16 condition_code = 0xffff;
};

struct CliOptions {
    std::string mode = kStatusMode;
    bool show_ui = false;
    DWORD timeout_ms = 15000;
    DWORD settle_after_enable_ms = kDefaultHeadlessSettleAfterEnableMs;
    bool settle_after_enable_explicit = false;
    std::filesystem::path output_dir;
};

struct CapabilityValues {
    bool ok = false;
    TW_UINT16 return_code = 0xffff;
    TW_UINT16 condition_code = 0xffff;
    TW_UINT16 container_type = 0xffff;
    TW_UINT16 item_type = 0xffff;
    std::vector<TW_UINT32> values;
};

struct FileTransferPlan {
    bool supported = false;
    TW_UINT16 format = TWFF_BMP;
    std::string extension = ".bmp";
    std::string format_name = "BMP";
};

struct BmpImageInfo {
    DWORD file_size = 0;
    DWORD dib_header_size = 0;
    LONG width = 0;
    LONG height = 0;
    WORD bits_per_pixel = 0;
    DWORD compression = BI_RGB;
    DWORD pixel_offset = 0;
};

struct CaptureAttemptResult {
    bool ok = false;
    std::string error_code = "transfer_failed";
    std::string message;
    std::string transfer_mechanism;
    std::filesystem::path output_path;
    std::uintmax_t output_size_bytes = 0;
    TW_UINT16 return_code = 0xffff;
    TW_UINT16 condition_code = 0xffff;
    std::string failure_stage;
    BmpImageInfo image;
    DWORD settle_after_enable_ms = 0;
    bool xferready_waited_for_settle = false;
};

std::filesystem::path g_capture_output_dir;

LRESULT CALLBACK HiddenWindowProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    return DefWindowProcA(hwnd, message, wparam, lparam);
}

template <typename CharT, size_t N>
void SetTwainString(CharT (&dest)[N], const char* value) {
    std::memset(dest, 0, sizeof(dest));
    const size_t len = std::min(std::strlen(value), N - 1);
    for (size_t i = 0; i < len; ++i) {
        dest[i] = static_cast<CharT>(value[i]);
    }
}

template <typename CharT, size_t N>
std::string TwainString(const CharT (&value)[N]) {
    size_t len = 0;
    while (len < N && value[len] != 0) {
        ++len;
    }

    std::string out;
    out.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        const unsigned char ch = static_cast<unsigned char>(value[i]);
        if (ch >= 0x20 || ch == '\t') {
            out.push_back(static_cast<char>(ch));
        }
    }

    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) {
        out.pop_back();
    }
    return out;
}

std::string JsonEscape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

std::string JsonString(const std::string& value) {
    return "\"" + JsonEscape(value) + "\"";
}

std::string LowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool IsBiometrikaCandidate(const SourceRecord& source) {
    const std::string haystack = LowerAscii(source.product_name + " " + source.manufacturer + " " + source.family);
    return haystack.find("biometrika") != std::string::npos;
}

bool IsExactBiometrikaSource(const SourceRecord& source) {
    return source.product_name == kBiometrikaSourceName;
}

std::string BoolJson(bool value) {
    return value ? "true" : "false";
}

void AddEvent(std::vector<std::string>& events, std::ofstream& log, const std::string& event) {
    events.push_back(event);
    log << "event=" << event << "\n";
}

std::string VersionString(const TW_VERSION& version) {
    std::ostringstream out;
    out << version.MajorNum << "." << version.MinorNum;
    const std::string info = TwainString(version.Info);
    if (!info.empty()) {
        out << " " << info;
    }
    return out.str();
}

SourceRecord ToSourceRecord(const TW_IDENTITY& identity) {
    SourceRecord record;
    record.identity = identity;
    record.product_name = TwainString(identity.ProductName);
    record.manufacturer = TwainString(identity.Manufacturer);
    record.family = TwainString(identity.ProductFamily);
    record.version = VersionString(identity.Version);
    return record;
}

TW_IDENTITY MakeAppIdentity() {
    TW_IDENTITY app{};
    app.Id = 0;
    app.Version.MajorNum = 1;
    app.Version.MinorNum = 0;
    app.Version.Language = TWLG_USA;
    app.Version.Country = TWCY_USA;
    SetTwainString(app.Version.Info, "Biometrika TWAIN capture helper");
    app.ProtocolMajor = TWON_PROTOCOLMAJOR;
    app.ProtocolMinor = TWON_PROTOCOLMINOR;
    app.SupportedGroups = DG_CONTROL | DG_IMAGE;
    SetTwainString(app.Manufacturer, "fingerprint-research");
    SetTwainString(app.ProductFamily, "scanner capture");
    SetTwainString(app.ProductName, "Biometrika TWAIN Capture Helper");
    return app;
}

HWND CreateHiddenParentWindow(std::vector<std::string>& notes) {
    HINSTANCE instance = GetModuleHandleA(nullptr);
    const char* class_name = "BiometrikaTwainCaptureHiddenWindow";

    WNDCLASSA wc{};
    wc.lpfnWndProc = HiddenWindowProc;
    wc.hInstance = instance;
    wc.lpszClassName = class_name;

    if (!RegisterClassA(&wc)) {
        const DWORD err = GetLastError();
        if (err != ERROR_CLASS_ALREADY_EXISTS) {
            std::ostringstream note;
            note << "RegisterClassA failed: " << err;
            notes.push_back(note.str());
            return nullptr;
        }
    }

    HWND hwnd = CreateWindowExA(
        0,
        class_name,
        "Biometrika TWAIN Capture Helper",
        WS_OVERLAPPED,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1,
        1,
        nullptr,
        nullptr,
        instance,
        nullptr);

    if (!hwnd) {
        std::ostringstream note;
        note << "CreateWindowExA failed: " << GetLastError();
        notes.push_back(note.str());
    }
    return hwnd;
}

std::filesystem::path ExecutablePath() {
    char buffer[MAX_PATH]{};
    const DWORD len = GetModuleFileNameA(nullptr, buffer, static_cast<DWORD>(sizeof(buffer)));
    if (len == 0 || len >= sizeof(buffer)) {
        return {};
    }
    return std::filesystem::path(buffer);
}

bool ContainsCaptureDirectory(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::exists(path / "tools" / "biometrika_capture", ec);
}

std::filesystem::path FindRepoRoot() {
    std::vector<std::filesystem::path> starts;
    std::error_code ec;
    starts.push_back(std::filesystem::current_path(ec));
    const std::filesystem::path exe = ExecutablePath();
    if (!exe.empty()) {
        starts.push_back(exe.parent_path());
    }

    for (auto start : starts) {
        if (start.empty()) {
            continue;
        }
        while (!start.empty()) {
            if (ContainsCaptureDirectory(start)) {
                return start;
            }
            const auto parent = start.parent_path();
            if (parent == start) {
                break;
            }
            start = parent;
        }
    }
    return starts.empty() ? std::filesystem::path(".") : starts.front();
}

std::filesystem::path DiagnosticsDir() {
    return FindRepoRoot() / "reports" / "diagnostics" / "biometrika_capture";
}

std::filesystem::path CaptureOutputDir() {
    return g_capture_output_dir.empty() ? (DiagnosticsDir() / "manual_output") : g_capture_output_dir;
}

std::string LocalTimestampForFilename() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    tm local_time{};
    localtime_s(&local_time, &now);
    std::ostringstream out;
    out << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return out.str();
}

std::filesystem::path UniqueCapturePath(const std::string& prefix, const std::string& extension) {
    const std::filesystem::path output_dir = CaptureOutputDir();
    std::error_code ec;

    const std::string stamp = LocalTimestampForFilename();
    const DWORD pid = GetCurrentProcessId();
    for (int i = 0; i < 1000; ++i) {
        std::ostringstream name;
        name << prefix << "_" << stamp << "_" << pid;
        if (i > 0) {
            name << "_" << i;
        }
        name << extension;
        std::filesystem::path candidate = output_dir / name.str();
        if (!std::filesystem::exists(candidate, ec)) {
            return candidate;
        }
    }
    return output_dir / (prefix + "_" + stamp + "_" + std::to_string(pid) + "_overflow" + extension);
}

std::ofstream OpenLog(const std::string& mode) {
    const auto report_dir = DiagnosticsDir();
    std::error_code ec;
    std::filesystem::create_directories(report_dir, ec);
    std::ofstream log(report_dir / "biometrika_twain_capture.log", std::ios::app);
    if (log) {
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        tm local_time{};
        localtime_s(&local_time, &now);
        log << "\n[" << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << "] mode=" << mode << "\n";
    }
    return log;
}

std::string LoadedTwainModulePath() {
    HMODULE module = GetModuleHandleA("twain_32.dll");
    if (!module) {
        return {};
    }

    char buffer[MAX_PATH]{};
    const DWORD len = GetModuleFileNameA(module, buffer, static_cast<DWORD>(sizeof(buffer)));
    if (len == 0 || len >= sizeof(buffer)) {
        return {};
    }
    return buffer;
}

TwainStatus ReadStatus(TW_IDENTITY* app, TW_IDENTITY* dest) {
    TW_STATUS status{};
    const TW_UINT16 rc = DSM_Entry(app, dest, DG_CONTROL, DAT_STATUS, MSG_GET, reinterpret_cast<TW_MEMREF>(&status));
    TwainStatus out;
    out.return_code = rc;
    out.condition_code = status.ConditionCode;
    return out;
}

std::string NotesJson(const std::vector<std::string>& notes);

std::string DiagnosticsJson(
    TW_UINT16 return_code,
    TW_UINT16 condition_code,
    const std::string& twain_module,
    const std::string& stage = "twain") {
    std::ostringstream out;
    out << "{";
    out << "\"stage\":" << JsonString(stage) << ",";
    out << "\"return_code\":" << return_code << ",";
    out << "\"condition_code\":" << condition_code;
    if (!twain_module.empty()) {
        out << ",\"twain_32_module\":" << JsonString(twain_module);
    }
    out << "}";
    return out.str();
}

std::string FailureJson(
    const std::string& mode,
    const std::string& error_code,
    const std::string& message,
    const std::string& diagnostics_json,
    long long duration_ms = 0,
    const std::vector<std::string>& events = std::vector<std::string>()) {
    std::ostringstream out;
    out << "{";
    out << "\"ok\":false,";
    out << "\"mode\":" << JsonString(mode) << ",";
    out << "\"provider\":\"twain\",";
    out << "\"error_code\":" << JsonString(error_code) << ",";
    out << "\"message\":" << JsonString(message) << ",";
    out << "\"duration_ms\":" << duration_ms << ",";
    out << "\"diagnostics\":" << diagnostics_json << ",";
    out << "\"events\":" << NotesJson(events);
    out << "}";
    return out.str();
}

std::string SourcesJson(const std::vector<SourceRecord>& sources) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < sources.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "{";
        out << "\"product_name\":" << JsonString(sources[i].product_name) << ",";
        out << "\"manufacturer\":" << JsonString(sources[i].manufacturer) << ",";
        out << "\"family\":" << JsonString(sources[i].family) << ",";
        out << "\"version\":" << JsonString(sources[i].version);
        out << "}";
    }
    out << "]";
    return out.str();
}

std::string NotesJson(const std::vector<std::string>& notes) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < notes.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << JsonString(notes[i]);
    }
    out << "]";
    return out.str();
}

std::string CaptureDiagnosticsJson(
    const std::string& stage,
    TW_UINT16 return_code,
    TW_UINT16 condition_code,
    const std::string& twain_module,
    const std::vector<std::pair<std::string, std::string>>& extra_json_fields = {}) {
    std::ostringstream out;
    out << "{";
    out << "\"stage\":" << JsonString(stage) << ",";
    out << "\"return_code\":" << return_code << ",";
    out << "\"condition_code\":" << condition_code;
    if (!twain_module.empty()) {
        out << ",\"twain_32_module\":" << JsonString(twain_module);
    }
    for (const auto& field : extra_json_fields) {
        out << "," << JsonString(field.first) << ":" << field.second;
    }
    out << "}";
    return out.str();
}

std::string CompressionName(DWORD compression) {
    switch (compression) {
        case BI_RGB:
            return "BI_RGB";
        case BI_RLE8:
            return "BI_RLE8";
        case BI_RLE4:
            return "BI_RLE4";
        case BI_BITFIELDS:
            return "BI_BITFIELDS";
#ifdef BI_JPEG
        case BI_JPEG:
            return "BI_JPEG";
#endif
#ifdef BI_PNG
        case BI_PNG:
            return "BI_PNG";
#endif
        default:
            return "UNKNOWN";
    }
}

std::string BmpImageInfoJson(const BmpImageInfo& image) {
    std::ostringstream out;
    out << "{";
    out << "\"format\":\"bmp\",";
    out << "\"width\":" << image.width << ",";
    out << "\"height\":" << (image.height < 0 ? -image.height : image.height) << ",";
    out << "\"bits_per_pixel\":" << image.bits_per_pixel << ",";
    out << "\"compression\":" << JsonString(CompressionName(image.compression));
    out << "}";
    return out.str();
}

std::string CaptureSuccessJson(
    bool show_ui,
    const std::string& transfer_mechanism,
    const std::filesystem::path& output_path,
    std::uintmax_t output_size_bytes,
    const BmpImageInfo& image,
    DWORD settle_after_enable_ms,
    bool xferready_waited_for_settle,
    long long duration_ms,
    const std::vector<std::string>& events) {
    std::ostringstream out;
    out << "{";
    out << "\"ok\":true,";
    out << "\"mode\":\"" << kCaptureMode << "\",";
    out << "\"provider\":\"twain\",";
    out << "\"architecture\":\"x86\",";
    out << "\"source_name\":" << JsonString(kBiometrikaSourceName) << ",";
    out << "\"show_ui\":" << BoolJson(show_ui) << ",";
    out << "\"transfer_mechanism\":" << JsonString(transfer_mechanism) << ",";
    out << "\"output_path\":" << JsonString(output_path.string()) << ",";
    out << "\"output_size_bytes\":" << output_size_bytes << ",";
    out << "\"settle_after_enable_ms\":" << settle_after_enable_ms << ",";
    out << "\"xferready_waited_for_settle\":" << BoolJson(xferready_waited_for_settle) << ",";
    out << "\"duration_ms\":" << duration_ms << ",";
    out << "\"image\":" << BmpImageInfoJson(image) << ",";
    out << "\"events\":" << NotesJson(events);
    out << "}";
    return out.str();
}

std::string CaptureFailureJson(
    const std::string& error_code,
    const std::string& message,
    bool show_ui,
    long long duration_ms,
    const std::vector<std::string>& events,
    const std::string& diagnostics_json) {
    std::ostringstream out;
    out << "{";
    out << "\"ok\":false,";
    out << "\"mode\":\"" << kCaptureMode << "\",";
    out << "\"provider\":\"twain\",";
    out << "\"error_code\":" << JsonString(error_code) << ",";
    out << "\"message\":" << JsonString(message) << ",";
    out << "\"show_ui\":" << BoolJson(show_ui) << ",";
    out << "\"duration_ms\":" << duration_ms << ",";
    out << "\"events\":" << NotesJson(events) << ",";
    out << "\"diagnostics\":" << diagnostics_json;
    out << "}";
    return out.str();
}

std::string StatusSuccessJson(
    const std::vector<SourceRecord>& sources,
    const SourceRecord* exact_source,
    const std::string& module_path,
    long long duration_ms,
    const std::vector<std::string>& events) {
    std::ostringstream out;
    out << "{";
    out << "\"ok\":true,";
    out << "\"mode\":\"" << kStatusMode << "\",";
    out << "\"architecture\":\"x86\",";
    out << "\"twain_available\":true,";
    out << "\"source_detected\":" << (exact_source ? "true" : "false") << ",";
    out << "\"source_name\":" << (exact_source ? JsonString(exact_source->product_name) : "null") << ",";
    out << "\"twain_32_module\":" << JsonString(module_path) << ",";
    out << "\"sources\":" << SourcesJson(sources) << ",";
    out << "\"duration_ms\":" << duration_ms << ",";
    out << "\"events\":" << NotesJson(events);
    out << "}";
    return out.str();
}

std::string ListSuccessJson(
    const std::string& mode,
    const std::vector<SourceRecord>& sources,
    const SourceRecord* candidate,
    long long duration_ms,
    const std::vector<std::string>& notes) {
    (void)mode;
    return StatusSuccessJson(sources, candidate, LoadedTwainModulePath(), duration_ms, notes);
}

std::string DryRunSuccessJson(
    const std::vector<SourceRecord>& sources,
    const SourceRecord& candidate,
    long long duration_ms,
    const std::vector<std::string>& notes) {
    std::ostringstream out;
    out << "{";
    out << "\"ok\":true,";
    out << "\"mode\":\"" << kDryRunMode << "\",";
    out << "\"architecture\":\"x86\",";
    out << "\"sources\":" << SourcesJson(sources) << ",";
    out << "\"biometrika_candidate_found\":true,";
    out << "\"biometrika_source\":" << JsonString(candidate.product_name) << ",";
    out << "\"dry_run_open_source_ok\":true,";
    out << "\"capture_attempted\":false,";
    out << "\"duration_ms\":" << duration_ms << ",";
    out << "\"notes\":" << NotesJson(notes);
    out << "}";
    return out.str();
}

bool OpenDsm(TW_IDENTITY& app, HWND hwnd, std::ofstream& log, std::string& failure_json, const std::string& mode) {
    TW_UINT16 rc = DSM_Entry(&app, nullptr, DG_CONTROL, DAT_PARENT, MSG_OPENDSM, reinterpret_cast<TW_MEMREF>(&hwnd));
    log << "MSG_OPENDSM rc=" << rc << "\n";
    if (rc != TWRC_SUCCESS) {
        const TwainStatus status = ReadStatus(&app, nullptr);
        log << "DAT_STATUS rc=" << status.return_code << " condition=" << status.condition_code << "\n";
        failure_json = FailureJson(
            mode,
            "twain_unavailable",
            "TWAIN DSM MSG_OPENDSM failed.",
            DiagnosticsJson(rc, status.condition_code, LoadedTwainModulePath(), "MSG_OPENDSM"));
        return false;
    }
    return true;
}

bool ListSources(
    TW_IDENTITY& app,
    std::ofstream& log,
    std::vector<SourceRecord>& sources,
    std::string& failure_json,
    const std::string& mode) {
    TW_IDENTITY source{};
    TW_UINT16 rc = DSM_Entry(&app, nullptr, DG_CONTROL, DAT_IDENTITY, MSG_GETFIRST, reinterpret_cast<TW_MEMREF>(&source));
    log << "MSG_GETFIRST rc=" << rc << "\n";

    if (rc == TWRC_ENDOFLIST) {
        return true;
    }

    if (rc != TWRC_SUCCESS) {
        const TwainStatus status = ReadStatus(&app, nullptr);
        log << "DAT_STATUS rc=" << status.return_code << " condition=" << status.condition_code << "\n";
        failure_json = FailureJson(
            mode,
            "twain_unavailable",
            "TWAIN source enumeration failed at MSG_GETFIRST.",
            DiagnosticsJson(rc, status.condition_code, LoadedTwainModulePath(), "MSG_GETFIRST"));
        return false;
    }

    while (rc == TWRC_SUCCESS) {
        SourceRecord record = ToSourceRecord(source);
        log << "source product=\"" << record.product_name << "\" manufacturer=\"" << record.manufacturer
            << "\" family=\"" << record.family << "\" version=\"" << record.version << "\"\n";
        sources.push_back(record);

        std::memset(&source, 0, sizeof(source));
        rc = DSM_Entry(&app, nullptr, DG_CONTROL, DAT_IDENTITY, MSG_GETNEXT, reinterpret_cast<TW_MEMREF>(&source));
        log << "MSG_GETNEXT rc=" << rc << "\n";
    }

    if (rc != TWRC_ENDOFLIST) {
        const TwainStatus status = ReadStatus(&app, nullptr);
        log << "DAT_STATUS rc=" << status.return_code << " condition=" << status.condition_code << "\n";
        failure_json = FailureJson(
            mode,
            "twain_unavailable",
            "TWAIN source enumeration failed at MSG_GETNEXT.",
            DiagnosticsJson(rc, status.condition_code, LoadedTwainModulePath(), "MSG_GETNEXT"));
        return false;
    }

    return true;
}

const SourceRecord* FindCandidate(const std::vector<SourceRecord>& sources) {
    for (const auto& source : sources) {
        if (IsBiometrikaCandidate(source)) {
            return &source;
        }
    }
    return nullptr;
}

const SourceRecord* FindExactBiometrikaSource(const std::vector<SourceRecord>& sources) {
    for (const auto& source : sources) {
        if (IsExactBiometrikaSource(source)) {
            return &source;
        }
    }
    return nullptr;
}

size_t TwainItemSize(TW_UINT16 item_type) {
    switch (item_type) {
        case TWTY_INT8:
        case TWTY_UINT8:
            return 1;
        case TWTY_INT16:
        case TWTY_UINT16:
        case TWTY_BOOL:
            return 2;
        case TWTY_INT32:
        case TWTY_UINT32:
        case TWTY_FIX32:
            return 4;
        default:
            return 0;
    }
}

bool ReadTwainItemValue(const TW_UINT8* data, TW_UINT16 item_type, TW_UINT32& value) {
    switch (item_type) {
        case TWTY_INT8: {
            TW_INT8 raw = 0;
            std::memcpy(&raw, data, sizeof(raw));
            value = static_cast<TW_UINT32>(raw);
            return true;
        }
        case TWTY_UINT8: {
            TW_UINT8 raw = 0;
            std::memcpy(&raw, data, sizeof(raw));
            value = raw;
            return true;
        }
        case TWTY_INT16: {
            TW_INT16 raw = 0;
            std::memcpy(&raw, data, sizeof(raw));
            value = static_cast<TW_UINT32>(raw);
            return true;
        }
        case TWTY_UINT16:
        case TWTY_BOOL: {
            TW_UINT16 raw = 0;
            std::memcpy(&raw, data, sizeof(raw));
            value = raw;
            return true;
        }
        case TWTY_INT32: {
            TW_INT32 raw = 0;
            std::memcpy(&raw, data, sizeof(raw));
            value = static_cast<TW_UINT32>(raw);
            return true;
        }
        case TWTY_UINT32: {
            TW_UINT32 raw = 0;
            std::memcpy(&raw, data, sizeof(raw));
            value = raw;
            return true;
        }
        default:
            return false;
    }
}

void FreeTwainHandle(TW_HANDLE handle) {
    if (handle) {
        GlobalFree(static_cast<HGLOBAL>(handle));
    }
}

CapabilityValues GetCapabilityValues(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    TW_UINT16 cap_id,
    const std::string& cap_name,
    std::ofstream& log,
    std::vector<std::string>& events) {
    CapabilityValues result;
    TW_CAPABILITY cap{};
    cap.Cap = cap_id;
    cap.ConType = TWON_DONTCARE16;
    cap.hContainer = nullptr;

    const TW_UINT16 rc = DSM_Entry(
        &app,
        &source,
        DG_CONTROL,
        DAT_CAPABILITY,
        MSG_GET,
        reinterpret_cast<TW_MEMREF>(&cap));
    result.return_code = rc;
    log << "MSG_GET " << cap_name << " rc=" << rc << " con_type=" << cap.ConType << "\n";

    if (rc != TWRC_SUCCESS) {
        const TwainStatus status = ReadStatus(&app, &source);
        result.condition_code = status.condition_code;
        log << "DAT_STATUS after MSG_GET " << cap_name << " rc=" << status.return_code
            << " condition=" << status.condition_code << "\n";
        AddEvent(events, log, "cap_get_" + cap_name + "_rc_" + std::to_string(rc) + "_cc_" +
                               std::to_string(status.condition_code));
        return result;
    }

    result.ok = true;
    result.condition_code = TWCC_SUCCESS;
    result.container_type = cap.ConType;
    AddEvent(events, log, "cap_get_" + cap_name + "_success");

    if (!cap.hContainer) {
        AddEvent(events, log, "cap_get_" + cap_name + "_empty_container");
        return result;
    }

    const HGLOBAL container = static_cast<HGLOBAL>(cap.hContainer);
    const SIZE_T container_size = GlobalSize(container);
    const auto* bytes = static_cast<const TW_UINT8*>(GlobalLock(container));
    if (!bytes) {
        AddEvent(events, log, "cap_get_" + cap_name + "_lock_failed");
        FreeTwainHandle(cap.hContainer);
        return result;
    }

    auto append_items = [&](const TW_UINT8* item_data, size_t count, TW_UINT16 item_type, SIZE_T available) {
        result.item_type = item_type;
        const size_t item_size = TwainItemSize(item_type);
        if (item_size == 0 || available < item_size) {
            return;
        }
        const size_t safe_count = std::min(count, static_cast<size_t>(available / item_size));
        for (size_t i = 0; i < safe_count; ++i) {
            TW_UINT32 value = 0;
            if (ReadTwainItemValue(item_data + (i * item_size), item_type, value)) {
                result.values.push_back(value);
            }
        }
    };

    if (cap.ConType == TWON_ONEVALUE && container_size >= sizeof(TW_ONEVALUE)) {
        const auto* one = reinterpret_cast<const TW_ONEVALUE*>(bytes);
        result.item_type = one->ItemType;
        result.values.push_back(one->Item);
    } else if (cap.ConType == TWON_ENUMERATION && container_size >= offsetof(TW_ENUMERATION, ItemList)) {
        const auto* enumeration = reinterpret_cast<const TW_ENUMERATION*>(bytes);
        const SIZE_T available = container_size - offsetof(TW_ENUMERATION, ItemList);
        append_items(enumeration->ItemList, static_cast<size_t>(enumeration->NumItems), enumeration->ItemType, available);
    } else if (cap.ConType == TWON_ARRAY && container_size >= offsetof(TW_ARRAY, ItemList)) {
        const auto* array = reinterpret_cast<const TW_ARRAY*>(bytes);
        const SIZE_T available = container_size - offsetof(TW_ARRAY, ItemList);
        append_items(array->ItemList, static_cast<size_t>(array->NumItems), array->ItemType, available);
    } else if (cap.ConType == TWON_RANGE && container_size >= sizeof(TW_RANGE)) {
        const auto* range = reinterpret_cast<const TW_RANGE*>(bytes);
        result.item_type = range->ItemType;
        result.values.push_back(range->CurrentValue);
        result.values.push_back(range->DefaultValue);
        result.values.push_back(range->MinValue);
        result.values.push_back(range->MaxValue);
    } else {
        AddEvent(events, log, "cap_get_" + cap_name + "_unparsed_container_" + std::to_string(cap.ConType));
    }

    GlobalUnlock(container);
    FreeTwainHandle(cap.hContainer);
    return result;
}

bool CapabilityContains(const CapabilityValues& capability, TW_UINT32 value) {
    return std::find(capability.values.begin(), capability.values.end(), value) != capability.values.end();
}

bool SetOneValueCapability(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    TW_UINT16 cap_id,
    TW_UINT16 item_type,
    TW_UINT32 item,
    const std::string& cap_name,
    std::ofstream& log,
    std::vector<std::string>& events,
    TW_UINT16& return_code,
    TW_UINT16& condition_code) {
    TW_HANDLE handle = GlobalAlloc(GHND, sizeof(TW_ONEVALUE));
    if (!handle) {
        return_code = TWRC_FAILURE;
        condition_code = TWCC_LOWMEMORY;
        AddEvent(events, log, "cap_set_" + cap_name + "_alloc_failed");
        return false;
    }

    auto* one = static_cast<TW_ONEVALUE*>(GlobalLock(static_cast<HGLOBAL>(handle)));
    if (!one) {
        FreeTwainHandle(handle);
        return_code = TWRC_FAILURE;
        condition_code = TWCC_LOWMEMORY;
        AddEvent(events, log, "cap_set_" + cap_name + "_lock_failed");
        return false;
    }
    one->ItemType = item_type;
    one->Item = item;
    GlobalUnlock(static_cast<HGLOBAL>(handle));

    TW_CAPABILITY cap{};
    cap.Cap = cap_id;
    cap.ConType = TWON_ONEVALUE;
    cap.hContainer = handle;

    const TW_UINT16 rc = DSM_Entry(
        &app,
        &source,
        DG_CONTROL,
        DAT_CAPABILITY,
        MSG_SET,
        reinterpret_cast<TW_MEMREF>(&cap));
    return_code = rc;
    log << "MSG_SET " << cap_name << " rc=" << rc << "\n";

    if (rc != TWRC_SUCCESS) {
        const TwainStatus status = ReadStatus(&app, &source);
        condition_code = status.condition_code;
        log << "DAT_STATUS after MSG_SET " << cap_name << " rc=" << status.return_code
            << " condition=" << status.condition_code << "\n";
        AddEvent(events, log, "cap_set_" + cap_name + "_rc_" + std::to_string(rc) + "_cc_" +
                               std::to_string(status.condition_code));
        FreeTwainHandle(handle);
        return false;
    }

    condition_code = TWCC_SUCCESS;
    AddEvent(events, log, "cap_set_" + cap_name + "_success");
    FreeTwainHandle(handle);
    return true;
}

bool CapabilityReadable(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    TW_UINT16 cap_id,
    const std::string& cap_name,
    std::ofstream& log,
    std::vector<std::string>& events) {
    CapabilityValues values = GetCapabilityValues(app, source, cap_id, cap_name, log, events);
    return values.ok;
}

void ConfigureXferCountOne(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    std::ofstream& log,
    std::vector<std::string>& events,
    std::vector<std::string>& notes) {
    if (!CapabilityReadable(app, source, CAP_XFERCOUNT, "CAP_XFERCOUNT", log, events)) {
        notes.push_back("CAP_XFERCOUNT was not readable; source may not support setting transfer count.");
        return;
    }

    TW_UINT16 rc = 0;
    TW_UINT16 cc = 0;
    if (!SetOneValueCapability(
            app,
            source,
            CAP_XFERCOUNT,
            TWTY_INT16,
            1,
            "CAP_XFERCOUNT",
            log,
            events,
            rc,
            cc)) {
        std::ostringstream note;
        note << "CAP_XFERCOUNT=1 request failed rc=" << rc << " cc=" << cc << "; continuing.";
        notes.push_back(note.str());
    }
}

FileTransferPlan DiscoverFileTransferPlan(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    std::ofstream& log,
    std::vector<std::string>& events,
    std::vector<std::string>& notes) {
    FileTransferPlan plan;
    CapabilityValues xfer_mech = GetCapabilityValues(app, source, ICAP_XFERMECH, "ICAP_XFERMECH", log, events);
    if (!xfer_mech.ok) {
        notes.push_back("ICAP_XFERMECH was not readable; file transfer fallback is not safely discoverable.");
        return plan;
    }

    if (!CapabilityContains(xfer_mech, TWSX_FILE)) {
        notes.push_back("ICAP_XFERMECH did not advertise TWSX_FILE.");
        return plan;
    }

    CapabilityValues formats = GetCapabilityValues(app, source, ICAP_IMAGEFILEFORMAT, "ICAP_IMAGEFILEFORMAT", log, events);
    if (!formats.ok) {
        notes.push_back("ICAP_IMAGEFILEFORMAT was not readable; file transfer fallback is not safely discoverable.");
        return plan;
    }

    if (CapabilityContains(formats, TWFF_BMP)) {
        plan.supported = true;
        plan.format = TWFF_BMP;
        plan.extension = ".bmp";
        plan.format_name = "BMP";
    } else if (CapabilityContains(formats, TWFF_PNG)) {
        plan.supported = true;
        plan.format = TWFF_PNG;
        plan.extension = ".png";
        plan.format_name = "PNG";
    } else if (CapabilityContains(formats, TWFF_TIFF)) {
        plan.supported = true;
        plan.format = TWFF_TIFF;
        plan.extension = ".tif";
        plan.format_name = "TIFF";
    }

    if (plan.supported) {
        notes.push_back("File transfer fallback is discoverable with " + plan.format_name + " output.");
        AddEvent(events, log, "file_transfer_supported_" + plan.format_name);
    } else {
        notes.push_back("File transfer was advertised, but BMP/PNG/TIFF formats were not advertised.");
    }
    return plan;
}

void PreferNativeTransfer(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    std::ofstream& log,
    std::vector<std::string>& events,
    std::vector<std::string>& notes) {
    CapabilityValues xfer_mech = GetCapabilityValues(app, source, ICAP_XFERMECH, "ICAP_XFERMECH", log, events);
    if (!xfer_mech.ok) {
        notes.push_back("ICAP_XFERMECH was not readable; leaving the source default and attempting native transfer.");
        return;
    }

    if (!CapabilityContains(xfer_mech, TWSX_NATIVE)) {
        notes.push_back("ICAP_XFERMECH did not advertise TWSX_NATIVE; native transfer will still be attempted first.");
        return;
    }

    TW_UINT16 rc = 0;
    TW_UINT16 cc = 0;
    if (!SetOneValueCapability(
            app,
            source,
            ICAP_XFERMECH,
            TWTY_UINT16,
            TWSX_NATIVE,
            "ICAP_XFERMECH_NATIVE",
            log,
            events,
            rc,
            cc)) {
        std::ostringstream note;
        note << "ICAP_XFERMECH=TWSX_NATIVE failed rc=" << rc << " cc=" << cc
             << "; native transfer will still be attempted.";
        notes.push_back(note.str());
    }
}

bool SetFileTransferCapabilities(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    const FileTransferPlan& plan,
    std::ofstream& log,
    std::vector<std::string>& events,
    TW_UINT16& return_code,
    TW_UINT16& condition_code) {
    if (!SetOneValueCapability(
            app,
            source,
            ICAP_XFERMECH,
            TWTY_UINT16,
            TWSX_FILE,
            "ICAP_XFERMECH_FILE",
            log,
            events,
            return_code,
            condition_code)) {
        return false;
    }

    if (!SetOneValueCapability(
            app,
            source,
            ICAP_IMAGEFILEFORMAT,
            TWTY_UINT16,
            plan.format,
            "ICAP_IMAGEFILEFORMAT",
            log,
            events,
            return_code,
            condition_code)) {
        return false;
    }

    return true;
}

bool ValidateFileExistsNonzero(const std::filesystem::path& path, std::uintmax_t& size_bytes) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || !std::filesystem::is_regular_file(path, ec)) {
        size_bytes = 0;
        return false;
    }
    size_bytes = std::filesystem::file_size(path, ec);
    return !ec && size_bytes > 0;
}

WORD ReadLe16(const std::vector<unsigned char>& data, size_t offset) {
    return static_cast<WORD>(data[offset] | (data[offset + 1] << 8));
}

DWORD ReadLe32(const std::vector<unsigned char>& data, size_t offset) {
    return static_cast<DWORD>(data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) |
                              (data[offset + 3] << 24));
}

LONG ReadLe32Signed(const std::vector<unsigned char>& data, size_t offset) {
    const DWORD raw = ReadLe32(data, offset);
    LONG value = 0;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

bool ValidateBmpFile(const std::filesystem::path& path, BmpImageInfo& image, std::string& message) {
    std::uintmax_t actual_size = 0;
    if (!ValidateFileExistsNonzero(path, actual_size)) {
        message = "BMP output file was missing or zero bytes.";
        return false;
    }
    if (actual_size < 54 || actual_size > static_cast<std::uintmax_t>(std::numeric_limits<DWORD>::max())) {
        message = "BMP output file size was outside the supported range.";
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> bytes(static_cast<size_t>(actual_size));
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in || bytes.size() < 54) {
        message = "BMP output file could not be read completely.";
        return false;
    }

    if (bytes[0] != 'B' || bytes[1] != 'M') {
        message = "BMP output file did not start with a BM signature.";
        return false;
    }

    image.file_size = ReadLe32(bytes, 2);
    image.pixel_offset = ReadLe32(bytes, 10);
    image.dib_header_size = ReadLe32(bytes, 14);
    if (image.file_size != static_cast<DWORD>(actual_size)) {
        message = "BMP file size field did not match the actual file size.";
        return false;
    }
    if (image.dib_header_size < sizeof(BITMAPCOREHEADER) || image.dib_header_size > image.file_size - 14) {
        message = "BMP DIB header size was invalid.";
        return false;
    }

    const DWORD minimum_pixel_offset = 14 + image.dib_header_size;
    if (image.pixel_offset < minimum_pixel_offset || image.pixel_offset >= image.file_size) {
        message = "BMP pixel data offset was invalid.";
        return false;
    }

    if (image.dib_header_size == sizeof(BITMAPCOREHEADER)) {
        if (bytes.size() < 14 + sizeof(BITMAPCOREHEADER)) {
            message = "BMP core DIB header was truncated.";
            return false;
        }
        image.width = static_cast<LONG>(ReadLe16(bytes, 18));
        image.height = static_cast<LONG>(ReadLe16(bytes, 20));
        const WORD planes = ReadLe16(bytes, 22);
        image.bits_per_pixel = ReadLe16(bytes, 24);
        image.compression = BI_RGB;
        if (planes != 1) {
            message = "BMP color plane count was invalid.";
            return false;
        }
    } else {
        if (image.dib_header_size < sizeof(BITMAPINFOHEADER) || bytes.size() < 14 + sizeof(BITMAPINFOHEADER)) {
            message = "BMP info DIB header was truncated or unsupported.";
            return false;
        }
        image.width = ReadLe32Signed(bytes, 18);
        image.height = ReadLe32Signed(bytes, 22);
        const WORD planes = ReadLe16(bytes, 26);
        image.bits_per_pixel = ReadLe16(bytes, 28);
        image.compression = ReadLe32(bytes, 30);
        if (planes != 1) {
            message = "BMP color plane count was invalid.";
            return false;
        }
    }

    if (image.width <= 0 || image.height == 0) {
        message = "BMP dimensions were invalid.";
        return false;
    }
    if (image.bits_per_pixel != 1 && image.bits_per_pixel != 4 && image.bits_per_pixel != 8 &&
        image.bits_per_pixel != 16 && image.bits_per_pixel != 24 && image.bits_per_pixel != 32) {
        message = "BMP bit depth was unsupported.";
        return false;
    }
    if (image.compression != BI_RGB) {
        message = "BMP compression was not BI_RGB.";
        return false;
    }
    return true;
}

bool ValidateBmpFileHeader(const std::filesystem::path& path) {
    BmpImageInfo image;
    std::string message;
    return ValidateBmpFile(path, image, message);
}

bool CalculateDibDataOffset(const TW_UINT8* data, SIZE_T dib_size, DWORD& data_offset, std::string& message) {
    if (!data || dib_size < sizeof(DWORD)) {
        message = "DIB handle was empty.";
        return false;
    }

    DWORD header_size = 0;
    std::memcpy(&header_size, data, sizeof(header_size));
    if (header_size < sizeof(BITMAPCOREHEADER) || header_size > dib_size) {
        message = "DIB header size was invalid.";
        return false;
    }

    if (header_size == sizeof(BITMAPCOREHEADER)) {
        if (dib_size < sizeof(BITMAPCOREHEADER)) {
            message = "DIB core header was truncated.";
            return false;
        }
        const auto* core = reinterpret_cast<const BITMAPCOREHEADER*>(data);
        DWORD palette_entries = 0;
        if (core->bcBitCount <= 8) {
            palette_entries = 1u << core->bcBitCount;
        }
        data_offset = header_size + (palette_entries * sizeof(RGBTRIPLE));
    } else {
        if (dib_size < sizeof(BITMAPINFOHEADER)) {
            message = "DIB info header was truncated.";
            return false;
        }
        const auto* info = reinterpret_cast<const BITMAPINFOHEADER*>(data);
        DWORD palette_entries = info->biClrUsed;
        if (palette_entries == 0 && info->biBitCount <= 8) {
            palette_entries = 1u << info->biBitCount;
        }

        DWORD mask_bytes = 0;
        if (info->biCompression == BI_BITFIELDS && header_size == sizeof(BITMAPINFOHEADER)) {
            mask_bytes = 3u * sizeof(DWORD);
        }
#ifdef BI_ALPHABITFIELDS
        if (info->biCompression == BI_ALPHABITFIELDS && header_size == sizeof(BITMAPINFOHEADER)) {
            mask_bytes = 4u * sizeof(DWORD);
        }
#endif
        data_offset = header_size + mask_bytes + (palette_entries * sizeof(RGBQUAD));
    }

    if (data_offset > dib_size) {
        message = "DIB pixel data offset exceeded handle size.";
        return false;
    }
    return true;
}

bool SaveNativeDibAsBmp(
    TW_HANDLE dib_handle,
    const std::filesystem::path& output_path,
    std::string& message,
    std::uintmax_t& size_bytes,
    BmpImageInfo& image) {
    if (!dib_handle) {
        message = "Native transfer returned a null DIB handle.";
        return false;
    }

    const HGLOBAL global = static_cast<HGLOBAL>(dib_handle);
    const SIZE_T dib_size = GlobalSize(global);
    if (dib_size == 0 || dib_size > static_cast<SIZE_T>(std::numeric_limits<DWORD>::max() - sizeof(BITMAPFILEHEADER))) {
        message = "Native DIB handle size was zero or too large for BMP.";
        return false;
    }

    const auto* data = static_cast<const TW_UINT8*>(GlobalLock(global));
    if (!data) {
        message = "GlobalLock failed for native DIB handle.";
        return false;
    }

    DWORD dib_data_offset = 0;
    if (!CalculateDibDataOffset(data, dib_size, dib_data_offset, message)) {
        GlobalUnlock(global);
        return false;
    }

    BITMAPFILEHEADER file_header{};
    file_header.bfType = 0x4d42;
    file_header.bfOffBits = static_cast<DWORD>(sizeof(BITMAPFILEHEADER) + dib_data_offset);
    file_header.bfSize = static_cast<DWORD>(sizeof(BITMAPFILEHEADER) + dib_size);

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        GlobalUnlock(global);
        message = "Could not open BMP output file for writing.";
        return false;
    }
    out.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(dib_size));
    out.close();
    GlobalUnlock(global);

    if (!out) {
        message = "Writing BMP output file failed.";
        return false;
    }

    if (!ValidateFileExistsNonzero(output_path, size_bytes)) {
        message = "BMP output file was missing or zero bytes after write.";
        return false;
    }
    if (!ValidateBmpFile(output_path, image, message)) {
        return false;
    }
    return true;
}

bool SetupFileTransfer(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    const std::filesystem::path& output_path,
    TW_UINT16 format,
    std::ofstream& log,
    std::vector<std::string>& events,
    TW_UINT16& return_code,
    TW_UINT16& condition_code) {
    TW_SETUPFILEXFER setup{};
    SetTwainString(setup.FileName, output_path.string().c_str());
    setup.Format = format;
    setup.VRefNum = 0;

    const TW_UINT16 rc = DSM_Entry(
        &app,
        &source,
        DG_CONTROL,
        DAT_SETUPFILEXFER,
        MSG_SET,
        reinterpret_cast<TW_MEMREF>(&setup));
    return_code = rc;
    log << "DAT_SETUPFILEXFER MSG_SET rc=" << rc << " path=\"" << output_path.string() << "\"\n";
    if (rc != TWRC_SUCCESS) {
        const TwainStatus status = ReadStatus(&app, &source);
        condition_code = status.condition_code;
        AddEvent(events, log, "setup_file_xfer_rc_" + std::to_string(rc) + "_cc_" +
                               std::to_string(status.condition_code));
        return false;
    }

    condition_code = TWCC_SUCCESS;
    AddEvent(events, log, "setup_file_xfer_success");
    return true;
}

bool EnableSource(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    HWND hwnd,
    bool show_ui,
    std::ofstream& log,
    std::vector<std::string>& events,
    TW_UINT16& return_code,
    TW_UINT16& condition_code) {
    TW_USERINTERFACE ui{};
    ui.ShowUI = show_ui ? TRUE : FALSE;
    ui.ModalUI = FALSE;
    ui.hParent = reinterpret_cast<TW_HANDLE>(hwnd);

    const TW_UINT16 rc = DSM_Entry(
        &app,
        &source,
        DG_CONTROL,
        DAT_USERINTERFACE,
        MSG_ENABLEDS,
        reinterpret_cast<TW_MEMREF>(&ui));
    return_code = rc;
    log << "MSG_ENABLEDS rc=" << rc << " show_ui=" << show_ui << "\n";
    if (rc != TWRC_SUCCESS) {
        const TwainStatus status = ReadStatus(&app, &source);
        condition_code = status.condition_code;
        AddEvent(events, log, "enable_ds_rc_" + std::to_string(rc) + "_cc_" +
                               std::to_string(status.condition_code));
        return false;
    }

    condition_code = TWCC_SUCCESS;
    AddEvent(events, log, "enable_ds_success");
    return true;
}

void DisableSourceIfEnabled(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    bool& enabled,
    bool& close_ds_ok_seen,
    std::ofstream& log,
    std::vector<std::string>& events) {
    if (!enabled || close_ds_ok_seen) {
        return;
    }

    TW_USERINTERFACE ui{};
    const TW_UINT16 rc = DSM_Entry(
        &app,
        &source,
        DG_CONTROL,
        DAT_USERINTERFACE,
        MSG_DISABLEDS,
        reinterpret_cast<TW_MEMREF>(&ui));
    log << "MSG_DISABLEDS rc=" << rc << "\n";
    if (rc == TWRC_SUCCESS) {
        AddEvent(events, log, "disable_ds_success");
        enabled = false;
    } else {
        const TwainStatus status = ReadStatus(&app, &source);
        AddEvent(events, log, "disable_ds_rc_" + std::to_string(rc) + "_cc_" +
                               std::to_string(status.condition_code));
    }
}

void EndPendingTransfers(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    std::ofstream& log,
    std::vector<std::string>& events) {
    TW_PENDINGXFERS pending{};
    const TW_UINT16 rc = DSM_Entry(
        &app,
        &source,
        DG_CONTROL,
        DAT_PENDINGXFERS,
        MSG_ENDXFER,
        reinterpret_cast<TW_MEMREF>(&pending));
    log << "MSG_ENDXFER rc=" << rc << " pending_count=" << pending.Count << "\n";
    if (rc == TWRC_SUCCESS) {
        AddEvent(events, log, "endxfer_success_pending_" + std::to_string(pending.Count));
    } else {
        const TwainStatus status = ReadStatus(&app, &source);
        AddEvent(events, log, "endxfer_rc_" + std::to_string(rc) + "_cc_" +
                               std::to_string(status.condition_code));
    }
}

bool TransferNativeImage(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    CaptureAttemptResult& result,
    std::ofstream& log,
    std::vector<std::string>& events) {
    TW_HANDLE dib_handle = nullptr;
    const TW_UINT16 rc = DSM_Entry(
        &app,
        &source,
        DG_IMAGE,
        DAT_IMAGENATIVEXFER,
        MSG_GET,
        reinterpret_cast<TW_MEMREF>(&dib_handle));
    result.return_code = rc;
    result.transfer_mechanism = "native";
    result.failure_stage = "DAT_IMAGENATIVEXFER";
    log << "DAT_IMAGENATIVEXFER MSG_GET rc=" << rc << " handle=" << dib_handle << "\n";

    if (rc != TWRC_XFERDONE) {
        const TwainStatus status = ReadStatus(&app, &source);
        result.condition_code = status.condition_code;
        result.error_code = (rc == TWRC_CANCEL) ? "user_cancelled" : "transfer_failed";
        result.message = "Native image transfer did not complete.";
        AddEvent(events, log, "native_transfer_rc_" + std::to_string(rc) + "_cc_" +
                               std::to_string(status.condition_code));
        if (dib_handle) {
            FreeTwainHandle(dib_handle);
        }
        return false;
    }

    result.output_path = UniqueCapturePath("biometrika_twain_capture", ".bmp");
    std::string save_message;
    if (!SaveNativeDibAsBmp(dib_handle, result.output_path, save_message, result.output_size_bytes, result.image)) {
        result.condition_code = TWCC_SUCCESS;
        result.error_code = "save_failed";
        result.message = save_message;
        AddEvent(events, log, "native_save_failed");
        FreeTwainHandle(dib_handle);
        return false;
    }

    FreeTwainHandle(dib_handle);
    result.condition_code = TWCC_SUCCESS;
    result.ok = true;
    result.message = "Native transfer completed.";
    AddEvent(events, log, "native_transfer_xferdone");
    AddEvent(events, log, "native_bmp_saved_" + std::to_string(result.output_size_bytes));
    return true;
}

bool TransferFileImage(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    const FileTransferPlan& plan,
    const std::filesystem::path& output_path,
    CaptureAttemptResult& result,
    std::ofstream& log,
    std::vector<std::string>& events) {
    const TW_UINT16 rc = DSM_Entry(
        &app,
        &source,
        DG_IMAGE,
        DAT_IMAGEFILEXFER,
        MSG_GET,
        nullptr);
    result.return_code = rc;
    result.transfer_mechanism = "file";
    result.failure_stage = "DAT_IMAGEFILEXFER";
    log << "DAT_IMAGEFILEXFER MSG_GET rc=" << rc << "\n";
    if (rc != TWRC_XFERDONE) {
        const TwainStatus status = ReadStatus(&app, &source);
        result.condition_code = status.condition_code;
        result.error_code = (rc == TWRC_CANCEL) ? "user_cancelled" : "transfer_failed";
        result.message = "File image transfer did not complete.";
        AddEvent(events, log, "file_transfer_rc_" + std::to_string(rc) + "_cc_" +
                               std::to_string(status.condition_code));
        return false;
    }

    std::uintmax_t size_bytes = 0;
    if (!ValidateFileExistsNonzero(output_path, size_bytes)) {
        result.condition_code = TWCC_SUCCESS;
        result.error_code = "save_failed";
        result.message = "File transfer completed, but the output file was missing or zero bytes.";
        AddEvent(events, log, "file_transfer_output_missing_or_empty");
        return false;
    }

    if (plan.format == TWFF_BMP && !ValidateBmpFileHeader(output_path)) {
        result.condition_code = TWCC_SUCCESS;
        result.error_code = "save_failed";
        result.message = "File transfer BMP output did not start with BM header.";
        AddEvent(events, log, "file_transfer_bmp_header_invalid");
        return false;
    }

    result.condition_code = TWCC_SUCCESS;
    result.ok = true;
    result.output_path = output_path;
    result.output_size_bytes = size_bytes;
    result.message = "File transfer completed.";
    AddEvent(events, log, "file_transfer_xferdone");
    AddEvent(events, log, "file_output_saved_" + std::to_string(result.output_size_bytes));
    return true;
}

bool WaitForTransferReadyAndTransfer(
    TW_IDENTITY& app,
    TW_IDENTITY& source,
    DWORD timeout_ms,
    DWORD settle_after_enable_ms,
    const std::chrono::steady_clock::time_point& enable_time,
    const std::string& transfer_mechanism,
    const FileTransferPlan* file_plan,
    const std::filesystem::path& file_output_path,
    CaptureAttemptResult& result,
    std::ofstream& log,
    std::vector<std::string>& events) {
    const auto wait_start = enable_time;
    bool xferready_seen = false;
    bool settle_elapsed_event_added = false;
    bool xferready_before_settle_event_added = false;

    result.settle_after_enable_ms = settle_after_enable_ms;

    auto add_settle_elapsed_event = [&]() {
        if (!settle_elapsed_event_added) {
            AddEvent(events, log, "settle_after_enable_elapsed");
            settle_elapsed_event_added = true;
        }
    };

    auto perform_transfer = [&]() -> bool {
        add_settle_elapsed_event();
        if (transfer_mechanism == "native") {
            return TransferNativeImage(app, source, result, log, events);
        }
        if (file_plan) {
            return TransferFileImage(app, source, *file_plan, file_output_path, result, log, events);
        }
        result.error_code = "unsupported_transfer";
        result.message = "File transfer was requested without a safe file transfer plan.";
        result.failure_stage = "transfer_plan";
        result.return_code = TWRC_FAILURE;
        result.condition_code = TWCC_BADVALUE;
        return false;
    };

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - wait_start);
        if (elapsed.count() >= static_cast<long long>(timeout_ms)) {
            result.error_code = "timeout";
            result.message = "Timed out waiting for MSG_XFERREADY.";
            result.failure_stage = "message_loop";
            result.return_code = TWRC_FAILURE;
            result.condition_code = TWCC_OPERATIONERROR;
            AddEvent(events, log, "timeout_waiting_for_xferready");
            return false;
        }

        const auto settle_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - enable_time);
        const bool settle_complete = settle_elapsed.count() >= static_cast<long long>(settle_after_enable_ms);
        if (settle_complete) {
            add_settle_elapsed_event();
        }
        if (xferready_seen && settle_complete) {
            return perform_transfer();
        }

        MSG msg{};
        bool saw_message = false;
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            saw_message = true;
            TW_EVENT tw_event{};
            tw_event.pEvent = reinterpret_cast<TW_MEMREF>(&msg);
            tw_event.TWMessage = MSG_NULL;

            const TW_UINT16 rc = DSM_Entry(
                &app,
                &source,
                DG_CONTROL,
                DAT_EVENT,
                MSG_PROCESSEVENT,
                reinterpret_cast<TW_MEMREF>(&tw_event));

            if (rc == TWRC_DSEVENT) {
                switch (tw_event.TWMessage) {
                    case MSG_XFERREADY:
                        AddEvent(events, log, "MSG_XFERREADY");
                        xferready_seen = true;
                        {
                            const auto xferready_settle_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - enable_time);
                            if (xferready_settle_elapsed.count() >= static_cast<long long>(settle_after_enable_ms)) {
                                return perform_transfer();
                            }
                        }
                        if (!xferready_before_settle_event_added) {
                            result.xferready_waited_for_settle = true;
                            AddEvent(events, log, "xferready_seen_before_settle_elapsed");
                            xferready_before_settle_event_added = true;
                        }
                        break;
                    case MSG_CLOSEDSREQ:
                        AddEvent(events, log, "MSG_CLOSEDSREQ");
                        result.error_code = "user_cancelled";
                        result.message = "Source requested close before transfer.";
                        result.failure_stage = "message_loop";
                        result.return_code = TWRC_CANCEL;
                        result.condition_code = TWCC_SUCCESS;
                        return false;
                    case MSG_CLOSEDSOK:
                        AddEvent(events, log, "MSG_CLOSEDSOK");
                        result.error_code = "user_cancelled";
                        result.message = "Source reported it is safe to close before transfer.";
                        result.failure_stage = "message_loop";
                        result.return_code = TWRC_CANCEL;
                        result.condition_code = TWCC_SUCCESS;
                        return false;
                    case MSG_DEVICEEVENT:
                        AddEvent(events, log, "MSG_DEVICEEVENT");
                        break;
                    case MSG_NULL:
                        break;
                    default:
                        AddEvent(events, log, "twain_event_msg_" + std::to_string(tw_event.TWMessage));
                        break;
                }
            } else if (rc == TWRC_NOTDSEVENT) {
                TranslateMessage(&msg);
                DispatchMessageA(&msg);
            } else if (rc != TWRC_SUCCESS) {
                const TwainStatus status = ReadStatus(&app, &source);
                AddEvent(events, log, "process_event_rc_" + std::to_string(rc) + "_cc_" +
                                       std::to_string(status.condition_code));
                TranslateMessage(&msg);
                DispatchMessageA(&msg);
            }
        }

        if (!saw_message) {
            const DWORD remaining = timeout_ms - static_cast<DWORD>(elapsed.count());
            DWORD wait_ms = std::min<DWORD>(remaining, 50);
            if (xferready_seen && !settle_complete) {
                const DWORD remaining_settle = settle_after_enable_ms - static_cast<DWORD>(settle_elapsed.count());
                wait_ms = std::min<DWORD>(wait_ms, remaining_settle);
            }
            MsgWaitForMultipleObjects(0, nullptr, FALSE, wait_ms, QS_ALLINPUT);
        }
    }
}

CaptureAttemptResult RunSingleCaptureAttempt(
    TW_IDENTITY& app,
    const TW_IDENTITY& source_identity,
    HWND hwnd,
    bool show_ui,
    DWORD timeout_ms,
    DWORD settle_after_enable_ms,
    const std::string& transfer_mechanism,
    const FileTransferPlan* file_plan,
    std::ofstream& log,
    std::vector<std::string>& events,
    std::vector<std::string>& notes) {
    CaptureAttemptResult result;
    result.transfer_mechanism = transfer_mechanism;
    result.settle_after_enable_ms = settle_after_enable_ms;
    TW_IDENTITY source = source_identity;
    bool source_open = false;
    bool enabled = false;
    bool close_ds_ok_seen = false;

    TW_UINT16 rc = DSM_Entry(
        &app,
        nullptr,
        DG_CONTROL,
        DAT_IDENTITY,
        MSG_OPENDS,
        reinterpret_cast<TW_MEMREF>(&source));
    log << "MSG_OPENDS rc=" << rc << " mechanism=" << transfer_mechanism << "\n";
    if (rc != TWRC_SUCCESS) {
        const TwainStatus status = ReadStatus(&app, nullptr);
        result.error_code = "open_source_failed";
        result.message = "TWAIN DSM found the Biometrika source, but MSG_OPENDS failed.";
        result.return_code = rc;
        result.condition_code = status.condition_code;
        result.failure_stage = "MSG_OPENDS";
        AddEvent(events, log, "open_ds_rc_" + std::to_string(rc) + "_cc_" + std::to_string(status.condition_code));
        return result;
    }
    source_open = true;
    AddEvent(events, log, "open_ds_success_" + transfer_mechanism);

    ConfigureXferCountOne(app, source, log, events, notes);

    std::filesystem::path file_output_path;
    if (transfer_mechanism == "native") {
        PreferNativeTransfer(app, source, log, events, notes);
    } else if (file_plan && file_plan->supported) {
        TW_UINT16 set_rc = 0;
        TW_UINT16 set_cc = 0;
        if (!SetFileTransferCapabilities(app, source, *file_plan, log, events, set_rc, set_cc)) {
            result.error_code = "unsupported_transfer";
            result.message = "File transfer was discoverable, but required file transfer capabilities could not be set.";
            result.return_code = set_rc;
            result.condition_code = set_cc;
            result.failure_stage = "file_transfer_capabilities";
            TW_UINT16 close_rc = DSM_Entry(
                &app,
                &source,
                DG_CONTROL,
                DAT_IDENTITY,
                MSG_CLOSEDS,
                reinterpret_cast<TW_MEMREF>(&source));
            log << "MSG_CLOSEDS rc=" << close_rc << "\n";
            return result;
        }

        file_output_path = UniqueCapturePath("capture_test_file", file_plan->extension);
        TW_UINT16 setup_rc = 0;
        TW_UINT16 setup_cc = 0;
        if (!SetupFileTransfer(app, source, file_output_path, file_plan->format, log, events, setup_rc, setup_cc)) {
            result.error_code = "unsupported_transfer";
            result.message = "DAT_SETUPFILEXFER failed for the planned output file.";
            result.return_code = setup_rc;
            result.condition_code = setup_cc;
            result.failure_stage = "DAT_SETUPFILEXFER";
            TW_UINT16 close_rc = DSM_Entry(
                &app,
                &source,
                DG_CONTROL,
                DAT_IDENTITY,
                MSG_CLOSEDS,
                reinterpret_cast<TW_MEMREF>(&source));
            log << "MSG_CLOSEDS rc=" << close_rc << "\n";
            return result;
        }
    } else {
        result.error_code = "unsupported_transfer";
        result.message = "Unsupported transfer mechanism requested.";
        result.return_code = TWRC_FAILURE;
        result.condition_code = TWCC_BADVALUE;
        result.failure_stage = "transfer_mechanism";
        TW_UINT16 close_rc = DSM_Entry(
            &app,
            &source,
            DG_CONTROL,
            DAT_IDENTITY,
            MSG_CLOSEDS,
            reinterpret_cast<TW_MEMREF>(&source));
        log << "MSG_CLOSEDS rc=" << close_rc << "\n";
        return result;
    }

    TW_UINT16 enable_rc = 0;
    TW_UINT16 enable_cc = 0;
    if (!EnableSource(app, source, hwnd, show_ui, log, events, enable_rc, enable_cc)) {
        result.error_code = "enable_source_failed";
        result.message = "DAT_USERINTERFACE / MSG_ENABLEDS failed.";
        result.return_code = enable_rc;
        result.condition_code = enable_cc;
        result.failure_stage = "MSG_ENABLEDS";
    } else {
        enabled = true;
        const auto enable_time = std::chrono::steady_clock::now();
        AddEvent(events, log, "settle_after_enable_start_" + std::to_string(settle_after_enable_ms) + "ms");
        if (!WaitForTransferReadyAndTransfer(
                app,
                source,
                timeout_ms,
                settle_after_enable_ms,
                enable_time,
                transfer_mechanism,
                file_plan,
                file_output_path,
                result,
                log,
                events)) {
            log << "capture attempt failed stage=" << result.failure_stage << " error=" << result.error_code << "\n";
        }
        if (result.failure_stage == "message_loop" && result.message.find("safe to close") != std::string::npos) {
            close_ds_ok_seen = true;
        }
        const bool transfer_phase =
            result.ok || result.failure_stage == "DAT_IMAGENATIVEXFER" || result.failure_stage == "DAT_IMAGEFILEXFER";
        if (transfer_phase) {
            EndPendingTransfers(app, source, log, events);
        }
    }

    DisableSourceIfEnabled(app, source, enabled, close_ds_ok_seen, log, events);

    if (source_open) {
        TW_UINT16 close_rc = DSM_Entry(
            &app,
            &source,
            DG_CONTROL,
            DAT_IDENTITY,
            MSG_CLOSEDS,
            reinterpret_cast<TW_MEMREF>(&source));
        log << "MSG_CLOSEDS rc=" << close_rc << "\n";
        if (close_rc == TWRC_SUCCESS) {
            AddEvent(events, log, "close_ds_success_" + transfer_mechanism);
        } else {
            const TwainStatus status = ReadStatus(&app, nullptr);
            AddEvent(events, log, "close_ds_rc_" + std::to_string(close_rc) + "_cc_" +
                                   std::to_string(status.condition_code));
        }
    }

    return result;
}

std::string RunCaptureTest(
    TW_IDENTITY& app,
    const SourceRecord& source,
    HWND hwnd,
    bool show_ui,
    DWORD timeout_ms,
    DWORD settle_after_enable_ms,
    const std::string& module_path,
    const std::chrono::steady_clock::time_point& start,
    std::ofstream& log,
    std::vector<std::string>& events,
    std::vector<std::string>& notes,
    int& exit_code) {
    AddEvent(events, log, "capture_start");
    std::error_code ec;
    const std::filesystem::path output_dir = CaptureOutputDir();
    if (!std::filesystem::exists(output_dir, ec) || !std::filesystem::is_directory(output_dir, ec)) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        exit_code = 1;
        return CaptureFailureJson(
            "invalid_output_dir",
            "Output directory is missing or is not a directory.",
            show_ui,
            duration.count(),
            events,
            CaptureDiagnosticsJson("validate_output_dir", TWRC_FAILURE, TWCC_BADVALUE, module_path));
    }

    CaptureAttemptResult native_result = RunSingleCaptureAttempt(
        app,
        source.identity,
        hwnd,
        show_ui,
        timeout_ms,
        settle_after_enable_ms,
        "native",
        nullptr,
        log,
        events,
        notes);

    if (native_result.ok) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        exit_code = 0;
        return CaptureSuccessJson(
            show_ui,
            "native",
            native_result.output_path,
            native_result.output_size_bytes,
            native_result.image,
            native_result.settle_after_enable_ms,
            native_result.xferready_waited_for_settle,
            duration.count(),
            events);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    exit_code = 1;
    return CaptureFailureJson(
        native_result.error_code,
        native_result.message.empty() ? "Capture test failed." : native_result.message,
        show_ui,
        duration.count(),
        events,
        CaptureDiagnosticsJson(
            native_result.failure_stage,
            native_result.return_code,
            native_result.condition_code,
            module_path,
            {{"transfer_mechanism", JsonString(native_result.transfer_mechanism)}}));
}

std::string UsageJson() {
    return FailureJson(
        kStatusMode,
        "unsupported_platform",
        "Usage: biometrika_twain_capture.exe --status OR biometrika_twain_capture.exe --capture --output-dir <dir> [--show-ui true|false] [--timeout-ms N] [--settle-after-enable-ms N]",
        DiagnosticsJson(TWRC_FAILURE, TWCC_BADVALUE, LoadedTwainModulePath(), "parse_args"));
}

bool ParseBool(const std::string& value, bool& out) {
    if (value == "true") {
        out = true;
        return true;
    }
    if (value == "false") {
        out = false;
        return true;
    }
    return false;
}

bool ParseTimeoutMs(const std::string& value, DWORD& out) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed == 0 || parsed > 600000) {
        return false;
    }
    out = static_cast<DWORD>(parsed);
    return true;
}

bool ParseSettleAfterEnableMs(const std::string& value, DWORD& out) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed > kMaxSettleAfterEnableMs) {
        return false;
    }
    out = static_cast<DWORD>(parsed);
    return true;
}

bool ValidateOutputDir(
    const std::filesystem::path& requested,
    std::filesystem::path& resolved,
    std::string& message) {
    if (requested.empty()) {
        message = "Capture mode requires --output-dir.";
        return false;
    }

    std::error_code ec;
    resolved = std::filesystem::weakly_canonical(requested, ec);
    if (ec) {
        resolved = std::filesystem::absolute(requested, ec);
    }
    if (ec || resolved.empty()) {
        message = "Output directory path could not be resolved.";
        return false;
    }
    if (!std::filesystem::exists(resolved, ec) || !std::filesystem::is_directory(resolved, ec)) {
        message = "Output directory is missing or is not a directory.";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(_M_IX86)
    std::cout << FailureJson(
                     kStatusMode,
                     "unsupported_platform",
                     "This TWAIN capture helper must be compiled and run as x86.",
                     DiagnosticsJson(TWRC_FAILURE, TWCC_BADVALUE, LoadedTwainModulePath(), "architecture"))
              << "\n";
    return 2;
#else
    CliOptions options;
    bool explicit_mode = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--status") {
            if (explicit_mode && options.mode != kStatusMode) {
                std::cout << UsageJson() << "\n";
                return 2;
            }
            options.mode = kStatusMode;
            explicit_mode = true;
        } else if (arg == "--capture") {
            if (explicit_mode && options.mode != kCaptureMode) {
                std::cout << UsageJson() << "\n";
                return 2;
            }
            options.mode = kCaptureMode;
            explicit_mode = true;
        } else if (arg == "--output-dir") {
            if (i + 1 >= argc) {
                std::cout << FailureJson(
                                 kCaptureMode,
                                 "invalid_output_dir",
                                 "Missing value for --output-dir.",
                                 DiagnosticsJson(TWRC_FAILURE, TWCC_BADVALUE, LoadedTwainModulePath(), "parse_args"))
                          << "\n";
                return 2;
            }
            options.output_dir = argv[++i];
        } else if (arg == "--show-ui") {
            if (i + 1 >= argc) {
                std::cout << UsageJson() << "\n";
                return 2;
            }
            const std::string value = argv[++i];
            if (!ParseBool(value, options.show_ui)) {
                std::cout << UsageJson() << "\n";
                return 2;
            }
        } else if (arg == "--timeout-ms") {
            if (i + 1 >= argc) {
                std::cout << UsageJson() << "\n";
                return 2;
            }
            if (!ParseTimeoutMs(argv[++i], options.timeout_ms)) {
                std::cout << UsageJson() << "\n";
                return 2;
            }
        } else if (arg == "--settle-after-enable-ms") {
            if (i + 1 >= argc) {
                std::cout << UsageJson() << "\n";
                return 2;
            }
            if (!ParseSettleAfterEnableMs(argv[++i], options.settle_after_enable_ms)) {
                std::cout << UsageJson() << "\n";
                return 2;
            }
            options.settle_after_enable_explicit = true;
        } else {
            std::cout << UsageJson() << "\n";
            return 2;
        }
    }

    if (options.mode == kCaptureMode && options.show_ui && !options.settle_after_enable_explicit) {
        options.settle_after_enable_ms = 0;
    }

    const auto start = std::chrono::steady_clock::now();
    std::ofstream log = OpenLog(options.mode);
    std::vector<std::string> notes;
    if (options.mode == kCaptureMode) {
        notes.push_back(options.show_ui ? "TWAIN source UI requested." : "TWAIN source UI suppressed by request.");
    } else {
        notes.push_back("No image acquisition requested.");
    }
    std::vector<std::string> events;

    if (options.mode == kCaptureMode) {
        std::filesystem::path resolved_output_dir;
        std::string output_dir_message;
        if (!ValidateOutputDir(options.output_dir, resolved_output_dir, output_dir_message)) {
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            const std::string json = CaptureFailureJson(
                "invalid_output_dir",
                output_dir_message,
                options.show_ui,
                duration.count(),
                events,
                CaptureDiagnosticsJson("validate_output_dir", TWRC_FAILURE, TWCC_BADVALUE, LoadedTwainModulePath()));
            log << "json=" << json << "\n";
            std::cout << json << "\n";
            return 2;
        }
        g_capture_output_dir = resolved_output_dir;
        log << "output_dir=" << g_capture_output_dir.string() << "\n";
    }

    HWND hwnd = CreateHiddenParentWindow(notes);
    if (!hwnd) {
        notes.push_back("Continuing with NULL parent window.");
    }

    TW_IDENTITY app = MakeAppIdentity();
    std::string failure_json;
    bool dsm_open = OpenDsm(app, hwnd, log, failure_json, options.mode);
    if (!dsm_open) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        failure_json = FailureJson(
            options.mode,
            "twain_unavailable",
            "TWAIN DSM MSG_OPENDSM failed.",
            DiagnosticsJson(TWRC_FAILURE, TWCC_OPERATIONERROR, LoadedTwainModulePath(), "MSG_OPENDSM"),
            duration.count(),
            events);
        if (hwnd) {
            DestroyWindow(hwnd);
        }
        log << "json=" << failure_json << "\n";
        std::cout << failure_json << "\n";
        return 1;
    }

    const std::string module_path = LoadedTwainModulePath();
    if (!module_path.empty()) {
        notes.push_back("twain_32.dll loaded from " + module_path);
        log << "twain_32_module=" << module_path << "\n";
    }

    std::vector<SourceRecord> sources;
    bool listed = ListSources(app, log, sources, failure_json, options.mode);
    const SourceRecord* exact_candidate = listed ? FindExactBiometrikaSource(sources) : nullptr;

    std::string json;
    int exit_code = 0;

    if (!listed) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        json = FailureJson(
            options.mode,
            "twain_unavailable",
            "TWAIN source enumeration failed.",
            DiagnosticsJson(TWRC_FAILURE, TWCC_OPERATIONERROR, module_path, "source_enumeration"),
            duration.count(),
            events);
        exit_code = 1;
    } else if (options.mode == kCaptureMode) {
        if (exact_candidate == nullptr) {
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            json = CaptureFailureJson(
                "source_not_found",
                "TWAIN source enumeration worked, but exact source 'TWAIN Biometrika Driver' was not found.",
                options.show_ui,
                duration.count(),
                events,
                CaptureDiagnosticsJson("source_enumeration", TWRC_ENDOFLIST, TWCC_SUCCESS, module_path));
            exit_code = 1;
        } else {
            AddEvent(events, log, "exact_source_found");
            json = RunCaptureTest(
                app,
                *exact_candidate,
                hwnd,
                options.show_ui,
                options.timeout_ms,
                options.settle_after_enable_ms,
                module_path,
                start,
                log,
                events,
                notes,
                exit_code);
        }
    } else {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        if (exact_candidate == nullptr) {
            json = FailureJson(
                kStatusMode,
                "source_not_found",
                "TWAIN source enumeration worked, but exact source 'TWAIN Biometrika Driver' was not found.",
                DiagnosticsJson(TWRC_ENDOFLIST, TWCC_SUCCESS, module_path, "source_enumeration"),
                duration.count(),
                events);
            exit_code = 1;
        } else {
            json = StatusSuccessJson(sources, exact_candidate, module_path, duration.count(), events);
        }
    }

    TW_UINT16 close_dsm_rc = DSM_Entry(&app, nullptr, DG_CONTROL, DAT_PARENT, MSG_CLOSEDSM, reinterpret_cast<TW_MEMREF>(&hwnd));
    log << "MSG_CLOSEDSM rc=" << close_dsm_rc << "\n";

    if (hwnd) {
        DestroyWindow(hwnd);
    }

    log << "json=" << json << "\n";
    std::cout << json << "\n";
    return exit_code;
#endif
}
