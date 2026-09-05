/// Launch-time inputs for the app.
///
/// Both values are supplied via `--dart-define` so the same mechanism works for
/// desktop (`flutter run -d macos`) and web (`flutter build web`) builds:
///
///   flutter run -d macos \
///     --dart-define=SEGFORGE_IMAGE_URL=https://example.com/cat.png \
///     --dart-define=SEGFORGE_SESSION_ID=my-session
///
/// `run.sh FE` forwards `SEGFORGE_IMAGE_URL` / `SEGFORGE_SESSION_ID` from the
/// environment, so in practice you export them instead of typing the flags.
///
/// [sessionId] is optional: when it is absent the backend allocates a session on
/// the first `/upload` and returns its id, which the app then adopts.
class LaunchConfig {
  const LaunchConfig._();

  static const String _imageUrl = String.fromEnvironment('SEGFORGE_IMAGE_URL');
  static const String _sessionId = String.fromEnvironment('SEGFORGE_SESSION_ID');

  /// Image to segment, or null if none was supplied at launch.
  static String? get imageUrl => _imageUrl.isEmpty ? null : _imageUrl;

  /// Session to work in, or null to let the backend allocate one on upload.
  static String? get sessionId => _sessionId.isEmpty ? null : _sessionId;

  /// Where the SegForge backend lives.
  ///
  /// 8401 keeps SegForge clear of DoubleNaught, which serves its own FastAPI
  /// backend on 8400, so the two can run side by side. `run.sh BE` passes the
  /// matching `--port`; keep the two in step.
  ///
  /// `127.0.0.1` rather than `localhost` on purpose: on macOS `localhost` can
  /// resolve to the IPv6 `::1` while uvicorn is bound to IPv4, which surfaces
  /// as a connection refusal.
  static const String backendUrl = String.fromEnvironment(
    'SEGFORGE_BACKEND_URL',
    defaultValue: 'http://127.0.0.1:8401',
  );
}
