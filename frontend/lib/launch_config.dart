import 'env/platform_env.dart'
    if (dart.library.io) 'env/platform_env_io.dart';

/// Launch-time inputs for the app.
///
/// Each value is looked up in two places, in order:
///
/// 1. The **process environment** — a runtime lookup, so a parent process can
///    hand a prebuilt bundle different inputs on every launch.
/// 2. A **`--dart-define`** value, which Dart folds in at compile time and
///    which therefore acts as the built-in default for that binary.
///
/// The environment has to come first because `--dart-define` cannot be varied
/// per launch: `String.fromEnvironment` is a const constructor evaluated when
/// the binary is built, not when it starts. DoubleNaught's Seg Forge node
/// spawns this app with `Process.start(..., environment: {...})` and depends on
/// the runtime path; `run.sh FE` uses the compile-time path.
///
///   flutter run -d macos \
///     --dart-define=SEGFORGE_IMAGE_URL=https://example.com/cat.png \
///     --dart-define=SEGFORGE_SESSION_ID=my-session
///
///   SEGFORGE_IMAGE_URL=... SEGFORGE_SESSION_ID=... ./SegForge.app/Contents/MacOS/frontend
///
/// [sessionId] is optional: when it is absent the backend allocates a session on
/// the first `/upload` and returns its id, which the app then adopts.
class LaunchConfig {
  const LaunchConfig._();

  static const String _imageUrl = String.fromEnvironment('SEGFORGE_IMAGE_URL');
  static const String _sessionId = String.fromEnvironment('SEGFORGE_SESSION_ID');
  static const String _backendUrl = String.fromEnvironment(
    'SEGFORGE_BACKEND_URL',
    defaultValue: 'http://127.0.0.1:8401',
  );

  /// Environment first, compile-time default second, null if neither is set.
  static String? _resolve(String name, String compiled) {
    final env = platformEnvironment[name];
    if (env != null && env.isNotEmpty) return env;
    return compiled.isEmpty ? null : compiled;
  }

  /// Image to segment, or null if none was supplied at launch.
  static String? get imageUrl => _resolve('SEGFORGE_IMAGE_URL', _imageUrl);

  /// Session to work in, or null to let the backend allocate one on upload.
  static String? get sessionId => _resolve('SEGFORGE_SESSION_ID', _sessionId);

  /// Where the SegForge backend lives.
  ///
  /// 8401 keeps SegForge clear of DoubleNaught, which serves its own FastAPI
  /// backend on 8400, so the two can run side by side. `run.sh BE` passes the
  /// matching `--port`; keep the two in step.
  ///
  /// `127.0.0.1` rather than `localhost` on purpose: on macOS `localhost` can
  /// resolve to the IPv6 `::1` while uvicorn is bound to IPv4, which surfaces
  /// as a connection refusal.
  static String get backendUrl =>
      _resolve('SEGFORGE_BACKEND_URL', _backendUrl) ?? _backendUrl;
}
