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
}
