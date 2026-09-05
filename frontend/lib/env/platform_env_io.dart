import 'dart:io' show Platform;

/// Process environment — `dart:io` implementation for desktop and mobile.
///
/// This is what lets a parent process hand SegForge per-launch inputs: a
/// prebuilt bundle cannot be given new `--dart-define` values (those are
/// compile-time constants), but it does inherit the environment it is spawned
/// with. DoubleNaught's Seg Forge node relies on this.
Map<String, String> get platformEnvironment => Platform.environment;
