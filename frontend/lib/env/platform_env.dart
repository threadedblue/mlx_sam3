/// Process environment — web stub.
///
/// `dart:io` does not exist on web, so [platformEnvironment] is empty there and
/// launch inputs fall back to their compile-time `--dart-define` values. The
/// `dart:io` implementation lives in `platform_env_io.dart`; callers pick
/// between them with a conditional import (see `launch_config.dart`).
Map<String, String> get platformEnvironment => const <String, String>{};
