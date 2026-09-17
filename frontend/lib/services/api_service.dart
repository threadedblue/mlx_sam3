import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../launch_config.dart';

/// Client for the SegForge FastAPI backend.
///
/// Only the endpoints the current UI uses are exposed here. The backend still
/// serves several others (`/updateState`, `/createSegments`, `/showSegments`,
/// the `/lora/*` family, `/process-image`, and `/inference/*`); they lost
/// their last caller when the corresponding cards were removed from the UI.
/// Session listing/creation (`/listSessions`, `/newSession`, `/initSession`)
/// regained a caller — [SessionPickerScreen] — for the standalone launch
/// path.
class ApiService {
  /// Backend address, overridable at launch with
  /// `--dart-define=SEGFORGE_BACKEND_URL=...`. See [LaunchConfig.backendUrl].
  final String baseUrl = LaunchConfig.backendUrl;

  Future<Map<String, dynamic>> checkHealth() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
    } catch (e) {
      // ignore error
    }
    return {"status": "offline", "model_loaded": false};
  }

  /// Reads back everything already stored for [sessionId]: its metadata
  /// (`name`, `description`, `image_url`), its image as `image_b64`, and any
  /// prompts/results it already holds.
  ///
  /// This is how a session handed to the app at launch fills the UI in. The
  /// caller that created the session — DoubleNaught's Seg Forge node — has
  /// already uploaded the image and named the session over `/initSession`, so
  /// re-downloading the image URL and re-uploading it would move the same
  /// bytes twice and still leave the name unknown.
  ///
  /// Returns null when the backend has nothing on file (404), which is the
  /// ordinary case for a session id that has not been uploaded to yet.
  Future<Map<String, dynamic>?> loadSession(String sessionId) async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/loadSession/$sessionId'));
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      if (response.statusCode == 404) return null;
      throw Exception('Load session failed: ${response.statusCode} ${response.body}');
    } catch (e) {
      debugPrint('Error loading session $sessionId: $e');
      rethrow;
    }
  }

  /// Uploads image bytes, optionally into an existing session.
  ///
  /// When [sessionId] is null the backend allocates a session and returns its
  /// id in the response, which the caller is expected to adopt.
  Future<Map<String, dynamic>?> uploadImageBytes(
    Uint8List bytes, {
    required String filename,
    String? sessionId,
  }) async {
    final uri = Uri.parse("$baseUrl/upload");

    final request = http.MultipartRequest("POST", uri);

    request.files.add(
      http.MultipartFile.fromBytes(
        "file",
        bytes,
        filename: filename,
      ),
    );
    if (sessionId != null) {
      request.fields['session_id'] = sessionId;
    }

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    } else {
      throw Exception("Upload failed: ${response.statusCode} ${response.body}");
    }
  }

  Future<Map<String, dynamic>?> segmentWithText(String sessionId, String prompt) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/segment/text'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'session_id': sessionId,
          'prompt': prompt,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('Text segmentation failed: ${response.body}');
    } catch (e) {
      debugPrint('Error text segment: $e');
      rethrow;
    }
  }

  /// [label] is SAM3's grounding polarity (Target/Avoid — see
  /// IncludeExcludeToggle). [textSubstitute] is an optional per-region tag;
  /// it is NOT a caption — captioning is a separate, later action via
  /// [attachCaption]. In/out marking no longer exists as a per-request
  /// field (v2): whether a selection is kept is `dataset_status`, set only
  /// by captioning, and whether it's queued for scrubbing is `held`, set
  /// only by [setMaskHeld].
  Future<Map<String, dynamic>?> segmentWithBox(
    String sessionId,
    List<double> box,
    bool label, {
    String? textSubstitute,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/segment/box'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'session_id': sessionId,
          'box': box, // [cx, cy, w, h] normalized
          'label': label,
          if (textSubstitute != null) 'text_substitute': textSubstitute,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('Box segmentation failed: ${response.body}');
    } catch (e) {
      debugPrint('Error box segment: $e');
      rethrow;
    }
  }

  Future<Map<String, dynamic>?> segmentWithPoint(
    String sessionId,
    List<double> point,
    bool label, {
    String? textSubstitute,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/segment/point'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'session_id': sessionId,
          'point': point, // [x, y] normalized
          'label': label,
          if (textSubstitute != null) 'text_substitute': textSubstitute,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('Point segmentation failed: ${response.body}');
    } catch (e) {
      debugPrint('Error point segment: $e');
      rethrow;
    }
  }

  /// Queues (or un-queues) a selection for the current pass's next scrub
  /// batch — `sf_engine.SFSession.set_held`. Independent of captioning.
  Future<Map<String, dynamic>?> setMaskHeld(String sessionId, String maskId, bool held) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/mask/hold'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'session_id': sessionId, 'mask_id': maskId, 'held': held}),
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      throw Exception('Set held failed: ${response.statusCode} ${response.body}');
    } catch (e) {
      debugPrint('Error setting held: $e');
      rethrow;
    }
  }

  /// Captions an existing selection, marking it `keep` —
  /// `sf_engine.SFSession.attach_caption`. This is the Prompt card's
  /// caption-mode call (see main.dart), distinct from [segmentWithText]'s
  /// SAM3 text-grounding call.
  Future<Map<String, dynamic>?> attachCaption(String sessionId, String maskId, String caption) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/mask/caption'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'session_id': sessionId, 'mask_id': maskId, 'caption': caption}),
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      throw Exception('Attach caption failed: ${response.statusCode} ${response.body}');
    } catch (e) {
      debugPrint('Error attaching caption: $e');
      rethrow;
    }
  }

  /// Triggers the LaMa scrub of the current pass's held masks
  /// (`SFSession.run_lama_pass`), advancing to the next pass. Repeatable
  /// without limit (v2) — there is no "already scrubbed" state.
  ///
  /// Unlike segmentWithText/Box/Point, this does not rethrow on failure —
  /// same defensive pattern as [checkHealth] — since scrubbing is a
  /// deliberate, occasional action whose card should show an inline error
  /// state rather than an uncaught exception if the backend endpoint isn't
  /// there yet.
  Future<Map<String, dynamic>?> scrubLamaBackground(String sessionId) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/lama/scrub'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'session_id': sessionId}),
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      debugPrint('LaMa scrub failed: ${response.statusCode} ${response.body}');
      return null;
    } catch (e) {
      debugPrint('Error scrubbing LaMa background: $e');
      return null;
    }
  }

  Future<Map<String, dynamic>?> resetPrompts(String sessionId) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/reset'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'session_id': sessionId}),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('Reset failed: ${response.body}');
    } catch (e) {
      debugPrint('Error resetting prompts: $e');
      rethrow;
    }
  }

  Future<Map<String, dynamic>?> saveMasks(String sessionId) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/saveMasks'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'session_id': sessionId}),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('Save masks failed: ${response.body}');
    } catch (e) {
      debugPrint('Error saving masks: $e');
      rethrow;
    }
  }

  /// Persists the session's Segment/Linkage/Registry AAs so it can be
  /// resumed later (standalone picker, or DoubleNaught re-launching the same
  /// session id). Distinct from [saveMasks], which only rasterizes mask PNGs
  /// for DoubleNaught's own Close-Forge collection flow.
  Future<Map<String, dynamic>?> saveSession(String sessionId) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/saveSession'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'session_id': sessionId}),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('Save session failed: ${response.body}');
    } catch (e) {
      debugPrint('Error saving session: $e');
      rethrow;
    }
  }

  /// Lists sessions SF has actually saved, for [SessionPickerScreen].
  ///
  /// Reads the same registry.parquet-backed listing DoubleNaught's own
  /// picklist reads — a session that was only ever staged (an id minted, no
  /// Save yet) does not appear here.
  Future<List<Map<String, dynamic>>> listSavedSessions() async {
    final response = await http.get(Uri.parse('$baseUrl/listSessions'));
    if (response.statusCode != 200) {
      throw Exception('List sessions failed: ${response.statusCode} ${response.body}');
    }
    final parsed = jsonDecode(response.body);
    debugPrint('listSavedSessions raw response: $parsed (type: ${parsed.runtimeType})');

    List<dynamic> sessionsList;

    // Handle both response shapes:
    // 1. Wrapped: {"sessions": [...]}
    // 2. Bare: [...]
    if (parsed is Map<String, dynamic>) {
      sessionsList = (parsed['sessions'] as List?) ?? [];
    } else if (parsed is List) {
      sessionsList = parsed;
    } else {
      throw Exception(
        'Expected Map or List response, got ${parsed.runtimeType}: $parsed'
      );
    }

    debugPrint(
      'sessions list: $sessionsList (${sessionsList.length} items, '
      'first: ${sessionsList.isNotEmpty ? sessionsList[0].runtimeType : "empty"})'
    );

    // Convert each session to a Map<String, dynamic>. Handle two cases:
    // 1. Already a dict: {"session_id": "...", "name": "...", ...}
    // 2. Just an ID string: "session-id-..." (legacy format, convert to minimal record)
    return [
      for (final s in sessionsList)
        if (s is Map<String, dynamic>)
          s
        else if (s is String)
          // Legacy: bare session ID string. Create minimal record with just the ID.
          {"session_id": s, "name": "", "description": "", "created_at": "", "image_url": ""}
        else
          (s as Map).cast<String, dynamic>(),
    ];
  }

  /// Allocates a new session id on the backend. Nothing is written under
  /// storage/sf/sessions/ until that session's first Save.
  Future<String?> newSession() async {
    final response = await http.post(Uri.parse('$baseUrl/newSession'));
    if (response.statusCode != 200) {
      throw Exception('New session failed: ${response.statusCode} ${response.body}');
    }
    final j = jsonDecode(response.body) as Map<String, dynamic>;
    return j['session_id'] as String?;
  }

  /// Registers name/description for a freshly allocated session — same
  /// `/initSession` call DoubleNaught's Open Forge flow uses.
  Future<void> initSessionMetadata({
    required String sessionId,
    required String name,
    required String description,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/initSession'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'session_id': sessionId,
        'name': name,
        'description': description,
        'image_url': '',
      }),
    );
    if (response.statusCode != 200) {
      throw Exception('Init session failed: ${response.statusCode} ${response.body}');
    }
  }

  Future<void> saveSessionSettings(String sessionId, Map<String, dynamic> settings) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/session/settings'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'session_id': sessionId,
          'settings': settings,
        }),
      );
      if (response.statusCode != 200) {
        debugPrint('Failed to save session settings: ${response.body}');
      }
    } catch (e) {
      debugPrint('Error saving session settings: $e');
    }
  }
}
