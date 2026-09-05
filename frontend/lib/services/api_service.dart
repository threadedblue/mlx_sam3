import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../launch_config.dart';

/// Client for the SegForge FastAPI backend.
///
/// Only the endpoints the current UI uses are exposed here. The backend still
/// serves several others (session listing/creation/deletion, `/updateState`,
/// `/createSegments`, `/showSegments`, the `/lora/*` family, `/process-image`,
/// and `/inference/*`); they lost their last caller when the corresponding
/// cards were removed from the UI.
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

  Future<Map<String, dynamic>?> segmentWithBox(String sessionId, List<double> box, bool label) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/segment/box'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'session_id': sessionId,
          'box': box, // [cx, cy, w, h] normalized
          'label': label,
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

  Future<Map<String, dynamic>?> segmentWithPoint(String sessionId, List<double> point, bool label) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/segment/point'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'session_id': sessionId,
          'point': point, // [x, y] normalized
          'label': label,
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
