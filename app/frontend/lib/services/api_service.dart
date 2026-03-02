import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;


class ApiService {
  // Set the base URL for the backend API.  Make sure this matches your backend.
  final String baseUrl = "http://localhost:8000";

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

  Future<Map<String, dynamic>?> uploadImageBytes(
    Uint8List bytes, {
    required String filename,
  }) async {
    final uri = Uri.parse("$baseUrl/upload"); // adjust endpoint if needed

    final request = http.MultipartRequest("POST", uri);

    request.files.add(
      http.MultipartFile.fromBytes(
        "file",        // must match your FastAPI field name
        bytes,
        filename: filename,
      ),
    );

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    } else {
      throw Exception("Upload failed: ${response.statusCode} ${response.body}");
    }
  }

  Future<Map<String, dynamic>?> uploadImage({
    required Uint8List fileBytes,
    required String fileName,
  }) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/upload'));
      request.files.add(
        http.MultipartFile.fromBytes('file', fileBytes, filename: fileName),
      );
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('Upload failed: ${response.body}');
    } catch (e) {
      print('Error uploading image: $e');
      rethrow;
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
      print('Error text segment: $e');
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
      print('Error box segment: $e');
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
      print('Error resetting prompts: $e');
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
      print('Error saving masks: $e');
      rethrow;
    }
  }

  Future<List<String>> listSessions() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/listSessions'));
      if (response.statusCode == 200) {
        final List<dynamic> list = jsonDecode(response.body);
        return list.cast<String>();
      }
      return [];
    } catch (e) {
      print('Error listing sessions: $e');
      return [];
    }
  }

  Future<String> newSession() async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/newSession'),
        headers: {'Content-Type': 'application/json'},
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['session_id'];
      }
      throw Exception('New session failed: ${response.body}');
    } catch (e) {
      print('Error creating session: $e');
      rethrow;
    }
  }

  Future<void> createSessionDirs(String sessionId) async {
    try {
      final response = await http.post(Uri.parse('$baseUrl/createSessionDirs/$sessionId'),
          headers: {'Content-Type': 'application/json'});
      if (response.statusCode != 200) {
        throw Exception('Failed to create session directories');
      }
    } catch (e) {
      print('Error creating session directories: $e');
      rethrow;
    }
  }

  Future<void> deleteSession(String sessionId) async {
    try {
      final response = await http.delete(Uri.parse('$baseUrl/deleteSession/$sessionId'));
      if (response.statusCode != 200) {
        throw Exception('Delete session failed: ${response.body}');
      }
    } catch (e) {
      print('Error deleting session: $e');
      rethrow;
    }
  }

  Future<Map<String, dynamic>?> createSegments(String sessionId) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/createSegments'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'session_id': sessionId}),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('Create segments failed: ${response.body}');
    } catch (e) {
      print('Error creating segments: $e');
      rethrow;
    }
  }

  Future<List<String>> showSegments(String sessionId) async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/showSegments/$sessionId'));
      if (response.statusCode == 200) {
        final List<dynamic> list = jsonDecode(response.body);
        return list.cast<String>().map((path) => '$baseUrl$path').toList();
      }
      return [];
    } catch (e) {
      print('Error showing segments: $e');
      return [];
    }
  }
}
