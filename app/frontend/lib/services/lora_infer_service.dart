import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

class InferRequest {
  final String loraPath;
  final String modelPath;
  final String prompt;
  final double loraStrength;
  final int steps;
  final double guidanceScale;
  final int seed;
  final String outputDir;
  final String? continuityImageB64;
  final double denoiseStrength;

  InferRequest({
    required this.loraPath,
    required this.modelPath,
    required this.prompt,
    this.loraStrength = 0.8,
    this.steps = 28,
    this.guidanceScale = 3.5,
    this.seed = 42,
    this.outputDir = '',
    this.continuityImageB64,
    this.denoiseStrength = 0.75,
  });

  Map<String, dynamic> toJson() => {
        'lora_path': loraPath,
        'model_path': modelPath,
        'prompt': prompt,
        'lora_strength': loraStrength,
        'steps': steps,
        'guidance_scale': guidanceScale,
        'seed': seed,
        'output_dir': outputDir,
        if (continuityImageB64 != null) 'continuity_image_b64': continuityImageB64,
        'denoise_strength': denoiseStrength,
      };
}

class InferStatus {
  final String status; // running | done | failed | cancelled
  final double progress; // 0.0 – 1.0
  final int currentStep;
  final int totalSteps;
  final int elapsedSeconds;
  final String? outputPath;
  final String? outputChecksum;
  final String? lastError;

  InferStatus({
    required this.status,
    required this.progress,
    required this.currentStep,
    required this.totalSteps,
    required this.elapsedSeconds,
    this.outputPath,
    this.outputChecksum,
    this.lastError,
  });

  factory InferStatus.fromJson(Map<String, dynamic> j) => InferStatus(
        status: j['status'] as String,
        progress: (j['progress'] as num?)?.toDouble() ?? 0.0,
        currentStep: j['current_step'] as int? ?? 0,
        totalSteps: j['total_steps'] as int? ?? 0,
        elapsedSeconds: j['elapsed_seconds'] as int? ?? 0,
        outputPath: j['output_path'] as String?,
        outputChecksum: j['output_checksum'] as String?,
        lastError: j['last_error'] as String?,
      );

  bool get isTerminal =>
      status == 'done' || status == 'failed' || status == 'cancelled';
}

// ---------------------------------------------------------------------------
// Provider models
// ---------------------------------------------------------------------------

class ProviderInfo {
  final String name;
  final bool active;
  final bool available;

  ProviderInfo({
    required this.name,
    required this.active,
    required this.available,
  });

  factory ProviderInfo.fromJson(Map<String, dynamic> j) => ProviderInfo(
        name: j['name'] as String,
        active: j['active'] as bool? ?? false,
        available: j['available'] as bool? ?? false,
      );

  String get label => switch (name) {
        'mlx'       => 'MLX',
        'cloud_run' => 'Cloud Run',
        _           => name,
      };
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

class LoraInferService {
  final String baseUrl;
  LoraInferService({this.baseUrl = 'http://localhost:8000'});

  Future<String> startGeneration(InferRequest req) async {
    final response = await http.post(
      Uri.parse('$baseUrl/inference/generate'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(req.toJson()),
    );
    if (response.statusCode == 200) {
      return (jsonDecode(response.body) as Map<String, dynamic>)['run_id']
          as String;
    }
    String detail = 'Generation start failed (${response.statusCode})';
    try {
      detail = (jsonDecode(response.body) as Map)['detail'] as String? ?? detail;
    } catch (_) {}
    throw Exception(detail);
  }

  Stream<InferStatus> watchStatus(String runId) async* {
    while (true) {
      await Future.delayed(const Duration(seconds: 1));
      try {
        final response = await http
            .get(Uri.parse('$baseUrl/inference/status/$runId'));
        if (response.statusCode == 200) {
          final status = InferStatus.fromJson(
              jsonDecode(response.body) as Map<String, dynamic>);
          yield status;
          if (status.isTerminal) break;
        }
      } catch (_) {}
    }
  }

  Future<Uint8List> fetchImage(String runId) async {
    final response =
        await http.get(Uri.parse('$baseUrl/inference/image/$runId'));
    if (response.statusCode == 200) return response.bodyBytes;
    throw Exception(
        'Failed to fetch image (${response.statusCode}): ${response.body}');
  }

  Future<void> cancel(String runId) async {
    try {
      await http.post(Uri.parse('$baseUrl/inference/cancel/$runId'));
    } catch (_) {}
  }

  // ── Provider management ─────────────────────────────────────────────────

  Future<List<ProviderInfo>> getProviders() async {
    final response =
        await http.get(Uri.parse('$baseUrl/inference/provider'));
    if (response.statusCode == 200) {
      final body = jsonDecode(response.body) as Map<String, dynamic>;
      return (body['providers'] as List)
          .map((p) => ProviderInfo.fromJson(p as Map<String, dynamic>))
          .toList();
    }
    return [];
  }

  Future<List<ProviderInfo>> setProvider(String name, {String url = ''}) async {
    final response = await http.post(
      Uri.parse('$baseUrl/inference/provider'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'provider': name, 'url': url}),
    );
    if (response.statusCode == 200) {
      final body = jsonDecode(response.body) as Map<String, dynamic>;
      return (body['providers'] as List)
          .map((p) => ProviderInfo.fromJson(p as Map<String, dynamic>))
          .toList();
    }
    String detail = 'Provider switch failed';
    try {
      detail = (jsonDecode(response.body) as Map)['detail'] as String? ?? detail;
    } catch (_) {}
    throw Exception(detail);
  }
}
