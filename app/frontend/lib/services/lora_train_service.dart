import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

class ReadinessResult {
  final bool ready;
  final int imageCount;
  final int fileSizeBytes;

  ReadinessResult({
    required this.ready,
    required this.imageCount,
    required this.fileSizeBytes,
  });

  factory ReadinessResult.fromJson(Map<String, dynamic> j) => ReadinessResult(
        ready: j['ready'] as bool,
        imageCount: j['image_count'] as int? ?? 0,
        fileSizeBytes: j['file_size_bytes'] as int? ?? 0,
      );
}

class TrainConfig {
  final String sessionId;
  final String modelPath;
  final String scriptPath;
  final int rank;
  final int alpha;
  final double learningRate;
  final int numTrainEpochs;
  final int batchSize;
  final int resolution;
  final String mixedPrecision;

  TrainConfig({
    required this.sessionId,
    this.modelPath = 'black-forest-labs/FLUX.1-dev',
    this.scriptPath = 'train_dreambooth_lora_flux.py',
    this.rank = 16,
    this.alpha = 16,
    this.learningRate = 1e-4,
    this.numTrainEpochs = 1,
    this.batchSize = 1,
    this.resolution = 1024,
    this.mixedPrecision = 'bf16',
  });

  Map<String, dynamic> toJson() => {
        'session_id': sessionId,
        'model_path': modelPath,
        'script_path': scriptPath,
        'rank': rank,
        'alpha': alpha,
        'learning_rate': learningRate,
        'num_train_epochs': numTrainEpochs,
        'batch_size': batchSize,
        'resolution': resolution,
        'mixed_precision': mixedPrecision,
      };
}

class TrainStatus {
  final String status; // running | done | failed
  final int currentStep;
  final int totalSteps;
  final int elapsedSeconds;
  final String? outputPath;
  final String? lastError;

  TrainStatus({
    required this.status,
    required this.currentStep,
    required this.totalSteps,
    required this.elapsedSeconds,
    this.outputPath,
    this.lastError,
  });

  factory TrainStatus.fromJson(Map<String, dynamic> j) => TrainStatus(
        status: j['status'] as String,
        currentStep: j['current_step'] as int? ?? 0,
        totalSteps: j['total_steps'] as int? ?? 0,
        elapsedSeconds: j['elapsed_seconds'] as int? ?? 0,
        outputPath: j['output_path'] as String?,
        lastError: j['last_error'] as String?,
      );

  bool get isTerminal => status == 'done' || status == 'failed';
}

class LoraTrainService {
  final String baseUrl;
  LoraTrainService({this.baseUrl = 'http://localhost:8000'});

  Future<ReadinessResult> checkReadiness(String sessionId) async {
    final uri = Uri.parse('$baseUrl/pipeline/lora_train/ready')
        .replace(queryParameters: {'session_id': sessionId});
    final response = await http.get(uri);
    if (response.statusCode == 200) {
      return ReadinessResult.fromJson(
          jsonDecode(response.body) as Map<String, dynamic>);
    }
    throw Exception('Readiness check failed (${response.statusCode}): ${response.body}');
  }

  Future<String> startTraining(TrainConfig config) async {
    final response = await http.post(
      Uri.parse('$baseUrl/pipeline/lora_train/run'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(config.toJson()),
    );
    if (response.statusCode == 200) {
      return (jsonDecode(response.body) as Map<String, dynamic>)['run_id']
          as String;
    }
    String detail = 'Training start failed';
    try {
      detail = (jsonDecode(response.body) as Map)['detail'] ?? detail;
    } catch (_) {}
    throw Exception(detail);
  }

  Stream<TrainStatus> watchStatus(String runId) async* {
    while (true) {
      await Future.delayed(const Duration(seconds: 2));
      try {
        final response = await http.get(
            Uri.parse('$baseUrl/pipeline/lora_train/status/$runId'));
        if (response.statusCode == 200) {
          final status = TrainStatus.fromJson(
              jsonDecode(response.body) as Map<String, dynamic>);
          yield status;
          if (status.isTerminal) break;
        }
      } catch (_) {}
    }
  }

  Future<List<String>> getLogs(String runId) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/pipeline/lora_train/logs/$runId'));
      if (response.statusCode == 200) {
        return (jsonDecode(response.body) as List).cast<String>();
      }
    } catch (_) {}
    return [];
  }
}
