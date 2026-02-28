// DROP-IN replacement for your file (Flutter Web-safe: no dart:io, no File/Image.file)
//
// What changed vs your version:
// - Removed dart:io usage (File) and switched to Uint8List + Image.memory (works on web).
// - _checkHealth() is wrapped in try/catch to avoid “white screen” on backend/CORS failures.
// - Canvas widget now takes imageBytes instead of imageFile.
// - Box drawing remains the same.
// - NOTE: Your ApiService must support uploading bytes. See the small adapter call below.
//
// If your current ApiService only accepts File, update it to accept bytes, or create an overload.

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

import 'services/api_service.dart';

void main() {
  runApp(const SamApp());
}

class SamApp extends StatelessWidget {
  const SamApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SAM3 Studio',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.indigo,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        cardTheme: const CardThemeData(
          elevation: 2,
          margin: EdgeInsets.zero,
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService _api = ApiService();
  final TextEditingController _textController = TextEditingController();

  // State
  String? _sessionId;

  Uint8List? _imageBytes; // Web-safe image data
  String? _imageName; // Optional, useful for upload filename
  Size? _imageSize; // Original size (from backend)

  Map<String, dynamic>? _result;
  bool _isLoading = false;
  String? _error;
  String _backendStatus = "checking";
  String _boxMode = "positive"; // "positive" or "negative"

  // Timing
  final List<Map<String, dynamic>> _timings = [];
  Timer? _healthCheckTimer;

  @override
  void initState() {
    super.initState();
    _checkHealth();
    _healthCheckTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => _checkHealth(),
    );
  }

  @override
  void dispose() {
    _healthCheckTimer?.cancel();
    _textController.dispose();
    super.dispose();
  }

  Future<void> _checkHealth() async {
    try {
      final health = await _api.checkHealth();
      if (!mounted) return;
      setState(() {
        _backendStatus = health['model_loaded'] == true ? "online" : "offline";
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _backendStatus = "offline";
        _error ??= "Health check failed: $e";
      });
    }
  }

  void _addTiming(String label, dynamic duration) {
    if (duration == null) return;
    setState(() {
      _timings.insert(0, {
        'label': label,
        'duration': duration,
        'timestamp': DateTime.now(),
      });
      if (_timings.length > 10) _timings.removeLast();
    });
  }

  Future<void> _pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true, // IMPORTANT: web needs bytes
    );

    final picked = result?.files.single;
    final bytes = picked?.bytes;

    if (bytes == null) return;

    setState(() {
      _isLoading = true;
      _error = null;
      _imageBytes = bytes;
      _imageName = picked?.name;
      _sessionId = null;
      _result = null;
      _imageSize = null;
    });

    try {
      // --- IMPORTANT ---
      // This requires ApiService.uploadImageBytes(...)
      // If you don't have it yet, add it (see comment below).
      final response = await _api.uploadImageBytes(bytes, filename: _imageName ?? "upload.png");

      if (response != null) {
        if (!mounted) return;
        setState(() {
          _sessionId = response['session_id'] as String?;
          _imageSize = Size(
            (response['width'] as num).toDouble(),
            (response['height'] as num).toDouble(),
          );
        });
        _addTiming("Image Encoding", response['processing_time_ms']);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  Future<void> _sendTextPrompt() async {
    if (_sessionId == null || _textController.text.isEmpty) return;
    setState(() => _isLoading = true);
    try {
      final response = await _api.segmentWithText(_sessionId!, _textController.text);
      if (!mounted) return;
      if (response != null) {
        setState(() => _result = response['results'] as Map<String, dynamic>?);
        _addTiming("Text: ${_textController.text}", response['processing_time_ms']);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  Future<void> _sendBoxPrompt(List<double> box) async {
    if (_sessionId == null) return;
    setState(() => _isLoading = true);
    try {
      final response = await _api.segmentWithBox(_sessionId!, box, _boxMode == "positive");
      if (!mounted) return;
      if (response != null) {
        setState(() => _result = response['results'] as Map<String, dynamic>?);
        _addTiming("Box ($_boxMode)", response['processing_time_ms']);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  Future<void> _reset() async {
    if (_sessionId == null) return;
    setState(() => _isLoading = true);
    try {
      final response = await _api.resetPrompts(_sessionId!);
      if (!mounted) return;
      if (response != null) {
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _textController.clear();
        });
        _addTiming("Reset Prompts", response['processing_time_ms']);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    // If you later want a responsive layout, you can use isWide.
    // final bool isWide = MediaQuery.of(context).size.width > 900;

    return Scaffold(
      appBar: AppBar(
        title: const Row(
          children: [
            Icon(Icons.auto_awesome, color: Colors.indigo),
            SizedBox(width: 10),
            Text('SAM3 Studio', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        actions: [
          _buildStatusBadge(),
          const SizedBox(width: 16),
        ],
      ),
      body: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Sidebar
          SizedBox(
            width: 340,
            child: ListView(
              padding: const EdgeInsets.all(16),
              children: [
                _buildUploadCard(),
                const SizedBox(height: 16),
                _buildTextPromptCard(),
                const SizedBox(height: 16),
                _buildBoxPromptCard(),
                const SizedBox(height: 16),
                _buildResultsCard(),
                const SizedBox(height: 16),
                _buildPerformanceCard(),
                if (_error != null) ...[
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.red.shade50,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.red.shade200),
                    ),
                    child: Text(
                      _error!,
                      style: TextStyle(color: Colors.red.shade800, fontSize: 12),
                    ),
                  ),
                ],
              ],
            ),
          ),

          // Main Canvas
          Expanded(
            child: Container(
              margin: const EdgeInsets.fromLTRB(0, 16, 16, 16),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey.shade300),
              ),
              clipBehavior: Clip.antiAlias,
              child: (_imageBytes == null)
                  ? Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.image_outlined, size: 64, color: Colors.grey.shade400),
                          const SizedBox(height: 16),
                          Text("Upload an image to start", style: TextStyle(color: Colors.grey.shade500)),
                        ],
                      ),
                    )
                  : (_imageSize == null)
                      ? const Center(child: CircularProgressIndicator())
                      : SegmentationCanvas(
                          imageBytes: _imageBytes!,
                          imageSize: _imageSize!,
                          result: _result,
                          isLoading: _isLoading,
                          onBoxDrawn: _sendBoxPrompt,
                        ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusBadge() {
    Color color;
    IconData icon;
    String text;

    switch (_backendStatus) {
      case "online":
        color = Colors.green;
        icon = Icons.check_circle;
        text = "Model Ready";
        break;
      case "offline":
        color = Colors.red;
        icon = Icons.error;
        text = "Backend Offline";
        break;
      default:
        color = Colors.orange;
        icon = Icons.hourglass_empty;
        text = "Connecting...";
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 6),
          Text(text, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  Widget _buildUploadCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.upload_file, size: 16),
                SizedBox(width: 8),
                Text("Image Source", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            InkWell(
              onTap: _isLoading ? null : _pickImage,
              borderRadius: BorderRadius.circular(8),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade300, style: BorderStyle.solid),
                  borderRadius: BorderRadius.circular(8),
                  color: Colors.grey.shade50,
                ),
                child: Column(
                  children: [
                    Icon(Icons.add_photo_alternate, size: 32, color: Colors.grey.shade400),
                    const SizedBox(height: 8),
                    Text("Click to upload", style: TextStyle(color: Colors.grey.shade600, fontSize: 12)),
                  ],
                ),
              ),
            ),
            if (_imageSize != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  "${_imageSize!.width.toInt()} × ${_imageSize!.height.toInt()} px",
                  style: TextStyle(fontSize: 11, color: Colors.grey.shade500),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTextPromptCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.text_fields, size: 16),
                SizedBox(width: 8),
                Text("Text Prompt", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _textController,
                    decoration: const InputDecoration(
                      hintText: 'e.g. "cat", "wheel"',
                      isDense: true,
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 12),
                    ),
                    enabled: _sessionId != null && !_isLoading,
                    onSubmitted: (_) => _sendTextPrompt(),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filled(
                  onPressed: (_sessionId == null || _isLoading) ? null : _sendTextPrompt,
                  icon: _isLoading
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Icon(Icons.send, size: 18),
                  style: IconButton.styleFrom(shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8))),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBoxPromptCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.crop_free, size: 16),
                SizedBox(width: 8),
                Text("Box Prompts", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            Text("Draw boxes to include/exclude regions", style: TextStyle(fontSize: 12, color: Colors.grey.shade600)),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: SegmentedButton<String>(
                    segments: const [
                      ButtonSegment(
                        value: "positive",
                        label: Text("Include"),
                        icon: Icon(Icons.add_box_outlined, size: 16),
                      ),
                      ButtonSegment(
                        value: "negative",
                        label: Text("Exclude"),
                        icon: Icon(Icons.indeterminate_check_box_outlined, size: 16),
                      ),
                    ],
                    selected: {_boxMode},
                    onSelectionChanged: (Set<String> newSelection) {
                      setState(() => _boxMode = newSelection.first);
                    },
                    style: const ButtonStyle(
                      visualDensity: VisualDensity.compact,
                      tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsCard() {
    final maskCount = (_result?['masks'] as List?)?.length ?? 0;
    final boxCount = (_result?['prompted_boxes'] as List?)?.length ?? 0;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.data_usage, size: 16),
                SizedBox(width: 8),
                Text("Results", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            _buildResultRow("Objects found", maskCount.toString()),
            _buildResultRow("Box prompts", boxCount.toString()),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: (_sessionId == null || _isLoading) ? null : _reset,
                icon: const Icon(Icons.delete_outline, size: 16),
                label: const Text("Clear All Prompts"),
                style: OutlinedButton.styleFrom(foregroundColor: Colors.red),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(fontSize: 13, color: Colors.grey.shade600)),
          Text(value, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildPerformanceCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.timer_outlined, size: 16),
                SizedBox(width: 8),
                Text("Performance", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            if (_timings.isEmpty)
              Text("No requests yet", style: TextStyle(fontSize: 12, color: Colors.grey.shade500))
            else
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _timings.length,
                itemBuilder: (context, index) {
                  final t = _timings[index];
                  final dur = t['duration'];
                  final durStr = (dur is num) ? dur.toStringAsFixed(1) : dur.toString();

                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Text(
                            t['label'].toString(),
                            style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        Text("$durStr ms", style: const TextStyle(fontSize: 12, fontFamily: 'monospace')),
                      ],
                    ),
                  );
                },
              ),
          ],
        ),
      ),
    );
  }
}

class SegmentationCanvas extends StatefulWidget {
  final Uint8List imageBytes;
  final Size imageSize;
  final Map<String, dynamic>? result;
  final bool isLoading;
  final Function(List<double>) onBoxDrawn;

  const SegmentationCanvas({
    super.key,
    required this.imageBytes,
    required this.imageSize,
    this.result,
    required this.isLoading,
    required this.onBoxDrawn,
  });

  @override
  State<SegmentationCanvas> createState() => _SegmentationCanvasState();
}

class _SegmentationCanvasState extends State<SegmentationCanvas> {
  Offset? _startDrag;
  Offset? _currentDrag;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final double scaleW = constraints.maxWidth / widget.imageSize.width;
        final double scaleH = constraints.maxHeight / widget.imageSize.height;
        final double scale = scaleW < scaleH ? scaleW : scaleH;

        final double displayedW = widget.imageSize.width * scale;
        final double displayedH = widget.imageSize.height * scale;

        final double offsetX = (constraints.maxWidth - displayedW) / 2;
        final double offsetY = (constraints.maxHeight - displayedH) / 2;

        return Stack(
          fit: StackFit.expand,
          children: [
            Center(child: Image.memory(widget.imageBytes, fit: BoxFit.contain)),

            CustomPaint(
              painter: SegmentationPainter(
                result: widget.result,
                imageSize: widget.imageSize,
                displaySize: Size(displayedW, displayedH),
                offset: Offset(offsetX, offsetY),
                scale: scale,
              ),
            ),

            GestureDetector(
              onPanStart: (details) {
                setState(() {
                  _startDrag = details.localPosition;
                  _currentDrag = details.localPosition;
                });
              },
              onPanUpdate: (details) {
                setState(() => _currentDrag = details.localPosition);
              },
              onPanEnd: (details) {
                if (_startDrag != null && _currentDrag != null) {
                  final rect = Rect.fromPoints(_startDrag!, _currentDrag!);

                  final double x = (rect.left - offsetX) / scale;
                  final double y = (rect.top - offsetY) / scale;
                  final double w = rect.width / scale;
                  final double h = rect.height / scale;

                  final double nx = (x + w / 2) / widget.imageSize.width;
                  final double ny = (y + h / 2) / widget.imageSize.height;
                  final double nw = w / widget.imageSize.width;
                  final double nh = h / widget.imageSize.height;

                  if (nw > 0.01 && nh > 0.01) {
                    widget.onBoxDrawn([nx, ny, nw, nh]);
                  }
                }
                setState(() {
                  _startDrag = null;
                  _currentDrag = null;
                });
              },
              child: Container(color: Colors.transparent),
            ),

            if (_startDrag != null && _currentDrag != null)
              CustomPaint(
                painter: DragBoxPainter(
                  rect: Rect.fromPoints(_startDrag!, _currentDrag!),
                ),
              ),

            if (widget.isLoading)
              Container(
                color: Colors.black12,
                child: const Center(child: CircularProgressIndicator()),
              ),
          ],
        );
      },
    );
  }
}

class SegmentationPainter extends CustomPainter {
  final Map<String, dynamic>? result;
  final Size imageSize;
  final Size displaySize;
  final Offset offset;
  final double scale;

  SegmentationPainter({
    required this.result,
    required this.imageSize,
    required this.displaySize,
    required this.offset,
    required this.scale,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (result == null) return;

    final boxes = result!['boxes'] as List?;
    if (boxes != null) {
      final paint = Paint()
        ..color = Colors.green.withOpacity(0.5)
        ..style = PaintingStyle.fill;

      final borderPaint = Paint()
        ..color = Colors.green
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      for (var b in boxes) {
        final list = (b as List).map((e) => (e as num).toDouble()).toList();
        final rect = Rect.fromLTRB(
          list[0] * scale + offset.dx,
          list[1] * scale + offset.dy,
          list[2] * scale + offset.dx,
          list[3] * scale + offset.dy,
        );
        canvas.drawRect(rect, paint);
        canvas.drawRect(rect, borderPaint);
      }
    }

    final promptedBoxes = result!['prompted_boxes'] as List?;
    if (promptedBoxes != null) {
      for (var pb in promptedBoxes) {
        final box = (pb['box'] as List).map((e) => (e as num).toDouble()).toList();
        final label = pb['label'] as bool;

        final paint = Paint()
          ..color = label ? Colors.blue : Colors.red
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0;

        final rect = Rect.fromLTRB(
          box[0] * scale + offset.dx,
          box[1] * scale + offset.dy,
          box[2] * scale + offset.dx,
          box[3] * scale + offset.dy,
        );
        canvas.drawRect(rect, paint);

        final iconPaint = Paint()..color = label ? Colors.blue : Colors.red;
        canvas.drawCircle(rect.topLeft, 4, iconPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class DragBoxPainter extends CustomPainter {
  final Rect rect;
  DragBoxPainter({required this.rect});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.blue
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawRect(rect, paint);

    final fillPaint = Paint()
      ..color = Colors.blue.withOpacity(0.1)
      ..style = PaintingStyle.fill;
    canvas.drawRect(rect, fillPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

/*
========================
ApiService NOTE (required)
========================

Your existing ApiService probably has something like:
  Future<Map<String,dynamic>?> uploadImage(File file)

For Flutter Web, you need an overload like:

  Future<Map<String,dynamic>?> uploadImageBytes(Uint8List bytes, {required String filename})

Implementation idea (using package:http):
- POST multipart/form-data
- add a MultipartFile.fromBytes('file', bytes, filename: filename)

If you paste your current ApiService, I’ll provide the exact drop-in update for it too.
*/