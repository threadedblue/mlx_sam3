// DROP-IN replacement for your file (Flutter Web-safe: no dart:io, no File/Image.file)
//
// What changed vs your version:
// - Refactored the main canvas into a layered architecture using a new `LayeredSegmentationCanvas` widget.
// - The old `SegmentationCanvas` is replaced with a new version that composes the layers and interaction controls.
// - The old `SegmentationPainter` is removed and its logic is split into `_updateSegmentsFromResult` (for data) and a new `PromptPainter` (for display).
// - State management is updated to handle `ui.Image` and `List<Segment>` for the new canvas.
// - The "Segment Layers" card now includes a toggle for the "Original" image layer.
//
// If your current ApiService only accepts File, update it to accept bytes, or create an overload.

import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
// import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:file_picker/file_picker.dart';

import 'services/api_service.dart';
import 'segment_layers_card.dart';
import 'layered_segmentation_canvas.dart';

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
  ui.Image? _uiImage; // Decoded image for canvas
  Size? _imageSize; // Original size

  Map<String, dynamic>? _result;
  List<Segment> _segments = [];
  bool _isLoading = false;
  String? _error;
  String _backendStatus = "checking";
  String _boxMode = "positive"; // "positive" or "negative"

  // Layer Visibility
  bool _showOriginal = true;
  bool _showMasks = true;
  bool _showRaw = true;
  bool _showFinal = true;

  List<String> _savedSessions = [];
  String? _selectedSavedSession;

  // Timing
  final List<Map<String, dynamic>> _timings = [];
  Timer? _healthCheckTimer;

  @override
  void initState() {
    super.initState();
    _checkHealth();
    _loadSavedSessions();
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
      debugPrint("Health check error: $e");
      if (!mounted) return;
      setState(() {
        _backendStatus = "offline";
        _error ??= "Health check failed: $e";
      });
    }
  }

  Future<void> _loadSavedSessions() async {
    final sessions = await _api.listSessions();
    if (!mounted) return;
    setState(() {
      _savedSessions = sessions;
      if (_savedSessions.isNotEmpty && _selectedSavedSession == null) {
        _selectedSavedSession = _savedSessions.first;
      } else if (!_savedSessions.contains(_selectedSavedSession)) {
        _selectedSavedSession = _savedSessions.isNotEmpty ? _savedSessions.first : null;
      }
    });
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

  Future<ui.Image> _decodeImage(Uint8List bytes) {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, (ui.Image img) {
      return completer.complete(img);
    });
    return completer.future;
  }

  void _updateSegmentsFromResult() {
    if (_result == null) {
      if (mounted) setState(() => _segments = []);
      return;
    }

    final List<Segment> newSegments = [];
    // The API result may contain 'masks' or 'boxes'. We check for both.
    final masks = _result!['masks'] as List? ?? _result!['boxes'] as List?;

    if (masks != null) {
      final colors = [
        Colors.red, Colors.blue, Colors.green, Colors.yellow, 
        Colors.purple, Colors.orange, Colors.cyan, Colors.pink
      ];
      int colorIndex = 0;
      for (var maskData in masks) {
        if (maskData is List && maskData.length == 4) {
          final list = maskData.map((e) => (e as num).toDouble()).toList();
          final rect = Rect.fromLTRB(list[0], list[1], list[2], list[3]);
          final path = Path()..addRect(rect);
          newSegments.add(Segment(
            path: path,
            color: colors[colorIndex % colors.length],
          ));
          colorIndex++;
        }
      }
    }
    if (mounted) setState(() => _segments = newSegments);
  }

  Future<void> _pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true, // IMPORTANT: web needs bytes
    );

    final picked = result?.files.single;
    final bytes = picked?.bytes;

    if (bytes == null) return;

    final decodedImage = await _decodeImage(bytes);

    setState(() {
      _isLoading = true;
      _error = null;
      _imageBytes = bytes;
      _imageName = picked?.name;
      _uiImage = decodedImage;
      _sessionId = null;
      _result = null;
      _imageSize = null;
      _segments = [];
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
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _updateSegmentsFromResult();
        });
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
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _updateSegmentsFromResult();
        });
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
          _updateSegmentsFromResult();
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

  Future<void> _saveMasks() async {
    if (_sessionId == null) return;
    setState(() => _isLoading = true);
    try {
      final response = await _api.saveMasks(_sessionId!);
      if (!mounted) return;
      if (response != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Masks saved successfully")),
        );
        _loadSavedSessions(); // Refresh list
        _addTiming("Save Masks", response['processing_time_ms']);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  Future<void> _handleCreateSegments() async {
    if (_sessionId == null) return;
    setState(() => _isLoading = true);
    try {
      final response = await _api.createSegments(_sessionId!);
      if (!mounted) return;
      if (response != null) {
        final count = response['segment_count'];
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("$count segments created successfully")),
        );
        _addTiming("Create Segments", response['processing_time_ms']);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  Future<void> _handleShowSegments() async {
    if (_sessionId == null) return;
    setState(() => _isLoading = true);
    List<String> segmentUrls = [];
    try {
      segmentUrls = await _api.showSegments(_sessionId!);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }

    if (!mounted) return;
    if (segmentUrls.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No segments found. Please create them first.")),
      );
      return;
    }

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Generated Segments"),
        content: SizedBox(
          width: double.maxFinite,
          child: GridView.builder(
            gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
              maxCrossAxisExtent: 150,
              childAspectRatio: 1,
              crossAxisSpacing: 10,
              mainAxisSpacing: 10,
            ),
            itemCount: segmentUrls.length,
            itemBuilder: (context, index) {
              return Image.network(segmentUrls[index], fit: BoxFit.cover);
            },
          ),
        ),
        actions: [TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text("Close"))],
      ),
    );
  }

  Future<void> _handleNewSession() async {
    setState(() => _isLoading = true);
    try {
      final newId = await _api.newSession();
      if (!mounted) return;
      setState(() {
        _sessionId = newId;
        _imageBytes = null;
        _uiImage = null;
        _imageSize = null;
        _result = null;
        _error = null;
        _textController.clear();
      });
      await _api.createSessionDirs(_sessionId!);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("New session created: $newId")),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  Future<void> _handleDeleteSession() async {
    if (_selectedSavedSession == null) return;
    setState(() => _isLoading = true);
    try {
      await _api.deleteSession(_selectedSavedSession!);
      await _loadSavedSessions();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Session deleted")),
      );
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
            Text('SAM3 MLX Studio', style: TextStyle(fontWeight: FontWeight.bold)),
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
                _buildSessionCard(),
                const SizedBox(height: 16),
                _buildUploadCard(),
                const SizedBox(height: 16),
                _buildTextPromptCard(),
                const SizedBox(height: 16),
                _buildBoxPromptCard(),
                const SizedBox(height: 16),
                _buildResultsCard(),
                const SizedBox(height: 16),
                _buildDownloadCard(),
                const SizedBox(height: 16),
                _buildSegmentsCard(),
                const SizedBox(height: 16),
                _buildSegmentLayersCard(),
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
                  : (_uiImage == null)
                      ? const Center(child: CircularProgressIndicator())
                      : SegmentationCanvas(
                          uiImage: _uiImage!,
                          segments: _segments,
                          result: _result,
                          isLoading: _isLoading,
                          onBoxDrawn: _sendBoxPrompt,
                          showOriginal: _showOriginal,
                          showMasks: _showMasks,
                          showRaw: _showRaw,
                          showFinals: _showFinal,
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

  Widget _buildSessionCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.folder_shared, size: 16),
                SizedBox(width: 8),
                Text("Session Management", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              value: _selectedSavedSession,
              isExpanded: true,
              decoration: const InputDecoration(
                labelText: "Saved Sessions",
                border: OutlineInputBorder(),
                contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              ),
              items: _savedSessions.map((s) {
                return DropdownMenuItem(value: s, child: Text(s, overflow: TextOverflow.ellipsis));
              }).toList(),
              onChanged: (val) {
                setState(() => _selectedSavedSession = val);
              },
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: FilledButton.tonal(
                    onPressed: _isLoading ? null : _handleNewSession,
                    child: const Text("New"),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton(
                    onPressed: (_selectedSavedSession == null || _isLoading) ? null : _handleDeleteSession,
                    style: OutlinedButton.styleFrom(foregroundColor: Colors.red),
                    child: const Text("Delete"),
                  ),
                ),
              ],
            ),
          ],
        ),
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

  Widget _buildDownloadCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.download, size: 16),
                SizedBox(width: 8),
                Text("Download", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              "Save the image and generated masks.",
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: (_sessionId == null || _isLoading) ? null : _saveMasks,
                icon: const Icon(Icons.save_alt, size: 16),
                label: const Text("Save Masks"),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSegmentsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.category_outlined, size: 16),
                SizedBox(width: 8),
                Text("Segments", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              "Create and view individual segment images.",
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: (_sessionId == null || _isLoading) ? null : _handleCreateSegments,
                    icon: const Icon(Icons.cut, size: 16),
                    label: const Text("Create Segments"),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: (_sessionId == null || _isLoading) ? null : _handleShowSegments,
                    icon: const Icon(Icons.image_search, size: 16),
                    label: const Text("Show Segments"),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSegmentLayersCard() {
    return SegmentLayersCard(
      showOriginal: _showOriginal,
      showMasks: _showMasks,
      showRaw: _showRaw,
      showFinal: _showFinal,
      onOriginalToggle: (val) => setState(() => _showOriginal = val),
      onMasksToggle: (val) => setState(() => _showMasks = val),
      onRawToggle: (val) => setState(() => _showRaw = val),
      onFinalToggle: (val) => setState(() => _showFinal = val),
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
  final ui.Image uiImage;
  final List<Segment> segments;
  final Map<String, dynamic>? result;
  final bool isLoading;
  final Function(List<double>) onBoxDrawn;
  final bool showOriginal;
  final bool showMasks;
  final bool showRaw;
  final bool showFinals;

  const SegmentationCanvas({
    super.key,
    required this.uiImage,
    required this.segments,
    this.result,
    required this.isLoading,
    required this.onBoxDrawn,
    required this.showOriginal,
    required this.showMasks,
    required this.showRaw,
    required this.showFinals,
  });

  @override
  State<SegmentationCanvas> createState() => _SegmentationCanvasState();
}

class _SegmentationCanvasState extends State<SegmentationCanvas> {
  Offset? _startDrag;
  Offset? _currentDrag;

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // The core display layers
        LayeredSegmentationCanvas(
          originalImage: widget.uiImage,
          segments: widget.segments,
          showOriginal: widget.showOriginal,
          showMasks: widget.showMasks,
          showRaw: widget.showRaw,
          showFinals: widget.showFinals,
        ),

        // The interaction and prompt overlay
        _buildInteractionOverlay(),

        // Loading indicator on top of everything
        if (widget.isLoading)
          Container(
            color: Colors.black12,
            child: const Center(child: CircularProgressIndicator()),
          ),
      ],
    );
  }

  Widget _buildInteractionOverlay() {
    // This overlay needs to scale and position itself exactly like the
    // content of LayeredSegmentationCanvas. We can achieve this by
    // wrapping it in an identical FittedBox/SizedBox structure.
    return FittedBox(
      fit: BoxFit.contain,
      child: SizedBox(
        width: widget.uiImage.width.toDouble(),
        height: widget.uiImage.height.toDouble(),
        child: LayoutBuilder(builder: (context, constraints) {
          // Inside this LayoutBuilder, the coordinate system matches the original image.
          return Stack(
            children: [
              // Painter for showing existing prompts (the blue/red boxes)
              CustomPaint(
                size: Size.infinite,
                painter: PromptPainter(result: widget.result),
              ),

              // Gesture detector for drawing new boxes
              GestureDetector(
                onPanStart: (details) => setState(() {
                  _startDrag = details.localPosition;
                  _currentDrag = details.localPosition;
                }),
                onPanUpdate: (details) => setState(() => _currentDrag = details.localPosition),
                onPanEnd: (details) {
                  if (_startDrag != null && _currentDrag != null) {
                    final rect = Rect.fromPoints(_startDrag!, _currentDrag!);

                    // Normalize coordinates for the API call.
                    final double nx = rect.center.dx / widget.uiImage.width;
                    final double ny = rect.center.dy / widget.uiImage.height;
                    final double nw = rect.width / widget.uiImage.width;
                    final double nh = rect.height / widget.uiImage.height;

                    if (nw > 0.005 && nh > 0.005) { // Avoid tiny boxes
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

              // Painter for the box being currently drawn
              if (_startDrag != null && _currentDrag != null)
                CustomPaint(
                  size: Size.infinite,
                  painter: DragBoxPainter(
                    rect: Rect.fromPoints(_startDrag!, _currentDrag!),
                  ),
                ),
            ],
          );
        }),
      ),
    );
  }
}

/// A painter for drawing the user's input prompts (positive/negative boxes).
class PromptPainter extends CustomPainter {
  final Map<String, dynamic>? result;

  PromptPainter({required this.result});

  @override
  void paint(Canvas canvas, Size size) {
    if (result == null) return;

    final promptedBoxes = result!['prompted_boxes'] as List?;
    if (promptedBoxes != null) {
      for (var pb in promptedBoxes) {
        final box = (pb['box'] as List).map((e) => (e as num).toDouble()).toList();
        final label = pb['label'] as bool;

        final paint = Paint()
          ..color = label ? Colors.blue : Colors.red
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0;

        // Coordinates are already in image space, no scaling needed.
        final rect = Rect.fromLTRB(box[0], box[1], box[2], box[3]);
        canvas.drawRect(rect, paint);

        final iconPaint = Paint()..color = label ? Colors.blue : Colors.red;
        canvas.drawCircle(rect.topLeft, 4, iconPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant PromptPainter oldDelegate) => result != oldDelegate.result;
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
  bool shouldRepaint(covariant DragBoxPainter oldDelegate) => rect != oldDelegate.rect;
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