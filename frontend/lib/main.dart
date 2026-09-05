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
import 'package:http/http.dart' as http;

import 'package:flutter/material.dart';

import 'services/api_service.dart';
import 'segment_layers_card.dart';
import 'package:provider/provider.dart';
import 'layered_segmentation_canvas.dart';
import 'layer_state.dart';
import 'models/result_datum.dart';
import 'widgets/result_cell.dart';
import 'widgets/include_exclude_toggle.dart';
import 'launch_config.dart';

void main() {
  runApp(const SamApp());
}

/// Which of the three prompt cards currently owns canvas interaction.
///
/// The radio buttons on the Prompt, Box Select and Point Select cards form one
/// mutually exclusive group over this enum; `null` means no mode is active and
/// the canvas ignores taps and drags.
enum SelectionMode { prompt, box, point }

class SamApp extends StatelessWidget {
  const SamApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (context) => LayerState()),
      ],
      child: MaterialApp(
        title: 'SegForge Studio',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.indigo,
            brightness: Brightness.light,
          ),
          useMaterial3: true,
          cardTheme: const CardThemeData(elevation: 2, margin: EdgeInsets.zero),
        ),
        darkTheme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.indigo,
            brightness: Brightness.dark,
          ),
          useMaterial3: true,
          cardTheme: const CardThemeData(elevation: 2, margin: EdgeInsets.zero),
        ),
        themeMode: ThemeMode.dark,
        home: const HomeScreen(),
      ),
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

  // State (supplied at launch, not user-entered)
  String? _sessionId;
  String? _imageUrl; // Supplied at launch, non-editable

  Uint8List? _imageBytes; // Web-safe image data
  ui.Image? _uiImage; // Decoded image for canvas
  Size? _imageSize; // Original size

  Map<String, dynamic>? _result;
  List<Segment> _segments = [];
  bool _isLoading = false;
  String? _error;
  String _backendStatus = "checking";
  SelectionMode? _selectedMode;
  String _boxMode = "positive"; // "positive" or "negative"
  String _pointMode = "positive"; // "positive" or "negative"

  /// Guards the one-shot auto-load of the launch-supplied image URL.
  bool _autoLoadStarted = false;

  // Per-card result state
  bool _imageSourceRunning = false;
  List<ResultDatum> _imageSourceResult = [];
  bool _textPromptRunning = false;
  List<ResultDatum> _textPromptResult = [];
  bool _boxRunning = false;
  List<ResultDatum> _boxResult = [];
  bool _pointRunning = false;
  List<ResultDatum> _pointResult = [];
  bool _resultsRunning = false;
  List<ResultDatum> _resultsResult = [];
  Timer? _healthCheckTimer;

  // Layer State
  LayerState? _layerState;

  @override
  void initState() {
    super.initState();
    // Session id and image URL are launch-time inputs, not user-entered.
    // A null session id is fine: /upload allocates one and returns it.
    _sessionId = LaunchConfig.sessionId;
    _imageUrl = LaunchConfig.imageUrl;
    // The Select button is disabled while the prompt is empty, so the field has
    // to trigger a rebuild as it is typed into.
    _textController.addListener(_onPromptTextChanged);
    _checkHealth();
    _healthCheckTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => _checkHealth(),
    );
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final newLayerState = Provider.of<LayerState>(context, listen: false);
    if (_layerState != newLayerState) {
      _layerState?.removeListener(_onLayerStateChanged);
      _layerState = newLayerState;
      _layerState?.addListener(_onLayerStateChanged);
    }
  }

  void _onLayerStateChanged() {
    _saveLayerState();
  }

  void _onPromptTextChanged() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _healthCheckTimer?.cancel();
    _textController.removeListener(_onPromptTextChanged);
    _textController.dispose();
    _layerState?.removeListener(_onLayerStateChanged);
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
    _maybeAutoLoadImage();
  }

  /// Uploads the launch-supplied image once the model is ready.
  ///
  /// /upload answers 503 until the model finishes loading, so this waits for
  /// the first "online" health check rather than firing from initState.
  void _maybeAutoLoadImage() {
    if (_autoLoadStarted) return;
    if (_backendStatus != "online") return;
    if (_imageUrl == null || _imageUrl!.isEmpty) return;
    _autoLoadStarted = true;
    _loadImageFromUrl();
  }


  Future<ui.Image> _decodeImage(Uint8List bytes) {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, (ui.Image img) {
      return completer.complete(img);
    });
    return completer.future;
  }

  void _updateSegmentsFromResult() {
    // This method is called inside setState(), so we update _segments directly.
    debugPrint('Updating segments from result...');
    if (_result == null) {
      _segments = [];
      return;
    }

    final List<Segment> newSegments = [];

    try {
      // 1. Try to load Masks (RLE)
      final masks = _result!['masks'] as List?;
      if (masks != null && masks.isNotEmpty && masks[0] is Map) {
        for (var maskData in masks) {
          try {
            final rle = maskData as Map;
            final counts = (rle['counts'] as List).cast<int>();
            final size = (rle['size'] as List).cast<int>(); // [H, W]
            final w = size[1];
            
            final path = Path();
            int p = 0;
            bool isForeground = false; // First run is always background (0) per backend logic

            for (final count in counts) {
              if (isForeground) {
                // Add rects for this run of 1s
                int start = p;
                int end = p + count;
                int curr = start;
                while (curr < end) {
                  int y = curr ~/ w;
                  int x = curr % w;
                  int endOfRow = (y + 1) * w;
                  int runEnd = (end < endOfRow) ? end : endOfRow;
                  int len = runEnd - curr;
                  path.addRect(Rect.fromLTWH(x.toDouble(), y.toDouble(), len.toDouble(), 1.0));
                  curr = runEnd;
                }
              }
              p += count;
              isForeground = !isForeground;
            }
            newSegments.add(Segment(path: path));
          } catch (e, st) {
            debugPrint('Error parsing RLE mask: $e\n$st');
          }
        }
      } 
      // 2. Fallback to Boxes if no RLE masks found
      else {
        final boxes = _result!['boxes'] as List? ?? _result!['masks'] as List?;
        if (boxes != null) {
          for (var maskData in boxes) {
            if (maskData is List && maskData.length == 4) {
              final list = maskData.map((e) => (e as num).toDouble()).toList();
              final rect = Rect.fromLTRB(list[0], list[1], list[2], list[3]);
              final path = Path()..addRect(rect);
              newSegments.add(Segment(path: path));
            }
          }
        }
      }
    } catch (e, st) {
      debugPrint('Error updating segments from result: $e\n$st');
    }
    
    _segments = newSegments;
    debugPrint('Finished updating segments. Found ${newSegments.length} segments.');
  }

  Future<void> _loadImageFromUrl() async {
    if (_imageUrl == null || _imageUrl!.isEmpty) return;
    final url = _imageUrl!;

    setState(() {
      _isLoading = true;
      _imageSourceRunning = true;
      _imageSourceResult = [];
      _error = null;
    });

    try {
      // Download image from URL
      final response = await http.get(Uri.parse(url));
      if (response.statusCode != 200) {
        throw Exception("Failed to download image: ${response.statusCode}");
      }

      final imageBytes = response.bodyBytes;
      final decodedImage = await _decodeImage(imageBytes);

      // Upload to backend
      final uploadResponse = await _api.uploadImageBytes(
        imageBytes,
        filename: url.split('/').last.isEmpty ? "image.png" : url.split('/').last,
        sessionId: _sessionId,
      );

      if (uploadResponse != null && mounted) {
        setState(() {
          _sessionId ??= uploadResponse['session_id'] as String?;
          _imageBytes = imageBytes;
          _uiImage = decodedImage;
          _imageSize = Size(
            (uploadResponse['width'] as num).toDouble(),
            (uploadResponse['height'] as num).toDouble(),
          );
          _result = null;
          _segments = [];
          _imageSourceResult = [ResultDatum(label: 'URL', value: url)];
        });
      }
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _imageSourceRunning = false; });
    }
  }

  Future<void> _sendTextPrompt() async {
    if (_sessionId == null || _textController.text.isEmpty) return;
    setState(() { _isLoading = true; _textPromptRunning = true; _textPromptResult = []; });
    try {
      final response = await _api.segmentWithText(_sessionId!, _textController.text);
      if (!mounted) return;
      if (response != null) {
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _updateSegmentsFromResult();
          _textPromptResult = [const ResultDatum(label: 'Status', value: 'Done')];
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _textPromptRunning = false; });
    }
  }

  Future<void> _sendBoxPrompt(List<double> box) async {
    if (_sessionId == null) return;
    setState(() { _isLoading = true; _boxRunning = true; _boxResult = []; });
    try {
      final response = await _api.segmentWithBox(_sessionId!, box, _boxMode == "positive");
      if (!mounted) return;
      if (response != null) {
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _updateSegmentsFromResult();
          _boxResult = [const ResultDatum(label: 'Status', value: 'Done')];
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _boxRunning = false; });
    }
  }

  Future<void> _sendPointPrompt(List<double> point) async {
    if (_sessionId == null) return;
    setState(() { _isLoading = true; _pointRunning = true; _pointResult = []; });
    try {
      final response = await _api.segmentWithPoint(_sessionId!, point, _pointMode == "positive");
      if (!mounted) return;
      if (response != null) {
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _updateSegmentsFromResult();
          _pointResult = [const ResultDatum(label: 'Status', value: 'Done')];
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _pointRunning = false; });
    }
  }

  Future<void> _reset() async {
    if (_sessionId == null) return;
    setState(() { _isLoading = true; _resultsRunning = true; _resultsResult = []; });
    try {
      final response = await _api.resetPrompts(_sessionId!);
      if (!mounted) return;
      if (response != null) {
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _textController.clear();
          _updateSegmentsFromResult();
          _resultsResult = [const ResultDatum(label: 'Status', value: 'Done')];
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _resultsRunning = false; });
    }
  }

  Future<void> _saveMasks() async {
    if (_sessionId == null) return;
    setState(() { _isLoading = true; });
    try {
      final response = await _api.saveMasks(_sessionId!);
      if (!mounted) return;
      if (response != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Masks saved successfully")),
        );
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; });
    }
  }


  Future<void> _saveLayerState() async {
    if (_sessionId == null || _layerState == null) return;
    try {
      // Note: Ensure ApiService has saveSessionSettings(String, Map)
      await _api.saveSessionSettings(_sessionId!, {
        'view_layers': {
          'original': _layerState!.showOriginal,
          'masks': _layerState!.showMasks,
          'raw': _layerState!.showRaw,
          'final': _layerState!.showFinal,
        }
      });
    } catch (e) {
      // Ignore errors for background saves
      debugPrint("Failed to save layer state: $e");
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
            Text('SegForge Studio', style: TextStyle(fontWeight: FontWeight.bold)),
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
          // Sidebar + middle result column (scroll together)
          SizedBox(
            width: 520,
            // One group spanning the Prompt, Box Select and Point Select cards,
            // so their radios are mutually exclusive. `toggleable: true` on each
            // radio reports null when the selected one is tapped again, which is
            // what clears the mode.
            child: RadioGroup<SelectionMode>(
              groupValue: _selectedMode,
              onChanged: (mode) => setState(() => _selectedMode = mode),
              child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildCardRow(_buildSessionCard(),       const ResultCell(isRunning: false, data: [])),
                  const SizedBox(height: 16),
                  _buildCardRow(_buildUploadCard(),        ResultCell(isRunning: _imageSourceRunning, data: _imageSourceResult)),
                  const SizedBox(height: 16),
                  _buildCardRow(_buildTextPromptCard(),    ResultCell(isRunning: _textPromptRunning, data: _textPromptResult)),
                  const SizedBox(height: 16),
                  _buildCardRow(_buildBoxPromptCard(),     ResultCell(isRunning: _boxRunning, data: _boxResult)),
                  const SizedBox(height: 16),
                  _buildCardRow(_buildPointPromptCard(),   ResultCell(isRunning: _pointRunning, data: _pointResult)),
                  const SizedBox(height: 16),
                  _buildCardRow(_buildResultsCard(),       ResultCell(isRunning: _resultsRunning, data: _resultsResult)),
                  const SizedBox(height: 16),
                  _buildCardRow(_buildSegmentLayersCard(), const ResultCell(isRunning: false, data: [])),
                  const SizedBox(height: 16),
                  _buildCardRow(_buildSaveCard(),          const ResultCell(isRunning: false, data: [])),
                  if (_error != null) ...[
                    const SizedBox(height: 16),
                    SizedBox(
                      width: 340,
                      child: Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.errorContainer,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Theme.of(context).colorScheme.error),
                        ),
                        child: Text(
                          _error!,
                          style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer, fontSize: 12),
                        ),
                      ),
                    ),
                  ],
                ],
              ),
              ),
            ),
          ),

          // Main Canvas
          Expanded(
            child: Container(
              margin: const EdgeInsets.fromLTRB(0, 16, 16, 16),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
              ),
              clipBehavior: Clip.antiAlias,
              child: (_imageBytes == null)
                      ? Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.image_outlined, size: 64, color: Theme.of(context).colorScheme.onSurfaceVariant),
                              const SizedBox(height: 16),
                              Text("Enter an Image URL to start", style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant)),
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
                              mode: _selectedMode,
                              onBoxDrawn: _sendBoxPrompt,
                              onPointDrawn: _sendPointPrompt,
                            ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCardRow(Widget card, Widget cell) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(width: 340, child: card),
        const SizedBox(width: 8),
        SizedBox(width: 140, child: cell),
      ],
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
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.3)),
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

  Widget _buildBorderedCard(Widget child) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.6),
          width: 1,
        ),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: child,
      ),
    );
  }

  Widget _buildSessionCard() {
    return _buildBorderedCard(
      Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Session ID", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
            const SizedBox(height: 8),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              decoration: BoxDecoration(
                border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                _sessionId ?? "(none)",
                style: TextStyle(
                  fontSize: 13,
                  color: _sessionId != null ? Theme.of(context).colorScheme.onSurface : Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUploadCard() {
    const greenColor = Color(0xFF007F00);

    return _buildBorderedCard(
      Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Image URL", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
            const SizedBox(height: 8),
            if (_imageUrl != null && _imageUrl!.isNotEmpty)
              GestureDetector(
                onTap: _isLoading ? null : _loadImageFromUrl,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    border: Border.all(color: greenColor, width: 1.5),
                    borderRadius: BorderRadius.circular(24),
                    color: Colors.transparent,
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.link, size: 14, color: greenColor),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          _imageUrl!,
                          style: const TextStyle(fontSize: 12, color: greenColor),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                decoration: BoxDecoration(
                  border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  "(no image URL)",
                  style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.onSurfaceVariant),
                ),
              ),
            if (_imageSize != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  "${_imageSize!.width.toInt()} × ${_imageSize!.height.toInt()} px",
                  style: TextStyle(fontSize: 11, color: Theme.of(context).colorScheme.onSurfaceVariant),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTextPromptCard() {
    return _buildBorderedCard(
      Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.text_fields, size: 16),
                const SizedBox(width: 8),
                const Text("Prompt", style: TextStyle(fontWeight: FontWeight.bold)),
                const Spacer(),
                const Radio<SelectionMode>(
                  value: SelectionMode.prompt,
                  toggleable: true,
                ),
              ],
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _textController,
              maxLines: null,
              decoration: const InputDecoration(
                hintText: 'e.g. "cat", "wheel"',
                isDense: true,
                border: OutlineInputBorder(),
                contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 12),
              ),
              enabled: _sessionId != null && !_isLoading,
              onSubmitted: (_) => _sendTextPrompt(),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton(
                onPressed: (_sessionId == null || _textController.text.isEmpty || _isLoading) ? null : _sendTextPrompt,
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.white,
                  side: const BorderSide(color: Colors.white),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                ),
                child: const Text("Select"),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBoxPromptCard() {
    return _buildBorderedCard(
      Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.crop_free, size: 16),
                const SizedBox(width: 8),
                const Text("Box Select", style: TextStyle(fontWeight: FontWeight.bold)),
                const Spacer(),
                const Radio<SelectionMode>(
                  value: SelectionMode.box,
                  toggleable: true,
                ),
              ],
            ),
            const SizedBox(height: 12),
            IncludeExcludeToggle(
              value: _boxMode == "positive",
              onChanged: (isPositive) => setState(() => _boxMode = isPositive ? "positive" : "negative"),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPointPromptCard() {
    return _buildBorderedCard(
      Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.touch_app, size: 16),
                const SizedBox(width: 8),
                const Text("Point Select", style: TextStyle(fontWeight: FontWeight.bold)),
                const Spacer(),
                const Radio<SelectionMode>(
                  value: SelectionMode.point,
                  toggleable: true,
                ),
              ],
            ),
            const SizedBox(height: 12),
            IncludeExcludeToggle(
              value: _pointMode == "positive",
              onChanged: (isPositive) => setState(() => _pointMode = isPositive ? "positive" : "negative"),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsCard() {
    final maskCount = (_result?['masks'] as List?)?.length ?? 0;

    return _buildBorderedCard(
      Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.data_usage, size: 16),
                SizedBox(width: 8),
                Text("Objects Selected", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            _buildResultRow("Object count", maskCount.toString()),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton(
                onPressed: (_sessionId == null || _isLoading) ? null : _reset,
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.white,
                  side: const BorderSide(color: Colors.white),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                ),
                child: const Text("Clear Prompts"),
              ),
            ),
          ],
        ),
      ),
    );
  }


  Widget _buildSegmentLayersCard() {
    // This widget now manages its own state via a Consumer<LayerState>
    return const SegmentLayersCard();
  }

  Widget _buildSaveCard() {
    return _buildBorderedCard(
      Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.save, size: 16),
                SizedBox(width: 8),
                Text("Save", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: (_sessionId == null || _isLoading) ? null : _saveMasks,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF007F00),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                ),
                child: const Text("Save"),
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
          Text(label, style: TextStyle(fontSize: 13, color: Theme.of(context).colorScheme.onSurfaceVariant)),
          Text(value, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

}

class SegmentationCanvas extends StatefulWidget {
  final ui.Image uiImage;
  final List<Segment> segments;
  final Map<String, dynamic>? result;
  final bool isLoading;

  /// Active selection mode. Only [SelectionMode.box] accepts drags and only
  /// [SelectionMode.point] accepts taps; anything else leaves the canvas inert.
  final SelectionMode? mode;

  final Function(List<double>) onBoxDrawn;
  final Function(List<double>) onPointDrawn;

  const SegmentationCanvas({
    super.key,
    required this.uiImage,
    required this.segments,
    this.result,
    required this.isLoading,
    required this.mode,
    required this.onBoxDrawn,
    required this.onPointDrawn,
  });

  @override
  State<SegmentationCanvas> createState() => _SegmentationCanvasState();
}

class _SegmentationCanvasState extends State<SegmentationCanvas> {
  Offset? _startDrag;
  Offset? _currentDrag;

  bool get _boxing => widget.mode == SelectionMode.box;
  bool get _pointing => widget.mode == SelectionMode.point;

  @override
  void didUpdateWidget(SegmentationCanvas oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Leaving box mode mid-drag would otherwise leave the rubber-band rect
    // painted with no way to finish or cancel it.
    if (!_boxing && (_startDrag != null || _currentDrag != null)) {
      _startDrag = null;
      _currentDrag = null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // The core display layers
        LayeredSegmentationCanvas(
          originalImage: widget.uiImage,
          segments: widget.segments,
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

              // Gesture detector for prompting. Only the handlers belonging to
              // the active mode are attached: leaving the pan recognizer live
              // in point mode would let it claim taps before onTapUp fires.
              GestureDetector(
                onTapUp: _pointing
                    ? (details) {
                        final local = details.localPosition;
                        final nx = local.dx / widget.uiImage.width;
                        final ny = local.dy / widget.uiImage.height;
                        // Simple bounds check to ensure we clicked inside
                        if (nx >= 0 && nx <= 1 && ny >= 0 && ny <= 1) {
                          widget.onPointDrawn([nx, ny]);
                        }
                      }
                    : null,
                onPanStart: _boxing
                    ? (details) => setState(() {
                          _startDrag = details.localPosition;
                          _currentDrag = details.localPosition;
                        })
                    : null,
                onPanUpdate: _boxing
                    ? (details) => setState(() => _currentDrag = details.localPosition)
                    : null,
                onPanEnd: _boxing
                    ? (details) {
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
                      }
                    : null,
                child: MouseRegion(
                  cursor: (_pointing || _boxing)
                      ? SystemMouseCursors.precise
                      : MouseCursor.defer,
                  child: Container(color: Colors.transparent),
                ),
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

    final promptedPoints = result!['prompted_points'] as List?;
    if (promptedPoints != null) {
      for (var pp in promptedPoints) {
        final point = (pp['point'] as List).map((e) => (e as num).toDouble()).toList();
        final label = pp['label'] as int; // 1 or 0

        final paint = Paint()
          ..color = (label == 1) ? Colors.green : Colors.red
          ..style = PaintingStyle.fill;

        // Draw a dot for the point
        canvas.drawCircle(Offset(point[0], point[1]), 5.0, paint);
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
      ..color = Colors.blue.withValues(alpha: 0.1)
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