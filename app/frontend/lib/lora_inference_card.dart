import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:desktop_drop/desktop_drop.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import 'services/lora_infer_service.dart';

enum _InferState { idle, running, done, failed }

class LoraInferenceCard extends StatefulWidget {
  final String? initialLoraPath;  // pre-populated from training card
  final VoidCallback? onRunning;
  final VoidCallback? onComplete;
  final void Function(Uint8List bytes)? onImageGenerated;

  const LoraInferenceCard({
    super.key,
    this.initialLoraPath,
    this.onRunning,
    this.onComplete,
    this.onImageGenerated,
  });

  @override
  State<LoraInferenceCard> createState() => _LoraInferenceCardState();
}

class _LoraInferenceCardState extends State<LoraInferenceCard> {
  final _svc = LoraInferService();

  // ── path / model ──────────────────────────────────────────────────────────
  final _loraCtrl   = TextEditingController();
  final _modelCtrl  = TextEditingController(text: 'runwayml/stable-diffusion-v1-5');
  final _promptCtrl = TextEditingController();
  final _outDirCtrl = TextEditingController();

  // ── inference params ──────────────────────────────────────────────────────
  double _loraStrength  = 0.8;
  final _stepsCtrl      = TextEditingController(text: '20');
  final _guidanceCtrl   = TextEditingController(text: '7.5');
  final _seedCtrl       = TextEditingController(text: '42');
  bool _showParams      = false;

  // ── continuity image (img2img) ────────────────────────────────────────────
  Uint8List? _continuityBytes;
  String?    _continuityName;
  double     _denoiseStrength = 0.75;
  bool       _isDragOver      = false;

  // ── runtime state ─────────────────────────────────────────────────────────
  _InferState _state    = _InferState.idle;
  double _progress      = 0.0;
  Uint8List? _imageBytes;
  String?    _imagePath;
  String?    _triggerHint;
  String?    _error;
  String?    _runId;
  StreamSubscription<InferStatus>? _statusSub;

  // ── provider state ────────────────────────────────────────────────────────
  List<ProviderInfo> _providers = [];
  String _activeProvider        = 'mlx';
  bool _switchingProvider       = false;
  // Cloud Run URL — shown in a dialog when the user selects cloud_run.
  final _cloudRunUrlCtrl = TextEditingController();

  @override
  void initState() {
    super.initState();
    if (widget.initialLoraPath != null) {
      _loraCtrl.text = widget.initialLoraPath!;
      _onLoraPathChanged(widget.initialLoraPath!);
    }
    _loadPersistedModel();
    _modelCtrl.addListener(_persistModel);
    _loadProviders();
  }

  @override
  void didUpdateWidget(LoraInferenceCard old) {
    super.didUpdateWidget(old);
    final incoming = widget.initialLoraPath;
    if (incoming != null && incoming != old.initialLoraPath && _loraCtrl.text.isEmpty) {
      _loraCtrl.text = incoming;
      _onLoraPathChanged(incoming);
    }
  }

  @override
  void dispose() {
    _statusSub?.cancel();
    _loraCtrl.dispose();
    _modelCtrl.dispose();
    _promptCtrl.dispose();
    _outDirCtrl.dispose();
    _stepsCtrl.dispose();
    _guidanceCtrl.dispose();
    _seedCtrl.dispose();
    _cloudRunUrlCtrl.dispose();
    super.dispose();
  }

  // ── persistence (path_provider + dart:io, no extra deps) ─────────────────

  Future<File> get _prefsFile async {
    final dir = await getApplicationSupportDirectory();
    return File('${dir.path}/lora_inference_prefs.json');
  }

  Future<void> _loadPersistedModel() async {
    try {
      final file = await _prefsFile;
      if (!await file.exists()) return;
      final data = json.decode(await file.readAsString()) as Map<String, dynamic>;
      final saved = data['model_path'] as String?;
      if (saved != null && saved.isNotEmpty && mounted) {
        setState(() => _modelCtrl.text = saved);
      }
    } catch (_) {}
  }

  Future<void> _persistModel() async {
    try {
      final file = await _prefsFile;
      await file.writeAsString(json.encode({'model_path': _modelCtrl.text}));
    } catch (_) {}
  }

  // ── provider management ───────────────────────────────────────────────────

  Future<void> _loadProviders() async {
    try {
      final list = await _svc.getProviders();
      if (!mounted) return;
      setState(() {
        _providers = list;
        final active = list.where((p) => p.active).firstOrNull;
        if (active != null) _activeProvider = active.name;
      });
    } catch (_) {}
  }

  Future<void> _switchProvider(String name) async {
    if (name == 'cloud_run' && _cloudRunUrlCtrl.text.trim().isEmpty) {
      // Ask for the URL before switching.
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Cloud Run URL'),
          content: TextField(
            controller: _cloudRunUrlCtrl,
            decoration: const InputDecoration(
              hintText: 'https://flux-infer-xxxx-uc.a.run.app',
              border: OutlineInputBorder(),
            ),
            autofocus: true,
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Connect'),
            ),
          ],
        ),
      );
      if (confirmed != true || !mounted) return;
    }

    setState(() => _switchingProvider = true);
    try {
      final list = await _svc.setProvider(
        name,
        url: name == 'cloud_run' ? _cloudRunUrlCtrl.text.trim() : '',
      );
      if (!mounted) return;
      setState(() {
        _providers = list;
        _activeProvider = name;
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Provider switch failed: $e'),
            duration: const Duration(seconds: 3)),
      );
    } finally {
      if (mounted) setState(() => _switchingProvider = false);
    }
  }

  // ── trigger word hint (reads metadata.jsonl sibling of the LoRA file) ─────

  Future<void> _onLoraPathChanged(String loraPath) async {
    if (loraPath.isEmpty) { setState(() => _triggerHint = null); return; }
    try {
      final metaFile = File('${File(loraPath).parent.path}/metadata.jsonl');
      if (!await metaFile.exists()) return;
      final lines = await metaFile.readAsLines();
      for (final line in lines) {
        if (line.trim().isEmpty) continue;
        final rec = json.decode(line.trim()) as Map<String, dynamic>;
        final text = rec['text'] as String? ?? '';
        if (text.isNotEmpty) {
          final words = text.trim().split(' ').where((w) => w.isNotEmpty).take(4).join(' ');
          if (mounted) setState(() => _triggerHint = words);
          return;
        }
      }
    } catch (_) {}
  }

  // ── file pickers ──────────────────────────────────────────────────────────

  Future<void> _pickLoraFile() async {
    final result = await FilePicker.platform.pickFiles(
      dialogTitle: 'Select LoRA .safetensors file',
      type: FileType.any,
    );
    final path = result?.files.single.path;
    if (path == null) return;
    setState(() => _loraCtrl.text = path);
    await _onLoraPathChanged(path);
  }

  Future<void> _pickOutputDir() async {
    final dir = await FilePicker.platform.getDirectoryPath(
      dialogTitle: 'Select output directory',
    );
    if (dir != null) setState(() => _outDirCtrl.text = dir);
  }

  Future<void> _saveAs() async {
    if (_imageBytes == null) return;
    final path = await FilePicker.platform.saveFile(
      dialogTitle: 'Save generated image',
      fileName: 'inference_output.png',
    );
    if (path == null) return;
    await File(path).writeAsBytes(_imageBytes!);
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Image saved'), duration: Duration(seconds: 2)),
      );
    }
  }

  Future<void> _pickContinuityImage() async {
    final result = await FilePicker.platform.pickFiles(
      dialogTitle: 'Select reference image',
      type: FileType.image,
      withData: true,
    );
    final f = result?.files.single;
    if (f == null) return;
    final bytes = f.bytes ?? (f.path != null ? await File(f.path!).readAsBytes() : null);
    if (bytes == null) return;
    setState(() { _continuityBytes = bytes; _continuityName = f.name; });
  }

  void _clearContinuityImage() =>
      setState(() { _continuityBytes = null; _continuityName = null; });

  // ── generation ────────────────────────────────────────────────────────────

  bool get _canGenerate =>
      _loraCtrl.text.trim().isNotEmpty &&
      _promptCtrl.text.trim().isNotEmpty &&
      _state != _InferState.running;

  InferRequest _buildRequest() {
    final String? continuityB64 = _continuityBytes != null
        ? base64Encode(_continuityBytes!)
        : null;
    return InferRequest(
      loraPath: _loraCtrl.text.trim(),
      modelPath: _modelCtrl.text.trim(),
      prompt: _promptCtrl.text.trim(),
      loraStrength: _loraStrength,
      steps: int.tryParse(_stepsCtrl.text) ?? 28,
      guidanceScale: double.tryParse(_guidanceCtrl.text) ?? 3.5,
      seed: int.tryParse(_seedCtrl.text) ?? 42,
      outputDir: _outDirCtrl.text.trim(),
      continuityImageB64: continuityB64,
      denoiseStrength: _denoiseStrength,
    );
  }

  Future<void> _generate() async {
    if (!_canGenerate) return;
    widget.onRunning?.call();
    _statusSub?.cancel();
    // Tell the backend to stop the previous run if one is active.
    if (_runId != null) {
      await _svc.cancel(_runId!);
    }
    setState(() {
      _state      = _InferState.running;
      _progress   = 0.0;
      _imageBytes = null;
      _imagePath  = null;
      _error      = null;
      _runId      = null;
    });

    try {
      final runId = await _svc.startGeneration(_buildRequest());
      if (!mounted) return;
      setState(() => _runId = runId);

      _statusSub = _svc.watchStatus(runId).listen(
        (status) async {
          if (!mounted) return;
          setState(() => _progress = status.progress);

          if (status.status == 'done') {
            _statusSub?.cancel();
            try {
              final bytes = await _svc.fetchImage(runId);
              if (!mounted) return;
              setState(() {
                _imageBytes = bytes;
                _imagePath  = status.outputPath;
                _state      = _InferState.done;
              });
              widget.onImageGenerated?.call(bytes);
              widget.onComplete?.call();
            } catch (e) {
              if (!mounted) return;
              setState(() { _state = _InferState.failed; _error = e.toString(); });
            }
          } else if (status.isTerminal) {
            _statusSub?.cancel();
            if (!mounted) return;
            setState(() {
              _state = _InferState.failed;
              _error = status.lastError ?? 'Generation failed';
            });
          }
        },
        onError: (e) {
          if (!mounted) return;
          setState(() { _state = _InferState.failed; _error = e.toString(); });
          _statusSub?.cancel();
        },
      );
    } catch (e) {
      if (!mounted) return;
      setState(() { _state = _InferState.failed; _error = e.toString(); });
    }
  }

  void _randomiseSeed() {
    setState(() => _seedCtrl.text = Random().nextInt(0x7FFFFFFF).toString());
  }

  // ── build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    const deco = InputDecoration(
      border: OutlineInputBorder(),
      contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      isDense: true,
    );

    return Card(
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _header(cs),
            const SizedBox(height: 14),

            // ── LoRA file ──
            _label('LoRA file (.safetensors)', cs),
            const SizedBox(height: 4),
            _pathRow(_loraCtrl, 'path/to/lora.safetensors', _pickLoraFile, deco),
            const SizedBox(height: 10),

            // ── Base model ──
            _label('Base model', cs),
            const SizedBox(height: 4),
            TextField(
              controller: _modelCtrl,
              decoration: deco.copyWith(hintText: 'runwayml/stable-diffusion-v1-5'),
              style: const TextStyle(fontSize: 12),
            ),
            const SizedBox(height: 10),

            // ── Prompt ──
            _label('Prompt', cs),
            const SizedBox(height: 4),
            TextField(
              controller: _promptCtrl,
              maxLines: 3,
              decoration: deco.copyWith(
                hintText: _triggerHint != null
                    ? 'e.g. "$_triggerHint, …"'
                    : 'Describe what to generate…',
              ),
              style: const TextStyle(fontSize: 12),
              onChanged: (_) => setState(() {}),
            ),
            const SizedBox(height: 12),

            // ── Continuity image ──
            _continuityZone(cs),
            const SizedBox(height: 12),

            // ── Parameters toggle ──
            GestureDetector(
              onTap: () => setState(() => _showParams = !_showParams),
              child: Row(children: [
                Icon(_showParams ? Icons.expand_less : Icons.expand_more,
                    size: 16, color: cs.onSurfaceVariant),
                const SizedBox(width: 4),
                Text('Parameters',
                    style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
              ]),
            ),
            if (_showParams) ...[
              const SizedBox(height: 10),
              _paramsSection(cs, deco),
            ],
            const SizedBox(height: 14),

            // ── Generate button ──
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: _canGenerate ? _generate : null,
                icon: _state == _InferState.running
                    ? const SizedBox(
                        width: 16, height: 16,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.white),
                      )
                    : const Icon(Icons.auto_awesome, size: 16),
                label: Text(
                  _state == _InferState.running ? 'Generating…' : 'Generate',
                ),
              ),
            ),

            // ── Progress ──
            if (_state == _InferState.running) ...[
              const SizedBox(height: 10),
              LinearProgressIndicator(value: _progress > 0 ? _progress : null),
              const SizedBox(height: 4),
              Text(
                'Step ${(_progress * (int.tryParse(_stepsCtrl.text) ?? 28)).round()}'
                ' / ${_stepsCtrl.text}',
                style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
              ),
            ],

            // ── Output image (inline only when no external canvas is wired up) ──
            if (_state == _InferState.done && _imageBytes != null) ...[
              const SizedBox(height: 14),
              if (widget.onImageGenerated == null)
                _outputSection(cs)
              else
                _outputDoneRow(cs),
            ],

            // ── Error ──
            if (_state == _InferState.failed && _error != null) ...[
              const SizedBox(height: 10),
              _errorSection(cs),
            ],
          ],
        ),
      ),
    );
  }

  // ── sub-widgets ───────────────────────────────────────────────────────────

  Widget _header(ColorScheme cs) {
    final (stateLabel, stateBg, stateFg) = switch (_state) {
      _InferState.idle    => ('Idle',       cs.surfaceContainerHighest,           cs.onSurfaceVariant),
      _InferState.running => ('Generating', Colors.orange.withValues(alpha: 0.2), Colors.orange),
      _InferState.done    => ('Done',       Colors.green.withValues(alpha: 0.15), Colors.green),
      _InferState.failed  => ('Failed',     cs.errorContainer,                    cs.onErrorContainer),
    };

    // Build provider label list for the dropdown.
    final providerItems = _providers.isEmpty
        ? [
            const DropdownMenuItem(value: 'mlx',       child: Text('MLX',        style: TextStyle(fontSize: 11))),
            const DropdownMenuItem(value: 'cloud_run', child: Text('Cloud Run',  style: TextStyle(fontSize: 11))),
          ]
        : _providers.map((p) => DropdownMenuItem(
              value: p.name,
              child: Text(p.label, style: const TextStyle(fontSize: 11)),
            )).toList();

    return Row(children: [
      const Icon(Icons.image_outlined, size: 16),
      const SizedBox(width: 8),
      const Expanded(
        child: Text('LoRA Inference',
            style: TextStyle(fontWeight: FontWeight.bold)),
      ),
      // ── provider picker ──
      if (_switchingProvider)
        const SizedBox(
          width: 14, height: 14,
          child: CircularProgressIndicator(strokeWidth: 2),
        )
      else
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
          decoration: BoxDecoration(
            border: Border.all(color: cs.outlineVariant),
            borderRadius: BorderRadius.circular(10),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: _activeProvider,
              isDense: true,
              style: TextStyle(fontSize: 11, color: cs.onSurface),
              items: providerItems,
              onChanged: _state == _InferState.running
                  ? null
                  : (v) { if (v != null) _switchProvider(v); },
            ),
          ),
        ),
      const SizedBox(width: 6),
      // ── state chip ──
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
        decoration: BoxDecoration(color: stateBg, borderRadius: BorderRadius.circular(12)),
        child: Text(stateLabel,
            style: TextStyle(fontSize: 11, color: stateFg, fontWeight: FontWeight.w600)),
      ),
    ]);
  }

  Widget _label(String text, ColorScheme cs) => Text(
        text,
        style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
      );

  Widget _pathRow(TextEditingController ctrl, String hint,
      VoidCallback onBrowse, InputDecoration base) {
    return Row(children: [
      Expanded(
        child: TextField(
          controller: ctrl,
          decoration: base.copyWith(hintText: hint),
          style: const TextStyle(fontSize: 12),
          onChanged: (v) => setState(() {}),
        ),
      ),
      const SizedBox(width: 6),
      IconButton.outlined(
        icon: const Icon(Icons.folder_open, size: 16),
        onPressed: onBrowse,
        tooltip: 'Browse',
        style: IconButton.styleFrom(
          padding: const EdgeInsets.all(8),
          minimumSize: const Size(36, 36),
        ),
      ),
    ]);
  }

  Widget _paramsSection(ColorScheme cs, InputDecoration base) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      // LoRA strength slider
      Row(children: [
        Text('LoRA strength', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
        const Spacer(),
        Text(_loraStrength.toStringAsFixed(2), style: const TextStyle(fontSize: 11)),
      ]),
      Slider(
        value: _loraStrength,
        min: 0.0, max: 1.0, divisions: 20,
        onChanged: (v) => setState(() => _loraStrength = v),
      ),
      const SizedBox(height: 4),
      // Steps + guidance
      Row(children: [
        Expanded(child: TextField(
          controller: _stepsCtrl,
          decoration: base.copyWith(labelText: 'Steps'),
          style: const TextStyle(fontSize: 12),
          keyboardType: TextInputType.number,
        )),
        const SizedBox(width: 6),
        Expanded(child: TextField(
          controller: _guidanceCtrl,
          decoration: base.copyWith(labelText: 'Guidance scale'),
          style: const TextStyle(fontSize: 12),
          keyboardType: const TextInputType.numberWithOptions(decimal: true),
        )),
      ]),
      const SizedBox(height: 6),
      // Seed + randomize
      Row(children: [
        Expanded(child: TextField(
          controller: _seedCtrl,
          decoration: base.copyWith(labelText: 'Seed'),
          style: const TextStyle(fontSize: 12),
          keyboardType: TextInputType.number,
        )),
        const SizedBox(width: 6),
        IconButton.outlined(
          icon: const Icon(Icons.casino_outlined, size: 16),
          onPressed: _randomiseSeed,
          tooltip: 'Randomise seed',
          style: IconButton.styleFrom(
            padding: const EdgeInsets.all(8),
            minimumSize: const Size(36, 36),
          ),
        ),
      ]),
      const SizedBox(height: 6),
      // Output directory
      _label('Output directory', cs),
      const SizedBox(height: 4),
      _pathRow(_outDirCtrl, 'Leave blank to use default storage dir', _pickOutputDir, base),
    ]);
  }

  Widget _continuityZone(ColorScheme cs) {
    // ── loaded: thumbnail + clear + denoise slider ────────────────────────
    if (_continuityBytes != null) {
      return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _label('Continuity image (img2img)', cs),
        const SizedBox(height: 4),
        Stack(clipBehavior: Clip.none, children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.memory(_continuityBytes!,
                height: 80, width: double.infinity, fit: BoxFit.cover),
          ),
          Positioned(
            top: 4, right: 4,
            child: GestureDetector(
              onTap: _clearContinuityImage,
              child: Container(
                padding: const EdgeInsets.all(3),
                decoration: BoxDecoration(
                  color: cs.errorContainer,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Icon(Icons.close, size: 12, color: cs.onErrorContainer),
              ),
            ),
          ),
        ]),
        if (_continuityName != null)
          Padding(
            padding: const EdgeInsets.only(top: 3),
            child: Text(_continuityName!,
                style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant)),
          ),
        const SizedBox(height: 10),
        // Denoise strength — only shown when a continuity image is loaded
        Row(children: [
          Text('Denoise strength',
              style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
          const Spacer(),
          Text(_denoiseStrength.toStringAsFixed(2),
              style: const TextStyle(fontSize: 11)),
        ]),
        Slider(
          value: _denoiseStrength,
          min: 0.0, max: 1.0, divisions: 20,
          label: _denoiseStrength.toStringAsFixed(2),
          onChanged: (v) => setState(() => _denoiseStrength = v),
        ),
      ]);
    }

    // ── empty: drag-and-drop zone with file picker fallback ───────────────
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _label('Continuity image — optional (img2img)', cs),
      const SizedBox(height: 4),
      DropTarget(
        onDragDone: (detail) async {
          final xfile = detail.files.firstOrNull;
          if (xfile == null) return;
          final bytes = await File(xfile.path).readAsBytes();
          if (mounted) {
            setState(() {
              _continuityBytes = bytes;
              _continuityName  = xfile.name;
              _isDragOver      = false;
            });
          }
        },
        onDragEntered: (_) => setState(() => _isDragOver = true),
        onDragExited:  (_) => setState(() => _isDragOver = false),
        child: GestureDetector(
          onTap: _pickContinuityImage,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 150),
            height: 60,
            decoration: BoxDecoration(
              border: Border.all(
                color: _isDragOver ? cs.primary : cs.outlineVariant,
                width: _isDragOver ? 2 : 1,
              ),
              borderRadius: BorderRadius.circular(8),
              color: _isDragOver
                  ? cs.primary.withValues(alpha: 0.08)
                  : cs.surfaceContainerLow,
            ),
            child: Center(
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.add_photo_alternate_outlined,
                    size: 18, color: cs.onSurfaceVariant),
                const SizedBox(width: 8),
                Text(
                  _isDragOver
                      ? 'Drop image here'
                      : 'Drop reference image or tap to browse',
                  style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant),
                ),
              ]),
            ),
          ),
        ),
      ),
    ]);
  }

  Widget _outputDoneRow(ColorScheme cs) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.green.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.green.withValues(alpha: 0.3)),
      ),
      child: Row(children: [
        const Icon(Icons.check_circle_outline, size: 14, color: Colors.green),
        const SizedBox(width: 8),
        const Expanded(
          child: Text('Image generated — see canvas',
              style: TextStyle(fontSize: 11, color: Colors.green)),
        ),
        IconButton(
          icon: const Icon(Icons.save_alt, size: 14),
          tooltip: 'Save image',
          visualDensity: VisualDensity.compact,
          onPressed: _saveAs,
        ),
      ]),
    );
  }

  Widget _outputSection(ColorScheme cs) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Image.memory(
          _imageBytes!,
          width: double.infinity,
          fit: BoxFit.contain,
        ),
      ),
      const SizedBox(height: 8),
      if (_imagePath != null)
        SelectableText(
          _imagePath!,
          style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant,
              fontFamily: 'monospace'),
        ),
      const SizedBox(height: 8),
      Row(children: [
        Expanded(
          child: OutlinedButton.icon(
            onPressed: _saveAs,
            icon: const Icon(Icons.save_alt, size: 14),
            label: const Text('Save As', style: TextStyle(fontSize: 12)),
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            ),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: OutlinedButton.icon(
            onPressed: _imagePath == null ? null : () {
              Clipboard.setData(ClipboardData(text: _imagePath!));
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Path copied'),
                    duration: Duration(seconds: 2)),
              );
            },
            icon: const Icon(Icons.copy, size: 14),
            label: const Text('Copy path', style: TextStyle(fontSize: 12)),
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            ),
          ),
        ),
      ]),
    ]);
  }

  Widget _errorSection(ColorScheme cs) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: cs.errorContainer,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(Icons.error_outline, size: 16, color: cs.onErrorContainer),
        const SizedBox(width: 8),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            SelectableText(_error!,
                style: TextStyle(fontSize: 11, color: cs.onErrorContainer)),
            const SizedBox(height: 8),
            OutlinedButton(
              onPressed: _generate,
              style: OutlinedButton.styleFrom(
                foregroundColor: cs.onErrorContainer,
                side: BorderSide(color: cs.onErrorContainer.withValues(alpha: 0.4)),
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                minimumSize: Size.zero,
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
              child: const Text('Retry', style: TextStyle(fontSize: 12)),
            ),
          ]),
        ),
      ]),
    );
  }
}
