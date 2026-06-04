import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:file_picker/file_picker.dart';
import 'services/lora_train_service.dart';

enum _TrainState { idle, running, done, failed }

class LoraTrainCard extends StatefulWidget {
  final VoidCallback? onRunning;
  final VoidCallback? onComplete;
  final void Function(String? path)? onCompleteWithPath;

  const LoraTrainCard({
    super.key,
    this.onRunning,
    this.onComplete,
    this.onCompleteWithPath,
  });

  @override
  State<LoraTrainCard> createState() => _LoraTrainCardState();
}

class _LoraTrainCardState extends State<LoraTrainCard> {
  final _svc = LoraTrainService();

  // Path controllers
  final _metaCtrl   = TextEditingController();   // metadata.jsonl full path
  final _outDirCtrl = TextEditingController();   // output directory

  // Hyperparameter controllers
  final _modelCtrl  = TextEditingController(text: 'black-forest-labs/FLUX.1-dev');
  final _scriptCtrl = TextEditingController(text: 'train_dreambooth_lora_flux.py');
  final _lrCtrl     = TextEditingController(text: '0.0001');
  final _epochsCtrl = TextEditingController(text: '1');
  final _batchCtrl  = TextEditingController(text: '1');
  final _rankCtrl   = TextEditingController(text: '16');
  final _alphaCtrl  = TextEditingController(text: '16');
  final _resCtrl    = TextEditingController(text: '1024');
  String _mixedPrecision = 'bf16';

  _TrainState _state = _TrainState.idle;
  TrainStatus? _status;
  List<String> _logs = [];
  String? _outputPath;
  String? _error;
  bool _showConfig = false;
  bool _showLogs = false;

  StreamSubscription<TrainStatus>? _statusSub;
  Timer? _logTimer;
  String? _runId;

  @override
  void dispose() {
    _statusSub?.cancel();
    _logTimer?.cancel();
    for (final c in [
      _metaCtrl, _outDirCtrl, _modelCtrl, _scriptCtrl,
      _lrCtrl, _epochsCtrl, _batchCtrl, _rankCtrl, _alphaCtrl, _resCtrl,
    ]) { c.dispose(); }
    super.dispose();
  }

  // ── file pickers ──────────────────────────────────────────────────────────

  Future<void> _pickMetadataFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      dialogTitle: 'Select metadata.jsonl',
    );
    final path = result?.files.single.path;
    if (path != null) setState(() => _metaCtrl.text = path);
  }

  Future<void> _pickOutputDir() async {
    final dir = await FilePicker.platform.getDirectoryPath(
      dialogTitle: 'Select output directory',
    );
    if (dir != null) setState(() => _outDirCtrl.text = dir);
  }

  // ── training ──────────────────────────────────────────────────────────────

  Future<void> _train() async {
    final metaPath = _metaCtrl.text.trim();
    final outDir   = _outDirCtrl.text.trim();
    if (metaPath.isEmpty || outDir.isEmpty) return;

    // Derive dataset_dir from the metadata.jsonl path
    final datasetDir = metaPath.contains('/')
        ? metaPath.substring(0, metaPath.lastIndexOf('/'))
        : '.';

    widget.onRunning?.call();
    setState(() {
      _state = _TrainState.running;
      _logs = [];
      _outputPath = null;
      _error = null;
    });

    try {
      final config = TrainConfig(
        datasetDir: datasetDir,
        outputDir: outDir,
        modelPath: _modelCtrl.text.trim(),
        scriptPath: _scriptCtrl.text.trim(),
        rank: int.tryParse(_rankCtrl.text) ?? 16,
        alpha: int.tryParse(_alphaCtrl.text) ?? 16,
        learningRate: double.tryParse(_lrCtrl.text) ?? 1e-4,
        numTrainEpochs: int.tryParse(_epochsCtrl.text) ?? 1,
        batchSize: int.tryParse(_batchCtrl.text) ?? 1,
        resolution: int.tryParse(_resCtrl.text) ?? 1024,
        mixedPrecision: _mixedPrecision,
      );

      _runId = await _svc.startTraining(config);
      _startLogPolling(_runId!);

      _statusSub = _svc.watchStatus(_runId!).listen(
        (status) {
          if (!mounted) return;
          setState(() => _status = status);
          if (status.status == 'done') {
            _outputPath = status.outputPath;
            setState(() => _state = _TrainState.done);
            _stopPolling();
            widget.onComplete?.call();
            widget.onCompleteWithPath?.call(_outputPath);
          } else if (status.status == 'failed') {
            _error = status.lastError;
            setState(() => _state = _TrainState.failed);
            _stopPolling();
          }
        },
        onError: (e) {
          if (!mounted) return;
          setState(() { _state = _TrainState.failed; _error = e.toString(); });
          _stopPolling();
        },
      );
    } catch (e) {
      if (!mounted) return;
      setState(() { _state = _TrainState.failed; _error = e.toString(); });
    }
  }

  void _startLogPolling(String runId) {
    _logTimer = Timer.periodic(const Duration(seconds: 3), (_) async {
      final lines = await _svc.getLogs(runId);
      if (mounted) setState(() => _logs = lines);
    });
  }

  void _stopPolling() {
    _logTimer?.cancel();
    _statusSub?.cancel();
  }

  // ── helpers ───────────────────────────────────────────────────────────────

  String _formatElapsed(int s) {
    final m = s ~/ 60;
    return m > 0 ? '${m}m ${s % 60}s' : '${s}s';
  }

  bool get _canTrain =>
      _metaCtrl.text.trim().isNotEmpty &&
      _outDirCtrl.text.trim().isNotEmpty &&
      _state != _TrainState.running;

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
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── header ──
            Row(
              children: [
                const Icon(Icons.model_training, size: 16),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text('Train LoRA',
                      style: TextStyle(fontWeight: FontWeight.bold)),
                ),
                _stateChip(cs),
              ],
            ),
            const SizedBox(height: 14),

            // ── metadata.jsonl path ──
            Text('metadata.jsonl',
                style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
            const SizedBox(height: 4),
            Row(children: [
              Expanded(
                child: TextField(
                  controller: _metaCtrl,
                  decoration: deco.copyWith(hintText: '/path/to/metadata.jsonl'),
                  style: const TextStyle(fontSize: 12),
                  onChanged: (_) => setState(() {}),
                ),
              ),
              const SizedBox(width: 6),
              _browseButton(_pickMetadataFile),
            ]),
            const SizedBox(height: 10),

            // ── output directory ──
            Text('Output directory',
                style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
            const SizedBox(height: 4),
            Row(children: [
              Expanded(
                child: TextField(
                  controller: _outDirCtrl,
                  decoration: deco.copyWith(hintText: '/path/to/output'),
                  style: const TextStyle(fontSize: 12),
                  onChanged: (_) => setState(() {}),
                ),
              ),
              const SizedBox(width: 6),
              _browseButton(_pickOutputDir),
            ]),
            const SizedBox(height: 12),

            // ── config toggle ──
            GestureDetector(
              onTap: () => setState(() => _showConfig = !_showConfig),
              child: Row(children: [
                Icon(_showConfig ? Icons.expand_less : Icons.expand_more,
                    size: 16, color: cs.onSurfaceVariant),
                const SizedBox(width: 4),
                Text('Hyperparameters',
                    style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
              ]),
            ),
            if (_showConfig) ...[
              const SizedBox(height: 10),
              _configField('Base model', _modelCtrl, deco),
              const SizedBox(height: 6),
              _configField('Script path', _scriptCtrl, deco),
              const SizedBox(height: 6),
              Row(children: [
                Expanded(child: _configField('Learning rate', _lrCtrl, deco,
                    type: const TextInputType.numberWithOptions(decimal: true))),
                const SizedBox(width: 6),
                Expanded(child: _configField('Epochs', _epochsCtrl, deco,
                    type: TextInputType.number)),
              ]),
              const SizedBox(height: 6),
              Row(children: [
                Expanded(child: _configField('Batch size', _batchCtrl, deco,
                    type: TextInputType.number)),
                const SizedBox(width: 6),
                Expanded(child: _configField('Resolution', _resCtrl, deco,
                    type: TextInputType.number)),
              ]),
              const SizedBox(height: 6),
              Row(children: [
                Expanded(child: _configField('Rank (r)', _rankCtrl, deco,
                    type: TextInputType.number)),
                const SizedBox(width: 6),
                Expanded(child: _configField('Alpha', _alphaCtrl, deco,
                    type: TextInputType.number)),
              ]),
              const SizedBox(height: 6),
              DropdownButtonFormField<String>(
                initialValue: _mixedPrecision,
                decoration: deco.copyWith(labelText: 'Mixed precision'),
                style: const TextStyle(fontSize: 12),
                items: const [
                  DropdownMenuItem(value: 'bf16', child: Text('bf16')),
                  DropdownMenuItem(value: 'fp16', child: Text('fp16')),
                  DropdownMenuItem(value: 'no',   child: Text('no')),
                ],
                onChanged: (v) { if (v != null) setState(() => _mixedPrecision = v); },
              ),
            ],
            const SizedBox(height: 14),

            // ── train button ──
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: _canTrain ? _train : null,
                icon: _state == _TrainState.running
                    ? const SizedBox(
                        width: 16, height: 16,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.white),
                      )
                    : const Icon(Icons.play_arrow, size: 16),
                label: Text(_state == _TrainState.running ? 'Training…' : 'Train'),
              ),
            ),

            // ── progress ──
            if (_state == _TrainState.running) ...[
              const SizedBox(height: 10),
              _progressSection(cs),
            ],

            // ── success ──
            if (_state == _TrainState.done) ...[
              const SizedBox(height: 12),
              _successSection(cs),
            ],

            // ── error ──
            if (_state == _TrainState.failed && _error != null) ...[
              const SizedBox(height: 10),
              _errorSection(cs),
            ],

            // ── logs ──
            if (_logs.isNotEmpty) ...[
              const SizedBox(height: 10),
              _logsSection(cs),
            ],
          ],
        ),
      ),
    );
  }

  // ── sub-widgets ───────────────────────────────────────────────────────────

  Widget _stateChip(ColorScheme cs) {
    final (label, bg, fg) = switch (_state) {
      _TrainState.idle    => ('Idle',     cs.surfaceContainerHighest, cs.onSurfaceVariant),
      _TrainState.running => ('Running',  Colors.orange.withValues(alpha: 0.2), Colors.orange),
      _TrainState.done    => ('Done',     Colors.green.withValues(alpha: 0.15), Colors.green),
      _TrainState.failed  => ('Failed',   cs.errorContainer, cs.onErrorContainer),
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(12)),
      child: Text(label, style: TextStyle(fontSize: 11, color: fg, fontWeight: FontWeight.w600)),
    );
  }

  Widget _browseButton(VoidCallback onTap) => IconButton.outlined(
        icon: const Icon(Icons.folder_open, size: 16),
        onPressed: onTap,
        tooltip: 'Browse',
        style: IconButton.styleFrom(
          padding: const EdgeInsets.all(8),
          minimumSize: const Size(36, 36),
        ),
      );

  Widget _configField(String label, TextEditingController ctrl,
      InputDecoration base, {TextInputType? type}) {
    return TextField(
      controller: ctrl,
      decoration: base.copyWith(labelText: label),
      style: const TextStyle(fontSize: 12),
      keyboardType: type,
    );
  }

  Widget _progressSection(ColorScheme cs) {
    final current = _status?.currentStep ?? 0;
    final total   = _status?.totalSteps  ?? 0;
    final elapsed = _status?.elapsedSeconds ?? 0;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      LinearProgressIndicator(value: total > 0 ? current / total : null),
      const SizedBox(height: 4),
      Text(
        total > 0
            ? 'Step $current / $total  ·  ${_formatElapsed(elapsed)}'
            : 'Starting…  ·  ${_formatElapsed(elapsed)}',
        style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
      ),
    ]);
  }

  Widget _successSection(ColorScheme cs) {
    final path = _outputPath ?? '(path not returned by backend)';
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.green.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.green.withValues(alpha: 0.3)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Row(children: [
          Icon(Icons.check_circle_outline, size: 14, color: Colors.green),
          SizedBox(width: 6),
          Text('Training complete',
              style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.green)),
        ]),
        const SizedBox(height: 6),
        SelectableText(path, style: const TextStyle(fontSize: 11, fontFamily: 'monospace')),
        const SizedBox(height: 8),
        OutlinedButton.icon(
          onPressed: () {
            Clipboard.setData(ClipboardData(text: path));
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Path copied to clipboard'), duration: Duration(seconds: 2)),
            );
          },
          icon: const Icon(Icons.copy, size: 14),
          label: const Text('Copy path', style: TextStyle(fontSize: 12)),
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            minimumSize: Size.zero,
            tapTargetSize: MaterialTapTargetSize.shrinkWrap,
          ),
        ),
      ]),
    );
  }

  Widget _errorSection(ColorScheme cs) => Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: cs.errorContainer,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(Icons.error_outline, size: 16, color: cs.onErrorContainer),
          const SizedBox(width: 8),
          Expanded(
            child: SelectableText(_error!,
                style: TextStyle(fontSize: 11, color: cs.onErrorContainer)),
          ),
        ]),
      );

  Widget _logsSection(ColorScheme cs) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          GestureDetector(
            onTap: () => setState(() => _showLogs = !_showLogs),
            child: Row(children: [
              Icon(_showLogs ? Icons.expand_less : Icons.expand_more,
                  size: 16, color: cs.onSurfaceVariant),
              const SizedBox(width: 4),
              Text('Logs (${_logs.length} lines)',
                  style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
            ]),
          ),
          if (_showLogs)
            Container(
              margin: const EdgeInsets.only(top: 6),
              height: 180,
              decoration: BoxDecoration(
                color: cs.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(6),
              ),
              child: _LogView(lines: _logs),
            ),
        ],
      );
}

// ── auto-scrolling log view ──────────────────────────────────────────────────

class _LogView extends StatefulWidget {
  final List<String> lines;
  const _LogView({required this.lines});

  @override
  State<_LogView> createState() => _LogViewState();
}

class _LogViewState extends State<_LogView> {
  final _scroll = ScrollController();

  @override
  void didUpdateWidget(_LogView old) {
    super.didUpdateWidget(old);
    if (widget.lines.length != old.lines.length) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scroll.hasClients) _scroll.jumpTo(_scroll.position.maxScrollExtent);
      });
    }
  }

  @override
  void dispose() { _scroll.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return ListView.builder(
      controller: _scroll,
      padding: const EdgeInsets.all(8),
      itemCount: widget.lines.length,
      itemBuilder: (_, i) => Text(
        widget.lines[i],
        style: TextStyle(fontSize: 10, fontFamily: 'monospace', color: cs.onSurfaceVariant),
      ),
    );
  }
}
