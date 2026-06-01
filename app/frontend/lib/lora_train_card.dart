import 'dart:async';
import 'package:flutter/material.dart';
import 'services/lora_train_service.dart';

enum _CardState { waiting, ready, running, done, failed }

class LoraTrainCard extends StatefulWidget {
  final VoidCallback? onRunning;
  final VoidCallback? onComplete;

  const LoraTrainCard({super.key, this.onRunning, this.onComplete});

  @override
  State<LoraTrainCard> createState() => _LoraTrainCardState();
}

class _LoraTrainCardState extends State<LoraTrainCard> {
  final _svc = LoraTrainService();

  // Config controllers
  final _datasetCtrl = TextEditingController();
  final _outputCtrl = TextEditingController();
  final _rankCtrl = TextEditingController(text: '16');
  final _lrCtrl = TextEditingController(text: '0.0001');
  final _epochsCtrl = TextEditingController(text: '1');
  final _resCtrl = TextEditingController(text: '1024');
  final _scriptCtrl = TextEditingController(text: 'train_dreambooth_lora_flux.py');
  String _mixedPrecision = 'bf16';

  _CardState _state = _CardState.waiting;
  ReadinessResult? _readiness;
  TrainStatus? _trainStatus;
  List<String> _logs = [];
  String? _error;
  bool _showLogs = false;

  Timer? _readinessTimer;
  Timer? _logTimer;
  StreamSubscription<TrainStatus>? _statusSub;

  @override
  void initState() {
    super.initState();
    _startReadinessPoll();
  }

  @override
  void dispose() {
    _readinessTimer?.cancel();
    _logTimer?.cancel();
    _statusSub?.cancel();
    _datasetCtrl.dispose();
    _outputCtrl.dispose();
    _rankCtrl.dispose();
    _lrCtrl.dispose();
    _epochsCtrl.dispose();
    _resCtrl.dispose();
    _scriptCtrl.dispose();
    super.dispose();
  }

  void _startReadinessPoll() {
    _checkReadiness();
    _readinessTimer = Timer.periodic(const Duration(seconds: 10), (_) {
      if (_state == _CardState.waiting) _checkReadiness();
    });
  }

  Future<void> _checkReadiness() async {
    final dir = _datasetCtrl.text.trim();
    if (dir.isEmpty) return;
    try {
      final r = await _svc.checkReadiness(dir);
      if (!mounted) return;
      setState(() {
        _readiness = r;
        if (r.ready && _state == _CardState.waiting) {
          _state = _CardState.ready;
          _readinessTimer?.cancel();
        }
      });
    } catch (_) {}
  }

  Future<void> _run() async {
    widget.onRunning?.call();
    setState(() {
      _state = _CardState.running;
      _trainStatus = null;
      _logs = [];
      _error = null;
    });

    try {
      final config = TrainConfig(
        datasetDir: _datasetCtrl.text.trim(),
        outputDir: _outputCtrl.text.trim(),
        scriptPath: _scriptCtrl.text.trim(),
        rank: int.tryParse(_rankCtrl.text) ?? 16,
        learningRate: double.tryParse(_lrCtrl.text) ?? 1e-4,
        numTrainEpochs: int.tryParse(_epochsCtrl.text) ?? 1,
        resolution: int.tryParse(_resCtrl.text) ?? 1024,
        mixedPrecision: _mixedPrecision,
      );

      final runId = await _svc.startTraining(config);
      _startLogPolling(runId);

      _statusSub = _svc.watchStatus(runId).listen(
        (status) {
          if (!mounted) return;
          setState(() {
            _trainStatus = status;
            if (status.status == 'done') {
              _state = _CardState.done;
              _stopPolling();
              widget.onComplete?.call();
            } else if (status.status == 'failed') {
              _state = _CardState.failed;
              _error = status.lastError;
              _stopPolling();
            }
          });
        },
        onError: (e) {
          if (!mounted) return;
          setState(() {
            _state = _CardState.failed;
            _error = e.toString();
          });
          _stopPolling();
        },
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _state = _CardState.failed;
        _error = e.toString();
      });
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

  String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  String _formatElapsed(int seconds) {
    final m = seconds ~/ 60;
    final s = seconds % 60;
    return m > 0 ? '${m}m ${s}s' : '${s}s';
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    Color? leftBorderColor;
    if (_state == _CardState.done) leftBorderColor = Colors.green;
    if (_state == _CardState.failed) leftBorderColor = cs.error;

    return Card(
      clipBehavior: Clip.antiAlias,
      shape: leftBorderColor != null
          ? Border(left: BorderSide(color: leftBorderColor, width: 4))
              .toShapeBorder()
          : null,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(cs),
            const SizedBox(height: 12),
            _buildBody(cs),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(ColorScheme cs) {
    return Row(
      children: [
        Icon(Icons.model_training, size: 16,
            color: _state == _CardState.waiting
                ? cs.onSurfaceVariant
                : cs.onSurface),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            'Train LoRA',
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: _state == _CardState.waiting ? cs.onSurfaceVariant : null,
            ),
          ),
        ),
        _buildStateChip(cs),
      ],
    );
  }

  Widget _buildStateChip(ColorScheme cs) {
    String label;
    Color bg;
    Color fg;
    switch (_state) {
      case _CardState.waiting:
        label = 'Waiting';
        bg = cs.surfaceContainerHighest;
        fg = cs.onSurfaceVariant;
      case _CardState.ready:
        label = 'Ready';
        bg = cs.primaryContainer;
        fg = cs.onPrimaryContainer;
      case _CardState.running:
        label = 'Running';
        bg = Colors.orange.withValues(alpha: 0.2);
        fg = Colors.orange;
      case _CardState.done:
        label = 'Done';
        bg = Colors.green.withValues(alpha: 0.15);
        fg = Colors.green;
      case _CardState.failed:
        label = 'Failed';
        bg = cs.errorContainer;
        fg = cs.onErrorContainer;
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(12)),
      child: Text(label, style: TextStyle(fontSize: 11, color: fg, fontWeight: FontWeight.w600)),
    );
  }

  Widget _buildBody(ColorScheme cs) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildDatasetDirField(cs),
        const SizedBox(height: 8),
        _buildSubtitle(cs),
        if (_state == _CardState.ready || _state == _CardState.waiting) ...[
          const SizedBox(height: 10),
          _buildConfigFields(cs),
        ],
        const SizedBox(height: 12),
        _buildRunButton(cs),
        if (_state == _CardState.running) ...[
          const SizedBox(height: 10),
          _buildProgress(cs),
        ],
        if (_state == _CardState.done && _trainStatus?.outputPath != null) ...[
          const SizedBox(height: 8),
          SelectableText(
            '✓ ${_trainStatus!.outputPath}',
            style: TextStyle(fontSize: 11, color: Colors.green),
          ),
        ],
        if (_error != null) ...[
          const SizedBox(height: 8),
          Text(_error!, style: TextStyle(fontSize: 11, color: cs.error)),
        ],
        if (_logs.isNotEmpty) ...[
          const SizedBox(height: 8),
          _buildLogsSection(cs),
        ],
      ],
    );
  }

  Widget _buildDatasetDirField(ColorScheme cs) {
    return TextField(
      controller: _datasetCtrl,
      decoration: InputDecoration(
        labelText: 'Dataset directory',
        hintText: '/path/to/dataset',
        border: const OutlineInputBorder(),
        contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        isDense: true,
        suffixIcon: IconButton(
          icon: const Icon(Icons.refresh, size: 16),
          tooltip: 'Check readiness',
          onPressed: _checkReadiness,
        ),
      ),
      style: const TextStyle(fontSize: 12),
      onSubmitted: (_) => _checkReadiness(),
    );
  }

  Widget _buildSubtitle(ColorScheme cs) {
    switch (_state) {
      case _CardState.waiting:
        return Text(
          'Waiting for metadata.jsonl',
          style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
        );
      case _CardState.ready:
        final r = _readiness!;
        return Text(
          '${r.imageCount} images · ${_formatBytes(r.fileSizeBytes)}',
          style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
        );
      case _CardState.running:
        final elapsed = _trainStatus?.elapsedSeconds ?? 0;
        return Text(
          'Running · ${_formatElapsed(elapsed)}',
          style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
        );
      case _CardState.done:
        final elapsed = _trainStatus?.elapsedSeconds ?? 0;
        return Text(
          'Completed in ${_formatElapsed(elapsed)}',
          style: const TextStyle(fontSize: 11, color: Colors.green),
        );
      case _CardState.failed:
        return Text(
          'Failed',
          style: TextStyle(fontSize: 11, color: cs.error),
        );
    }
  }

  Widget _buildConfigFields(ColorScheme cs) {
    const inputDeco = InputDecoration(
      border: OutlineInputBorder(),
      contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      isDense: true,
    );
    return Column(
      children: [
        TextField(
          controller: _outputCtrl,
          decoration: inputDeco.copyWith(labelText: 'Output directory'),
          style: const TextStyle(fontSize: 12),
        ),
        const SizedBox(height: 6),
        TextField(
          controller: _scriptCtrl,
          decoration: inputDeco.copyWith(labelText: 'Script path'),
          style: const TextStyle(fontSize: 12),
        ),
        const SizedBox(height: 6),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _rankCtrl,
                decoration: inputDeco.copyWith(labelText: 'Rank'),
                style: const TextStyle(fontSize: 12),
                keyboardType: TextInputType.number,
              ),
            ),
            const SizedBox(width: 6),
            Expanded(
              child: TextField(
                controller: _epochsCtrl,
                decoration: inputDeco.copyWith(labelText: 'Epochs'),
                style: const TextStyle(fontSize: 12),
                keyboardType: TextInputType.number,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _lrCtrl,
                decoration: inputDeco.copyWith(labelText: 'Learning rate'),
                style: const TextStyle(fontSize: 12),
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
              ),
            ),
            const SizedBox(width: 6),
            Expanded(
              child: TextField(
                controller: _resCtrl,
                decoration: inputDeco.copyWith(labelText: 'Resolution'),
                style: const TextStyle(fontSize: 12),
                keyboardType: TextInputType.number,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        DropdownButtonFormField<String>(
          initialValue: _mixedPrecision,
          decoration: inputDeco.copyWith(labelText: 'Mixed precision'),
          style: const TextStyle(fontSize: 12),
          items: const [
            DropdownMenuItem(value: 'bf16', child: Text('bf16')),
            DropdownMenuItem(value: 'fp16', child: Text('fp16')),
            DropdownMenuItem(value: 'no', child: Text('no')),
          ],
          onChanged: (v) { if (v != null) setState(() => _mixedPrecision = v); },
        ),
      ],
    );
  }

  Widget _buildRunButton(ColorScheme cs) {
    final canRun = (_state == _CardState.ready) &&
        _outputCtrl.text.trim().isNotEmpty;
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        onPressed: canRun ? _run : null,
        icon: _state == _CardState.running
            ? const SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
              )
            : const Icon(Icons.play_arrow, size: 16),
        label: const Text('Run'),
      ),
    );
  }

  Widget _buildProgress(ColorScheme cs) {
    final current = _trainStatus?.currentStep ?? 0;
    final total = _trainStatus?.totalSteps ?? 0;
    final progress = (total > 0) ? current / total : null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        LinearProgressIndicator(value: progress),
        if (total > 0)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              'Step $current / $total',
              style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
            ),
          ),
      ],
    );
  }

  Widget _buildLogsSection(ColorScheme cs) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        GestureDetector(
          onTap: () => setState(() => _showLogs = !_showLogs),
          child: Row(
            children: [
              Icon(_showLogs ? Icons.expand_less : Icons.expand_more, size: 16,
                  color: cs.onSurfaceVariant),
              const SizedBox(width: 4),
              Text('Logs (${_logs.length} lines)',
                  style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
            ],
          ),
        ),
        if (_showLogs)
          Container(
            margin: const EdgeInsets.only(top: 6),
            height: 160,
            decoration: BoxDecoration(
              color: cs.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(6),
            ),
            child: _LogView(lines: _logs),
          ),
      ],
    );
  }
}

// Auto-scrolling log view
class _LogView extends StatefulWidget {
  final List<String> lines;
  const _LogView({required this.lines});

  @override
  State<_LogView> createState() => _LogViewState();
}

class _LogViewState extends State<_LogView> {
  final _scroll = ScrollController();

  @override
  void didUpdateWidget(_LogView oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.lines.length != oldWidget.lines.length) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scroll.hasClients) {
          _scroll.jumpTo(_scroll.position.maxScrollExtent);
        }
      });
    }
  }

  @override
  void dispose() {
    _scroll.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return ListView.builder(
      controller: _scroll,
      padding: const EdgeInsets.all(8),
      itemCount: widget.lines.length,
      itemBuilder: (_, i) => Text(
        widget.lines[i],
        style: TextStyle(
          fontSize: 10,
          fontFamily: 'monospace',
          color: cs.onSurfaceVariant,
        ),
      ),
    );
  }
}

extension on Border {
  ShapeBorder toShapeBorder() => _BorderShape(this);
}

class _BorderShape extends ShapeBorder {
  final Border border;
  const _BorderShape(this.border);

  @override
  EdgeInsetsGeometry get dimensions => EdgeInsets.only(left: border.left.width);

  @override
  Path getInnerPath(Rect rect, {TextDirection? textDirection}) =>
      Path()..addRect(rect);

  @override
  Path getOuterPath(Rect rect, {TextDirection? textDirection}) =>
      Path()..addRect(rect);

  @override
  void paint(Canvas canvas, Rect rect, {TextDirection? textDirection}) {
    border.paint(canvas, rect);
  }

  @override
  ShapeBorder scale(double t) => this;
}
