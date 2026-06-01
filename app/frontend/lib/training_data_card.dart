import 'package:flutter/material.dart';
import 'services/api_service.dart';

class TrainingDataCard extends StatefulWidget {
  final String? sessionId;
  final int segmentCount;
  final String currentPrompt;
  final VoidCallback? onRunning;
  final VoidCallback? onComplete;

  const TrainingDataCard({
    super.key,
    required this.sessionId,
    required this.segmentCount,
    required this.currentPrompt,
    this.onRunning,
    this.onComplete,
  });

  @override
  State<TrainingDataCard> createState() => _TrainingDataCardState();
}

class _TrainingDataCardState extends State<TrainingDataCard> {
  final ApiService _api = ApiService();
  int _selectedIndex = 0;
  String? _refinedCaption;
  String? _savedFilename;
  int _entryCount = 0;
  bool _isLoading = false;
  String? _error;

  @override
  void didUpdateWidget(TrainingDataCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.segmentCount != oldWidget.segmentCount && _selectedIndex >= widget.segmentCount) {
      setState(() => _selectedIndex = 0);
    }
  }

  Future<void> _run() async {
    if (widget.sessionId == null || widget.currentPrompt.isEmpty) return;
    widget.onRunning?.call();
    setState(() {
      _isLoading = true;
      _error = null;
      _refinedCaption = null;
      _savedFilename = null;
    });
    try {
      final result = await _api.appendLoraEntry(
        widget.sessionId!,
        _selectedIndex,
        widget.currentPrompt,
      );
      if (!mounted) return;
      setState(() {
        _refinedCaption = result['refined_prompt'] as String?;
        _savedFilename = result['jsonl_filename'] as String?;
        _entryCount = result['entry_count'] as int? ?? _entryCount + 1;
      });
      widget.onComplete?.call();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final hasSession = widget.sessionId != null;
    final hasSegments = widget.segmentCount > 0;
    final hasPrompt = widget.currentPrompt.isNotEmpty;
    final canRun = hasSession && hasSegments && hasPrompt && !_isLoading;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.model_training_outlined, size: 16),
                SizedBox(width: 8),
                Text("LoRA Training Prep", style: TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 12),
            if (hasPrompt)
              Text(
                'Prompt: "${widget.currentPrompt}"',
                style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
                overflow: TextOverflow.ellipsis,
              ),
            if (hasSegments) ...[
              const SizedBox(height: 10),
              Row(
                children: [
                  Text("Segment:", style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant)),
                  const SizedBox(width: 8),
                  Expanded(
                    child: DropdownButtonFormField<int>(
                      initialValue: _selectedIndex,
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        isDense: true,
                      ),
                      items: List.generate(
                        widget.segmentCount,
                        (i) => DropdownMenuItem(
                          value: i,
                          child: Text('segment_${i.toString().padLeft(3, '0')}'),
                        ),
                      ),
                      onChanged: _isLoading
                          ? null
                          : (v) { if (v != null) setState(() => _selectedIndex = v); },
                    ),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: canRun ? _run : null,
                icon: _isLoading
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                      )
                    : const Icon(Icons.play_arrow, size: 16),
                label: const Text("Run"),
              ),
            ),
            if (!hasSession)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text("No active session.", style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
              )
            else if (!hasPrompt)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text("Enter a text prompt above.", style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
              )
            else if (!hasSegments)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text("Run 'Create Segments' first.", style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
              ),
            if (_refinedCaption != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: cs.primaryContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SelectableText(
                      _refinedCaption!,
                      style: TextStyle(fontSize: 12, color: cs.onPrimaryContainer),
                    ),
                    if (_savedFilename != null) ...[
                      const SizedBox(height: 6),
                      Text(
                        'Saved to $_savedFilename ($_entryCount entries)',
                        style: TextStyle(fontSize: 10, color: cs.onPrimaryContainer.withValues(alpha: 0.7)),
                      ),
                    ],
                  ],
                ),
              ),
            ],
            if (_isLoading) ...[
              const SizedBox(height: 10),
              Text(
                "Calling Gemini — this may take a few seconds…",
                style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
              ),
            ],
            if (_error != null) ...[
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: cs.errorContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.error_outline, size: 16, color: cs.onErrorContainer),
                    const SizedBox(width: 8),
                    Expanded(
                      child: SelectableText(
                        _error!,
                        style: TextStyle(fontSize: 11, color: cs.onErrorContainer),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
