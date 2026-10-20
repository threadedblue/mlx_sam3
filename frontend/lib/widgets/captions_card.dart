import 'action_button_style.dart';
import 'package:flutter/material.dart';

/// "Captioner" card: one button that runs the captioner over every segment
/// that has no caption or draft yet. Results land in the Captions tab as
/// drafts to review.
///
/// Stateless by design, like [SaveCard]/[PanelsCard] — the caller (main.dart's
/// state) owns the run. [progress] is shown on the button while [running];
/// [runError] says why a run could not start or finish.
class CaptionsCard extends StatelessWidget {
  final bool hasSession;

  /// Segments a run would caption (no caption, no draft). Zero disables the button.
  final int blank;
  final VoidCallback onRun;
  final bool running;
  final String? progress;
  final String? runError;

  /// Segments marked "never needs", and segments already captioned, for the
  /// line that explains a disabled button.
  final int never;
  final int captioned;

  const CaptionsCard({
    super.key,
    required this.hasSession,
    required this.blank,
    required this.onRun,
    this.running = false,
    this.progress,
    this.runError,
    this.never = 0,
    this.captioned = 0,
  });

  /// Why the button is off, or null when it is on / running.
  String? get _whyOff {
    if (running) return null;
    if (!hasSession) return 'Open a session first.';
    if (blank > 0) return null;
    final parts = <String>[
      if (captioned > 0) '$captioned captioned',
      if (never > 0) '$never marked Never',
    ];
    return parts.isEmpty
        ? 'No segments yet.'
        : 'Nothing left to caption (${parts.join(', ')}). Untick Never on a segment to caption it.';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final label = running
        ? 'Captioning${progress == null ? '…' : ' $progress'}'
        : (blank > 0 ? 'Run captioner ($blank)' : 'Run captioner');
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.white.withValues(alpha: 0.6), width: 1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  key: const ValueKey('captions-card-run'),
                  onPressed: (!hasSession || running || blank == 0) ? null : onRun,
                  style: actionButtonStyle(busy: running),
                  child: Text(label),
                ),
              ),
              if (_whyOff != null) ...[
                const SizedBox(height: 8),
                Text(
                  _whyOff!,
                  key: const ValueKey('captions-card-why-off'),
                  style: theme.textTheme.bodySmall,
                ),
              ],
              if (runError != null) ...[
                const SizedBox(height: 8),
                Text(
                  runError!,
                  key: const ValueKey('captions-card-error'),
                  style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.error),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
