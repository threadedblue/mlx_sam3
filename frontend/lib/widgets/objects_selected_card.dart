import 'package:flutter/material.dart';

import '../layered_segmentation_canvas.dart';

/// "Objects Selected" card: object count, one Hold checkbox per currently
/// selected mask, and a Clear Prompts button.
///
/// Stateless by design, like [LBSCard]/`ResultCell` — the caller
/// (main.dart's state) owns [segments] and the actual `/mask/hold` call.
///
/// One checkbox per mask, not a single one for "the" focused mask (the
/// pre-fix behaviour): a text prompt returns several masks at once with
/// none of them focused — only box/point selection sets a focused mask id,
/// since text has no single "the" instance to point at. Gating the
/// checkbox on a focused mask left a multi-object text selection with no
/// way to hold ANY of its results without individually re-clicking each
/// one via box/point first, which is what actually happened: the checkbox
/// disappeared entirely after a 4-result text search, not just for one of
/// the four.
class ObjectsSelectedCard extends StatelessWidget {
  final int maskCount;
  final List<Segment> segments;
  final bool enabled;
  final void Function(String maskId, bool held) onSetHeld;
  final VoidCallback onClearPrompts;

  const ObjectsSelectedCard({
    super.key,
    required this.maskCount,
    required this.segments,
    required this.enabled,
    required this.onSetHeld,
    required this.onClearPrompts,
  });

  @override
  Widget build(BuildContext context) {
    // Only segments with a v2 identity can be held at all (e.g. not the
    // bbox-fallback path, which carries no mask_id) — same guard the
    // pre-fix single-checkbox version implicitly had via _focusedSegment
    // (never non-null without a maskId).
    final holdable = segments.where((s) => s.maskId != null).toList();

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
              const Row(
                children: [
                  Icon(Icons.data_usage, size: 16),
                  SizedBox(width: 8),
                  Text("Objects Selected", style: TextStyle(fontWeight: FontWeight.bold)),
                ],
              ),
              const SizedBox(height: 12),
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text("Object count",
                        style: TextStyle(fontSize: 13, color: Theme.of(context).colorScheme.onSurfaceVariant)),
                    Text(maskCount.toString(), style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
                  ],
                ),
              ),
              if (holdable.isNotEmpty) ...[
                const SizedBox(height: 8),
                for (final seg in holdable)
                  CheckboxListTile(
                    value: seg.held,
                    onChanged: enabled ? (checked) => onSetHeld(seg.maskId!, checked ?? false) : null,
                    controlAffinity: ListTileControlAffinity.leading,
                    contentPadding: EdgeInsets.zero,
                    dense: true,
                    title: Text(holdable.length == 1
                        ? "Hold for next scrub"
                        : "Hold ${_shortMaskId(seg.maskId!)} for next scrub"),
                  ),
              ],
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton(
                  onPressed: enabled ? onClearPrompts : null,
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                    side: const BorderSide(color: Colors.white),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                  ),
                  child: const Text("Clear Prompts"),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// The part of a mask id worth showing next to its checkbox — real ids are
/// `f"{pass}:{uuid4()}"` (backend/sf_engine.py), always well over 8 chars
/// after the colon, but a bare `.substring(0, 8)` still isn't safe against
/// anything shorter (any real id is fine; only a hand-built/mocked one
/// could be), so this clamps instead of trusting that length.
String _shortMaskId(String maskId) {
  final tail = maskId.split(':').last;
  return tail.length <= 8 ? tail : tail.substring(0, 8);
}
