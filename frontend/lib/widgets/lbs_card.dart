import 'package:flutter/material.dart';

/// LaMa Background Scrub card: triggers a scrub of the current pass's held
/// masks (backend's `/lama/scrub`), advancing to the next pass — repeatable
/// without limit (sf-model-v2-design.md §3/§4), not a one-shot Pass 1 -> 2
/// transition.
///
/// Stateless by design, like [ResultCell]/`IncludeExcludeToggle` — the
/// caller (main.dart's state) owns [pendingCount]/[lastScrubLabel]/
/// [isScrubbing] and the actual API call, the same split already used for
/// every other per-card widget in this app. `SegmentLayersCard` is the one
/// exception (a `Provider`-backed `ChangeNotifier`) because layer
/// visibility is also read by the canvas elsewhere; nothing else needs to
/// read this card's state, so that heavier pattern isn't warranted here.
class LBSCard extends StatelessWidget {
  /// Regions currently held for the next scrub batch (`held == true` in
  /// the current pass) — a true "awaiting scrub" count, not a proxy.
  final int pendingCount;

  /// Null before the first successful scrub this session.
  final String? lastScrubLabel;

  final bool isScrubbing;
  final VoidCallback onScrub;

  const LBSCard({
    super.key,
    required this.pendingCount,
    required this.lastScrubLabel,
    required this.isScrubbing,
    required this.onScrub,
  });

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.white.withValues(alpha: 0.6), width: 1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('LaMa Background Scrub', style: textTheme.titleMedium),
              const SizedBox(height: 12),
              Text('$pendingCount pending regions to review', style: textTheme.bodySmall),
              // Guessed, not sourced: no Blender/Save-button precedent gave
              // this gap a real value — 8 picked as the nearest step on
              // this app's 4/8/12/16/24 spacing scale.
              const SizedBox(height: 8),
              Text(
                lastScrubLabel == null ? 'No scrub yet' : 'Last scrub: $lastScrubLabel',
                style: textTheme.bodySmall,
              ),
              // Guessed: no existing precedent for "gap before a card's
              // own action button" to copy: reused the card's own 16px
              // padding value for rhythm, per the spec's own suggestion.
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                // Same three ButtonStyle properties as main.dart's Save
                // button (_buildSaveCard) — not a new style. Save's own
                // height is theme-default (no explicit height set), which
                // is why this doesn't set one either.
                child: ElevatedButton(
                  onPressed: isScrubbing ? null : onScrub,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF007F00),
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                  ),
                  child: isScrubbing
                      ? const SizedBox(
                          // Same 20x20/strokeWidth 2 as ResultCell's spinner.
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Text('Scrub Selected Regions'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
