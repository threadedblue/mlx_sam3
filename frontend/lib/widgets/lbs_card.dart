import 'package:flutter/material.dart';

/// LaMa Background Scrub card: triggers a scrub of the current pass's held
/// masks (backend's `/lama/scrub`), advancing to the next pass — repeatable
/// without limit (sf-model-v2-design.md §3/§4), not a one-shot Pass 1 -> 2
/// transition.
///
/// Stateless by design, like [ResultCell]/`IncludeExcludeToggle` — the
/// caller (main.dart's state) owns [pendingCount]/[hasSelection]/
/// [lastScrubLabel]/[isScrubbing] and the actual API call, the same split
/// already used for every other per-card widget in this app.
/// `SegmentLayersCard` is the one exception (a `Provider`-backed
/// `ChangeNotifier`) because layer visibility is also read by the canvas
/// elsewhere; nothing else needs to read this card's state, so that
/// heavier pattern isn't warranted here.
class LBSCard extends StatelessWidget {
  /// Regions currently held for the next scrub batch (`held == true` in
  /// the current pass) — a true "awaiting scrub" count, not a proxy.
  ///
  /// sf-display-and-workflow-v3-spec.md §5: no longer what gates the
  /// button (see [hasSelection] for that) — held now means "still checked
  /// in the opt-out list below," so this is what will ACTUALLY be sent to
  /// LaMa if Scrub is pressed right now, shown so unchecking a row is
  /// visibly reflected here before the click.
  final int pendingCount;

  /// Whether one or more objects are currently selected in the current
  /// pass — via Prompt, Box, or Point, regardless of each one's held
  /// (checked/unchecked) flag. Drives the button's enabled state (§5):
  /// Scrub no longer requires a separate Hold step, only a selection to
  /// review:  it must stay enabled even if the user has unchecked every
  /// row, since pressing it then is a legitimate (if pointless) empty
  /// scrub — [pendingCount] is what actually gates what gets sent, not
  /// this.
  final bool hasSelection;

  /// Null before the first successful scrub this session.
  final String? lastScrubLabel;

  final bool isScrubbing;
  final VoidCallback onScrub;

  const LBSCard({
    super.key,
    required this.pendingCount,
    required this.hasSelection,
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
                  // §5: gated on SELECTION, not on anything being held —
                  // Hold is retired as a scrub-eligibility gate. Still
                  // disabled with nothing selected at all, not just while a
                  // scrub is in flight — /lama/scrub still succeeds and
                  // advances the pass counter on an empty batch
                  // (sf-model-v2-design.md §3/§4: no "already scrubbed"
                  // state to reject), so without this a stray click with no
                  // selection on screen silently walks the pass forward for
                  // no visible effect. Falls out naturally once
                  // _scrubLamaBackground clears `_segments` post-scrub.
                  onPressed: (isScrubbing || !hasSelection) ? null : onScrub,
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
