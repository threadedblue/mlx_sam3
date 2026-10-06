import 'package:flutter/material.dart';

/// Balloon Scrub card: runs the comic speech-balloon detector over the current
/// pass's image (backend's `/segment/balloons`) and holds every detection for
/// review.
///
/// Detection only — it never scrubs. Results land in the existing per-object
/// checkbox list, and the user scrubs them with [LBSCard] as usual; there is
/// deliberately no second scrub path.
///
/// Stateless by design, like [LBSCard]/[ResultCell]/`IncludeExcludeToggle` —
/// the caller (main.dart's state) owns [isDetecting]/[hasImage]/
/// [lastDetectedCount] and the actual API call, the same split already used
/// for every other per-card widget in this app.
class BalloonScrubCard extends StatelessWidget {
  /// Whether an image is loaded at all. Gates the button the way
  /// [LBSCard.hasSelection] does: with nothing on screen there is nothing to
  /// detect against, and the backend would only answer 400.
  final bool hasImage;

  /// Null before the first detection run this session — distinct from `0`,
  /// which means a run happened and genuinely found nothing. Always wins
  /// over [isResumedSession]/[resumedWbScrubPass] below when non-null: a
  /// live detection this session is more specific than anything restored.
  final int? lastDetectedCount;

  /// True iff this card's session came from a resumed /loadSession, not a
  /// fresh upload (bug-3 audit finding #2/#3) — distinguishes "no scrub
  /// history restored" from "no history exists at all", which
  /// [lastDetectedCount] alone cannot, since both start as `null`.
  final bool isResumedSession;

  /// The resumed session's `wb_scrub_pass` (same value backing the WB-scrub
  /// layer toggle) — non-null means a balloon scrub genuinely ran at some
  /// point in this session's history, whichever pass it landed on.
  /// `MaskRecord.source` is never persisted, so the ORIGINAL detection
  /// count cannot be recovered — this says only THAT one ran, not how many
  /// balloons it found.
  final int? resumedWbScrubPass;

  final bool isDetecting;
  final VoidCallback onDetect;

  const BalloonScrubCard({
    super.key,
    required this.hasImage,
    required this.lastDetectedCount,
    this.isResumedSession = false,
    this.resumedWbScrubPass,
    required this.isDetecting,
    required this.onDetect,
  });

  /// Neutral before the first run, and distinguishes "found none" from
  /// "haven't looked" — the same distinction [LBSCard] draws with its
  /// 'No scrub yet'.
  ///
  /// A fresh session (never resumed) keeps the original two-way text
  /// unchanged. A resumed session with no live detection yet this run adds
  /// a third: "ran previously" vs "nothing restored to report" — NOT the
  /// fresh-session default, which would misread as "never looked" when it
  /// might just be unrecoverable (see [resumedWbScrubPass]'s doc).
  String get _statusLine {
    final count = lastDetectedCount;
    if (count != null) {
      if (count == 0) return 'No balloons detected';
      return '$count balloon${count == 1 ? '' : 's'} detected';
    }
    if (isResumedSession) {
      final pass = resumedWbScrubPass;
      if (pass != null) return 'Balloons scrubbed at pass $pass';
      return 'Not run in this session';
    }
    return 'No detection run yet';
  }

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
              Text('Balloon Scrub', style: textTheme.titleMedium),
              // 12 and 16 are LBSCard's own title gap and pre-button gap,
              // reused unchanged. LBSCard's 8 has no counterpart here: it
              // separated its two status lines, and this card has one.
              const SizedBox(height: 12),
              Text(_statusLine, style: textTheme.bodySmall),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                // Same three ButtonStyle properties as LBSCard's Scrub button
                // and main.dart's Save button — not a new style.
                child: ElevatedButton(
                  // Mirrors LBSCard's `isScrubbing || !hasSelection`: disabled
                  // while a call is in flight, and disabled with no image to
                  // detect against.
                  onPressed: (isDetecting || !hasImage) ? null : onDetect,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF007F00),
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                  ),
                  child: isDetecting
                      ? const SizedBox(
                          // Same 20x20/strokeWidth 2 as LBSCard's and
                          // ResultCell's spinner; white for contrast on green.
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Text('Detect Balloons'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
