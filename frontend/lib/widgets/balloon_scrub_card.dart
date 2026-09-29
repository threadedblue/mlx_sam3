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
  /// which means a run happened and genuinely found nothing.
  final int? lastDetectedCount;

  final bool isDetecting;
  final VoidCallback onDetect;

  const BalloonScrubCard({
    super.key,
    required this.hasImage,
    required this.lastDetectedCount,
    required this.isDetecting,
    required this.onDetect,
  });

  /// Neutral before the first run, and distinguishes "found none" from
  /// "haven't looked" — the same distinction [LBSCard] draws with its
  /// 'No scrub yet'.
  String get _statusLine {
    final count = lastDetectedCount;
    if (count == null) return 'No detection run yet';
    if (count == 0) return 'No balloons detected';
    return '$count balloon${count == 1 ? '' : 's'} detected';
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
