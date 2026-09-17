// lib/layered_segmentation_canvas.dart

import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'package:provider/provider.dart';
import 'layer_state.dart';
/// A data class to hold information about a single segmented area.
/// [path] defines the shape of the segment.
/// [color] is used for the mask layer.
/// [retouchedImage] is an optional image for the 'finals' layer.
class Segment {
  final Path path;
  final ui.Image? retouchedImage;

  /// Backend `sf_engine.MaskRecord.mask_id` — what the hold/caption
  /// endpoints (`/mask/hold`, `/mask/caption`) need to name which record a
  /// UI action applies to. Null for a segment with no v2 identity behind it
  /// (e.g. the bbox fallback path).
  final String? maskId;

  /// `"unassigned"` | `"keep"` (sf_engine.DatasetStatus) — whether this
  /// selection has been captioned yet. Null when unknown (no maskId).
  final String? datasetStatus;

  /// Whether this record is currently queued for the next scrub batch.
  final bool held;

  Segment({
    required this.path,
    this.retouchedImage,
    this.maskId,
    this.datasetStatus,
    this.held = false,
  });
}

/// A widget that displays an image and its segmentations in four distinct,
/// toggleable layers:
/// 1. [Original]: The base image.
/// 2. [Masks]: Colored overlays representing the segmentation masks.
/// 3. [Raw]: The parts of the original image "cut out" by the masks.
/// 4. [Finals]: Retouched images displayed within their segment boundaries.
class LayeredSegmentationCanvas extends StatelessWidget {
  final ui.Image originalImage;
  final List<Segment> segments;

  const LayeredSegmentationCanvas({
    super.key,
    required this.originalImage,
    required this.segments,
  });

  @override
  Widget build(BuildContext context) {
    // Use a FittedBox to ensure the canvas scales to fit its container
    // while maintaining the original image's aspect ratio.
    return FittedBox(
      fit: BoxFit.contain,
      child: SizedBox(
        width: originalImage.width.toDouble(),
        height: originalImage.height.toDouble(),
        child: Consumer<LayerState>(builder: (context, layerState, _) {
          return Stack(
            children: [
              // Layer 1: Original Image
              Visibility(
                visible: layerState.showOriginal,
                child: CustomPaint(
                  painter: OriginalImagePainter(image: originalImage),
                  size: Size.infinite, // Expands to the Stack's constraints
                ),
              ),
              // Layer 2: Segmentation Masks
              Visibility(
                visible: layerState.showMasks,
                child: CustomPaint(
                  painter: MasksPainter(
                    segments: segments,
                    showOriginal: layerState.showOriginal,
                  ),
                  size: Size.infinite,
                ),
              ),
              // Layer 3: Raw Cutouts from Original
              Visibility(
                visible: layerState.showRaw,
                child: CustomPaint(
                  painter: RawCutoutsPainter(
                    originalImage: originalImage,
                    segments: segments,
                  ),
                  size: Size.infinite,
                ),
              ),
              // Layer 4: Final Retouched Images
              Visibility(
                visible: layerState.showFinal,
                child: CustomPaint(
                  painter: FinalsPainter(segments: segments),
                  size: Size.infinite,
                ),
              ),
            ],
          );
        }),
      ),
    );
  }
}

//--- Custom Painters for Each Layer ---

/// Layer 1: Draws the original image.
class OriginalImagePainter extends CustomPainter {
  final ui.Image image;
  
  OriginalImagePainter({required this.image});

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawImage(image, Offset.zero, Paint());
  }

  @override
  bool shouldRepaint(covariant OriginalImagePainter oldDelegate) {
    return image != oldDelegate.image;
  }
}

/// Layer 2: Draws semi-transparent colored masks for each segment.
class MasksPainter extends CustomPainter {
  final List<Segment> segments;
  final bool showOriginal;

  MasksPainter({required this.segments, required this.showOriginal});

  /// Green for a captioned keeper, red for a record queued in the pending
  /// scrub batch is a SEPARATE, independent cue (the border stroke below) —
  /// the two axes are independent in the v2 model (a captioned keeper can
  /// still be held for a later scrub) and were colliding when `held` also
  /// overrode fill color: both used red, and red is already owned by
  /// Target/Avoid prompt markers (a third, unrelated meaning for the same
  /// hue). Fill color no longer varies with `held` at all — see [paint]'s
  /// border pass for how holding is actually shown.
  ///
  /// Green is the same literal the "keep"/caption-mode UI already uses
  /// elsewhere, so the canvas and the controls agree. `keep` gets a
  /// stronger alpha than the neutral fallback — 0x007F00 at the old 0.2
  /// fill is nearly invisible over artwork, and this needs to be read at a
  /// glance to be useful. That alpha is my choice, not sourced from a
  /// mockup. Neutral (unassigned, not held) is UNCHANGED from the original
  /// pre-v2 painter: dark grey with Original shown, bright cyan without.
  ({ui.Color color, double fill, double stripe, double border}) _paletteFor(Segment segment) {
    const green = ui.Color(0xFF007F00);
    if (segment.datasetStatus == 'keep') {
      return (color: green, fill: showOriginal ? 0.28 : 0.6,
              stripe: showOriginal ? 0.7 : 0.9, border: showOriginal ? 0.9 : 1.0);
    }
    final neutral = showOriginal
        ? const ui.Color.fromARGB(255, 48, 42, 42)     // Dark grey for subtle overlay
        : const ui.Color.fromARGB(255, 100, 200, 255); // Bright cyan standalone
    return (color: neutral, fill: showOriginal ? 0.2 : 0.6,
            stripe: showOriginal ? 0.5 : 0.8, border: showOriginal ? 0.3 : 1.0);
  }

  // Amber, not red or green — held is its own axis from dataset_status, and
  // red/green are both already spoken for (Target/Avoid markers; keep
  // fill). Solid rather than dashed: no path-dashing utility exists in this
  // codebase yet, and drawing one (or adding a package for it) is more
  // than a color cue needs — the color change alone reads clearly at a
  // glance against every fill state.
  static const _heldBorderColor = ui.Color(0xFFFFA000);

  @override
  void paint(Canvas canvas, Size size) {
    for (final segment in segments) {
      final palette = _paletteFor(segment);

      final fillPaint = Paint()
        ..color = palette.color.withValues(alpha: palette.fill)
        ..style = PaintingStyle.fill;

      // Border stroke to outline each mask. Always the dataset-status hue —
      // the held cue is drawn separately at the end of this loop, for the
      // reason documented there.
      final borderPaint = Paint()
        ..color = palette.color.withValues(alpha: palette.border)
        ..strokeWidth = 2.0
        ..style = PaintingStyle.stroke;

      final stripePaint = Paint()
        ..color = palette.color.withValues(alpha: palette.stripe)
        ..strokeWidth = 2.0
        ..style = PaintingStyle.stroke;

      // First, draw the fill.
      canvas.drawPath(segment.path, fillPaint);

      // Then, clip to the path and draw a stripe pattern on top.
      // This makes the masks distinguishable from a solid color overlay.
      canvas.save();
      canvas.clipPath(segment.path);

      // Draw diagonal stripes across the whole canvas; they will be clipped.
      // The pattern will be consistent across all segments.
      for (double i = -size.height; i < size.width; i += 8) {
        canvas.drawLine(
          Offset(i, 0),
          Offset(i + size.height, size.height),
          stripePaint,
        );
      }

      canvas.restore();

      // Finally, draw a border around the mask for clear edge definition.
      canvas.drawPath(segment.path, borderPaint);

      // Held cue: an amber outline around the mask's BOUNDS, deliberately
      // not a stroke of segment.path. That path is built scanline by
      // scanline — one 1px-tall addRect per RLE run, thousands of them
      // (main.dart's _updateSegmentsFromResult) — so stroking it strokes
      // every one of those rects and floods the whole mask with the stroke
      // colour rather than outlining it. That flooding is invisible while
      // the stroke matches the fill hue (it just reads as a more saturated
      // mask, which is what every pass above relies on), but an amber
      // stroke painted the entire mask amber and buried the green `keep`
      // fill under it: a captioned mask still looked held-only, which is
      // exactly the bug this cue caused. Bounds are the one outline that
      // survives that path shape without a contour trace.
      if (segment.held) {
        canvas.drawRect(
          segment.path.getBounds(),
          Paint()
            ..color = _heldBorderColor.withValues(alpha: showOriginal ? 0.9 : 1.0)
            ..strokeWidth = 2.0
            ..style = PaintingStyle.stroke,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant MasksPainter oldDelegate) {
    // For better performance, consider a deep list comparison or versioning.
    return !listEquals(segments, oldDelegate.segments) ||
        showOriginal != oldDelegate.showOriginal;
  }
}


/// Layer 3: Draws the parts of the original image that correspond to the masks.
class RawCutoutsPainter extends CustomPainter {
  final ui.Image originalImage;
  final List<Segment> segments;

  RawCutoutsPainter({required this.originalImage, required this.segments});

  @override
  void paint(Canvas canvas, Size size) {
    final imageRect = Rect.fromLTWH(0, 0, size.width, size.height);

    for (final segment in segments) {
      // Create a temporary drawing layer.
      canvas.saveLayer(imageRect, Paint());

      // Draw the original image into the temporary layer.
      canvas.drawImage(originalImage, Offset.zero, Paint());

      // Use BlendMode.dstIn to keep the destination (image) pixels only where
      // the source (mask path) is drawn, effectively creating a cutout.
      final maskPaint = Paint()..blendMode = BlendMode.dstIn;
      canvas.drawPath(segment.path, maskPaint);

      // Composite the temporary layer back onto the main canvas.
      canvas.restore();
    }
  }

  @override
  bool shouldRepaint(covariant RawCutoutsPainter oldDelegate) {
    return originalImage != oldDelegate.originalImage ||
        !listEquals(segments, oldDelegate.segments);
  }
}

/// Layer 4: Draws the final, retouched images, clipped to their segment path.
class FinalsPainter extends CustomPainter {
  final List<Segment> segments;

  FinalsPainter({required this.segments});

  @override
  void paint(Canvas canvas, Size size) {
    for (final segment in segments) {
      if (segment.retouchedImage != null) {
        // Save the current canvas state and clip the drawing area to the path.
        canvas.save();
        canvas.clipPath(segment.path);

        // Draw the retouched image. It will only be visible inside the clipped path.
        canvas.drawImage(segment.retouchedImage!, Offset.zero, Paint());

        // Restore the canvas to its original state (removes the clip).
        canvas.restore();
      }
    }
  }

  @override
  bool shouldRepaint(covariant FinalsPainter oldDelegate) {
    return !listEquals(segments, oldDelegate.segments);
  }
}
