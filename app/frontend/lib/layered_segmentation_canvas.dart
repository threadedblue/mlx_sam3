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

  Segment({required this.path, this.retouchedImage});
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
                  painter: MasksPainter(segments: segments),
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

  MasksPainter({required this.segments});

  @override
  void paint(Canvas canvas, Size size) {
    const singleColor = Colors.blue;

    final fillPaint = Paint()
      ..color = singleColor.withOpacity(0.2)
      ..style = PaintingStyle.fill;

    final stripePaint = Paint()
      ..color = singleColor.withOpacity(0.5)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    for (final segment in segments) {
      // First, draw a light, uniform fill.
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
    }
  }

  @override
  bool shouldRepaint(covariant MasksPainter oldDelegate) {
    // For better performance, consider a deep list comparison or versioning.
    return !listEquals(segments, oldDelegate.segments);
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
