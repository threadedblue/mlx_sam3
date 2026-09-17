import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';
import 'package:frontend/layered_segmentation_canvas.dart';
import 'package:frontend/widgets/include_exclude_toggle.dart';
import 'package:frontend/widgets/lbs_card.dart';

/// A solid-colour image to hand to [SegmentationCanvas].
Future<ui.Image> _solidImage(int width, int height) {
  final recorder = ui.PictureRecorder();
  Canvas(recorder).drawRect(
    Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
    Paint()..color = const Color(0xFF202020),
  );
  return recorder.endRecording().toImage(width, height);
}

/// The canvas layers read [LayerState] through a Consumer, so the provider has
/// to be in scope even for widgets that do not obviously need it.
Widget _wrap(Widget child) => ChangeNotifierProvider<LayerState>(
      create: (_) => LayerState(),
      child: MaterialApp(home: Scaffold(body: child)),
    );

void main() {
  group('IncludeExcludeToggle', () {
    // Labels are "Target"/"Avoid" (SAM3 grounding polarity), not the
    // original "Include"/"Exclude" — relabeled to stop reading as a
    // duplicate of the held/dataset_status axis (see the widget's own
    // doc comment). This group name is the class name, not the visible
    // text, so it's left as-is.
    testWidgets('renders both halves', (tester) async {
      await tester.pumpWidget(_wrap(
        IncludeExcludeToggle(value: true, onChanged: (_) {}),
      ));

      expect(find.text('Target'), findsOneWidget);
      expect(find.text('Avoid'), findsOneWidget);
    });

    testWidgets('tapping Avoid reports false', (tester) async {
      final calls = <bool>[];
      await tester.pumpWidget(_wrap(
        IncludeExcludeToggle(value: true, onChanged: calls.add),
      ));

      await tester.tap(find.text('Avoid'));
      expect(calls, [false]);
    });

    testWidgets('tapping Target reports true', (tester) async {
      final calls = <bool>[];
      await tester.pumpWidget(_wrap(
        IncludeExcludeToggle(value: false, onChanged: calls.add),
      ));

      await tester.tap(find.text('Target'));
      expect(calls, [true]);
    });
  });

  group('LBSCard', () {
    // Fix regression: /lama/scrub succeeds and advances the pass counter
    // even with an empty held set (no "already scrubbed" state to reject
    // per sf-model-v2-design.md §3/§4) -- the button itself has to be the
    // guard against a stray click doing that silently.
    testWidgets('scrub button disabled with 0 pending regions, even when not scrubbing', (tester) async {
      var scrubCalls = 0;
      await tester.pumpWidget(_wrap(LBSCard(
        pendingCount: 0,
        lastScrubLabel: null,
        isScrubbing: false,
        onScrub: () => scrubCalls++,
      )));

      await tester.tap(find.text('Scrub Selected Regions'));
      await tester.pump();
      expect(scrubCalls, 0, reason: 'a click with nothing held must not invoke onScrub');
    });

    testWidgets('scrub button enabled once something is held', (tester) async {
      var scrubCalls = 0;
      await tester.pumpWidget(_wrap(LBSCard(
        pendingCount: 1,
        lastScrubLabel: null,
        isScrubbing: false,
        onScrub: () => scrubCalls++,
      )));

      await tester.tap(find.text('Scrub Selected Regions'));
      await tester.pump();
      expect(scrubCalls, 1);
    });

    testWidgets('scrub button stays disabled while scrubbing even with pending regions', (tester) async {
      var scrubCalls = 0;
      await tester.pumpWidget(_wrap(LBSCard(
        pendingCount: 3,
        lastScrubLabel: null,
        isScrubbing: true,
        onScrub: () => scrubCalls++,
      )));

      expect(find.text('Scrub Selected Regions'), findsNothing); // spinner replaces the label
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });
  });

  group('MasksPainter fill vs held border', () {
    // Fix 2 regression: `held` used to override fill color to red, colliding
    // with red already meaning Target/Avoid. Fill must depend only on
    // dataset_status; held gets its own independent cue.
    const size = Size(100, 100);
    const maskRect = Rect.fromLTWH(20, 20, 60, 60);
    const fillPoint = Offset(50, 50); // well inside the mask, away from any stroke
    const borderPoint = Offset(50, 20); // on the mask's top bound, where the held cue draws

    /// The path shape main.dart's _updateSegmentsFromResult ACTUALLY
    /// produces: one 1px-tall addRect per RLE run, thousands of them — not
    /// a single rect. These tests originally used a single addRect, and
    /// that is precisely what let a real bug through: stroking this path
    /// strokes every one of those rects and floods the mask interior with
    /// the stroke colour, which a single-rect path never reveals (its
    /// stroke only touches the perimeter). Any test of fill colour has to
    /// use this shape to mean anything.
    Path scanlinePath(Rect r) {
      final path = Path();
      for (double y = r.top; y < r.bottom; y += 1.0) {
        path.addRect(Rect.fromLTWH(r.left, y, r.width, 1.0));
      }
      return path;
    }

    Segment segmentWith({String? datasetStatus, bool held = false}) => Segment(
          path: scanlinePath(maskRect),
          datasetStatus: datasetStatus,
          held: held,
        );

    // dart:ui image rasterization (toImage/toByteData) needs the real raster
    // thread, which never advances inside testWidgets' fake-async test zone
    // — awaiting it directly hangs forever. tester.runAsync escapes that
    // zone for exactly this kind of real I/O.
    Future<Color> renderAndSample(WidgetTester tester, Segment segment, Offset point) async {
      return (await tester.runAsync(() async {
        final recorder = ui.PictureRecorder();
        final canvas = Canvas(recorder, Rect.fromLTWH(0, 0, size.width, size.height));
        // Opaque backdrop so alpha-blended fill/stroke colors read back
        // deterministically instead of blending against a transparent pixel.
        canvas.drawRect(Offset.zero & size, Paint()..color = Colors.black);
        MasksPainter(segments: [segment], showOriginal: false).paint(canvas, size);
        final image = await recorder.endRecording().toImage(size.width.toInt(), size.height.toInt());
        final bytes = (await image.toByteData(format: ui.ImageByteFormat.rawRgba))!.buffer.asUint8List();
        final offset = (point.dy.toInt() * size.width.toInt() + point.dx.toInt()) * 4;
        return Color.fromARGB(bytes[offset + 3], bytes[offset], bytes[offset + 1], bytes[offset + 2]);
      }))!;
    }

    testWidgets('held alone does not change fill color', (tester) async {
      final unheldFill = await renderAndSample(tester, segmentWith(held: false), fillPoint);
      final heldFill = await renderAndSample(tester, segmentWith(held: true), fillPoint);
      expect(heldFill, unheldFill, reason: 'holding a mask must not affect its fill colour');
    });

    testWidgets('held draws its own amber cue on the mask bounds, distinct from the fill', (tester) async {
      final unheldBorder = await renderAndSample(tester, segmentWith(held: false), borderPoint);
      final heldBorder = await renderAndSample(tester, segmentWith(held: true), borderPoint);
      expect(heldBorder, isNot(unheldBorder));
      // 0xFFFFA000 (amber): red and green both high, blue low.
      // Color.r/g/b are 0.0-1.0 doubles (the non-deprecated replacement for
      // the old 0-255 .red/.green/.blue getters); 150/255 ~= 0.588 etc.
      expect(heldBorder.r, greaterThan(150 / 255));
      expect(heldBorder.g, greaterThan(80 / 255));
      expect(heldBorder.b, lessThan(80 / 255));
    });

    testWidgets('fill is never red (Target/Avoid markers own that hue)', (tester) async {
      for (final status in [null, 'unassigned', 'keep']) {
        for (final held in [false, true]) {
          final fill = await renderAndSample(tester, segmentWith(datasetStatus: status, held: held), fillPoint);
          expect(
            fill.r > fill.g && fill.r > fill.b,
            isFalse,
            reason: 'status=$status held=$held: fill must not be red-dominant',
          );
        }
      }
    });

    testWidgets('hold -> caption -> scrub sequence: fill and border change independently', (tester) async {
      // Step 1: selected, held, not yet captioned.
      final s1Fill = await renderAndSample(tester, segmentWith(datasetStatus: 'unassigned', held: true), fillPoint);
      final s1Border = await renderAndSample(tester, segmentWith(datasetStatus: 'unassigned', held: true), borderPoint);

      // Step 2: captioned (dataset_status -> keep) while still held.
      final s2Fill = await renderAndSample(tester, segmentWith(datasetStatus: 'keep', held: true), fillPoint);
      final s2Border = await renderAndSample(tester, segmentWith(datasetStatus: 'keep', held: true), borderPoint);
      expect(s2Fill, isNot(s1Fill), reason: 'captioning should turn the fill green');
      expect(s2Fill.g, greaterThan(s2Fill.r));
      expect(s2Border, s1Border, reason: 'held border cue must be unchanged by captioning');

      // Step 3: scrub runs, clearing held; caption (and its green fill) survives.
      final s3Fill = await renderAndSample(tester, segmentWith(datasetStatus: 'keep', held: false), fillPoint);
      final s3Border = await renderAndSample(tester, segmentWith(datasetStatus: 'keep', held: false), borderPoint);
      expect(s3Fill, s2Fill, reason: 'clearing held must not affect fill colour');
      expect(s3Border, isNot(s2Border), reason: 'held border cue should disappear once held clears');
    });

    testWidgets('captioning a HELD mask shows green fill, not the amber cue', (tester) async {
      // The reported bug: Save Caption reported "Status: Captioned" and the
      // backend really did return dataset_status "keep", but the mask stayed
      // amber. Cause was entirely in this painter — the held cue used to
      // stroke segment.path, and on the real scanline path that floods the
      // interior, burying the green fill the caption had correctly set.
      final fill = await renderAndSample(tester, segmentWith(datasetStatus: 'keep', held: true), fillPoint);

      expect(fill.g, greaterThan(fill.r), reason: 'a captioned mask must read green even while held');
      expect(fill.g, greaterThan(fill.b));

      // Pin the specific wrong colour, so a regression cannot pass by being
      // merely "not amber enough": 0xFFFFA000 is red-dominant.
      expect(fill.r > fill.g && fill.g > fill.b, isFalse, reason: 'must not be the amber held cue');

      // And it must match what the same caption looks like unheld: holding
      // changes the cue, never the fill.
      final unheldFill = await renderAndSample(tester, segmentWith(datasetStatus: 'keep', held: false), fillPoint);
      expect(fill, unheldFill);
    });
  });

  group('SegmentationCanvas selection mode', () {
    late ui.Image image;

    setUpAll(() async {
      image = await _solidImage(200, 200);
    });

    Future<void> pumpCanvas(
      WidgetTester tester, {
      required SelectionMode? mode,
      required List<List<double>> points,
      required List<List<double>> boxes,
    }) async {
      await tester.pumpWidget(_wrap(SegmentationCanvas(
        uiImage: image,
        segments: const <Segment>[],
        isLoading: false,
        mode: mode,
        onPointDrawn: points.add,
        onBoxDrawn: boxes.add,
      )));
    }

    testWidgets('point mode forwards a tap, box mode does not', (tester) async {
      final points = <List<double>>[];
      final boxes = <List<double>>[];

      await pumpCanvas(tester, mode: SelectionMode.point, points: points, boxes: boxes);
      await tester.tapAt(tester.getCenter(find.byType(SegmentationCanvas)));
      await tester.pump();
      expect(points, hasLength(1), reason: 'point mode should accept taps');

      points.clear();
      await pumpCanvas(tester, mode: SelectionMode.box, points: points, boxes: boxes);
      await tester.tapAt(tester.getCenter(find.byType(SegmentationCanvas)));
      await tester.pump();
      expect(points, isEmpty, reason: 'box mode should ignore taps');
    });

    testWidgets('box mode forwards a drag, point mode does not', (tester) async {
      final points = <List<double>>[];
      final boxes = <List<double>>[];
      Offset centre() => tester.getCenter(find.byType(SegmentationCanvas));

      await pumpCanvas(tester, mode: SelectionMode.box, points: points, boxes: boxes);
      await tester.dragFrom(centre() - const Offset(40, 40), const Offset(80, 80));
      await tester.pump();
      expect(boxes, hasLength(1), reason: 'box mode should accept drags');

      boxes.clear();
      await pumpCanvas(tester, mode: SelectionMode.point, points: points, boxes: boxes);
      await tester.dragFrom(centre() - const Offset(40, 40), const Offset(80, 80));
      await tester.pump();
      expect(boxes, isEmpty, reason: 'point mode should ignore drags');
    });

    testWidgets('no active mode ignores both taps and drags', (tester) async {
      final points = <List<double>>[];
      final boxes = <List<double>>[];

      await pumpCanvas(tester, mode: null, points: points, boxes: boxes);
      final centre = tester.getCenter(find.byType(SegmentationCanvas));
      await tester.tapAt(centre);
      await tester.dragFrom(centre - const Offset(40, 40), const Offset(80, 80));
      await tester.pump();

      expect(points, isEmpty);
      expect(boxes, isEmpty);
    });
  });
}
