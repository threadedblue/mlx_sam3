import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';
import 'package:frontend/layered_segmentation_canvas.dart';
import 'package:frontend/widgets/include_exclude_toggle.dart';

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
    testWidgets('renders both halves', (tester) async {
      await tester.pumpWidget(_wrap(
        IncludeExcludeToggle(value: true, onChanged: (_) {}),
      ));

      expect(find.text('Include'), findsOneWidget);
      expect(find.text('Exclude'), findsOneWidget);
    });

    testWidgets('tapping Exclude reports false', (tester) async {
      final calls = <bool>[];
      await tester.pumpWidget(_wrap(
        IncludeExcludeToggle(value: true, onChanged: calls.add),
      ));

      await tester.tap(find.text('Exclude'));
      expect(calls, [false]);
    });

    testWidgets('tapping Include reports true', (tester) async {
      final calls = <bool>[];
      await tester.pumpWidget(_wrap(
        IncludeExcludeToggle(value: false, onChanged: calls.add),
      ));

      await tester.tap(find.text('Include'));
      expect(calls, [true]);
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
