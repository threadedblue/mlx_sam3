import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';
import 'package:frontend/layered_segmentation_canvas.dart';
import 'package:frontend/widgets/include_exclude_toggle.dart';
import 'package:frontend/widgets/lbs_card.dart';
import 'package:frontend/widgets/objects_selected_card.dart';

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
    testWidgets('scrub button disabled with nothing selected, even when not scrubbing', (tester) async {
      var scrubCalls = 0;
      await tester.pumpWidget(_wrap(LBSCard(
        pendingCount: 0,
        hasSelection: false,
        lastScrubLabel: null,
        isScrubbing: false,
        onScrub: () => scrubCalls++,
      )));

      await tester.tap(find.text('Scrub Selected Regions'));
      await tester.pump();
      expect(scrubCalls, 0, reason: 'a click with no selection must not invoke onScrub');
    });

    // v3 spec §5: Hold is retired as a scrub-eligibility gate -- SELECTION
    // alone enables the button now, across all three selection modes
    // (Prompt/Box/Point all populate main.dart's `_segments` the same way,
    // which is what `hasSelection` is computed from — see _buildLBSCard).
    testWidgets('scrub button enabled the moment something is selected, with nothing held/checked yet',
        (tester) async {
      var scrubCalls = 0;
      await tester.pumpWidget(_wrap(LBSCard(
        pendingCount: 0, // nothing checked — proves this does NOT gate the button anymore
        hasSelection: true,
        lastScrubLabel: null,
        isScrubbing: false,
        onScrub: () => scrubCalls++,
      )));

      await tester.tap(find.text('Scrub Selected Regions'));
      await tester.pump();
      expect(scrubCalls, 1);
    });

    testWidgets('scrub button stays disabled while scrubbing even with a selection', (tester) async {
      var scrubCalls = 0;
      await tester.pumpWidget(_wrap(LBSCard(
        pendingCount: 3,
        hasSelection: true,
        lastScrubLabel: null,
        isScrubbing: true,
        onScrub: () => scrubCalls++,
      )));

      expect(find.text('Scrub Selected Regions'), findsNothing); // spinner replaces the label
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });

    testWidgets('scrub button disables again once the post-scrub rebuild clears the selection', (tester) async {
      // Models _scrubLamaBackground's own setState: `_segments = []` after
      // a completed scrub, which is what main.dart derives `hasSelection`
      // from (_buildLBSCard) — this is the rebuild that follows, not a new
      // mechanism of LBSCard's own.
      var scrubCalls = 0;
      await tester.pumpWidget(_wrap(LBSCard(
        pendingCount: 2,
        hasSelection: true,
        lastScrubLabel: null,
        isScrubbing: false,
        onScrub: () => scrubCalls++,
      )));
      expect(find.byType(ElevatedButton).evaluate().single.widget, isA<ElevatedButton>()
          .having((b) => b.onPressed, 'onPressed', isNotNull));

      await tester.pumpWidget(_wrap(LBSCard(
        pendingCount: 0,
        hasSelection: false,
        lastScrubLabel: 'Pass 0 → Pass 1',
        isScrubbing: false,
        onScrub: () => scrubCalls++,
      )));

      expect(find.byType(ElevatedButton).evaluate().single.widget, isA<ElevatedButton>()
          .having((b) => b.onPressed, 'onPressed', isNull));
    });
  });

  group('ObjectsSelectedCard', () {
    // Fix regression: the Hold checkbox used to be gated on a single
    // focused mask, which a text prompt never sets (only box/point
    // selection does) — a multi-object text search made the checkbox
    // disappear ENTIRELY, not just for the unfocused objects, leaving no
    // way to hold any of the results. One checkbox per mask fixes this.
    Segment segmentWith(String maskId, {bool held = false}) =>
        Segment(path: Path(), maskId: maskId, datasetStatus: 'unassigned', held: held);

    testWidgets('a multi-object selection shows a checkbox for every mask, not zero', (tester) async {
      final calls = <(String, bool)>[];
      await tester.pumpWidget(_wrap(ObjectsSelectedCard(
        maskCount: 4,
        segments: [
          // Realistic shape: f"{pass}:{uuid4()}" (backend/sf_engine.py).
          segmentWith('0:80aa20b5-2554-4dee-b46a-06cdd3dc9bb7'),
          segmentWith('0:1a2b3c4d-0000-0000-0000-000000000001'),
          segmentWith('0:1a2b3c4d-0000-0000-0000-000000000002'),
          segmentWith('0:1a2b3c4d-0000-0000-0000-000000000003'),
        ],
        enabled: true,
        onSetHeld: (id, held) => calls.add((id, held)),
        onClearPrompts: () {},
      )));

      expect(find.byType(CheckboxListTile), findsNWidgets(4));
    });

    testWidgets('checking one mask\'s box reports that mask\'s id, not another one\'s', (tester) async {
      final calls = <(String, bool)>[];
      await tester.pumpWidget(_wrap(ObjectsSelectedCard(
        maskCount: 2,
        segments: [
          segmentWith('0:1a2b3c4d-0000-0000-0000-0000000000f1'),
          segmentWith('0:1a2b3c4d-0000-0000-0000-0000000000f2'),
        ],
        enabled: true,
        onSetHeld: (id, held) => calls.add((id, held)),
        onClearPrompts: () {},
      )));

      final checkboxes = find.byType(CheckboxListTile);
      await tester.tap(checkboxes.at(1)); // the SECOND mask's checkbox
      await tester.pump();

      expect(calls, [('0:1a2b3c4d-0000-0000-0000-0000000000f2', true)]);
    });

    testWidgets('a mask id shorter than the truncation length does not crash '
        '(hand-built/mocked ids only -- real ids are always well over 8 chars)', (tester) async {
      await tester.pumpWidget(_wrap(ObjectsSelectedCard(
        maskCount: 2,
        segments: [segmentWith('0:a'), segmentWith('0:bb')],
        enabled: true,
        onSetHeld: (_, __) {},
        onClearPrompts: () {},
      )));

      expect(tester.takeException(), isNull);
      expect(find.textContaining('Include a in next scrub'), findsOneWidget);
      expect(find.textContaining('Include bb in next scrub'), findsOneWidget);
    });

    testWidgets('a single-object selection still works (no regression for the common case)', (tester) async {
      final calls = <(String, bool)>[];
      await tester.pumpWidget(_wrap(ObjectsSelectedCard(
        maskCount: 1,
        segments: [segmentWith('0:only', held: true)],
        enabled: true,
        onSetHeld: (id, held) => calls.add((id, held)),
        onClearPrompts: () {},
      )));

      final checkbox = find.byType(CheckboxListTile);
      expect(checkbox, findsOneWidget);
      expect(tester.widget<CheckboxListTile>(checkbox).value, true);

      await tester.tap(checkbox);
      await tester.pump();
      expect(calls, [('0:only', false)]);
    });

    testWidgets('segments with no v2 identity (no maskId) get no checkbox', (tester) async {
      await tester.pumpWidget(_wrap(ObjectsSelectedCard(
        maskCount: 1,
        segments: [Segment(path: Path())], // bbox-fallback path: no maskId
        enabled: true,
        onSetHeld: (_, __) {},
        onClearPrompts: () {},
      )));

      expect(find.byType(CheckboxListTile), findsNothing);
    });

    testWidgets('disabled disables every checkbox and the Clear Prompts button', (tester) async {
      var setHeldCalls = 0;
      var clearCalls = 0;
      await tester.pumpWidget(_wrap(ObjectsSelectedCard(
        maskCount: 2,
        segments: [segmentWith('0:a'), segmentWith('0:b')],
        enabled: false,
        onSetHeld: (_, __) => setHeldCalls++,
        onClearPrompts: () => clearCalls++,
      )));

      for (final cb in tester.widgetList<CheckboxListTile>(find.byType(CheckboxListTile))) {
        expect(cb.onChanged, isNull);
      }
      await tester.tap(find.text('Clear Prompts'));
      await tester.pump();

      expect(setHeldCalls, 0);
      expect(clearCalls, 0);
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
        MasksPainter(segments: [segment], backgroundVisible: false).paint(canvas, size);
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

  group('LayeredSegmentationCanvas: Original survives a scrub', () {
    // Regression test for the bug this fix exists for: `originalImage`
    // (Original layer) and `currentImage` (Raw/Current layers) used to be
    // ONE slot in main.dart (`_imageBytes`/`_uiImage`), unconditionally
    // overwritten by every scrub — so Original silently showed the current
    // pass, not pass 0, for any session with a scrub behind it. Reference
    // identity is the right thing to assert here, not pixel content:
    // main.dart never re-decodes pass 0's bytes on a scrub, it keeps the
    // same ui.Image object pinned in `_originalUiImage` (see its field doc
    // comment) — the bug this guards against is exactly that pinning being
    // silently dropped, which reference identity catches directly.
    late ui.Image pass0;
    late ui.Image pass1;

    setUpAll(() async {
      pass0 = await _solidImage(64, 64);
      pass1 = await _solidImage(64, 64);
    });

    T painterOf<T extends CustomPainter>(WidgetTester tester) => tester
        .widgetList<CustomPaint>(find.descendant(
          of: find.byType(LayeredSegmentationCanvas),
          matching: find.byType(CustomPaint),
        ))
        .map((w) => w.painter)
        .whereType<T>()
        .single;

    testWidgets(
        'Original stays pinned to pass 0 across a simulated scrub; Raw and Current follow the current pass',
        (tester) async {
      // "Load": pass 0 and the current pass start out identical, matching
      // main.dart setting _originalUiImage/_uiImage from the same decoded
      // bytes at the same upload-completion point.
      await tester.pumpWidget(_wrap(LayeredSegmentationCanvas(
        originalImage: pass0,
        currentImage: pass0,
        segments: const <Segment>[],
      )));

      expect(painterOf<OriginalImagePainter>(tester).image, same(pass0));
      expect(painterOf<CurrentImagePainter>(tester).image, same(pass0));
      expect(painterOf<RawCutoutsPainter>(tester).image, same(pass0));

      // "Scrub": only the current-pass slot moves, exactly like
      // _scrubLamaBackground's setState block in main.dart — originalImage
      // is never touched.
      await tester.pumpWidget(_wrap(LayeredSegmentationCanvas(
        originalImage: pass0,
        currentImage: pass1,
        segments: const <Segment>[],
      )));

      expect(painterOf<OriginalImagePainter>(tester).image, same(pass0),
          reason: 'Original must still show pass 0, unchanged, after a scrub');
      expect(painterOf<CurrentImagePainter>(tester).image, same(pass1),
          reason: 'Current must reflect the post-scrub image');
      expect(painterOf<RawCutoutsPainter>(tester).image, same(pass1),
          reason: 'Raw cuts from the current pass, not pass 0 — SETTLED in the v3 spec');
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
        originalImage: image,
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

  group('Prompt card Select button mode-exclusivity (v3 spec §4)', () {
    // Drives the actual `promptCardCanSubmit` function main.dart's Select
    // button calls, through real radio taps and text entry — not a
    // re-implementation of the button's condition. The surrounding
    // RadioGroup/Radio/TextField shell here mirrors _buildTextPromptCard's
    // real wiring (one RadioGroup<SelectionMode> spanning Prompt/Box/Point,
    // same as main.dart ~1232) closely enough to exercise it without
    // dragging in HomeScreen's network/timer setup — same reasoning
    // DisplayAreaTabs was split out for; see its own doc comment.
    Future<void> pumpPromptCard(
      WidgetTester tester, {
      required SelectionMode? initialMode,
    }) async {
      final controller = TextEditingController();
      SelectionMode? mode = initialMode;
      await tester.pumpWidget(_wrap(StatefulBuilder(
        builder: (context, setState) => RadioGroup<SelectionMode>(
          groupValue: mode,
          onChanged: (m) => setState(() => mode = m),
          child: Column(
            children: [
              const Radio<SelectionMode>(value: SelectionMode.prompt),
              const Radio<SelectionMode>(value: SelectionMode.box),
              const Radio<SelectionMode>(value: SelectionMode.point),
              TextField(controller: controller),
              ValueListenableBuilder<TextEditingValue>(
                valueListenable: controller,
                builder: (context, value, _) => OutlinedButton(
                  onPressed: promptCardCanSubmit(
                        hasSession: true,
                        promptEmpty: value.text.isEmpty,
                        isLoading: false,
                        captionMode: false,
                        selectedMode: mode,
                      )
                      ? () {}
                      : null,
                  child: const Text('Select'),
                ),
              ),
            ],
          ),
        ),
      )));
    }

    Finder radioFor(SelectionMode value) => find.byWidgetPredicate(
        (w) => w is Radio<SelectionMode> && w.value == value);

    bool selectEnabled(WidgetTester tester) =>
        tester.widget<OutlinedButton>(find.byType(OutlinedButton)).onPressed != null;

    testWidgets('enabled in Prompt mode with non-empty text — matches the pre-existing condition',
        (tester) async {
      await pumpPromptCard(tester, initialMode: SelectionMode.prompt);
      await tester.enterText(find.byType(TextField), 'cat');
      await tester.pump();

      expect(selectEnabled(tester), isTrue);
    });

    testWidgets('disabled in Prompt mode with empty text — pre-existing condition, unaffected by §4',
        (tester) async {
      await pumpPromptCard(tester, initialMode: SelectionMode.prompt);

      expect(selectEnabled(tester), isFalse);
    });

    testWidgets('disabled while Box mode is active, regardless of prompt text', (tester) async {
      await pumpPromptCard(tester, initialMode: SelectionMode.box);
      await tester.enterText(find.byType(TextField), 'cat');
      await tester.pump();

      expect(selectEnabled(tester), isFalse);
    });

    testWidgets('disabled while Point mode is active, regardless of prompt text', (tester) async {
      await pumpPromptCard(tester, initialMode: SelectionMode.point);
      await tester.enterText(find.byType(TextField), 'cat');
      await tester.pump();

      expect(selectEnabled(tester), isFalse);
    });

    testWidgets('re-enables on switching back to Prompt mode — mode-exclusivity, not a permanent disable',
        (tester) async {
      await pumpPromptCard(tester, initialMode: SelectionMode.box);
      await tester.enterText(find.byType(TextField), 'cat');
      await tester.pump();
      expect(selectEnabled(tester), isFalse, reason: 'sanity check: starts disabled in Box mode');

      await tester.tap(radioFor(SelectionMode.prompt));
      await tester.pump();

      expect(selectEnabled(tester), isTrue);
    });
  });

  group('Box/point status label (v3 spec §3 Fix 2 surfacing)', () {
    // Pure-function logic (no widget tree needed) -- same rationale as
    // promptCardCanSubmit's own tests: exercises the REAL
    // positiveClickStatusLabel/promptStatusLabel production code calls,
    // not a re-implementation.
    Map<String, dynamic> response({String? selectedMaskId, required List maskIds}) => {
          'selected_mask_id': selectedMaskId,
          'results': {'mask_ids': maskIds},
        };

    test('a positive click with no selected_mask_id reports "0 objects found"', () {
      // The actual bug this closes: previously a failed click could echo
      // a stale id from an EARLIER call, or `results.mask_ids` would
      // still be non-empty from prior selections, both of which made
      // promptStatusLabel alone report "Done" for a click that found
      // nothing.
      final r = response(selectedMaskId: null, maskIds: ['0:earlier-selection']);

      expect(positiveClickStatusLabel(r, true), '0 objects found');
    });

    test('a positive click that finds something reports "Done"', () {
      final r = response(selectedMaskId: '0:abc', maskIds: ['0:abc']);

      expect(positiveClickStatusLabel(r, true), 'Done');
    });

    test('a negative (Avoid) click with a null selected_mask_id still reports "Done" '
        'when the pass already has selections', () {
      // selected_mask_id carries no success/failure meaning for Avoid --
      // it's left untouched by _select_single by design, so a null value
      // here (e.g. the very first action in a session) must NOT be read
      // as "this Avoid click failed". promptStatusLabel's own whole-pass
      // check is what applies instead.
      final r = response(selectedMaskId: null, maskIds: ['0:already-selected']);

      expect(positiveClickStatusLabel(r, false), 'Done');
    });

    test('a negative (Avoid) click reports "0 objects found" only via the whole-pass check, '
        'not selected_mask_id', () {
      final r = response(selectedMaskId: null, maskIds: []);

      expect(positiveClickStatusLabel(r, false), '0 objects found');
    });

    test('promptStatusLabel alone: empty mask_ids reports "0 objects found"', () {
      expect(promptStatusLabel({'mask_ids': []}), '0 objects found');
      expect(promptStatusLabel({'mask_ids': null}), '0 objects found');
      expect(promptStatusLabel(null), '0 objects found');
    });

    test('promptStatusLabel alone: non-empty mask_ids reports "Done"', () {
      expect(promptStatusLabel({'mask_ids': ['0:abc']}), 'Done');
    });
  });

  group('sessionImageUrlFrom (/loadSession Image URL field wiring)', () {
    // Regression test for a real bug: /loadSession's response has carried
    // `image_url` from registry.parquet all along (services.py's
    // load_session_from_disk), but _loadLaunchSession never read it -- the
    // Image URL field showed its "(no image URL)" placeholder even for a
    // session that genuinely has one on disk. Confirmed via
    // DoubleNaught's seg_forge_node_widget.dart: a session's `image_url` is
    // set once at its original creation (New Session from a URL or
    // uploaded-bytes image port) and persists across every later resume
    // (DN deliberately passes an empty SEGFORGE_IMAGE_URL when resuming,
    // trusting SF to read the session's own stored value) -- so a
    // resumed session with a real, previously-set image_url is exactly
    // the case this fixes.
    test('a session with a real image_url displays it', () {
      final session = {
        'name': 'My Session',
        'image_url': 'https://example.com/cat.png',
      };

      expect(sessionImageUrlFrom(session), 'https://example.com/cat.png');
    });

    test('a session with a real image_url with surrounding whitespace is trimmed', () {
      final session = {'image_url': '  https://example.com/cat.png  '};

      expect(sessionImageUrlFrom(session), 'https://example.com/cat.png');
    });

    test('a session with no image_url at all returns null, not an empty string', () {
      // The other, legitimate half of this fix: a session that has
      // genuinely never had a URL (e.g. resumed, or uploaded via SF's own
      // standalone file picker) must still fall through to the
      // "(no image URL)" placeholder, not show a blank/broken value.
      final session = {'name': 'My Session'};

      expect(sessionImageUrlFrom(session), isNull);
    });

    test('a session with an empty-string image_url returns null', () {
      final session = {'image_url': ''};

      expect(sessionImageUrlFrom(session), isNull);
    });

    test('a session with a whitespace-only image_url returns null', () {
      final session = {'image_url': '   '};

      expect(sessionImageUrlFrom(session), isNull);
    });
  });

  group('DisplayAreaTabs', () {
    // Tests the tab wiring + live-reactivity mechanism this widget adds,
    // NOT the shared aa_preview_table package's own rendering/pagination
    // logic (that has its own 14-test suite) and not SegmentationCanvas's
    // own selection-mode behaviour (covered above, unchanged by this
    // widget existing).
    late ui.Image image;

    setUpAll(() async {
      image = await _solidImage(200, 200);
    });

    Widget pumpTabs({
      Map<String, dynamic>? result,
      Map<String, dynamic>? aaPreviewData,
      Uint8List? imageBytes,
      ui.Image? currentImage,
      List<Segment> segments = const <Segment>[],
    }) =>
        _wrap(DisplayAreaTabs(
          imageBytes: imageBytes,
          uiImage: currentImage ?? image,
          originalUiImage: image,
          segments: segments,
          result: result,
          aaPreviewData: aaPreviewData,
          isLoading: false,
          mode: null,
          onBoxDrawn: (_) {},
          onPointDrawn: (_) {},
        ));

    // AaPreviewTableWidget reloads via AaPreviewWorker's real Isolate.run()
    // on every non-empty payload (initial mount AND update, via
    // didUpdateWidget) — genuine async I/O that, like dart:ui rasterization
    // elsewhere in this file, never advances inside testWidgets' fake-async
    // zone. While it's pending, the widget shows an indeterminate spinner
    // (CircularProgressIndicator on first load, LinearProgressIndicator on
    // a reload), which tickers forever and makes a bare pumpAndSettle()
    // time out. Only needed once the payload is non-empty — the empty-AA
    // short-circuit in AaPreviewTableWidget.build() never reaches that
    // spinner code at all, which is why the earlier empty-state tests don't
    // need this.
    Future<void> settleAfterAaLoad(WidgetTester tester) async {
      await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 500)));
      await tester.pumpAndSettle();
    }

    testWidgets('Canvas tab is selected by default and shows the canvas', (tester) async {
      await tester.pumpWidget(pumpTabs(imageBytes: Uint8List(0), result: null));
      await tester.pumpAndSettle();

      expect(find.text('Canvas'), findsOneWidget);
      expect(find.text('AA Preview'), findsOneWidget);
      expect(find.byType(SegmentationCanvas), findsOneWidget);
    });

    testWidgets('AA Preview tab regression check: zero masks shows the '
        "shared widget's existing empty state", (tester) async {
      await tester.pumpWidget(pumpTabs(imageBytes: Uint8List(0), result: null));
      await tester.pumpAndSettle();

      await tester.tap(find.text('AA Preview'));
      await tester.pumpAndSettle();

      expect(find.textContaining('Empty associative array'), findsOneWidget);
    });

    testWidgets('AA Preview tab updates live on rebuild, without leaving and returning to the tab',
        (tester) async {
      // Starts empty (no selections yet).
      await tester.pumpWidget(pumpTabs(imageBytes: Uint8List(0), aaPreviewData: null));
      await tester.pumpAndSettle();
      await tester.tap(find.text('AA Preview'));
      await tester.pumpAndSettle();
      expect(find.textContaining('Empty associative array'), findsOneWidget);

      // Simulates what _sendBoxPrompt's setState() does in main.dart: a new
      // `aaPreviewData` map reaches this widget via an ordinary rebuild —
      // nothing re-taps into the AA Preview tab, it's already the active one.
      await tester.pumpWidget(pumpTabs(
        imageBytes: Uint8List(0),
        aaPreviewData: {
          'mask_ids': ['0:abc', '0:def'],
          'dataset_statuses': ['keep', 'unassigned'],
          'held_flags': [true, false],
          'passes': [0, 0],
        },
      ));
      await settleAfterAaLoad(tester);

      expect(find.textContaining('Empty associative array'), findsNothing);
      expect(find.textContaining('2 rows'), findsOneWidget);
    });

    testWidgets('a scrub does NOT clear the AA Preview tab, even though it clears the canvas',
        (tester) async {
      // Regression: a captioned/kept mask used to vanish from this tab the
      // moment its pass was scrubbed past, because the adapter read
      // `result` (current-pass-only, nulled by _scrubLamaBackground's own
      // setState on every scrub) instead of the session-wide ledger.
      // Confirmed live: "I did a selection with a prompt... I did the
      // scrub. AA preview shows nothing" -- reproduced with a genuinely
      // captioned mask, not just an uncaptioned one, since the report
      // didn't distinguish. `aaPreviewData` (this test's whole point) is
      // main.dart's `_aaPreviewData` -- deliberately NOT reset alongside
      // `result`/`_segments` in _scrubLamaBackground; see its own comment.
      // First build of AaPreviewTableWidget MUST be empty: its Isolate.run()
      // reload never resolves in this test harness when kicked off from
      // initState() (a hard limitation of this environment, confirmed
      // directly — real dart:ui rasterization has the same class of issue
      // elsewhere in this file), only when kicked off from didUpdateWidget()
      // on an already-mounted widget. The empty-AA short-circuit in
      // AaPreviewTableWidget.build() means an empty initState() load is
      // harmless (never reaches the spinner, so nothing needs waiting on) —
      // so the actual "starts with real data" case is modelled by mounting
      // empty, then rebuilding with data, exactly like the live app's own
      // sequence (nothing is ever selected before the first render either).
      await tester.pumpWidget(pumpTabs(imageBytes: Uint8List(0), result: null, aaPreviewData: null));
      await tester.pumpAndSettle();
      await tester.tap(find.text('AA Preview'));
      await tester.pumpAndSettle(); // empty payload — trivial, no spinner involved
      expect(find.textContaining('Empty associative array'), findsOneWidget);

      // Regression: a captioned/kept mask used to vanish from this tab the
      // moment its pass was scrubbed past, because the adapter read
      // `result` (current-pass-only, nulled by _scrubLamaBackground's own
      // setState on every scrub) instead of the session-wide ledger.
      // Confirmed live: "I did a selection with a prompt... I did the
      // scrub. AA preview shows nothing" -- reproduced here with a
      // genuinely captioned mask, not just an uncaptioned one, since the
      // report didn't distinguish.
      await tester.pumpWidget(pumpTabs(
        imageBytes: Uint8List(0),
        result: {
          'mask_ids': ['0:abc'],
          'dataset_statuses': ['keep'],
          'passes': [0],
        },
        aaPreviewData: {
          'mask_ids': ['0:abc'],
          'dataset_statuses': ['keep'],
          'captions': ['a grey egg'],
          'passes': [0],
        },
      ));
      await settleAfterAaLoad(tester);
      expect(find.textContaining('1 rows'), findsOneWidget);

      // Simulates _scrubLamaBackground's own setState(): `result` resets to
      // null (correct — the new pass has no live geometry yet, matches
      // serialize_sf_masks' current-pass-only contract) but `aaPreviewData`
      // (main.dart's `_aaPreviewData`) carries forward the fresh `all_masks`
      // response, which still includes the captioned mask from the pass
      // that was just scrubbed — deliberately NOT reset alongside
      // `result`/`_segments`, see that setState block's own comment.
      await tester.pumpWidget(pumpTabs(
        imageBytes: Uint8List(0),
        result: null,
        aaPreviewData: {
          'mask_ids': ['0:abc'],
          'dataset_statuses': ['keep'],
          'captions': ['a grey egg'],
          'passes': [0],
        },
      ));
      await settleAfterAaLoad(tester);

      expect(find.textContaining('Empty associative array'), findsNothing);
      expect(find.textContaining('1 rows'), findsOneWidget);
    });

    testWidgets('an uncaptioned, scrubbed-away mask shows as unassigned, not absent '
        '— this is the real audit trail, not the bug', (tester) async {
      // The flip side of the fix above: the v2 model's implicit discard
      // (held + scrubbed, never captioned) is real and intentional
      // (backend/sf_engine.py's TestExport) — but it's an EXPORT-time
      // filter (build_sf_payload's keep-only pass), never a deletion from
      // SFSession.masks. Confirmed directly against the backend: the
      // record stays forever as an audit trail (DatasetStatus has no
      // discard value at all, only unassigned/keep), so `all_masks` keeps
      // reporting it — correctly as unassigned/unheld, not absent. An
      // earlier version of this test assumed it would disappear and was
      // simply wrong; fixed after checking the real backend response
      // rather than a hand-built assumption.
      // First build empty, then tap, then rebuild with data — see the
      // sibling test above for why (Isolate.run() from initState() never
      // resolves in this harness; only a didUpdateWidget()-triggered
      // reload does).
      await tester.pumpWidget(pumpTabs(imageBytes: Uint8List(0), aaPreviewData: null));
      await tester.pumpAndSettle();
      await tester.tap(find.text('AA Preview'));
      await tester.pumpAndSettle();
      expect(find.textContaining('Empty associative array'), findsOneWidget);

      await tester.pumpWidget(pumpTabs(
        imageBytes: Uint8List(0),
        aaPreviewData: {
          'mask_ids': ['0:abc'],
          'dataset_statuses': ['unassigned'],
          'held_flags': [true],
          'passes': [0],
        },
      ));
      await settleAfterAaLoad(tester);
      expect(find.textContaining('1 rows'), findsOneWidget);

      // Backend's all_masks after the scrub: the record survives with
      // held cleared to false (matching a real /lama/scrub response —
      // confirmed backend-side, see test_sf_wiring.py's
      // test_all_masks_shows_an_uncaptioned_scrubbed_mask_as_unassigned_not_absent).
      await tester.pumpWidget(pumpTabs(
        imageBytes: Uint8List(0),
        aaPreviewData: {
          'mask_ids': ['0:abc'],
          'dataset_statuses': ['unassigned'],
          'held_flags': [false],
          'passes': [0],
        },
      ));
      await settleAfterAaLoad(tester);

      expect(find.textContaining('Empty associative array'), findsNothing);
      expect(find.textContaining('1 rows'), findsOneWidget);
    });

    testWidgets('AA Preview tab renders a mask\'s cached crop inline, not as base64 text',
        (tester) async {
      // A valid, minimal (1×1 transparent) PNG, base64-encoded — matches
      // the shape services.py's crop cache actually returns.
      const tinyPngBase64 =
          'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A'
          'AQUBAScY42YAAAAASUVORK5CYII=';

      await tester.pumpWidget(pumpTabs(imageBytes: Uint8List(0), aaPreviewData: null));
      await tester.pumpAndSettle();
      await tester.tap(find.text('AA Preview'));
      await tester.pumpAndSettle();
      expect(find.textContaining('Empty associative array'), findsOneWidget);

      await tester.pumpWidget(pumpTabs(
        imageBytes: Uint8List(0),
        aaPreviewData: {
          'mask_ids': ['0:abc'],
          'dataset_statuses': ['keep'],
          'held_flags': [false],
          'passes': [0],
          'crop_png_bytes': [tinyPngBase64],
        },
      ));
      await settleAfterAaLoad(tester);

      expect(find.byType(Image), findsOneWidget);
      expect(find.textContaining(tinyPngBase64), findsNothing);
    });
  });
}
