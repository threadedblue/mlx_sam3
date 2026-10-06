// Manual panel-bounds tool (v4 §4), frontend half.
//
// A panel is a LAYOUT REGION, not an object. The properties pinned here are
// the structural ones: a panel drag produces NO mask (nothing in Objects
// Selected, nothing in `results`, no /segment/* call at all), the Panels
// card is Comix-only, and authoring one dirties the session so the Unsaved
// Changes dialog stays correct.
//
// Drives the real HomeScreen through runWithClient/MockClient, the pattern
// established in resume_wires_aa_preview_test.dart (no real socket I/O
// works here — see that file's header for why).
import 'dart:convert';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';
import 'package:frontend/sf_mode.dart';
import 'package:frontend/layered_segmentation_canvas.dart';
import 'package:frontend/widgets/panels_card.dart';
import 'package:frontend/widgets/objects_selected_card.dart';

const _sid = 'panels-test-session';

/// A real 200x200 PNG, COLOURED and NON-UNIFORM (a per-pixel gradient), not
/// a flat or black square: the canvas has to decode a genuine image for a
/// drag to produce real page-pixel coordinates, and a flat fixture would
/// make a wrong-coordinate bug invisible.
const _imageB64 =
    'iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAIAAAAiOjnJAAAD/ElEQVR4nO3WgQkCQRTE0KxnCdZkYVZuEyfBIZAKdh6fPa/35+JUL3DdyuDJ40D1AudeBsGK1PnFZQlWsE6wQnD+5d/SxfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlb8BiwfI3YLFg+RuwWLD8DVgsWP4GLBYsfwMWC5a/AYsFy9+AxYLlbzBZsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/AxYLlr8BiwXL34DFguVvwGLB8jdgsWD5G7BYsPwNWCxY/gYsFix/Axb7AgIFpBwbTmFCAAAAAElFTkSuQmCC';

Widget _wrap(Widget child) => ChangeNotifierProvider<LayerState>(
      create: (_) => LayerState(),
      child: MaterialApp(home: child),
    );

Map<String, dynamic> _emptyAllMasks() => {
      'mask_ids': <dynamic>[], 'dataset_statuses': <dynamic>[],
      'held_flags': <dynamic>[], 'captions': <dynamic>[],
      'text_tags': <dynamic>[], 'scores': <dynamic>[], 'boxes': <dynamic>[],
      'crop_png_bytes': <dynamic>[], 'passes': <dynamic>[],
    };

Map<String, dynamic> _emptyResults() => {
      'mask_ids': <dynamic>[], 'dataset_statuses': <dynamic>[],
      'held_flags': <dynamic>[], 'masks': <dynamic>[], 'boxes': <dynamic>[],
      'scores': <dynamic>[],
    };

/// Records every path the app hit, so "no /segment/* call fired" is asserted
/// against the fake rather than inferred.
class _Calls {
  final List<String> paths = [];
  final List<Map<String, dynamic>> addPanelBodies = [];
}

MockClient _client(
  _Calls calls, {
  required List<Map<String, dynamic>> panels,
  String mode = 'comix',
  bool saveSucceeds = true,
}) {
  // Mutable so /panel/add and /panel/delete behave like the real backend:
  // each returns the list AFTER the change.
  final live = [...panels];
  return MockClient((request) async {
    final p = request.url.path;
    calls.paths.add(p);
    Map<String, dynamic>? body;
    if (request.body.isNotEmpty) {
      try {
        body = jsonDecode(request.body) as Map<String, dynamic>;
      } catch (_) {}
    }

    if (request.method == 'GET' && p == '/loadSession/$_sid') {
      return http.Response(
        jsonEncode({
          'session_id': _sid,
          'image_b64': _imageB64,
          'width': 200, 'height': 200,
          'results': _emptyResults(),
          'all_masks': _emptyAllMasks(),
          'prompts': <dynamic>[],
          'created_at': '2026-01-01T00:00:00',
          'name': 'panels test', 'description': '', 'image_url': null,
          'mode': mode,
          'pass': 0,
          'panels': live,
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    }
    if (p == '/panel/add') {
      calls.addPanelBodies.add(body!);
      final added = {
        'panel_id': 'panel-${live.length + 1}',
        'x0': body['x0'], 'y0': body['y0'], 'x1': body['x1'], 'y1': body['y1'],
      };
      live.add(added);
      return http.Response(
        jsonEncode({
          'session_id': _sid, 'panel_id': added['panel_id'], 'panels': live,
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    }
    if (p == '/panel/delete') {
      live.removeWhere((x) => x['panel_id'] == body!['panel_id']);
      return http.Response(
        jsonEncode({'session_id': _sid, 'panels': live}),
        200,
        headers: {'content-type': 'application/json'},
      );
    }
    if (p == '/saveSession') {
      if (saveSucceeds) {
        return http.Response(
          jsonEncode({'message': 'Session saved', 'session_id': _sid}),
          200, headers: {'content-type': 'application/json'});
      }
      return http.Response(jsonEncode({'detail': 'simulated save failure'}), 500);
    }
    return http.Response('not found', 404);
  });
}

Future<void> _pumpHome(WidgetTester tester, MockClient client) async {
  await tester.binding.setSurfaceSize(const Size(1600, 1200));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  await http.runWithClient(() async {
    // image_b64 goes through ui.instantiateImageCodec, which only ever
    // completes under runAsync — without it `_uiImage` stays null, and
    // _loadLaunchSession's setState (which restores `mode`, and therefore
    // decides whether the Panels card exists at all) is never reached. Same
    // pattern and same reason as balloon_scrub_card_test.dart.
    await tester.runAsync(() async {
      await tester.pumpWidget(_wrap(const HomeScreen(initialSessionId: _sid)));
      await tester.pump();
      await Future<void>.delayed(const Duration(milliseconds: 300));
    });
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

Finder get _segmentCheckboxes => find.descendant(
      of: find.byType(ObjectsSelectedCard),
      matching: find.byType(CheckboxListTile),
    );

Future<bool> _tapSwitchSessionAndCheckDialog(
    WidgetTester tester, MockClient client) async {
  await http.runWithClient(() async {
    final btn = find.widgetWithText(OutlinedButton, 'Switch Session');
    await tester.ensureVisible(btn);
    await tester.pump();
    await tester.tap(btn, warnIfMissed: false);
    await tester.pump();
  }, () => client);
  final shown = find.text('Unsaved Changes').evaluate().isNotEmpty;
  if (shown) {
    await tester.tap(find.text('Keep Working'));
    await tester.pump();
  }
  return shown;
}

/// Selects the Panels card's radio, putting the canvas in Panel mode.
Future<void> _selectPanelMode(WidgetTester tester, MockClient client) async {
  await http.runWithClient(() async {
    final radio = find.descendant(
      of: find.byType(PanelsCard),
      matching: find.byType(Radio<SelectionMode>),
    );
    await tester.ensureVisible(radio);
    await tester.pump();
    await tester.tap(radio, warnIfMissed: false);
    await tester.pump();
  }, () => client);
}

/// Authors a panel through the REAL callback main.dart wires into the
/// canvas (`onPanelDrawn: (b) => _addPanel(...)`), driving everything from
/// that closure onward: /panel/add, the panel-list refresh, and the
/// mutation counter.
///
/// Deliberately not a synthetic `tester.dragFrom` here: inside HomeScreen
/// the canvas sits in a TabBarView, whose horizontal PageView drag
/// recognizer wins the gesture arena against the canvas's own pan — so a
/// drag in this tree tests Flutter's arena, not this feature. The gesture
/// half (drag -> onPanelDrawn, in page pixels, only in Panel mode) is
/// covered directly in the "SegmentationCanvas in Panel mode" group above,
/// against a canvas with no PageView over it.
Future<void> _authorPanel(
    WidgetTester tester, MockClient client, List<int> box) async {
  final canvas = tester.widget<SegmentationCanvas>(
      find.byType(SegmentationCanvas));
  await http.runWithClient(() async {
    canvas.onPanelDrawn(box);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

Future<void> _deleteFirstPanel(WidgetTester tester, MockClient client) async {
  await http.runWithClient(() async {
    final btn = find.byTooltip('Delete Panel 1');
    await tester.ensureVisible(btn);
    await tester.pump();
    await tester.tap(btn, warnIfMissed: false);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

Future<void> _tapSave(WidgetTester tester, MockClient client) async {
  await http.runWithClient(() async {
    final btn = find.widgetWithText(ElevatedButton, 'Save');
    await tester.ensureVisible(btn);
    await tester.pump();
    await tester.tap(btn, warnIfMissed: false);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

void main() {
  group('mode gating', () {
    test('Panels is Comix-only; Poster and Cine stay empty', () {
      expect(cardsFor(SfMode.comix), contains(SfCard.panels));
      expect(cardsFor(SfMode.poster), isEmpty);
      expect(cardsFor(SfMode.cine), isEmpty);
    });

    test('the unfiltered/legacy list does not gain Panels', () {
      // kAllCards is defined as "unchanged from pre-mode behaviour"; Panels
      // is the first genuinely mode-specific card.
      expect(cardsFor(null), isNot(contains(SfCard.panels)));
      expect(cardsFor(null), kAllCards);
    });
  });

  group('PanelsCard rendering', () {
    testWidgets('shows the count and one row per panel, each with a delete control',
        (tester) async {
      final deleted = <String>[];
      await tester.pumpWidget(_wrap(Scaffold(
        body: RadioGroup<SelectionMode>(
          groupValue: null,
          onChanged: (_) {},
          child: PanelsCard(
            panels: const [
              {'panel_id': 'a', 'x0': 0, 'y0': 0, 'x1': 100, 'y1': 120},
              {'panel_id': 'b', 'x0': 110, 'y0': 10, 'x1': 200, 'y1': 140},
            ],
            hasImage: true,
            onDeletePanel: deleted.add,
          ),
        ),
      )));

      expect(find.text('Panels'), findsOneWidget);
      expect(find.text('2 panels'), findsOneWidget);
      expect(find.textContaining('Panel 1'), findsWidgets);
      expect(find.textContaining('Panel 2'), findsWidgets);

      await tester.tap(find.byTooltip('Delete Panel 2'));
      await tester.pump();
      // Routed by id, never by index — deleting row 2 must name panel 'b'.
      expect(deleted, ['b']);
    });

    testWidgets('an empty session shows zero panels and no rows', (tester) async {
      await tester.pumpWidget(_wrap(Scaffold(
        body: RadioGroup<SelectionMode>(
          groupValue: null,
          onChanged: (_) {},
          child: PanelsCard(
            panels: const [],
            hasImage: true,
            onDeletePanel: (_) {},
          ),
        ),
      )));

      expect(find.text('0 panels'), findsOneWidget);
      expect(find.byTooltip('Delete Panel 1'), findsNothing);
    });

    testWidgets('singular label for exactly one panel', (tester) async {
      await tester.pumpWidget(_wrap(Scaffold(
        body: RadioGroup<SelectionMode>(
          groupValue: null,
          onChanged: (_) {},
          child: PanelsCard(
            panels: const [{'panel_id': 'a', 'x0': 0, 'y0': 0, 'x1': 50, 'y1': 50}],
            hasImage: true,
            onDeletePanel: (_) {},
          ),
        ),
      )));
      expect(find.text('1 panel'), findsOneWidget);
    });
  });

  group('Panel mode', () {
    test('Select is disabled in Panel mode, as it is for Box and Point', () {
      bool canSubmit(SelectionMode? m) => promptCardCanSubmit(
            hasSession: true, promptEmpty: false, isLoading: false,
            captionMode: false, selectedMode: m,
          );
      expect(canSubmit(SelectionMode.panel), isFalse);
      expect(canSubmit(SelectionMode.box), isFalse);
      expect(canSubmit(SelectionMode.point), isFalse);
      expect(canSubmit(SelectionMode.prompt), isTrue);
      expect(canSubmit(null), isTrue);
    });

    test('Save Caption stays enabled in Panel mode', () {
      // Captioning the already-focused mask is a normal next step and is
      // deliberately unaffected by mode, same as for Box/Point.
      expect(
        promptCardCanSubmit(
          hasSession: true, promptEmpty: false, isLoading: false,
          captionMode: true, selectedMode: SelectionMode.panel,
        ),
        isTrue,
      );
    });
  });

  group('the Panels card in the real HomeScreen', () {
    testWidgets('resumed panels are listed', (tester) async {
      final calls = _Calls();
      await _pumpHome(tester, _client(calls, panels: [
        {'panel_id': 'p1', 'x0': 5, 'y0': 6, 'x1': 105, 'y1': 126},
      ]));

      expect(find.byType(PanelsCard), findsOneWidget);
      expect(find.text('1 panel'), findsOneWidget);
      expect(find.textContaining('100x120 at 5,6'), findsWidgets);
    });

    testWidgets('a panel creates no mask: Objects Selected and results unchanged',
        (tester) async {
      final calls = _Calls();
      final client = _client(calls, panels: [
        {'panel_id': 'p1', 'x0': 5, 'y0': 6, 'x1': 105, 'y1': 126},
      ]);
      await _pumpHome(tester, client);

      expect(_segmentCheckboxes, findsNothing,
          reason: 'a panel must never appear as a selectable object');
      // The Scrub button keys off `hasSelection`, which panels must not feed.
      final scrub = find.widgetWithText(ElevatedButton, 'Scrub Selected Regions');
      expect(tester.widget<ElevatedButton>(scrub).onPressed, isNull);
      // And nothing in the selection pipeline was ever contacted.
      expect(calls.paths.where((p) => p.startsWith('/segment/')), isEmpty);
    });

    testWidgets('deleting a panel calls /panel/delete and never /segment/*',
        (tester) async {
      final calls = _Calls();
      final client = _client(calls, panels: [
        {'panel_id': 'p1', 'x0': 5, 'y0': 6, 'x1': 105, 'y1': 126},
      ]);
      await _pumpHome(tester, client);
      expect(find.text('1 panel'), findsOneWidget);

      await http.runWithClient(() async {
        final btn = find.byTooltip('Delete Panel 1');
        await tester.ensureVisible(btn);
        await tester.pump();
        await tester.tap(btn, warnIfMissed: false);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 500));
      }, () => client);

      expect(find.text('0 panels'), findsOneWidget);
      expect(calls.paths, contains('/panel/delete'));
      expect(calls.paths.where((p) => p.startsWith('/segment/')), isEmpty);
    });
  });

  group('SegmentationCanvas in Panel mode', () {
    late ui.Image image;

    setUpAll(() async {
      // Non-uniform on purpose: a flat fixture would hide a coordinate bug.
      final recorder = ui.PictureRecorder();
      final c = Canvas(recorder);
      for (var i = 0; i < 20; i++) {
        c.drawRect(
          Rect.fromLTWH(i * 10.0, 0, 10, 200),
          Paint()..color = Color(0xFF000000 | (i * 0x0C1123)),
        );
      }
      image = await recorder.endRecording().toImage(200, 200);
    });

    Future<List<List<int>>> dragInMode(
        WidgetTester tester, SelectionMode? mode) async {
      final panels = <List<int>>[];
      final boxes = <List<double>>[];
      await tester.pumpWidget(_wrap(Scaffold(
        body: SegmentationCanvas(
          uiImage: image,
          originalImage: image,
          segments: const <Segment>[],
          isLoading: false,
          mode: mode,
          onBoxDrawn: boxes.add,
          onPointDrawn: (_) {},
          onPanelDrawn: panels.add,
        ),
      )));
      final centre = tester.getCenter(find.byType(SegmentationCanvas));
      await tester.dragFrom(centre - const Offset(40, 40), const Offset(80, 80));
      await tester.pump();
      // A panel drag must never also register as a box prompt.
      if (mode == SelectionMode.panel) expect(boxes, isEmpty);
      return panels;
    }

    testWidgets('a drag in Panel mode reports PAGE PIXELS, not normalised',
        (tester) async {
      final panels = await dragInMode(tester, SelectionMode.panel);

      expect(panels, hasLength(1));
      final b = panels.single;
      // 200x200 image; the drag spans 80 logical px about the centre. The
      // values must be page pixels (tens), never the 0..1 cx/cy/w/h a box
      // prompt sends.
      expect(b, hasLength(4));
      expect(b[2] - b[0], greaterThan(10),
          reason: 'page pixels, so a width of ~80 — not a fraction');
      expect(b[3] - b[1], greaterThan(10));
      for (final v in b) {
        expect(v, isA<int>());
        expect(v, inInclusiveRange(0, 200));
      }
    });

    testWidgets('a drag in Box mode still produces a box prompt, not a panel',
        (tester) async {
      final panels = await dragInMode(tester, SelectionMode.box);
      expect(panels, isEmpty,
          reason: 'Panel and Box must stay mutually exclusive');
    });

    testWidgets('no mode means no panel', (tester) async {
      expect(await dragInMode(tester, null), isEmpty);
    });

    testWidgets('panel outlines are painted in the interaction overlay',
        (tester) async {
      await tester.pumpWidget(_wrap(Scaffold(
        body: SegmentationCanvas(
          uiImage: image,
          originalImage: image,
          segments: const <Segment>[],
          isLoading: false,
          mode: null,
          panels: const [
            {'panel_id': 'a', 'x0': 10, 'y0': 20, 'x1': 110, 'y1': 140},
          ],
          onBoxDrawn: (_) {},
          onPointDrawn: (_) {},
          onPanelDrawn: (_) {},
        ),
      )));

      final painter = tester
          .widgetList<CustomPaint>(find.byType(CustomPaint))
          .map((w) => w.painter)
          .whereType<PanelOutlinePainter>()
          .single;
      expect(painter.panels, hasLength(1));
      expect(painter.panels.single['x0'], 10);
    });
  });

  group('end to end in the real HomeScreen', () {
    testWidgets('an authored panel reaches both the card and the canvas outline',
        (tester) async {
      final calls = _Calls();
      final client = _client(calls, panels: []);
      await _pumpHome(tester, client);
      expect(find.text('0 panels'), findsOneWidget);

      await _selectPanelMode(tester, client);
      await _authorPanel(tester, client, [10, 20, 110, 140]);

      // The card lists it...
      expect(find.text('1 panel'), findsOneWidget);
      expect(find.textContaining('100x120 at 10,20'), findsWidgets);

      // ...and the canvas outlines it, from the same state.
      final painter = tester
          .widgetList<CustomPaint>(find.byType(CustomPaint))
          .map((w) => w.painter)
          .whereType<PanelOutlinePainter>()
          .single;
      expect(painter.panels.single['x0'], 10);
      expect(painter.panels.single['y1'], 140);

      // And still no mask was created by any of it.
      expect(calls.paths.where((p) => p.startsWith('/segment/')), isEmpty);
      expect(_segmentCheckboxes, findsNothing);
    });
  });

  group('unsaved changes', () {
    // NOTE: _tapSwitchSessionAndCheckDialog navigates AWAY when no dialog
    // appears, destroying the HomeScreen — so each test gets at most one
    // "clean" assertion, and it has to come last.

    testWidgets('adding a panel by dragging the canvas dirties the session',
        (tester) async {
      final calls = _Calls();
      final client = _client(calls, panels: []);
      await _pumpHome(tester, client);
      await _selectPanelMode(tester, client);
      await _authorPanel(tester, client, [10, 20, 110, 140]);

      expect(calls.paths, contains('/panel/add'));
      expect(calls.addPanelBodies.single, {
        'session_id': _sid, 'x0': 10, 'y0': 20, 'x1': 110, 'y1': 140,
      });
      expect(await _tapSwitchSessionAndCheckDialog(tester, client), isTrue,
          reason: 'an authored panel is persisted state, so it must dirty');
    });

    testWidgets('Save after a panel add clears the dirt', (tester) async {
      final calls = _Calls();
      final client = _client(calls, panels: []);
      await _pumpHome(tester, client);
      await _selectPanelMode(tester, client);
      await _authorPanel(tester, client, [10, 20, 110, 140]);
      await _tapSave(tester, client);

      expect(await _tapSwitchSessionAndCheckDialog(tester, client), isFalse,
          reason: 'Save must clear panel-authored dirt like any other');
    });

    testWidgets('deleting a panel dirties the session', (tester) async {
      final calls = _Calls();
      final client = _client(calls, panels: [
        {'panel_id': 'p1', 'x0': 5, 'y0': 6, 'x1': 105, 'y1': 126},
      ]);
      await _pumpHome(tester, client);
      await _deleteFirstPanel(tester, client);

      expect(await _tapSwitchSessionAndCheckDialog(tester, client), isTrue);
    });

    testWidgets('Save after a panel delete clears the dirt', (tester) async {
      final calls = _Calls();
      final client = _client(calls, panels: [
        {'panel_id': 'p1', 'x0': 5, 'y0': 6, 'x1': 105, 'y1': 126},
      ]);
      await _pumpHome(tester, client);
      await _deleteFirstPanel(tester, client);
      await _tapSave(tester, client);

      expect(await _tapSwitchSessionAndCheckDialog(tester, client), isFalse);
    });
  });
}
