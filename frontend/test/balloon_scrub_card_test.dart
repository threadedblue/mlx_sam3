// Coverage for the Balloon Scrub card and the detect-and-hold flow behind it.
//
// No card in this app had a test before this one (LBSCard included), so this
// establishes the shape: the two pure-presentation properties (button gating,
// in-flight spinner) drive the real `BalloonScrubCard` directly, and everything
// that depends on backend wiring drives the real `HomeScreen` through
// `runWithClient`/`MockClient` — the pattern from
// resume_wires_aa_preview_test.dart and unsaved_changes_dirty_tracking_test.dart
// (no real socket I/O works here; see that file's header for why).
//
// The card only detects and holds. It must never scrub: /lama/scrub belongs to
// LaMa Background Scrub, and a second scrub path is exactly what this card was
// specified not to build.
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';
import 'package:frontend/widgets/balloon_scrub_card.dart';

const _sid = 'balloon-card-test-session';
const _otherSid = 'some-other-session-that-must-not-be-touched';
const _maskA = '0:balloon-a';
const _maskB = '0:balloon-b';

/// A real 10x10 PNG, so `_uiImage` actually decodes and the card's
/// `hasImage` gate is genuinely satisfied rather than faked.
const _png =
    'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGOsCNBgwA2Y8MgxjFxpAC5xAQSHxVkEAAAAAElFTkSuQmCC';

/// Convention (services.py/aa_persistence.py): first run is the background
/// count; [0, 100] on a 10x10 canvas is "all foreground".
const _rle = {
  'counts': [0, 100],
  'size': [10, 10],
};

Widget _wrap(Widget child) => ChangeNotifierProvider<LayerState>(
      create: (_) => LayerState(),
      child: MaterialApp(home: child),
    );

Map<String, dynamic> _emptyAllMasks() => {
      'mask_ids': <dynamic>[],
      'dataset_statuses': <dynamic>[],
      'held_flags': <dynamic>[],
      'captions': <dynamic>[],
      'text_tags': <dynamic>[],
      'scores': <dynamic>[],
      'boxes': <dynamic>[],
      'crop_png_bytes': <dynamic>[],
      'passes': <dynamic>[],
    };

/// Two detected balloons, exactly as /segment/balloons reports them: held,
/// and never captioned — unassigned is the implicit-discard state.
Map<String, dynamic> _twoBalloons() => {
      'mask_ids': [_maskA, _maskB],
      'dataset_statuses': ['unassigned', 'unassigned'],
      'held_flags': [true, true],
      'captions': [null, null],
      'text_tags': [null, null],
      'scores': [0.91, 0.62],
      'boxes': [
        [1.0, 1.0, 4.0, 4.0],
        [5.0, 5.0, 9.0, 9.0],
      ],
      'crop_png_bytes': [null, null],
      'passes': [0, 0],
    };

/// Records every request so a test can assert what was and was not called.
class _Calls {
  final List<String> paths = [];
  final List<String> sessionIds = [];
}

MockClient _buildClient(_Calls calls, {required int detectionCount}) {
  return MockClient((request) async {
    final p = request.url.path;
    calls.paths.add(p);
    if (request.body.isNotEmpty) {
      final body = jsonDecode(request.body);
      if (body is Map && body['session_id'] is String) {
        calls.sessionIds.add(body['session_id'] as String);
      }
    }

    if (p == '/loadSession/$_sid') {
      return http.Response(
        jsonEncode({
          'session_id': _sid,
          'image_b64': _png,
          'width': 10,
          'height': 10,
          'results': {'masks': <dynamic>[], 'boxes': <dynamic>[], 'scores': <dynamic>[]},
          'all_masks': _emptyAllMasks(),
          'prompts': <dynamic>[],
          'created_at': '2026-01-01T00:00:00',
          'name': 'balloon card probe',
          'description': '',
          'image_url': null,
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    }

    if (p == '/segment/balloons') {
      final none = detectionCount == 0;
      return http.Response(
        jsonEncode({
          'session_id': _sid,
          'detection_count': detectionCount,
          'held_mask_ids': none ? <dynamic>[] : [_maskA, _maskB],
          'results': none
              ? {
                  'original_width': 10,
                  'original_height': 10,
                  'masks': <dynamic>[],
                  'boxes': <dynamic>[],
                  'scores': <dynamic>[],
                  'mask_ids': <dynamic>[],
                  'dataset_statuses': <dynamic>[],
                  'held_flags': <dynamic>[],
                  'captions': <dynamic>[],
                  'text_tags': <dynamic>[],
                  'crop_png_bytes': <dynamic>[],
                  'passes': <dynamic>[],
                }
              : {
                  'original_width': 10,
                  'original_height': 10,
                  'masks': [_rle, _rle],
                  'boxes': [
                    [1.0, 1.0, 4.0, 4.0],
                    [5.0, 5.0, 9.0, 9.0],
                  ],
                  'scores': [0.91, 0.62],
                  'mask_ids': [_maskA, _maskB],
                  'dataset_statuses': ['unassigned', 'unassigned'],
                  'held_flags': [true, true],
                  'captions': [null, null],
                  'text_tags': [null, null],
                  'crop_png_bytes': [null, null],
                  'passes': [0, 0],
                },
          'all_masks': none ? _emptyAllMasks() : _twoBalloons(),
          'selected_mask_id': none ? null : _maskB,
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    }

    // Everything else (health, upload) 404s, keeping _maybeAutoLoadImage inert.
    return http.Response('{}', 404, headers: {'content-type': 'application/json'});
  });
}

Future<void> _pumpHomeScreen(WidgetTester tester, MockClient client) async {
  await tester.binding.setSurfaceSize(const Size(1600, 1400));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  await http.runWithClient(() async {
    // The session's image_b64 goes through ui.instantiateImageCodec, which
    // only ever completes under runAsync. Without it `_uiImage` stays null,
    // the card's hasImage gate never opens, and the button stays disabled --
    // which is why unsaved_changes_dirty_tracking_test.dart sends a null
    // image_b64 instead. This card needs a real loaded image, so it pays the
    // runAsync cost rather than faking the gate.
    await tester.runAsync(() async {
      await tester.pumpWidget(_wrap(const HomeScreen(initialSessionId: _sid)));
      await tester.pump();
      await Future<void>.delayed(const Duration(milliseconds: 300));
    });
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

Finder get _detectButton => find.byWidgetPredicate(
      (w) => w is ElevatedButton && w.child is Text && (w.child as Text).data == 'Detect Balloons',
    );

/// The card's button whatever its current child — the label is swapped for a
/// spinner in flight, so a Text-based finder cannot find it then.
Finder get _cardButton => find.descendant(
      of: find.byType(BalloonScrubCard),
      matching: find.byType(ElevatedButton),
    );

Future<void> _tapDetect(WidgetTester tester, MockClient client) async {
  await http.runWithClient(() async {
    await tester.ensureVisible(_detectButton);
    await tester.pump();
    await tester.tap(_detectButton, warnIfMissed: false);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

void main() {
  group('the card itself', () {
    Future<void> pumpCard(
      WidgetTester tester, {
      required bool hasImage,
      required bool isDetecting,
      int? lastDetectedCount,
      VoidCallback? onDetect,
    }) =>
        tester.pumpWidget(MaterialApp(
          home: Scaffold(
            body: BalloonScrubCard(
              hasImage: hasImage,
              lastDetectedCount: lastDetectedCount,
              isDetecting: isDetecting,
              onDetect: onDetect ?? () {},
            ),
          ),
        ));

    testWidgets('the button is disabled with no image loaded', (tester) async {
      var taps = 0;
      await pumpCard(tester,
          hasImage: false, isDetecting: false, onDetect: () => taps++);

      expect(tester.widget<ElevatedButton>(_cardButton).onPressed, isNull);

      await tester.tap(_cardButton, warnIfMissed: false);
      await tester.pump();
      expect(taps, 0, reason: 'a disabled button must not fire its callback');
    });

    testWidgets('the button is enabled once an image is loaded',
        (tester) async {
      await pumpCard(tester, hasImage: true, isDetecting: false);
      expect(tester.widget<ElevatedButton>(_cardButton).onPressed, isNotNull);
    });

    testWidgets('while detecting the button is disabled and the label is '
        'replaced by the spinner', (tester) async {
      await pumpCard(tester, hasImage: true, isDetecting: true);

      expect(tester.widget<ElevatedButton>(_cardButton).onPressed, isNull);

      // Replaced, not overlaid — the label is gone entirely.
      expect(find.text('Detect Balloons'), findsNothing);

      // The exact spinner LBSCard and ResultCell use.
      final spinner = find.descendant(
        of: _cardButton,
        matching: find.byType(CircularProgressIndicator),
      );
      expect(spinner, findsOneWidget);
      expect(tester.widget<CircularProgressIndicator>(spinner).strokeWidth, 2);
      expect(tester.widget<CircularProgressIndicator>(spinner).color,
          Colors.white);

      final box = tester.widget<SizedBox>(find.ancestor(
        of: spinner,
        matching: find.byType(SizedBox),
      ).first);
      expect(box.width, 20);
      expect(box.height, 20);
    });

    testWidgets('the status line distinguishes "never run" from "found none"',
        (tester) async {
      await pumpCard(tester, hasImage: true, isDetecting: false);
      expect(find.text('No detection run yet'), findsOneWidget);

      await pumpCard(tester,
          hasImage: true, isDetecting: false, lastDetectedCount: 0);
      expect(find.text('No balloons detected'), findsOneWidget);

      await pumpCard(tester,
          hasImage: true, isDetecting: false, lastDetectedCount: 1);
      expect(find.text('1 balloon detected'), findsOneWidget);

      await pumpCard(tester,
          hasImage: true, isDetecting: false, lastDetectedCount: 12);
      expect(find.text('12 balloons detected'), findsOneWidget);
    });
  });

  group('wired into HomeScreen', () {
    testWidgets('zero detections is a clean no-op', (tester) async {
      final calls = _Calls();
      final client = _buildClient(calls, detectionCount: 0);
      await _pumpHomeScreen(tester, client);

      await _tapDetect(tester, client);

      expect(calls.paths, contains('/segment/balloons'));
      expect(find.text('No balloons detected'), findsOneWidget);
      // A no-op, not a failure.
      expect(find.textContaining('failed'), findsNothing);
      // And emphatically not a scrub.
      expect(calls.paths, isNot(contains('/lama/scrub')));
    });

    testWidgets('detections land as held, unassigned objects through the '
        'existing selection path', (tester) async {
      final calls = _Calls();
      final client = _buildClient(calls, detectionCount: 2);
      await _pumpHomeScreen(tester, client);

      await _tapDetect(tester, client);

      expect(find.text('2 balloons detected'), findsOneWidget);

      // The existing per-object checkbox list is what renders held objects;
      // both arrive checked (held) and uncaptioned, which is the
      // implicit-discard state a word balloon belongs in.
      final checkboxes = find.byType(CheckboxListTile);
      expect(checkboxes, findsWidgets);
      final held = tester
          .widgetList<CheckboxListTile>(checkboxes)
          .where((c) => c.value == true)
          .length;
      expect(held, greaterThanOrEqualTo(2),
          reason: 'both detections should arrive held for review');
    });

    testWidgets('detecting never triggers a scrub', (tester) async {
      final calls = _Calls();
      final client = _buildClient(calls, detectionCount: 2);
      await _pumpHomeScreen(tester, client);

      await _tapDetect(tester, client);

      // The card detects and holds only; scrubbing stays with LBSCard.
      expect(calls.paths, isNot(contains('/lama/scrub')));
    });

    testWidgets('no session other than the current one is touched',
        (tester) async {
      final calls = _Calls();
      final client = _buildClient(calls, detectionCount: 2);
      await _pumpHomeScreen(tester, client);

      await _tapDetect(tester, client);

      expect(calls.sessionIds, isNotEmpty);
      expect(
        calls.sessionIds.where((s) => s != _sid),
        isEmpty,
        reason: 'every request must carry only the open session id, '
            'but saw ${calls.sessionIds}',
      );
      expect(
        calls.paths.where((p) => p.contains(_otherSid)),
        isEmpty,
      );
    });
  });
}
