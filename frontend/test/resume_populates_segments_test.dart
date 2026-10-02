// Bug 2 regression test: `_loadLaunchSession` set `_imageBytes`/`_uiImage`/
// `_aaPreviewData` on resume but never `_result`/`_segments` — the exact
// fields ObjectsSelectedCard's checkboxes and LBSCard's `hasSelection` key
// off. A resumed session's Scrub button was permanently disabled no matter
// how many masks the session actually had, because nothing had ever wired
// the HTTP response's `results` field into the widget tree at all.
//
// Same `runWithClient`/MockClient pattern as resume_wires_aa_preview_test.dart
// (no real socket I/O works here — see that file's own header for why).
//
// Also covers the explicitly-out-of-scope case this fix must NOT touch: a
// session whose masks sit at an earlier pass than its current one (bug 3)
// must still resume to a cleanly DISABLED Scrub button and zero checkboxes —
// not a crash, and not an accidental, coincidental success.
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';
import 'package:frontend/widgets/objects_selected_card.dart';

Widget _wrap(Widget child) => ChangeNotifierProvider<LayerState>(
      create: (_) => LayerState(),
      child: MaterialApp(home: child),
    );

const _sessionId = 'resume-segments-test-session';

/// A fully-foreground 10x10 mask — `counts: [0, 100]` is 0 background then
/// 100 foreground, the convention `mask_to_rle` produces for a mask whose
/// first pixel is already foreground (see that function's own comment).
const _rle = {'counts': [0, 100], 'size': [10, 10]};

Map<String, dynamic> _emptyAllMasks() => {
      'mask_ids': <dynamic>[], 'dataset_statuses': <dynamic>[],
      'held_flags': <dynamic>[], 'captions': <dynamic>[],
      'text_tags': <dynamic>[], 'scores': <dynamic>[], 'boxes': <dynamic>[],
      'crop_png_bytes': <dynamic>[], 'passes': <dynamic>[],
    };

MockClient _clientWithResults(Map<String, dynamic> results, Map<String, dynamic> allMasks) {
  return MockClient((request) async {
    if (request.method == 'GET' &&
        request.url.path == '/loadSession/$_sessionId') {
      return http.Response(
        jsonEncode({
          'session_id': _sessionId,
          'image_b64': null,
          'width': 10,
          'height': 10,
          'results': results,
          'all_masks': allMasks,
          'prompts': <dynamic>[],
          'created_at': '2026-01-01T00:00:00',
          'name': 'resume segments test',
          'description': '',
          'image_url': null,
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    }
    return http.Response('not found', 404);
  });
}

Future<void> _pumpResumedSession(WidgetTester tester, MockClient client) async {
  await tester.binding.setSurfaceSize(const Size(1600, 1000));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  await http.runWithClient(() async {
    await tester.pumpWidget(_wrap(const HomeScreen(initialSessionId: _sessionId)));
    await tester.pump();
    await tester.pump();
  }, () => client);
}

Finder get _scrubButton => find.widgetWithText(ElevatedButton, 'Scrub Selected Regions');

/// Scoped to ObjectsSelectedCard specifically: SegmentLayersCard's own
/// Display Layers checkboxes (Original/Masks/Raw/Current) use the same
/// widget type and are always present regardless of session state, so an
/// unscoped find.byType(CheckboxListTile) would always find 4 even with
/// zero segments.
Finder get _segmentCheckboxes => find.descendant(
      of: find.byType(ObjectsSelectedCard),
      matching: find.byType(CheckboxListTile),
    );

void main() {
  testWidgets(
      'a resumed session with a current-pass mask populates _segments and '
      'enables Scrub', (tester) async {
    final results = {
      'mask_ids': ['1:a-real-mask'],
      'dataset_statuses': ['unassigned'],
      'held_flags': [true],
      'masks': [_rle],
      'boxes': [
        [1.0, 1.0, 9.0, 9.0]
      ],
      'scores': [0.9],
    };
    final allMasks = Map<String, dynamic>.from(_emptyAllMasks())
      ..['mask_ids'] = ['1:a-real-mask']
      ..['dataset_statuses'] = ['unassigned']
      ..['held_flags'] = [true]
      ..['scores'] = [0.9]
      ..['boxes'] = [
        [1.0, 1.0, 9.0, 9.0]
      ]
      ..['crop_png_bytes'] = [null]
      ..['passes'] = [1];

    await _pumpResumedSession(tester, _clientWithResults(results, allMasks));

    expect(tester.takeException(), isNull);
    // ObjectsSelectedCard renders one checkbox per segment carrying a
    // maskId — confirms `_updateSegmentsFromResult` actually ran and built
    // a real Segment, not just that `_result` was assigned.
    expect(_segmentCheckboxes, findsWidgets);
    // LBSCard's `hasSelection` (`_segments.any((s) => s.maskId != null)`)
    // must be true, so the button is genuinely clickable, not merely
    // visible.
    expect(_scrubButton, findsOneWidget);
    expect(tester.widget<ElevatedButton>(_scrubButton).onPressed, isNotNull);
  });

  testWidgets(
      'a resumed session with no current-pass masks (bug 3 shape) resumes '
      'cleanly with Scrub disabled — not a crash, not a false success',
      (tester) async {
    // The real shape a WB-scrubbed-then-resumed session has today: masks
    // exist (all_masks is non-empty, same as the AA Preview tab still
    // showing them) but none sit at the session's current pass, so
    // `results` -- what the canvas selects from -- is empty.
    final results = {
      'mask_ids': <dynamic>[], 'dataset_statuses': <dynamic>[],
      'held_flags': <dynamic>[], 'masks': <dynamic>[], 'boxes': <dynamic>[],
      'scores': <dynamic>[],
    };
    final allMasks = Map<String, dynamic>.from(_emptyAllMasks())
      ..['mask_ids'] = ['0:an-earlier-pass-mask']
      ..['dataset_statuses'] = ['unassigned']
      ..['held_flags'] = [false]
      ..['scores'] = [0.9]
      ..['boxes'] = [
        [1.0, 1.0, 9.0, 9.0]
      ]
      ..['crop_png_bytes'] = [null]
      ..['passes'] = [0];

    await _pumpResumedSession(tester, _clientWithResults(results, allMasks));

    expect(tester.takeException(), isNull, reason: 'must not crash');
    expect(_segmentCheckboxes, findsNothing,
        reason: 'no current-pass mask to select from');
    expect(_scrubButton, findsOneWidget);
    expect(tester.widget<ElevatedButton>(_scrubButton).onPressed, isNull,
        reason: 'cleanly disabled, not a silent no-op success');
  });
}
