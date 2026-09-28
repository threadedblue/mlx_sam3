// Regression coverage for the "Unsaved Changes" dialog's dirty-tracking.
//
// Was `_hasUnsavedWork => _result != null`, which is true the instant
// anything is selected and NEVER goes back to false — including right
// after a successful Save, since Save never touched `_result`. Confirmed
// live: the dialog fired on every session switch after the first
// selection, saved or not.
//
// Fixed with a real mutation counter (`_mutationCount`) bumped by every
// handler that changes state `/saveSession` would persist, compared
// against the counter's value as of the last successful save
// (`_savedAtMutationCount`). This file drives the real `HomeScreen` and
// checks the dialog's actual presence/absence — the one thing a unit test
// on a private field can't do — via `runWithClient`/`MockClient`, the
// pattern established in resume_wires_aa_preview_test.dart (no real
// socket I/O works here; see that file's header comment for why).
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';

const _sid = 'dirty-tracking-test-session';
const _maskId = '0:probe-mask';

Widget _wrap(Widget child) => ChangeNotifierProvider<LayerState>(
      create: (_) => LayerState(),
      child: MaterialApp(home: child),
    );

/// A fully-filled 10x10 RLE mask -- shape doesn't matter here, only that
/// _updateSegmentsFromResult can build one real Segment with a maskId from
/// it. Convention (services.py/aa_persistence.py): first run is the
/// background (0) count; [0, 100] on a 10x10 canvas is "all foreground".
const _rle = {
  'counts': [0, 100],
  'size': [10, 10],
};

Map<String, dynamic> _allMasks({required bool held, required String status}) => {
      'mask_ids': [_maskId],
      'dataset_statuses': [status],
      'held_flags': [held],
      'captions': [status == 'keep' ? 'a probe object' : null],
      'text_tags': [null],
      'scores': [0.9],
      'boxes': [
        [1.0, 1.0, 9.0, 9.0]
      ],
      'crop_png_bytes': [null],
      'passes': [0],
    };

/// Builds a MockClient for one HomeScreen test run.
///
/// - `/loadSession/$_sid`: always answers with the empty session below
///   (image_b64 null -- no canvas rendering needed for this test).
/// - `/segment/text`: always succeeds, returns one auto-held, unassigned
///   mask (matches the real backend's auto-hold-on-selection behaviour --
///   sf-display-and-workflow-v3-spec.md §5).
/// - `/mask/hold`: echoes back whatever `held` was posted (real backend
///   behaviour), so a second toggle in the same test sees the updated
///   state.
/// - `/mask/caption`: always succeeds, marks the mask `keep`.
/// - `/saveSession`: 200 if `saveSucceeds`, 500 otherwise.
/// - anything else (health, upload): 404, keeping `_maybeAutoLoadImage`
///   inert so no `/upload` call is ever needed.
MockClient _buildClient({bool saveSucceeds = true}) {
  var currentlyHeld = true; // matches the auto-hold this mock's /segment/text reports
  return MockClient((request) async {
    final p = request.url.path;
    Map<String, dynamic>? body;
    if (request.body.isNotEmpty) {
      body = jsonDecode(request.body) as Map<String, dynamic>;
    }

    if (p == '/loadSession/$_sid') {
      return http.Response(
        jsonEncode({
          'session_id': _sid,
          'image_b64': null,
          'width': 100,
          'height': 100,
          'results': {'masks': <dynamic>[], 'boxes': <dynamic>[], 'scores': <dynamic>[]},
          'all_masks': {
            'mask_ids': <dynamic>[],
            'dataset_statuses': <dynamic>[],
            'held_flags': <dynamic>[],
            'captions': <dynamic>[],
            'text_tags': <dynamic>[],
            'scores': <dynamic>[],
            'boxes': <dynamic>[],
            'crop_png_bytes': <dynamic>[],
            'passes': <dynamic>[],
          },
          'prompts': <dynamic>[],
          'created_at': '2026-01-01T00:00:00',
          'name': 'dirty tracking probe',
          'description': '',
          'image_url': null,
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    }

    if (p == '/segment/text') {
      currentlyHeld = true;
      return http.Response(
        jsonEncode({
          'session_id': _sid,
          'results': {
            'original_width': 100,
            'original_height': 100,
            'masks': [_rle],
            'boxes': [
              [1.0, 1.0, 9.0, 9.0]
            ],
            'scores': [0.9],
            'mask_ids': [_maskId],
            'dataset_statuses': ['unassigned'],
            'held_flags': [true],
            'captions': [null],
            'text_tags': [null],
            'crop_png_bytes': [null],
            'passes': [0],
          },
          'all_masks': _allMasks(held: true, status: 'unassigned'),
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    }

    if (p == '/mask/hold') {
      currentlyHeld = body!['held'] as bool;
      return http.Response(
        jsonEncode({
          'session_id': _sid,
          'mask_id': _maskId,
          'held': currentlyHeld,
          'all_masks': _allMasks(held: currentlyHeld, status: 'unassigned'),
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    }

    if (p == '/mask/caption') {
      return http.Response(
        jsonEncode({
          'session_id': _sid,
          'mask_id': _maskId,
          'dataset_status': 'keep',
          'caption': body!['caption'],
          'all_masks': _allMasks(held: currentlyHeld, status: 'keep'),
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    }

    if (p == '/saveSession') {
      if (saveSucceeds) {
        return http.Response(
          jsonEncode({'message': 'Session saved', 'session_id': _sid}),
          200,
          headers: {'content-type': 'application/json'},
        );
      }
      return http.Response(jsonEncode({'detail': 'simulated save failure'}), 500);
    }

    return http.Response('not found', 404);
  });
}

Future<void> _pumpFreshHomeScreen(WidgetTester tester, MockClient client) async {
  await tester.binding.setSurfaceSize(const Size(1600, 1000));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  await http.runWithClient(() async {
    await tester.pumpWidget(_wrap(const HomeScreen(initialSessionId: _sid)));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

/// Types [text] into the Prompt card and taps its Select/Save Caption
/// button (same OutlinedButton, its label and target endpoint both follow
/// `_isCaptionMode` -- see main.dart's `_buildTextPromptCard`).
Future<void> _submitPrompt(WidgetTester tester, MockClient client, String text) async {
  await http.runWithClient(() async {
    await tester.enterText(find.byType(TextField).first, text);
    await tester.pump();
    final btn = find.byWidgetPredicate(
        (w) => w is OutlinedButton && w.child is Text && ((w.child as Text).data == 'Select' || (w.child as Text).data == 'Save Caption'));
    await tester.ensureVisible(btn);
    await tester.pump();
    await tester.tap(btn, warnIfMissed: false);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

Future<void> _toggleHoldCheckbox(WidgetTester tester, MockClient client) async {
  await http.runWithClient(() async {
    // find.byType(CheckboxListTile) alone matches too many: the Display
    // Layers card (segment_layers_card.dart) uses the same widget type for
    // its Original/Masks/Raw/Current toggles, always present regardless of
    // session state. Only one segment is selected in this test, so its
    // row's title is exactly "Include in next scrub" (see
    // ObjectsSelectedCard's singular-vs-plural label).
    final box = find.widgetWithText(CheckboxListTile, 'Include in next scrub');
    await tester.ensureVisible(box);
    await tester.pump();
    await tester.tap(box, warnIfMissed: false);
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

/// Taps "Switch Session" and reports whether the "Unsaved Changes" dialog
/// appeared. If it did, dismisses via "Keep Working" so the widget tree
/// survives for further steps in the same test (tapping "Discard &
/// Switch" -- or not showing at all -- both navigate away immediately).
Future<bool> _tapSwitchSessionAndCheckDialog(WidgetTester tester, MockClient client) async {
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

void main() {
  testWidgets('a freshly resumed session with nothing selected is clean',
      (tester) async {
    final client = _buildClient();
    await _pumpFreshHomeScreen(tester, client);

    final dialogShown = await _tapSwitchSessionAndCheckDialog(tester, client);
    expect(dialogShown, isFalse,
        reason: 'nothing was ever selected; the resumed session has no unsaved work');
  });

  testWidgets('selecting an object dirties the session', (tester) async {
    final client = _buildClient();
    await _pumpFreshHomeScreen(tester, client);

    await _submitPrompt(tester, client, 'boy');

    final dialogShown = await _tapSwitchSessionAndCheckDialog(tester, client);
    expect(dialogShown, isTrue,
        reason: 'a /segment/text selection is a mutation the backend would persist');
  });

  testWidgets('a successful Save clears the dirty flag', (tester) async {
    final client = _buildClient(saveSucceeds: true);
    await _pumpFreshHomeScreen(tester, client);

    await _submitPrompt(tester, client, 'boy');
    await _tapSave(tester, client);

    final dialogShown = await _tapSwitchSessionAndCheckDialog(tester, client);
    expect(dialogShown, isFalse,
        reason: 'Save succeeded after the only mutation so far; nothing is unsaved');
  });

  testWidgets('a mutation after a successful Save dirties the session again',
      (tester) async {
    final client = _buildClient(saveSucceeds: true);
    await _pumpFreshHomeScreen(tester, client);

    await _submitPrompt(tester, client, 'boy'); // dirty #1
    await _tapSave(tester, client); // clean

    // Toggle held twice: the mock's mask starts held=true (auto-hold), so
    // the first tap unchecks it (held=false) AND sets focus (_setMaskHeld
    // always does, unlike a bare text-select) -- the second tap re-checks
    // it (held=true), which is what _isCaptionMode requires alongside
    // focus + dataset_status=='unassigned' to route the Prompt card's
    // button into /mask/caption instead of another /segment/text call.
    await _toggleHoldCheckbox(tester, client); // dirty #2 (hold/include toggle)
    await _toggleHoldCheckbox(tester, client); // dirty #3 (hold/include toggle)

    expect(find.widgetWithText(OutlinedButton, 'Save Caption'), findsOneWidget,
        reason: 'focused + held + unassigned must have entered caption mode');

    await _submitPrompt(tester, client, 'a probe object'); // dirty #4 (caption commit)

    final dialogShown = await _tapSwitchSessionAndCheckDialog(tester, client);
    expect(dialogShown, isTrue,
        reason: 'captioning after the earlier Save is new, unsaved work');
  });

  testWidgets('a failed Save leaves the session dirty', (tester) async {
    final client = _buildClient(saveSucceeds: false);
    await _pumpFreshHomeScreen(tester, client);

    await _submitPrompt(tester, client, 'boy');
    await _tapSave(tester, client); // 500 -> ApiService.saveSession throws

    final dialogShown = await _tapSwitchSessionAndCheckDialog(tester, client);
    expect(dialogShown, isTrue,
        reason: 'the save never succeeded; _savedAtMutationCount must not have moved');
  });
}
