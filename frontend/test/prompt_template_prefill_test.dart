// Prompt templating v1, frontend half: entering caption mode pre-fills the
// EXISTING caption box from the session's template.
//
// BASELINE THIS MUST PRESERVE: typed text is NOT cleared when focus moves.
// `_textController` is only cleared by a successful Save Caption and by Clear
// Prompts. So "no template means don't touch the controller" genuinely means
// the user's text survives, exactly as it did before this feature existed.
//
// No new card, no new layer, no new persisted field: the draft is computed on
// demand and becomes permanent only via the existing Save Caption path.
import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';

const _sid = 'prefill-test-session';
const _maskId = '0:probe-mask';

const _rle = {'counts': [0, 100], 'size': [10, 10]};

Widget _wrap(Widget child) => ChangeNotifierProvider<LayerState>(
      create: (_) => LayerState(),
      child: MaterialApp(home: child),
    );

Map<String, dynamic> _allMasks({required bool held}) => {
      'mask_ids': [_maskId],
      'dataset_statuses': const ['unassigned'],
      'held_flags': [held],
      'captions': const [null],
      'text_tags': const ['nemo'],
      'scores': const [0.9],
      'boxes': const [[1.0, 1.0, 9.0, 9.0]],
      'crop_png_bytes': const [null],
      'passes': const [0],
    };

Map<String, dynamic> _results({required bool held}) => {
      'mask_ids': [_maskId],
      'dataset_statuses': const ['unassigned'],
      'held_flags': [held],
      'masks': const [_rle],
      'boxes': const [[1.0, 1.0, 9.0, 9.0]],
      'scores': const [0.9],
      'captions': const [null],
      'text_tags': const ['nemo'],
      'passes': const [0],
    };

class _Calls {
  final List<String> paths = [];
  final List<Map<String, dynamic>> captionBodies = [];
}

MockClient _client(
  _Calls calls, {
  required String draft,
  Completer<void>? gate,
  bool draftFails = false,
}) {
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
          'session_id': _sid, 'image_b64': null, 'width': 10, 'height': 10,
          'results': _results(held: false),
          'all_masks': _allMasks(held: false),
          'prompts': <dynamic>[], 'created_at': '2026-01-01T00:00:00',
          'name': 'prefill test', 'description': '', 'image_url': null,
          'pass': 0, 'panels': <dynamic>[],
        }),
        200, headers: {'content-type': 'application/json'},
      );
    }
    if (p == '/mask/hold') {
      return http.Response(
        jsonEncode({
          'session_id': _sid, 'mask_id': _maskId,
          'held': body!['held'],
          'all_masks': _allMasks(held: body['held'] as bool),
        }),
        200, headers: {'content-type': 'application/json'},
      );
    }
    if (p == '/mask/prompt-draft') {
      if (gate != null) await gate.future;
      if (draftFails) return http.Response('boom', 500);
      return http.Response(jsonEncode({'draft': draft}), 200,
          headers: {'content-type': 'application/json'});
    }
    if (p == '/mask/caption') {
      calls.captionBodies.add(body!);
      return http.Response(
        jsonEncode({
          'session_id': _sid, 'mask_id': _maskId, 'dataset_status': 'keep',
          'caption': body['caption'],
          'all_masks': _allMasks(held: true),
        }),
        200, headers: {'content-type': 'application/json'},
      );
    }
    if (p == '/saveSession') {
      return http.Response(
          jsonEncode({'message': 'Session saved', 'session_id': _sid}), 200,
          headers: {'content-type': 'application/json'});
    }
    return http.Response('not found', 404);
  });
}

Future<void> _pumpHome(WidgetTester tester, MockClient client) async {
  await tester.binding.setSurfaceSize(const Size(1600, 1200));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  await http.runWithClient(() async {
    await tester.pumpWidget(_wrap(const HomeScreen(initialSessionId: _sid)));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

/// The caption box is the Prompt card's TextField — the first one on screen.
String get _promptText {
  final f = find.byType(TextField).evaluate().first.widget as TextField;
  return f.controller!.text;
}

Future<void> _setPromptText(WidgetTester tester, String text) async {
  await tester.enterText(find.byType(TextField).first, text);
  await tester.pump();
}

/// Checks the hold box, which both holds the mask and focuses it — the
/// documented transition into caption mode (held && unassigned).
Future<void> _enterCaptionMode(WidgetTester tester, MockClient client) async {
  await http.runWithClient(() async {
    final box = find.widgetWithText(CheckboxListTile, 'Include in next scrub');
    await tester.ensureVisible(box);
    await tester.pump();
    await tester.tap(box, warnIfMissed: false);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
  }, () => client);
}

Future<bool> _switchSessionShowsDialog(
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

void main() {
  testWidgets('entering caption mode pre-fills the caption box from the template',
      (tester) async {
    final calls = _Calls();
    final client = _client(calls, draft: 'describe nemo at 1,1,9,9');
    await _pumpHome(tester, client);
    expect(_promptText, isEmpty);

    await _enterCaptionMode(tester, client);

    expect(calls.paths, contains('/mask/prompt-draft'));
    expect(_promptText, 'describe nemo at 1,1,9,9');
  });

  testWidgets('no template: the caption box is not touched at all',
      (tester) async {
    final calls = _Calls();
    final client = _client(calls, draft: '');
    await _pumpHome(tester, client);

    await _enterCaptionMode(tester, client);

    expect(calls.paths, contains('/mask/prompt-draft'));
    expect(_promptText, isEmpty, reason: "today's behaviour, unchanged");
  });

  testWidgets('no template: text the user already typed survives focus change',
      (tester) async {
    final calls = _Calls();
    final client = _client(calls, draft: '');
    await _pumpHome(tester, client);
    await _setPromptText(tester, 'typed by hand');

    await _enterCaptionMode(tester, client);

    expect(_promptText, 'typed by hand');
  });

  testWidgets('text typed while the draft is in flight wins over the draft',
      (tester) async {
    final calls = _Calls();
    final gate = Completer<void>();
    final client = _client(calls, draft: 'the draft', gate: gate);
    await _pumpHome(tester, client);

    await http.runWithClient(() async {
      final box = find.widgetWithText(CheckboxListTile, 'Include in next scrub');
      await tester.ensureVisible(box);
      await tester.pump();
      await tester.tap(box, warnIfMissed: false);
      await tester.pump();
      await tester.enterText(find.byType(TextField).first, 'user wins');
      await tester.pump();
      gate.complete();
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 500));
    }, () => client);

    expect(_promptText, 'user wins',
        reason: 'a draft must never overwrite text typed since the request');
  });

  testWidgets('a failed draft call leaves the box untouched', (tester) async {
    final calls = _Calls();
    final client = _client(calls, draft: 'unused', draftFails: true);
    await _pumpHome(tester, client);
    await _setPromptText(tester, 'still mine');

    await _enterCaptionMode(tester, client);

    expect(_promptText, 'still mine');
  });

  testWidgets('pre-filling does not bump _mutationCount', (tester) async {
    final calls = _Calls();
    final client = _client(calls, draft: 'a draft');
    await _pumpHome(tester, client);

    await _enterCaptionMode(tester, client);
    expect(_promptText, 'a draft');

    // Clear the dirt the HOLD itself legitimately created, so what remains
    // measures the pre-fill alone.
    await http.runWithClient(() async {
      final btn = find.widgetWithText(ElevatedButton, 'Save');
      await tester.ensureVisible(btn);
      await tester.pump();
      await tester.tap(btn, warnIfMissed: false);
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 500));
    }, () => client);

    expect(await _switchSessionShowsDialog(tester, client), isFalse,
        reason: 'the pre-filled text alone must not count as unsaved work');
  });

  testWidgets('Save Caption sends the EDITED draft, not the original',
      (tester) async {
    final calls = _Calls();
    final client = _client(calls, draft: 'draft text');
    await _pumpHome(tester, client);
    await _enterCaptionMode(tester, client);
    expect(_promptText, 'draft text');

    await _setPromptText(tester, 'draft text, edited by the user');
    await http.runWithClient(() async {
      final btn = find.byWidgetPredicate((w) =>
          w is OutlinedButton &&
          w.child is Text &&
          (w.child as Text).data == 'Save Caption');
      await tester.ensureVisible(btn);
      await tester.pump();
      await tester.tap(btn, warnIfMissed: false);
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 500));
    }, () => client);

    expect(calls.captionBodies.single['caption'],
        'draft text, edited by the user');
  });
}
