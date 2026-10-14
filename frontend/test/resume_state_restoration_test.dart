// Bug-3-audit fixes #1/#2/#3 regression tests: three pieces of state
// /loadSession's response carried (or now carries) but the frontend never
// restored on resume, leaving a resumed, previously-scrubbed session
// looking identical to one that had never been touched — "No scrub yet",
// "No detection run yet", and layer checkboxes silently back at their
// LayerState defaults regardless of what was actually saved.
//
// Same MockClient/runWithClient pattern as resume_populates_segments_test
// and resume_wires_aa_preview_test (no real socket I/O works here — see
// resume_wires_aa_preview_test.dart's own header for why).
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';

const _sessionId = 'resume-state-restoration-session';

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

/// Builds a /loadSession response. [extra] layers additional/overriding
/// top-level keys on top of the ordinary shape (e.g. `pass`,
/// `wb_scrub_pass`, `view_layers`) — same convention as the other resume
/// tests' `_clientWithResults`, generalised to cover these three fields.
Map<String, dynamic> _sessionPayload({Map<String, dynamic>? extra}) => {
      'session_id': _sessionId,
      'image_b64': null,
      'width': 10,
      'height': 10,
      'results': _emptyResults(),
      'all_masks': _emptyAllMasks(),
      'prompts': <dynamic>[],
      'created_at': '2026-01-01T00:00:00',
      'name': 'resume state restoration test',
      'description': '',
      'image_url': null,
      ...?extra,
    };

/// Tracks whether POST /session/settings ever fired — the proof that
/// restoring view_layers on load must not itself trigger a save.
class _SettingsCallTracker {
  int calls = 0;
}

MockClient _client(Map<String, dynamic>? sessionPayload, _SettingsCallTracker tracker) {
  return MockClient((request) async {
    if (request.method == 'GET' &&
        request.url.path == '/loadSession/$_sessionId') {
      if (sessionPayload == null) {
        return http.Response('not found', 404);
      }
      return http.Response(
        jsonEncode(sessionPayload),
        200,
        headers: {'content-type': 'application/json'},
      );
    }
    if (request.method == 'POST' && request.url.path == '/session/settings') {
      tracker.calls += 1;
      return http.Response(jsonEncode({'ok': true}), 200,
          headers: {'content-type': 'application/json'});
    }
    return http.Response('not found', 404);
  });
}

Widget _wrap(Widget child, LayerState layerState) =>
    ChangeNotifierProvider<LayerState>.value(
      value: layerState,
      child: MaterialApp(home: child),
    );

/// Loads [sessionId] and hands back the live [LayerState] plus a tracker of
/// how many POST /session/settings calls fired — both still live inside the
/// MockClient zone when [afterLoad] runs, so a test that needs to make its
/// OWN HTTP-triggering call after the initial load (e.g. a post-resume
/// checkbox toggle) can do so there: outside runWithClient's zone,
/// flutter_test routes any http call to a real HttpClient that always
/// 400s, which would silently hide the very call such a test is trying to
/// observe.
class _ResumedSession {
  final LayerState layerState;
  final _SettingsCallTracker settingsCalls;
  _ResumedSession(this.layerState, this.settingsCalls);
}

Future<_ResumedSession> _pumpResumedSession(
  WidgetTester tester, {
  Map<String, dynamic>? sessionPayload,
  String? initialSessionId = _sessionId,
  Future<void> Function(LayerState layerState, _SettingsCallTracker tracker)?
      afterLoad,
}) async {
  await tester.binding.setSurfaceSize(const Size(1600, 1000));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  final layerState = LayerState();
  final tracker = _SettingsCallTracker();
  await http.runWithClient(() async {
    await tester.pumpWidget(_wrap(
      HomeScreen(initialSessionId: initialSessionId),
      layerState,
    ));
    await tester.pump();
    await tester.pump();
    if (afterLoad != null) {
      await afterLoad(layerState, tracker);
      await tester.pump();
      await tester.pump();
    }
  }, () => _client(sessionPayload, tracker));
  return _ResumedSession(layerState, tracker);
}

void main() {
  group('Fix 1: scrub label restored on resume', () {
    testWidgets('a scrubbed session resumes showing "Last scrub: Pass 0 → Pass 1"',
        (tester) async {
      await _pumpResumedSession(tester,
          sessionPayload: _sessionPayload(extra: {'pass': 1}));

      expect(tester.takeException(), isNull);
      expect(find.text('Last scrub: Pass 0 → Pass 1'), findsOneWidget);
      expect(find.text('No scrub yet'), findsNothing);
    });

    testWidgets('a session resumed at pass 0 (never scrubbed) still shows "No scrub yet"',
        (tester) async {
      await _pumpResumedSession(tester,
          sessionPayload: _sessionPayload(extra: {'pass': 0}));

      expect(tester.takeException(), isNull);
      expect(find.text('No scrub yet'), findsOneWidget);
    });

    testWidgets('a session scrubbed twice resumes showing only the latest transition',
        (tester) async {
      await _pumpResumedSession(tester,
          sessionPayload: _sessionPayload(extra: {'pass': 2}));

      expect(find.text('Last scrub: Pass 1 → Pass 2'), findsOneWidget);
    });

    testWidgets('a fresh session (no launch session id at all) shows "No scrub yet"',
        (tester) async {
      await _pumpResumedSession(tester, initialSessionId: null);

      expect(tester.takeException(), isNull);
      expect(find.text('No scrub yet'), findsOneWidget);
    });
  });

  group('Fix 2: Balloon Scrub status text on resume', () {
    testWidgets('resumed, wb_scrub_pass set -> "Balloons scrubbed at pass N"',
        (tester) async {
      await _pumpResumedSession(tester,
          sessionPayload: _sessionPayload(extra: {'pass': 3, 'wb_scrub_pass': 2}));

      expect(tester.takeException(), isNull);
      expect(find.text('Balloons scrubbed at pass 2'), findsOneWidget);
      expect(find.text('No detection run yet'), findsNothing);
    });

    testWidgets('resumed, wb_scrub_pass null -> "Not run in this session"',
        (tester) async {
      await _pumpResumedSession(tester,
          sessionPayload: _sessionPayload(extra: {'pass': 1}));

      expect(tester.takeException(), isNull);
      expect(find.text('Not run in this session'), findsOneWidget);
      // The resumed-but-unknown state must never read as "never looked" --
      // that's specifically the ambiguity this fix exists to avoid.
      expect(find.text('No detection run yet'), findsNothing);
    });

    testWidgets('a fresh session (never resumed) still shows "No detection run yet"',
        (tester) async {
      await _pumpResumedSession(tester, initialSessionId: null);

      expect(tester.takeException(), isNull);
      expect(find.text('No detection run yet'), findsOneWidget);
      expect(find.text('Not run in this session'), findsNothing);
    });
  });

  group('Fix 3: view_layers restored on resume', () {
    testWidgets(
        'a legacy raw key is ignored; other keys round-trip and missing ones keep defaults',
        (tester) async {
      final resumed = await _pumpResumedSession(tester,
          sessionPayload: _sessionPayload(extra: {
            'pass': 0,
            'view_layers': {
              'original': true,
              'masks': false,
              'raw': false,
              // 'current' deliberately omitted -- must keep its default.
              // Legacy key from before the Final -> Current rename --
              // must be ignored, not crash and not be misread as "current".
              'final': true,
            },
          }));

      expect(tester.takeException(), isNull);
      final layerState = resumed.layerState;
      expect(layerState.showOriginal, isTrue, reason: 'explicit true round-trips');
      expect(layerState.showMasks, isFalse, reason: 'explicit false round-trips');
      // 'raw' (the real stored shape these sessions carry) is read and
      // dropped: the Raw layer no longer exists, and must not crash a resume.
      expect(layerState.showCurrent, isTrue,
          reason: 'missing key must keep LayerState\'s own default');
    });

    testWidgets('a session with no view_layers at all keeps every LayerState default',
        (tester) async {
      final resumed = await _pumpResumedSession(tester,
          sessionPayload: _sessionPayload(extra: {'pass': 0}));

      expect(tester.takeException(), isNull);
      final layerState = resumed.layerState;
      expect(layerState.showOriginal, isTrue);
      expect(layerState.showMasks, isTrue);
      expect(layerState.showCurrent, isTrue);
    });

    testWidgets('applying view_layers on load never calls POST /session/settings',
        (tester) async {
      final resumed = await _pumpResumedSession(tester,
          sessionPayload: _sessionPayload(extra: {
            'pass': 0,
            'view_layers': {'original': false, 'masks': false, 'raw': false, 'current': false},
          }));

      expect(tester.takeException(), isNull);
      expect(resumed.settingsCalls.calls, 0,
          reason: 'loading a session must not write its own loaded state back to disk');
    });

    testWidgets('toggling a layer checkbox AFTER resume still saves normally',
        (tester) async {
      final resumed = await _pumpResumedSession(
        tester,
        sessionPayload: _sessionPayload(extra: {'pass': 0}),
        afterLoad: (layerState, tracker) async {
          expect(tracker.calls, 0,
              reason: 'the restoration itself must not have saved anything');
          layerState.setMasks(false);
        },
      );

      expect(resumed.settingsCalls.calls, 1,
          reason: 'the restoration-time listener detach must not leak past load');
    });
  });
}
