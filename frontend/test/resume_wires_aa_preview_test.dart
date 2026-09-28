// Fix 1 regression test: `/loadSession`'s `all_masks` field being present
// in the HTTP response was never the gap — `_loadLaunchSession` fetched it
// and simply never assigned `_aaPreviewData` from it. A test on the HTTP
// response alone (or on the adapter in isolation) cannot see that gap: it
// has to drive the real `HomeScreen.initState` -> `_loadLaunchSession` ->
// `setState` path and check what actually reaches the widget tree.
//
// Uses `package:http`'s own `runWithClient` to zone-scope a `MockClient`
// for `ApiService`'s top-level `http.get`/`http.post` calls — no change to
// ApiService itself, and no real socket I/O (which cannot be made to work
// here: a real HttpClient call fired unawaited from initState, as
// _loadLaunchSession does, starts outside `tester.runAsync`'s real-event-
// loop window and never gets a chance to complete afterward either,
// confirmed via a minimal repro during this investigation — only a
// same-isolate, no-socket fake response resolves under flutter_test's
// fake-async zone without that problem).
//
// This is also the session's final end-to-end proof: a resumed session
// with real masks (8, across 3 passes, real captured crop_png_bytes) must
// render images in the AA Preview tab immediately on resume, no exception,
// no empty state.
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:provider/provider.dart';

import 'package:frontend/main.dart';
import 'package:frontend/layer_state.dart';

Widget _wrap(Widget child) => ChangeNotifierProvider<LayerState>(
      create: (_) => LayerState(),
      child: MaterialApp(home: child),
    );

const _sessionId = 'resume-test-session';

void main() {
  testWidgets(
      'resuming a session populates _aaPreviewData and the AA Preview tab '
      'renders real images — not just present in the HTTP response',
      (tester) async {
    // test/fixtures/real_all_masks.json is genuine all_masks captured live
    // from the backend (8 masks, 3 passes, real crop_png_bytes) — captured
    // before Fix 2, so `passes` is still the numeric-string form on disk.
    // Coerced to int here to match what /loadSession actually sends now
    // that Fix 2 is in place.
    final rawAllMasks = jsonDecode(
      File('test/fixtures/real_all_masks.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    final allMasks = Map<String, dynamic>.from(rawAllMasks)
      ..['passes'] = (rawAllMasks['passes'] as List)
          .map((p) => p is int ? p : int.parse(p.toString()))
          .toList();

    final mockClient = MockClient((request) async {
      if (request.method == 'GET' &&
          request.url.path == '/loadSession/$_sessionId') {
        return http.Response(
          jsonEncode({
            'session_id': _sessionId,
            'image_b64': null,
            'width': 400,
            'height': 300,
            'results': {'masks': <dynamic>[], 'boxes': <dynamic>[], 'scores': <dynamic>[]},
            'all_masks': allMasks,
            'prompts': <dynamic>[],
            'created_at': '2026-01-01T00:00:00',
            'name': 'resume test session',
            'description': 'fixture-backed resume',
            'image_url': null,
          }),
          200,
          headers: {'content-type': 'application/json'},
        );
      }
      // /health and anything else this HomeScreen instance calls (the
      // health-check timer, etc.) — 404 keeps _backendStatus off "online",
      // so _maybeAutoLoadImage never fires and no /upload call is needed.
      return http.Response('not found', 404);
    });

    // HomeScreen's full layout (side cards + canvas + tabs) needs more
    // width than flutter_test's default surface — the AA Preview table
    // overflowed and pumpAndSettle hung on the resulting error banner
    // without this.
    await tester.binding.setSurfaceSize(const Size(1600, 1000));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    await http.runWithClient(() async {
      await tester.pumpWidget(
        _wrap(const HomeScreen(initialSessionId: _sessionId)),
      );
      // MockClient's response resolves via an ordinary Future — no real
      // socket I/O — so a couple of pumps let the setState from
      // _loadLaunchSession's completed await flush through normally.
      await tester.pump();
      await tester.pump();

      await tester.tap(find.text('AA Preview'));
      // TabBarView's PageView lazily builds the newly-visible page, so the
      // tap alone doesn't materialize AaPreviewTableWidget yet — a couple
      // more timed pumps are needed before it, and its own real
      // Isolate.run() worker, actually appear in the tree. Confirmed by
      // direct observation during this investigation: a single pump()
      // right after tap found DisplayAreaTabs present but
      // AaPreviewTableWidget absent (count 0), and only appeared once
      // these extra pumps ran.
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 500));
      await tester.pump(const Duration(milliseconds: 500));
      // Not pumpAndSettle(): HomeScreen's own 10s health-check
      // Timer.periodic is real, outstanding, pending work as far as
      // pumpAndSettle is concerned, so it never considers the tree
      // settled and hangs regardless of period length.
      await tester.runAsync(() => Future<void>.delayed(const Duration(seconds: 3)));
      await tester.pump();
      await tester.pump();

      expect(tester.takeException(), isNull);
      expect(find.textContaining('Empty associative'), findsNothing);
      expect(find.textContaining('8 rows'), findsOneWidget);
      expect(find.textContaining('nemo'), findsOneWidget); // the real captioned mask
      expect(find.byType(Image), findsNWidgets(8)); // every mask's crop painted
    }, () => mockClient);
  });
}
