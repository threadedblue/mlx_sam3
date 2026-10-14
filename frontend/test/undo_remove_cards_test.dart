// Remove-a-selection and undo-last-scrub controls (segmentation overhaul, step 1).
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:frontend/layered_segmentation_canvas.dart';
import 'package:frontend/widgets/lbs_card.dart';
import 'package:frontend/widgets/objects_selected_card.dart';

Widget _host(Widget child) => MaterialApp(
      theme: ThemeData.dark(),
      home: Scaffold(body: SingleChildScrollView(child: child)),
    );

Segment _seg(String id) =>
    Segment(path: Path(), maskId: id, datasetStatus: 'unassigned', held: true);

void main() {
  group('ObjectsSelectedCard remove', () {
    testWidgets('shows a remove control per selection and reports which', (tester) async {
      final removed = <String>[];
      await tester.pumpWidget(_host(ObjectsSelectedCard(
        maskCount: 2,
        segments: [_seg('0:aaaaaaaa'), _seg('0:bbbbbbbb')],
        enabled: true,
        onSetHeld: (_, __) {},
        onClearPrompts: () {},
        onRemove: removed.add,
      )));
      expect(find.byIcon(Icons.close), findsNWidgets(2));
      await tester.tap(find.byKey(const Key('remove-mask-0:bbbbbbbb')));
      expect(removed, ['0:bbbbbbbb']);
    });

    testWidgets('no remove control when the callback is not wired', (tester) async {
      await tester.pumpWidget(_host(ObjectsSelectedCard(
        maskCount: 1,
        segments: [_seg('0:aaaaaaaa')],
        enabled: true,
        onSetHeld: (_, __) {},
        onClearPrompts: () {},
      )));
      expect(find.byIcon(Icons.close), findsNothing);
    });

    testWidgets('remove is disabled while busy', (tester) async {
      var called = false;
      await tester.pumpWidget(_host(ObjectsSelectedCard(
        maskCount: 1,
        segments: [_seg('0:aaaaaaaa')],
        enabled: false,
        onSetHeld: (_, __) {},
        onClearPrompts: () {},
        onRemove: (_) => called = true,
      )));
      await tester.tap(find.byKey(const Key('remove-mask-0:aaaaaaaa')), warnIfMissed: false);
      expect(called, isFalse);
    });
  });

  group('LBSCard undo', () {
    LBSCard card({bool canUndo = true, bool scrubbing = false, VoidCallback? onUndo, String? msg}) => LBSCard(
          pendingCount: 1,
          hasSelection: true,
          lastScrubLabel: 'Pass 0 → Pass 1',
          isScrubbing: scrubbing,
          onScrub: () {},
          canUndo: canUndo,
          onUndo: onUndo,
          undoMessage: msg,
        );

    testWidgets('no undo button when undo is not wired', (tester) async {
      await tester.pumpWidget(_host(card(onUndo: null)));
      expect(find.byKey(const Key('undo-scrub')), findsNothing);
    });

    testWidgets('undo is enabled only when there is a scrub to undo', (tester) async {
      var taps = 0;
      await tester.pumpWidget(_host(card(canUndo: false, onUndo: () => taps++)));
      await tester.tap(find.byKey(const Key('undo-scrub')), warnIfMissed: false);
      expect(taps, 0);

      await tester.pumpWidget(_host(card(canUndo: true, onUndo: () => taps++)));
      await tester.tap(find.byKey(const Key('undo-scrub')));
      expect(taps, 1);
    });

    testWidgets('undo is off while a scrub is running', (tester) async {
      var taps = 0;
      await tester.pumpWidget(_host(card(scrubbing: true, onUndo: () => taps++)));
      await tester.tap(find.byKey(const Key('undo-scrub')), warnIfMissed: false);
      expect(taps, 0);
    });

    testWidgets('a refusal reason is shown', (tester) async {
      await tester.pumpWidget(_host(card(onUndo: () {}, msg: 'cannot undo: 1 captioned selection(s)')));
      expect(find.byKey(const Key('undo-message')), findsOneWidget);
      expect(find.textContaining('1 captioned'), findsOneWidget);
    });
  });
}
