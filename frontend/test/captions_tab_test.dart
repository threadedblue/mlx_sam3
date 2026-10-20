// The caption table: segments and captions side by side, editable, saved back.
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:frontend/layer_state.dart';
import 'package:frontend/main.dart' show DisplayAreaTabs;
import 'package:frontend/widgets/captions_card.dart';
import 'package:frontend/widgets/captions_tab.dart';

// 1x1 transparent PNG, base64.
const _png =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

const _a = '0:aaaaaaaa'; // captioned
const _b = '0:bbbbbbbb'; // never captioned
const _c = '1:cccccccc'; // captioned, pass 1

Map<String, dynamic> _masks({
  List<String?> captions = const ['small man, knight armor', null, 'clown, sitting'],
}) =>
    {
      'mask_ids': const [_a, _b, _c],
      'dataset_statuses': [for (final c in captions) c == null ? 'unassigned' : 'keep'],
      'captions': captions,
      'passes': const [0, 0, 1],
      'text_tags': const ['knight', null, 'clown'],
      'thumb_png_bytes': const [_png, _png, _png],
    };

Widget _host(
  Map<String, dynamic>? data, {
  SaveCaption? onSave,
  SaveSession? onSaveSession,
  double width = 900,
}) =>
    MaterialApp(
      home: Scaffold(
        body: SizedBox(
          width: width,
          height: 700,
          child: CaptionsTab(aaPreviewData: data, onSave: onSave, onSaveSession: onSaveSession),
        ),
      ),
    );

String _text(WidgetTester t, String id) =>
    t.widget<TextField>(find.byKey(ValueKey('caption-field-$id'))).controller!.text;

Future<void> _edit(WidgetTester t, String id, String text) async {
  await t.enterText(find.byKey(ValueKey('caption-field-$id')), text);
  await t.pump();
}

FilledButton _saveButton(WidgetTester t) =>
    t.widget<FilledButton>(find.byKey(const ValueKey('captions-save')));

Future<void> _showAll(WidgetTester t) async {
  await t.tap(find.textContaining('All segments'));
  await t.pump();
}

void main() {
  group('segmentEntriesFrom', () {
    test('lists every segment, or only the captioned ones, with pass and tag', () {
      final all = segmentEntriesFrom(_masks());
      expect(all.map((e) => e.maskId), [_a, _b, _c]);
      expect(all.map((e) => e.pass), [0, 0, 1]);
      expect(all.map((e) => e.captioned), [true, false, true]);
      expect(all.map((e) => e.tag), ['knight', null, 'clown']);
      expect(captionEntriesFrom(_masks()).map((e) => e.maskId), [_a, _c]);
      expect(segmentEntriesFrom(null), isEmpty);
    });
  });

  group('table', () {
    testWidgets('shows captioned segments by default, with column headers and counts', (t) async {
      await t.pumpWidget(_host(_masks()));
      for (final h in ['Segment', 'Caption', 'Pass', 'Tag', 'Status']) {
        expect(find.text(h), findsOneWidget);
      }
      expect(find.text('Captioned (2)'), findsOneWidget);
      expect(find.text('All segments (3)'), findsOneWidget);
      expect(_text(t, _a), 'small man, knight armor');
      expect(_text(t, _c), 'clown, sitting');
      expect(find.byKey(const ValueKey('caption-row-$_b')), findsNothing);
      expect(find.byKey(const ValueKey('caption-thumb-$_a')), findsOneWidget);
    });

    testWidgets('All segments adds the uncaptioned ones, ready for a caption', (t) async {
      await t.pumpWidget(_host(_masks()));
      await _showAll(t);
      expect(_text(t, _b), '');
      expect(find.text('no caption yet'), findsOneWidget);
      expect(find.byKey(const ValueKey('caption-status-$_b')), findsOneWidget);
    });

    testWidgets('the Caption column sits to the left of the Segment column', (t) async {
      await t.pumpWidget(_host(_masks()));
      final captionX = t.getTopLeft(find.text('Caption')).dx;
      final segmentX = t.getTopLeft(find.text('Segment')).dx;
      expect(captionX, lessThan(segmentX));
      expect(
        t.getTopLeft(find.byKey(const ValueKey('caption-field-$_a'))).dx,
        lessThan(t.getTopLeft(find.byKey(const ValueKey('caption-thumb-$_a'))).dx),
      );
    });

    testWidgets('narrow widths drop the Tag and Status columns, keep Segment and Caption', (t) async {
      await t.pumpWidget(_host(_masks(), width: 500));
      expect(find.text('Segment'), findsOneWidget);
      expect(find.text('Caption'), findsOneWidget);
      expect(find.text('Tag'), findsNothing);
      expect(find.text('Status'), findsNothing);
    });

    testWidgets('with no segments at all it says so', (t) async {
      await t.pumpWidget(_host(null));
      expect(find.byKey(const ValueKey('captions-empty')), findsOneWidget);
    });

    testWidgets('with segments but none captioned it opens on Not captioned, listing them all', (t) async {
      await t.pumpWidget(_host(_masks(captions: const [null, null, null])));
      expect(find.text('Not captioned (3)'), findsOneWidget);
      expect(find.byKey(const ValueKey('caption-row-$_a')), findsOneWidget);
      expect(find.byKey(const ValueKey('caption-row-$_b')), findsOneWidget);
    });

    testWidgets('Captioned with nothing in it explains itself', (t) async {
      await t.pumpWidget(_host(_masks(captions: const [null, null, null])));
      await t.tap(find.text('Captioned (0)'));
      await t.pump();
      expect(find.byKey(const ValueKey('captions-none-captioned')), findsOneWidget);
    });
  });

  group('not captioned lane', () {
    // Pretends to be the backend: delete returns the segment to uncaptioned.
    Widget host(List<String?> captions, {ClearCaptions? onClear, SetNever? onNever}) => MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 900,
              height: 700,
              child: CaptionsTab(aaPreviewData: _masks(captions: captions), onClear: onClear, onNever: onNever),
            ),
          ),
        );

    testWidgets('Not captioned lists exactly the uncaptioned segments; Captioned the rest', (t) async {
      await t.pumpWidget(host(const ['one', null, 'three']));
      expect(find.text('Not captioned (1)'), findsOneWidget);
      expect(find.text('Captioned (2)'), findsOneWidget);
      await t.tap(find.text('Not captioned (1)'));
      await t.pump();
      expect(find.byKey(const ValueKey('caption-row-$_b')), findsOneWidget);
      expect(find.byKey(const ValueKey('caption-row-$_a')), findsNothing);
    });

    testWidgets('a segment moves lanes when its caption is added', (t) async {
      await t.pumpWidget(host(const ['one', null, 'three']));
      await t.pumpWidget(host(const ['one', 'now captioned', 'three']));
      expect(find.text('Not captioned (0)'), findsOneWidget);
      expect(find.text('Captioned (3)'), findsOneWidget);
    });

    testWidgets('Delete caption calls the handler and the segment moves back', (t) async {
      final cleared = <List<String>>[];
      await t.pumpWidget(host(const ['one', null, 'three'], onClear: (ids) async {
        cleared.add(ids);
        return null;
      }));
      await t.tap(find.text('Captioned (2)'));
      await t.pump();
      await t.tap(find.byKey(const ValueKey('caption-delete-$_a')));
      await t.pump();
      expect(cleared, [
        [_a]
      ]);
      // The parent refreshes the payload with the caption gone.
      await t.pumpWidget(host(const [null, null, 'three'], onClear: (ids) async => null));
      expect(find.text('Not captioned (2)'), findsOneWidget);
      expect(find.text('Captioned (1)'), findsOneWidget);
    });

    testWidgets('emptying a captioned row and pressing Save deletes the caption', (t) async {
      final cleared = <List<String>>[];
      final saved = <String>[];
      await t.pumpWidget(MaterialApp(
        home: Scaffold(
          body: SizedBox(
            width: 900,
            height: 700,
            child: CaptionsTab(
              aaPreviewData: _masks(),
              onSave: (id, text) async {
                saved.add(id);
                return null;
              },
              onClear: (ids) async {
                cleared.add(ids);
                return null;
              },
            ),
          ),
        ),
      ));
      await _edit(t, _a, '   ');
      expect(find.text('Save will delete this caption'), findsOneWidget);
      expect(find.text('a caption cannot be empty'), findsNothing);
      await t.tap(find.byKey(const ValueKey('captions-save')));
      await t.pump();
      expect(cleared, [
        [_a]
      ]);
      expect(saved, isEmpty);
    });

    testWidgets('no Delete caption button without a handler, nor on an uncaptioned row', (t) async {
      await t.pumpWidget(host(const ['one', null, 'three']));
      await t.tap(find.text('Captioned (2)'));
      await t.pump();
      expect(find.byKey(const ValueKey('caption-delete-$_a')), findsNothing);
    });

    testWidgets('the Never column is the last one in the header', (t) async {
      await t.pumpWidget(host(const ['one', null, 'three'], onNever: (ids, never) async => null));
      final neverX = t.getTopLeft(find.text('Never')).dx;
      expect(neverX, greaterThan(t.getTopLeft(find.text('Status')).dx));
      expect(neverX, greaterThan(t.getTopLeft(find.text('Pass')).dx));
      expect(neverX, greaterThan(t.getTopLeft(find.text('Segment')).dx));
    });
  });

  group('saving back', () {
    testWidgets('Save is off until a caption is edited, then applies it and saves the session',
        (t) async {
      final saved = <(String, String)>[];
      var sessionSaves = 0;
      await t.pumpWidget(_host(_masks(), onSave: (id, text) async {
        saved.add((id, text));
        return null;
      }, onSaveSession: () async {
        sessionSaves++;
        return null;
      }));
      expect(_saveButton(t).onPressed, isNull);

      await _edit(t, _a, '  small man, grey knight armor  ');
      expect(_saveButton(t).onPressed, isNotNull);
      expect(find.text('Save 1 change'), findsOneWidget);
      expect(find.text('edited'), findsOneWidget);

      await t.tap(find.byKey(const ValueKey('captions-save')));
      await t.pumpAndSettle();

      expect(saved, [(_a, 'small man, grey knight armor')]);
      expect(sessionSaves, 1);
      expect(find.text('Saved 1 caption and the session.'), findsOneWidget);
      expect(_text(t, _a), 'small man, grey knight armor'); // trimmed in place
      expect(_saveButton(t).onPressed, isNull); // clean again
    });

    testWidgets('several edits are all applied, then the session is saved once', (t) async {
      final saved = <String>[];
      var sessionSaves = 0;
      await t.pumpWidget(_host(_masks(), onSave: (id, text) async {
        saved.add(id);
        return null;
      }, onSaveSession: () async {
        sessionSaves++;
        return null;
      }));
      await _edit(t, _a, 'one');
      await _edit(t, _c, 'two');
      expect(find.text('Save 2 changes'), findsOneWidget);
      await t.tap(find.byKey(const ValueKey('captions-save')));
      await t.pumpAndSettle();
      expect(saved, [_a, _c]);
      expect(sessionSaves, 1);
    });

    testWidgets('typing a caption on an uncaptioned segment captions it', (t) async {
      final saved = <(String, String)>[];
      await t.pumpWidget(_host(_masks(), onSave: (id, text) async {
        saved.add((id, text));
        return null;
      }));
      await _showAll(t);
      await _edit(t, _b, 'a new caption');
      await t.tap(find.byKey(const ValueKey('captions-save')));
      await t.pumpAndSettle();
      expect(saved, [(_b, 'a new caption')]);
    });

    testWidgets('a row that fails says why, and nothing is written to disk', (t) async {
      var sessionSaves = 0;
      await t.pumpWidget(_host(
        _masks(),
        onSave: (id, text) async => id == _c ? 'Session not found' : null,
        onSaveSession: () async {
          sessionSaves++;
          return null;
        },
      ));
      await _edit(t, _a, 'fine');
      await _edit(t, _c, 'will fail');
      await t.tap(find.byKey(const ValueKey('captions-save')));
      await t.pumpAndSettle();

      expect(find.text('Session not found'), findsOneWidget);
      expect(sessionSaves, 0);
      final summary = t.widget<Text>(find.byKey(const ValueKey('captions-summary'))).data!;
      expect(summary, contains('1 of 2 captions could not be saved'));
      expect(summary, contains('nothing was written to disk'));
      expect(_text(t, _c), 'will fail'); // the edit is kept
      // The good row went through; the failed one is still pending.
      expect(find.text('Save 1 change'), findsOneWidget);
    });

    testWidgets('if only the session save fails, the table says so', (t) async {
      await t.pumpWidget(_host(
        _masks(),
        onSave: (id, text) async => null,
        onSaveSession: () async => 'disk full',
      ));
      await _edit(t, _a, 'edited');
      await t.tap(find.byKey(const ValueKey('captions-save')));
      await t.pumpAndSettle();
      final summary = t.widget<Text>(find.byKey(const ValueKey('captions-summary'))).data!;
      expect(summary, contains('saving the session failed: disk full'));
    });

    testWidgets('a blanked caption cannot be saved and is flagged', (t) async {
      var calls = 0;
      await t.pumpWidget(_host(_masks(), onSave: (id, text) async {
        calls++;
        return null;
      }));
      await _edit(t, _a, '   ');
      expect(find.text('a caption cannot be empty'), findsOneWidget);
      expect(_saveButton(t).onPressed, isNull);
      expect(calls, 0);
    });

    testWidgets('Revert puts every stored caption back', (t) async {
      await t.pumpWidget(_host(_masks(), onSave: (id, text) async => null));
      await _edit(t, _a, 'changed a');
      await _edit(t, _c, 'changed c');
      await t.tap(find.byKey(const ValueKey('captions-revert')));
      await t.pump();
      expect(_text(t, _a), 'small man, knight armor');
      expect(_text(t, _c), 'clown, sitting');
      expect(_saveButton(t).onPressed, isNull);
    });

    testWidgets('a refresh updates a row you have not touched, never one you are editing',
        (t) async {
      await t.pumpWidget(_host(_masks(), onSave: (id, text) async => null));
      await _edit(t, _c, 'my unsaved words');

      await t.pumpWidget(_host(
        _masks(captions: const ['small man, GREY knight armor', null, 'clown, running']),
        onSave: (id, text) async => null,
      ));
      expect(_text(t, _a), 'small man, GREY knight armor'); // untouched: follows
      expect(_text(t, _c), 'my unsaved words'); // edited: kept
    });

    testWidgets('edits survive switching between Captioned and All segments', (t) async {
      await t.pumpWidget(_host(_masks(), onSave: (id, text) async => null));
      await _edit(t, _a, 'kept across the filter');
      await _showAll(t);
      await t.tap(find.textContaining('Captioned ('));
      await t.pump();
      expect(_text(t, _a), 'kept across the filter');
    });
  });

  group('DisplayAreaTabs', () {
    late ui.Image image;

    setUpAll(() async {
      final recorder = ui.PictureRecorder();
      Canvas(recorder).drawRect(
        const Rect.fromLTWH(0, 0, 50, 50),
        Paint()..color = const Color(0xFF808080),
      );
      image = await recorder.endRecording().toImage(50, 50);
    });

    Widget tabs({ValueNotifier<int>? request}) => ChangeNotifierProvider<LayerState>(
          create: (_) => LayerState(),
          child: MaterialApp(
            home: Scaffold(
              body: SizedBox(
                width: 900,
                height: 700,
                child: DisplayAreaTabs(
                  imageBytes: Uint8List(0),
                  uiImage: image,
                  originalUiImage: image,
                  segments: const [],
                  result: null,
                  aaPreviewData: _masks(),
                  isLoading: false,
                  mode: null,
                  onBoxDrawn: (_) {},
                  onPointDrawn: (_) {},
                  onPanelDrawn: (_) {},
                  onCaptionSave: (id, text) async => null,
                  openCaptionsRequest: request,
                ),
              ),
            ),
          ),
        );

    testWidgets('Captions sits before AA Preview in the tab bar', (t) async {
      await t.pumpWidget(tabs());
      expect(t.getTopLeft(find.text('Canvas')).dx, lessThan(t.getTopLeft(find.text('Captions')).dx));
      expect(t.getTopLeft(find.text('Captions')).dx, lessThan(t.getTopLeft(find.text('AA Preview')).dx));
    });

    testWidgets('there is a Captions tab and it shows the table', (t) async {
      await t.pumpWidget(tabs());
      expect(find.byKey(const ValueKey('captions-save')), findsNothing);
      await t.tap(find.text('Captions'));
      await t.pump(const Duration(seconds: 1));
      await t.pump(const Duration(seconds: 1));
      expect(find.byKey(const ValueKey('captions-save')), findsOneWidget);
    });

    testWidgets('the card\'s request switches to the Captions tab', (t) async {
      final request = ValueNotifier<int>(0);
      await t.pumpWidget(tabs(request: request));
      expect(find.byKey(const ValueKey('captions-save')), findsNothing);
      request.value++;
      await t.pump(const Duration(seconds: 1));
      await t.pump(const Duration(seconds: 1));
      expect(find.byKey(const ValueKey('captions-save')), findsOneWidget);
    });
  });

  group('never needs', () {
    // _a captioned by hand, _b blank, _c marked never needs.
    Map<String, dynamic> neverMasks() => {
          ..._masks(captions: const ['small man, knight armor', null, null]),
          'dataset_statuses': const ['keep', 'unassigned', 'discard'],
          'draft_captions': const [null, 'clown, sitting', 'a title banner'],
        };

    Widget hostWithNever(Future<String?> Function(List<String>, bool) onNever) => MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 900,
              height: 700,
              child: CaptionsTab(aaPreviewData: neverMasks(), onNever: onNever),
            ),
          ),
        );

    test('a discard segment is never, and never reviewable, even with a draft', () {
      final all = segmentEntriesFrom(neverMasks());
      expect(all.map((e) => e.never), [false, false, true]);
      expect(all.map((e) => e.needsReview), [false, true, false]);
      expect(all.map((e) => e.reviewState), ['captioned', 'draft', 'never needs']);
    });

    testWidgets('no Never column unless the table is given a handler', (t) async {
      await t.pumpWidget(_host(neverMasks()));
      expect(find.text('Never'), findsNothing);
      expect(find.byKey(const ValueKey('caption-never-$_b')), findsNothing);
    });

    testWidgets('ticking a row marks just that segment', (t) async {
      final calls = <(List<String>, bool)>[];
      await t.pumpWidget(hostWithNever((ids, never) async {
        calls.add((ids, never));
        return null;
      }));
      await _showAll(t);
      await t.tap(find.byKey(const ValueKey('caption-never-$_b')));
      await t.pump();
      expect(calls, [([_b], true)]);
    });

    testWidgets('a captioned row cannot be ticked', (t) async {
      await t.pumpWidget(hostWithNever((ids, never) async => null));
      await _showAll(t);
      expect(t.widget<Checkbox>(find.byKey(const ValueKey('caption-never-$_a'))).onChanged, isNull);
    });

    testWidgets('a never row shows ticked and its caption field is off', (t) async {
      await t.pumpWidget(hostWithNever((ids, never) async => null));
      await _showAll(t);
      expect(t.widget<Checkbox>(find.byKey(const ValueKey('caption-never-$_c'))).value, isTrue);
      expect(t.widget<TextField>(find.byKey(const ValueKey('caption-field-$_c'))).enabled, isFalse);
    });

    testWidgets('the header box ticks every shown row that can take the mark', (t) async {
      final calls = <(List<String>, bool)>[];
      await t.pumpWidget(hostWithNever((ids, never) async {
        calls.add((ids, never));
        return null;
      }));
      await _showAll(t);
      await t.tap(find.byKey(const ValueKey('captions-never-all')));
      await t.pump();
      // _a is captioned (left out); _b and _c are eligible, _c already marked.
      expect(calls.single.$1, [_b, _c]);
      expect(calls.single.$2, isTrue);
    });

    testWidgets('a Never needs view lists just the marked segments', (t) async {
      await t.pumpWidget(hostWithNever((ids, never) async => null));
      expect(find.text('Never needs (1)'), findsOneWidget);
      await t.tap(find.text('Never needs (1)'));
      await t.pump();
      expect(find.byKey(const ValueKey('caption-row-$_c')), findsOneWidget);
      expect(find.byKey(const ValueKey('caption-row-$_b')), findsNothing);
      expect(find.byKey(const ValueKey('caption-row-$_a')), findsNothing);
    });

    testWidgets('a bulk tick can be undone', (t) async {
      final calls = <(List<String>, bool)>[];
      await t.pumpWidget(hostWithNever((ids, never) async {
        calls.add((ids, never));
        return null;
      }));
      await _showAll(t);
      await t.tap(find.byKey(const ValueKey('captions-never-all')));
      await t.pump();
      expect(find.byKey(const ValueKey('captions-never-undo')), findsOneWidget);
      await t.tap(find.byKey(const ValueKey('captions-never-undo')));
      await t.pump();
      expect(calls.length, 2);
      expect(calls[1].$1, calls[0].$1);
      expect(calls[1].$2, !calls[0].$2);
    });

    testWidgets('a never segment is not in the Needs review list', (t) async {
      await t.pumpWidget(hostWithNever((ids, never) async => null));
      expect(find.byKey(const ValueKey('caption-row-$_b')), findsOneWidget);
      expect(find.byKey(const ValueKey('caption-row-$_c')), findsNothing);
    });
  });

  group('model drafts', () {
    // _a captioned by a person, _b has only a model draft, _c nothing yet.
    Map<String, dynamic> draftMasks({String? aDraft}) => {
          ..._masks(captions: const ['small man, knight armor', null, null]),
          'draft_captions': [aDraft, 'clown, sitting', null],
        };

    test('a draft is reviewable until a person captions it', () {
      final all = segmentEntriesFrom(draftMasks());
      expect(all.map((e) => e.needsReview), [false, true, false]);
      expect(all.map((e) => e.reviewState), ['captioned', 'draft', 'no caption']);
    });

    test('a kept caption is approved when it equals the draft, corrected when not', () {
      final same = segmentEntriesFrom(draftMasks(aDraft: 'small man, knight armor'));
      expect(same.first.reviewState, 'approved');
      final diff = segmentEntriesFrom(draftMasks(aDraft: 'a man'));
      expect(diff.first.reviewState, 'corrected');
    });

    testWidgets('opens on Needs review and shows the draft in the field', (t) async {
      await t.pumpWidget(_host(draftMasks()));
      expect(find.text('Needs review (1)'), findsOneWidget);
      expect(_text(t, _b), 'clown, sitting');
      expect(find.byKey(const ValueKey('caption-row-$_a')), findsNothing);
    });

    testWidgets('an untouched draft is not saved', (t) async {
      final saved = <String>[];
      await t.pumpWidget(_host(draftMasks(), onSave: (id, c) async {
        saved.add(id);
        return null;
      }));
      expect(_saveButton(t).onPressed, isNull);
      expect(saved, isEmpty);
    });

    testWidgets('approving as written saves the draft text unchanged', (t) async {
      final saved = <String, String>{};
      await t.pumpWidget(_host(draftMasks(), onSave: (id, c) async {
        saved[id] = c;
        return null;
      }));
      await t.tap(find.byKey(const ValueKey('caption-approve-$_b')));
      await t.pump();
      expect(_saveButton(t).onPressed, isNotNull);
      await t.tap(find.byKey(const ValueKey('captions-save')));
      await t.pump();
      expect(saved, {_b: 'clown, sitting'});
    });

    testWidgets('editing a draft saves the edit', (t) async {
      final saved = <String, String>{};
      await t.pumpWidget(_host(draftMasks(), onSave: (id, c) async {
        saved[id] = c;
        return null;
      }));
      await _edit(t, _b, 'clown, sitting, crosshatching');
      await t.tap(find.byKey(const ValueKey('captions-save')));
      await t.pump();
      expect(saved, {_b: 'clown, sitting, crosshatching'});
    });

    testWidgets('Approve drafts shown ticks every draft in view', (t) async {
      final saved = <String>[];
      await t.pumpWidget(_host(draftMasks(), onSave: (id, c) async {
        saved.add(id);
        return null;
      }));
      await t.tap(find.byKey(const ValueKey('captions-approve-all')));
      await t.pump();
      await t.tap(find.byKey(const ValueKey('captions-save')));
      await t.pump();
      expect(saved, [_b]);
    });
  });

  group('Captioner card', () {
    Widget card({int blank = 2, bool running = false, bool hasSession = true, VoidCallback? onRun, String? error, String? progress}) =>
        MaterialApp(
          home: Scaffold(
            body: CaptionsCard(
              hasSession: hasSession,
              blank: blank,
              onRun: onRun ?? () {},
              running: running,
              progress: progress,
              runError: error,
            ),
          ),
        );

    FilledButton button(WidgetTester t) => t.widget<FilledButton>(find.byKey(const ValueKey('captions-card-run')));

    testWidgets('one button runs the captioner and says how many segments it will caption', (t) async {
      var runs = 0;
      await t.pumpWidget(card(onRun: () => runs++));
      expect(find.text('Run captioner (2)'), findsOneWidget);
      await t.tap(find.byKey(const ValueKey('captions-card-run')));
      expect(runs, 1);
    });

    testWidgets('is off with nothing blank, no session, or a run in progress', (t) async {
      await t.pumpWidget(card(blank: 0));
      expect(button(t).onPressed, isNull);
      await t.pumpWidget(card(hasSession: false));
      expect(button(t).onPressed, isNull);
      await t.pumpWidget(card(running: true, progress: '3 of 9'));
      expect(button(t).onPressed, isNull);
      expect(find.text('Captioning 3 of 9'), findsOneWidget);
    });

    testWidgets('says why it is off when everything is captioned or marked Never', (t) async {
      await t.pumpWidget(MaterialApp(
        home: Scaffold(
          body: CaptionsCard(hasSession: true, blank: 0, onRun: () {}, never: 16, captioned: 9),
        ),
      ));
      expect(find.byKey(const ValueKey('captions-card-why-off')), findsOneWidget);
      expect(find.textContaining('16 marked Never'), findsOneWidget);
      expect(find.textContaining('9 captioned'), findsOneWidget);
    });

    testWidgets('says nothing about being off while it is on', (t) async {
      await t.pumpWidget(card());
      expect(find.byKey(const ValueKey('captions-card-why-off')), findsNothing);
    });

    testWidgets('shows why a run failed', (t) async {
      await t.pumpWidget(card(error: 'No captioner available'));
      expect(find.text('No captioner available'), findsOneWidget);
    });
  });

}
