import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:aa_preview_table/aa_preview_table.dart';
import 'package:frontend/services/sf_aa_preview_adapter.dart';

void main() {
  group('sfResultToAaPayload', () {
    test('null result produces an empty payload', () {
      final aa = sfResultToAaPayload(null);

      expect(aa.rows, isEmpty);
      expect(aa.cols, isEmpty);
      expect(aa.vals, isEmpty);
    });

    test('one row per mask, one triple per present field', () {
      final result = {
        'mask_ids': ['0:abc', '0:def'],
        'dataset_statuses': ['keep', 'unassigned'],
        'held_flags': [true, false],
        'captions': ['a red egg', null],
        'text_tags': [null, 'figure'],
        'scores': [0.91, 0.77],
        'boxes': [
          [10.0, 20.0, 30.0, 40.0],
          [1.0, 2.0, 3.0, 4.0],
        ],
        'passes': [0, 0],
      };

      final aa = sfResultToAaPayload(result);

      Object? cell(String row, String col) {
        for (var i = 0; i < aa.rows.length; i++) {
          if (aa.rows[i] == row && aa.cols[i] == col) return aa.vals[i];
        }
        return null;
      }

      // Captioned, held keeper: every field present.
      expect(cell('0:abc', 'dataset_status'), 'keep');
      expect(cell('0:abc', 'held'), true);
      expect(cell('0:abc', 'caption'), 'a red egg');
      expect(cell('0:abc', 'text_tag'), null); // never emitted — no triple
      expect(cell('0:abc', 'score'), 0.91);
      expect(cell('0:abc', 'bbox'), '[10.0, 20.0, 30.0, 40.0]');
      expect(cell('0:abc', 'pass'), 0);

      // Unassigned, unheld: caption absent (not an empty-string triple).
      expect(cell('0:def', 'dataset_status'), 'unassigned');
      expect(cell('0:def', 'held'), false);
      expect(cell('0:def', 'text_tag'), 'figure');
    });

    test('absent optional fields never produce a triple, not a null/empty one', () {
      final result = {
        'mask_ids': ['0:xyz'],
        'dataset_statuses': ['unassigned'],
        // held_flags/captions/text_tags/scores/boxes all absent entirely.
      };

      final aa = sfResultToAaPayload(result);

      expect(aa.rows, ['0:xyz']);
      expect(aa.cols, ['dataset_status']);
      expect(aa.vals, ['unassigned']);
    });

    test('held is emitted even when false, unlike the other optional fields', () {
      final result = {
        'mask_ids': ['0:xyz'],
        'dataset_statuses': ['unassigned'],
        'held_flags': [false],
      };

      final aa = sfResultToAaPayload(result);

      expect(aa.rows, contains('0:xyz'));
      final heldIndex = aa.cols.indexOf('held');
      expect(heldIndex, isNot(-1), reason: 'an explicit false must still produce a triple');
      expect(aa.vals[heldIndex], false);
    });

    test('a mask with no fields at all still gets no row (nothing to emit)', () {
      final result = {
        'mask_ids': ['0:xyz'],
        'dataset_statuses': [null],
      };

      final aa = sfResultToAaPayload(result);

      expect(aa.rows, isEmpty);
    });

    test('output feeds the shared package AaPreviewWorker without error', () async {
      final result = {
        'mask_ids': ['0:a', '0:b', '0:c'],
        'dataset_statuses': ['keep', 'keep', 'unassigned'],
        'held_flags': [false, true, true],
        'captions': ['egg one', 'egg two', null],
      };

      final aa = sfResultToAaPayload(result);
      const worker = AaPreviewWorker();
      final page = await worker.processPreviewPage(aa);

      expect(page.totalRowCount, 3);
      expect(page.columns, containsAll(['dataset_status', 'held', 'caption']));
      final rowA = page.rows.firstWhere((r) => r.rowKey == '0:a');
      expect(rowA.cells['dataset_status'], 'keep');
      expect(rowA.cells['held'], 'false');
      expect(rowA.cells['caption'], 'egg one');
    });

    test('pass is emitted as its own column, present-only like the other fields', () {
      final result = {
        'mask_ids': ['0:abc', '0:def'],
        'dataset_statuses': ['keep', 'unassigned'],
        'passes': [0, 0],
      };

      final aa = sfResultToAaPayload(result);

      final passTriples = [
        for (var i = 0; i < aa.rows.length; i++)
          if (aa.cols[i] == 'pass') (row: aa.rows[i], val: aa.vals[i]),
      ];
      expect(passTriples, hasLength(2));
      expect(passTriples.map((t) => t.row), containsAll(['0:abc', '0:def']));
      expect(passTriples.every((t) => t.val == 0), isTrue);

      // Absent entirely (older backend, or a field genuinely missing) ->
      // no pass triples at all, same present-only convention as
      // dataset_status/caption/text_tag/score/bbox.
      final withoutPasses = sfResultToAaPayload({
        'mask_ids': ['0:xyz'],
        'dataset_statuses': ['unassigned'],
      });
      expect(withoutPasses.cols, isNot(contains('pass')));
    });

    test('crop_png_bytes is emitted as its own column, present-only like the other fields', () {
      final result = {
        'mask_ids': ['0:abc', '0:def'],
        'dataset_statuses': ['keep', 'unassigned'],
        'crop_png_bytes': ['aGVsbG8=', null],
      };

      final aa = sfResultToAaPayload(result);

      Object? cell(String row, String col) {
        for (var i = 0; i < aa.rows.length; i++) {
          if (aa.rows[i] == row && aa.cols[i] == col) return aa.vals[i];
        }
        return null;
      }

      expect(cell('0:abc', 'crop_png_bytes'), 'aGVsbG8=');
      expect(cell('0:def', 'crop_png_bytes'), null); // never emitted — no triple

      // Absent entirely (older backend response) -> no crop_png_bytes
      // triples at all, same present-only convention as the other fields.
      final withoutCrops = sfResultToAaPayload({
        'mask_ids': ['0:xyz'],
        'dataset_statuses': ['unassigned'],
      });
      expect(withoutCrops.cols, isNot(contains('crop_png_bytes')));
    });

    test('a scrub advancing the pass changes what later rows report', () {
      final passZero = sfResultToAaPayload({
        'mask_ids': ['0:abc'],
        'dataset_statuses': ['keep'],
        'passes': [0],
      });
      final passOne = sfResultToAaPayload({
        'mask_ids': ['1:def'],
        'dataset_statuses': ['unassigned'],
        'passes': [1],
      });

      final passIdx0 = passZero.cols.indexOf('pass');
      final passIdx1 = passOne.cols.indexOf('pass');
      expect(passZero.vals[passIdx0], 0);
      expect(passOne.vals[passIdx1], 1);
    });

    test('round-trips a REAL serialize_sf_masks response (captured from the '
        'live engine: select -> hold -> caption, see services.py)', () async {
      // Captured verbatim from a real SFSession.add_box_selection ->
      // attach_caption -> set_held -> serialize_sf_masks call — not
      // hand-authored — with only the RLE `masks` bytes stripped (opaque,
      // irrelevant to this adapter). Re-generate by running
      // serialize_sf_masks against a live SFSession if this ever needs
      // updating.
      final result = {
        'original_width': 100,
        'original_height': 100,
        'boxes': [
          [10.0, 10.0, 40.0, 40.0],
        ],
        'scores': [0.9],
        'mask_ids': ['0:80aa20b5-2554-4dee-b46a-06cdd3dc9bb7'],
        'dataset_statuses': ['keep'],
        'held_flags': [true],
        'captions': ['a red egg'],
        'text_tags': [null],
        'passes': [0],
      };

      final aa = sfResultToAaPayload(result);
      const worker = AaPreviewWorker();
      final page = await worker.processPreviewPage(aa);

      expect(page.totalRowCount, 1);
      expect(page.columns, containsAll(['dataset_status', 'held', 'caption', 'score', 'bbox', 'pass']));
      expect(page.columns, isNot(contains('text_tag'))); // null in the source -> never emitted
      final row = page.rows.single;
      expect(row.rowKey, '0:80aa20b5-2554-4dee-b46a-06cdd3dc9bb7');
      expect(row.cells['dataset_status'], 'keep');
      expect(row.cells['held'], 'true');
      expect(row.cells['caption'], 'a red egg');
      expect(row.cells['pass'], '0');
    });
  });

  group('sfResultToAaPayload — passes type contract (root cause of the AA '
      'Preview tab rendering nothing at all on resume)', () {
    Map<String, dynamic> realAllMasksWithPasses(List<dynamic> passes) {
      final raw = jsonDecode(
        File('test/fixtures/real_all_masks.json').readAsStringSync(),
      ) as Map<String, dynamic>;
      return Map<String, dynamic>.from(raw)..['passes'] = passes;
    }

    test(
        'string passes (load_session_from_disk\'s pre-fix numeric-string '
        'form) throws — must fail if this string form is ever reintroduced',
        () {
      final result = realAllMasksWithPasses(
        List<dynamic>.generate(8, (i) => '${[0, 0, 0, 0, 1, 2, 2, 2][i]}'),
      );

      expect(() => sfResultToAaPayload(result), throwsA(isA<TypeError>()));
    });

    test(
        'int passes (every live /segment/*, /mask/*, /lama/scrub response, '
        'and now load_session_from_disk too) does not throw', () {
      final result = realAllMasksWithPasses([0, 0, 0, 0, 1, 2, 2, 2]);

      final aa = sfResultToAaPayload(result);

      expect(aa.rows, isNotEmpty);
      expect(aa.cols, contains('pass'));
      expect(aa.cols, contains('crop_png_bytes'));
    });
  });
}
