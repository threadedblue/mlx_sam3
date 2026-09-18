import 'package:aa_preview_table/aa_preview_table.dart';

/// Converts SF's `_result` map — the wire shape `serialize_sf_masks`
/// returns from every `/segment/*` call (see `services.py`) — into the
/// shared `aa_preview_table` package's generic [AaPayload]: one row per
/// mask, one column per field.
///
/// This is the one place SF's own domain (mask records: `mask_id`,
/// `dataset_status`, `held`, `caption`, `text_tag`, `score`, bbox, pass)
/// meets
/// the shared package's generic rows/cols/vals shape — the same principle
/// applied to DoubleNaught's own `AaPayload` when this widget was
/// extracted: the shared package knows nothing about SF's domain model,
/// so the conversion happens here, in SF's own codebase, not there.
///
/// `result['mask_ids']`/`dataset_statuses`/`held_flags`/`captions`/
/// `text_tags`/`scores`/`boxes`/`passes` are parallel lists (one entry per
/// *current-pass* mask — `serialize_sf_masks` never returns another
/// pass's records, so every value in `passes` is the same constant,
/// repeated once per row — but it's still surfaced as its own column
/// rather than assumed, since nothing else in this payload tells a viewer
/// which pass they're looking at).
///
/// A field is only emitted as a triple when it has a real value — an
/// unassigned mask's absent caption, for instance, produces no `caption`
/// row rather than an empty-string one — so the table's columns stay
/// exactly as sparse as the underlying AA convention intends.
AaPayload sfResultToAaPayload(Map<String, dynamic>? result) {
  if (result == null) return const AaPayload();

  final maskIds = (result['mask_ids'] as List?) ?? const [];
  final datasetStatuses = (result['dataset_statuses'] as List?) ?? const [];
  final heldFlags = (result['held_flags'] as List?) ?? const [];
  final captions = (result['captions'] as List?) ?? const [];
  final textTags = (result['text_tags'] as List?) ?? const [];
  final scores = (result['scores'] as List?) ?? const [];
  final boxes = (result['boxes'] as List?) ?? const [];
  final passes = (result['passes'] as List?) ?? const [];

  final rows = <String>[];
  final cols = <String>[];
  final vals = <Object>[];

  void emit(String rowKey, String col, Object? value) {
    if (value == null) return;
    rows.add(rowKey);
    cols.add(col);
    vals.add(value);
  }

  for (var i = 0; i < maskIds.length; i++) {
    final rowKey = maskIds[i] as String? ?? 'mask_$i';

    emit(rowKey, 'dataset_status', i < datasetStatuses.length ? datasetStatuses[i] as String? : null);
    // held is meaningful even when false (an explicit "not queued", not
    // "unknown"), so it's emitted whenever the parallel list has an entry
    // at all — unlike the other fields, where absence means "no value".
    if (i < heldFlags.length) emit(rowKey, 'held', heldFlags[i] as bool? ?? false);
    emit(rowKey, 'caption', i < captions.length ? captions[i] as String? : null);
    emit(rowKey, 'text_tag', i < textTags.length ? textTags[i] as String? : null);
    emit(rowKey, 'score', i < scores.length ? scores[i] as num? : null);
    if (i < boxes.length && boxes[i] != null) {
      emit(rowKey, 'bbox', (boxes[i] as List).toString());
    }
    emit(rowKey, 'pass', i < passes.length ? passes[i] as int? : null);
  }

  return AaPayload(rows: rows, cols: cols, vals: vals);
}
