import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

/// Saves [caption] on mask [maskId]; returns an error message, or null on success.
typedef SaveCaption = Future<String?> Function(String maskId, String caption);

/// Marks segments [maskIds] "never needs" ([never] true) or takes the mark back;
/// returns an error message, or null on success.
typedef SetNever = Future<String?> Function(List<String> maskIds, bool never);

/// Deletes the captions of [maskIds] (they return to uncaptioned); returns an
/// error message, or null on success.
typedef ClearCaptions = Future<String?> Function(List<String> maskIds);

/// Writes the session to storage; returns an error message, or null on success.
typedef SaveSession = Future<String?> Function();

/// One row of the table: a segment of the session, captioned or not.
class SegmentEntry {
  final String maskId;
  final int pass;

  /// The saved caption, or null for a segment that has none.
  final String? caption;

  /// `keep` (captioned) or `unassigned`.
  final String status;

  /// The SAM3 prompt or substitute tag the segment was selected with, if any.
  final String? tag;

  /// Base64 PNG of the object trimmed to its bounding box (`thumb_png_bytes`),
  /// or null when the backend sent none.
  final String? thumbB64;

  /// The captioner's unreviewed first attempt, if it ran on this segment. Kept
  /// after a person reviews it, so a corrected caption can be told from one
  /// approved as written.
  final String? draft;

  const SegmentEntry({
    required this.maskId,
    required this.pass,
    required this.status,
    this.caption,
    this.tag,
    this.thumbB64,
    this.draft,
  });

  bool get captioned => status == 'keep' && caption != null;

  /// Marked "never needs" (`discard`): scrubbed away or otherwise not caption
  /// material. Not captioned, not in the review queue, not in a captioner run.
  bool get never => status == 'discard';

  /// A model draft waiting for a person: not captioned yet, but has a draft.
  bool get needsReview => !captioned && !never && draft != null && draft!.trim().isNotEmpty;

  /// `captioned`, `draft`, `approved` (kept exactly as the model wrote it),
  /// `corrected` (kept after a person changed the model's draft) or `no caption`.
  String get reviewState {
    if (captioned) {
      if (draft == null) return 'captioned';
      return caption!.trim() == draft!.trim() ? 'approved' : 'corrected';
    }
    if (never) return 'never needs';
    return needsReview ? 'draft' : 'no caption';
  }
}

/// The segments in [allMasks] (the session-wide, every-pass ledger), in the
/// payload's order. [captionedOnly] keeps just the captioned (`keep`) ones.
List<SegmentEntry> segmentEntriesFrom(Map<String, dynamic>? allMasks, {bool captionedOnly = false}) {
  if (allMasks == null) return const [];
  final ids = (allMasks['mask_ids'] as List?) ?? const [];
  final statuses = (allMasks['dataset_statuses'] as List?) ?? const [];
  final captions = (allMasks['captions'] as List?) ?? const [];
  final passes = (allMasks['passes'] as List?) ?? const [];
  final tags = (allMasks['text_tags'] as List?) ?? const [];
  final drafts = (allMasks['draft_captions'] as List?) ?? const [];
  final thumbs = (allMasks['thumb_png_bytes'] as List?) ?? const [];
  final out = <SegmentEntry>[];
  for (var i = 0; i < ids.length; i++) {
    final entry = SegmentEntry(
      maskId: '${ids[i]}',
      pass: i < passes.length ? ((passes[i] as num?)?.toInt() ?? 0) : 0,
      status: i < statuses.length ? '${statuses[i]}' : 'unassigned',
      caption: i < captions.length ? captions[i] as String? : null,
      tag: i < tags.length ? tags[i] as String? : null,
      draft: i < drafts.length ? drafts[i] as String? : null,
      thumbB64: () {
        final t = i < thumbs.length ? thumbs[i] as String? : null;
        return (t == null || t.isEmpty) ? null : t;
      }(),
    );
    if (captionedOnly && !entry.captioned) continue;
    out.add(entry);
  }
  return out;
}

/// Just the captioned segments (the table's default view, and the count the
/// sidebar card shows).
List<SegmentEntry> captionEntriesFrom(Map<String, dynamic>? allMasks) =>
    segmentEntriesFrom(allMasks, captionedOnly: true);

/// The session's segments as an editable table: picture and caption side by
/// side, one row per segment.
///
/// Why it exists: once a mask is captioned (`keep`), the Canvas no longer offers
/// the caption box for it, so a caption could not be reviewed or corrected. The
/// backend already overwrites a caption on a re-save (`/mask/caption`), so this
/// only needs a place to do it.
///
/// Edit captions in place. **Save** applies every edited row through [onSave]
/// (a caption on an uncaptioned segment makes it captioned, exactly as on the
/// Canvas), then writes the session through [onSaveSession] so the changes reach
/// `segment.parquet`. If any row fails, nothing is written to disk and the
/// failed rows say why. Saving is always an explicit click; there is no autosave.
///
/// A field you have edited is never overwritten by a refresh; one you have not
/// touched follows the stored caption.
enum _Filter { notCaptioned, review, captioned, never, all }

class CaptionsTab extends StatefulWidget {
  final Map<String, dynamic>? aaPreviewData;
  final SaveCaption? onSave;
  final SaveSession? onSaveSession;

  /// Ticks / unticks the "Never" checkbox. Null hides the column.
  final SetNever? onNever;

  /// Deletes a caption. Null hides the "Delete caption" button.
  final ClearCaptions? onClear;

  const CaptionsTab({
    super.key,
    required this.aaPreviewData,
    this.onSave,
    this.onSaveSession,
    this.onNever,
    this.onClear,
  });

  @override
  State<CaptionsTab> createState() => _CaptionsTabState();
}

class _CaptionsTabState extends State<CaptionsTab> {
  final Map<String, TextEditingController> _controllers = {};
  final Map<String, String> _stored = {}; // what the field is compared against: the saved caption, else the model's draft, else ''
  final Set<String> _approved = {}; // draft rows the user ticked as right as written
  final Map<String, bool> _hasDraft = {};
  final Map<String, bool> _wasCaptioned = {};
  // The last Never change (ticks or unticks), so a bulk tick can be undone.
  ({List<String> ids, bool never})? _lastNever;
  final Map<String, String> _errors = {};
  final Map<String, ({String b64, Uint8List bytes})> _thumbs = {};
  _Filter? _filter; // null until chosen: needs-review when drafts exist, else captioned
  bool _saving = false;
  String? _summary;
  bool _summaryIsError = false;

  @override
  void initState() {
    super.initState();
    _sync();
  }

  @override
  void didUpdateWidget(CaptionsTab old) {
    super.didUpdateWidget(old);
    _sync();
  }

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  /// Bring the controllers in line with the payload: new segments get a field; a
  /// field that still shows the old stored caption follows the new one; a field
  /// the user has edited is left alone.
  void _sync() {
    final entries = segmentEntriesFrom(widget.aaPreviewData);
    final live = {for (final e in entries) e.maskId};
    for (final id in _controllers.keys.where((k) => !live.contains(k)).toList()) {
      _controllers.remove(id)!.dispose();
      _stored.remove(id);
      _errors.remove(id);
      _thumbs.remove(id);
      _approved.remove(id);
      _hasDraft.remove(id);
      _wasCaptioned.remove(id);
    }
    for (final e in entries) {
      // A draft row shows the model's draft in the field, but it is NOT saved
      // until the user edits it or ticks it: the baseline is the draft, so an
      // untouched draft is not "dirty" and Save leaves it alone.
      final stored = e.caption ?? (e.needsReview ? e.draft! : '');
      _hasDraft[e.maskId] = e.needsReview;
      _wasCaptioned[e.maskId] = e.captioned;
      if (!e.needsReview) _approved.remove(e.maskId);
      final c = _controllers[e.maskId];
      if (c == null) {
        _controllers[e.maskId] = TextEditingController(text: stored);
      } else if (c.text == _stored[e.maskId] && c.text != stored) {
        c.text = stored;
      }
      _stored[e.maskId] = stored;
    }
  }

  Uint8List? _thumbFor(SegmentEntry e) {
    final b64 = e.thumbB64;
    if (b64 == null) return null;
    final cached = _thumbs[e.maskId];
    if (cached != null && cached.b64 == b64) return cached.bytes;
    try {
      final bytes = base64Decode(b64);
      _thumbs[e.maskId] = (b64: b64, bytes: bytes);
      return bytes;
    } catch (_) {
      return null;
    }
  }

  String _text(String id) => _controllers[id]?.text.trim() ?? '';
  bool _dirty(String id) => _text(id) != (_stored[id] ?? '').trim();

  /// A captioned row whose field was emptied: Save deletes its caption.
  bool _isDelete(String id) =>
      widget.onClear != null && (_wasCaptioned[id] ?? false) && _dirty(id) && _text(id).isEmpty;

  /// Edited rows with something to save. An emptied caption on a captioned row
  /// is a delete; any other blank edit is not saveable (the backend refuses an
  /// empty caption).
  List<String> _pending() => [
        for (final id in _controllers.keys)
          if (_isDelete(id) ||
              (_text(id).isNotEmpty && (_dirty(id) || (_approved.contains(id) && (_hasDraft[id] ?? false)))))
            id,
      ];

  int get _editedCount => _controllers.keys.where(_dirty).length;

  Future<void> _saveAll() async {
    final save = widget.onSave;
    if (save == null || _saving) return;
    final ids = _pending();
    if (ids.isEmpty) return;
    setState(() {
      _saving = true;
      _summary = null;
      for (final id in ids) {
        _errors.remove(id);
      }
    });
    var failed = 0;
    for (final id in ids) {
      final text = _text(id);
      final deleting = _isDelete(id);
      final error = deleting ? await widget.onClear!([id]) : await save(id, text);
      if (!mounted) return;
      setState(() {
        if (error != null) {
          failed++;
          _errors[id] = error;
        } else {
          _stored[id] = text;
          _controllers[id]?.text = text;
          _approved.remove(id);
        }
      });
    }
    String? sessionError;
    if (failed == 0 && widget.onSaveSession != null) {
      sessionError = await widget.onSaveSession!();
      if (!mounted) return;
    }
    setState(() {
      _saving = false;
      if (failed > 0) {
        _summaryIsError = true;
        _summary = '$failed of ${ids.length} captions could not be saved; nothing was written to disk. '
            'Fix the rows marked in red and press Save again.';
      } else if (sessionError != null) {
        _summaryIsError = true;
        _summary = 'Captions updated, but saving the session failed: $sessionError';
      } else {
        _summaryIsError = false;
        _summary = widget.onSaveSession == null
            ? 'Updated ${ids.length} caption${ids.length == 1 ? '' : 's'}. Use Save to write the session.'
            : 'Saved ${ids.length} caption${ids.length == 1 ? '' : 's'} and the session.';
      }
    });
  }

  void _revertAll() {
    setState(() {
      for (final id in _controllers.keys) {
        _controllers[id]!.text = _stored[id] ?? '';
      }
      _approved.clear();
      _errors.clear();
      _summary = null;
    });
  }

  /// Tick or untick "never needs" on [ids]. Applies at once (it is a status, not
  /// text), through the backend, which refuses a captioned segment.
  Future<void> _setNever(List<String> ids, bool never) async {
    final cb = widget.onNever;
    if (cb == null || ids.isEmpty || _saving) return;
    final error = await cb(ids, never);
    if (!mounted) return;
    setState(() {
      _summaryIsError = error != null;
      _summary = error;
      if (error == null) _lastNever = (ids: List<String>.of(ids), never: never);
    });
  }

  /// Reverse the last Never change: untick what was ticked, tick what was unticked.
  Future<void> _undoNever() async {
    final last = _lastNever;
    if (last == null) return;
    await _setNever(last.ids, !last.never);
    if (mounted && _summary == null) setState(() => _lastNever = null);
  }

  /// Delete the caption of [id]: the segment leaves Captioned and returns to
  /// Not captioned. Applies at once, like Never.
  Future<void> _clearCaption(String id) async {
    final cb = widget.onClear;
    if (cb == null || _saving) return;
    final error = await cb([id]);
    if (!mounted) return;
    setState(() {
      _summaryIsError = error != null;
      _summary = error;
      if (error == null) {
        // Drop any unsaved edit: the field now belongs to an uncaptioned segment.
        _controllers[id]?.text = '';
        _stored[id] = '';
        _errors.remove(id);
        _approved.remove(id);
      }
    });
  }

  void _toggleApprove(String id) => setState(() {
        if (!_approved.remove(id)) _approved.add(id);
      });

  /// Tick every draft in the current view as right as written.
  void _approveAllDrafts(Iterable<SegmentEntry> rows) => setState(() {
        for (final e in rows) {
          if (e.needsReview) _approved.add(e.maskId);
        }
      });

  void _enlarge(SegmentEntry e, Uint8List bytes) {
    showDialog<void>(
      context: context,
      builder: (ctx) => Dialog(
        child: GestureDetector(
          onTap: () => Navigator.of(ctx).pop(),
          child: Container(
            color: const Color(0xFFEDEDED),
            padding: const EdgeInsets.all(16),
            child: Image.memory(bytes, key: ValueKey('caption-enlarged-${e.maskId}'), fit: BoxFit.contain),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final all = segmentEntriesFrom(widget.aaPreviewData);
    final captionedCount = all.where((e) => e.captioned).length;
    final uncaptionedCount = all.where((e) => !e.captioned && !e.never).length;
    final reviewCount = all.where((e) => e.needsReview).length;
    final neverCount = all.where((e) => e.never).length;
    final filter = _filter ??
        (reviewCount > 0
            ? _Filter.review
            : (captionedCount > 0 || uncaptionedCount == 0 ? _Filter.captioned : _Filter.notCaptioned));
    final visible = switch (filter) {
      _Filter.notCaptioned => all.where((e) => !e.captioned && !e.never).toList(),
      _Filter.review => all.where((e) => e.needsReview).toList(),
      _Filter.captioned => all.where((e) => e.captioned).toList(),
      _Filter.never => all.where((e) => e.never).toList(),
      _Filter.all => all,
    };
    final pending = _pending().length;
    final edited = _editedCount;

    if (all.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            'No segments in this session yet.',
            key: const ValueKey('captions-empty'),
            style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
          ),
        ),
      );
    }

    return LayoutBuilder(builder: (context, box) {
      final wide = box.maxWidth >= 640;
      return Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 6),
            child: Wrap(
              spacing: 12,
              runSpacing: 8,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                SegmentedButton<_Filter>(
                  key: const ValueKey('captions-filter'),
                  showSelectedIcon: false,
                  segments: [
                    if (reviewCount > 0 || filter == _Filter.review)
                      ButtonSegment(value: _Filter.review, label: Text('Needs review ($reviewCount)')),
                    ButtonSegment(value: _Filter.notCaptioned, label: Text('Not captioned ($uncaptionedCount)')),
                    ButtonSegment(value: _Filter.captioned, label: Text('Captioned ($captionedCount)')),
                    if (widget.onNever != null && (neverCount > 0 || filter == _Filter.never))
                      ButtonSegment(value: _Filter.never, label: Text('Never needs ($neverCount)')),
                    ButtonSegment(value: _Filter.all, label: Text('All segments (${all.length})')),
                  ],
                  selected: {filter},
                  onSelectionChanged: (v) => setState(() => _filter = v.first),
                ),
                if (reviewCount > 0)
                  OutlinedButton(
                    key: const ValueKey('captions-approve-all'),
                    onPressed: _saving ? null : () => _approveAllDrafts(visible),
                    child: const Text('Approve drafts shown'),
                  ),
                FilledButton(
                  key: const ValueKey('captions-save'),
                  onPressed: (pending > 0 && !_saving && widget.onSave != null) ? _saveAll : null,
                  child: Text(_saving ? 'Saving…' : (pending > 0 ? 'Save $pending change${pending == 1 ? '' : 's'}' : 'Save')),
                ),
                if (_lastNever != null)
                  TextButton(
                    key: const ValueKey('captions-never-undo'),
                    onPressed: _saving ? null : _undoNever,
                    child: Text(_lastNever!.never
                        ? 'Undo: marked ${_lastNever!.ids.length} Never'
                        : 'Undo: cleared ${_lastNever!.ids.length} Never'),
                  ),
                TextButton(
                  key: const ValueKey('captions-revert'),
                  onPressed: (edited > 0 && !_saving) ? _revertAll : null,
                  child: const Text('Revert'),
                ),
              ],
            ),
          ),
          if (_summary != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 6),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  _summary!,
                  key: const ValueKey('captions-summary'),
                  style: theme.textTheme.labelMedium?.copyWith(
                    color: _summaryIsError ? theme.colorScheme.error : theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
            ),
          _header(theme, wide, visible),
          const Divider(height: 1),
          Expanded(
            child: visible.isEmpty
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.all(24),
                      child: Text(
                        'Nothing to show here. A segment appears under Captioned once it has a caption, '
                        'and returns to Not captioned if you delete the caption.',
                        key: const ValueKey('captions-none-captioned'),
                        textAlign: TextAlign.center,
                        style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                      ),
                    ),
                  )
                : ListView.separated(
                    itemCount: visible.length,
                    separatorBuilder: (_, __) => const Divider(height: 1),
                    itemBuilder: (context, i) => _row(context, visible[i], wide),
                  ),
          ),
        ],
      );
    });
  }

  static const double _neverCol = 88;
  static const double _imageCol = 112;
  static const double _passCol = 52;
  static const double _tagCol = 120;
  static const double _statusCol = 92;

  Widget _header(ThemeData theme, bool wide, List<SegmentEntry> visible) {
    final style = theme.textTheme.labelMedium?.copyWith(fontWeight: FontWeight.bold);
    // The header box acts on the rows shown that can take the mark (a captioned
    // segment cannot). Ticked when all are, empty when none, dash in between.
    final eligible = visible.where((e) => !e.captioned).toList();
    final ids = [for (final e in eligible) e.maskId];
    final marked = eligible.where((e) => e.never).length;
    final headerNever = eligible.isEmpty
        ? null
        : (ids: ids, value: marked == 0 ? false : (marked == eligible.length ? true : null as bool?));
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(
        children: [
          Expanded(child: Text('Caption', style: style)),
          SizedBox(width: _imageCol, child: Text('Segment', style: style)),
          SizedBox(width: _passCol, child: Text('Pass', style: style)),
          if (wide) SizedBox(width: _tagCol, child: Text('Tag', style: style)),
          if (wide) SizedBox(width: _statusCol, child: Text('Status', style: style)),
          if (widget.onNever != null)
            SizedBox(
              width: _neverCol,
              child: Row(children: [
                Tooltip(
                  message: 'Never needs: scrubbed or otherwise never to be captioned. '
                      'The box in this header ticks every segment shown.',
                  child: Text('Never', style: style),
                ),
                if (headerNever != null)
                  SizedBox(
                    width: 28,
                    child: Checkbox(
                      key: const ValueKey('captions-never-all'),
                      tristate: true,
                      visualDensity: VisualDensity.compact,
                      value: headerNever.value,
                      onChanged: _saving
                          ? null
                          : (v) => _setNever(headerNever.ids, v == true),
                    ),
                  ),
              ]),
            ),
        ],
      ),
    );
  }

  Widget _row(BuildContext context, SegmentEntry e, bool wide) {
    final theme = Theme.of(context);
    final id = e.maskId;
    final controller = _controllers[id]!;
    final dirty = _dirty(id);
    final deleting = _isDelete(id);
    final blankEdit = dirty && controller.text.trim().isEmpty && !deleting;
    final thumb = _thumbFor(e);
    final error = _errors[id] ??
        (blankEdit ? 'a caption cannot be empty' : null);
    final muted = theme.textTheme.labelSmall?.copyWith(color: theme.colorScheme.onSurfaceVariant);

    return Container(
      key: ValueKey('caption-row-$id'),
      color: e.never
          ? theme.colorScheme.onSurface.withValues(alpha: 0.04)
          : ((dirty || _approved.contains(id)) ? theme.colorScheme.primary.withValues(alpha: 0.06) : null),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(right: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  TextField(
                    key: ValueKey('caption-field-$id'),
                    controller: controller,
                    enabled: !_saving && !e.never,
                    minLines: 2,
                    maxLines: 5,
                    onChanged: (_) => setState(() {}),
                    decoration: InputDecoration(
                      isDense: true,
                      border: const OutlineInputBorder(),
                      hintText: e.captioned ? null : (e.never ? 'never needs a caption' : 'no caption yet'),
                      errorText: error,
                      helperText: deleting ? 'Save will delete this caption' : null,
                      errorMaxLines: 3,
                    ),
                  ),
                  if (e.captioned && widget.onClear != null)
                    Align(
                      alignment: Alignment.centerLeft,
                      child: TextButton.icon(
                        key: ValueKey('caption-delete-$id'),
                        onPressed: _saving ? null : () => _clearCaption(id),
                        icon: const Icon(Icons.delete_outline, size: 16),
                        label: const Text('Delete caption'),
                      ),
                    ),
                  if (e.needsReview)
                    Align(
                      alignment: Alignment.centerLeft,
                      child: TextButton.icon(
                        key: ValueKey('caption-approve-$id'),
                        onPressed: _saving ? null : () => _toggleApprove(id),
                        icon: Icon(
                          _approved.contains(id) ? Icons.check_circle : Icons.check_circle_outline,
                          size: 16,
                        ),
                        label: Text(_approved.contains(id) ? 'Approved' : 'Approve as written'),
                      ),
                    ),
                ],
              ),
            ),
          ),
          SizedBox(
            width: _imageCol,
            child: Align(
              alignment: Alignment.centerLeft,
              child: GestureDetector(
                onTap: thumb == null ? null : () => _enlarge(e, thumb),
                child: Container(
                  width: 96,
                  height: 96,
                  decoration: BoxDecoration(
                    // A light backing: the object is a transparent PNG, often dark ink.
                    color: const Color(0xFFEDEDED),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: thumb == null
                      ? const Icon(Icons.image_not_supported_outlined, color: Colors.black38)
                      : Image.memory(thumb, key: ValueKey('caption-thumb-$id'), fit: BoxFit.contain),
                ),
              ),
            ),
          ),
          SizedBox(width: _passCol, child: Text('${e.pass}', style: muted)),
          if (wide)
            SizedBox(
              width: _tagCol,
              child: Text(e.tag ?? '', style: muted, maxLines: 3, overflow: TextOverflow.ellipsis),
            ),
          if (wide)
            SizedBox(
              width: _statusCol,
              child: Text(
                deleting ? 'will delete' : dirty ? 'edited' : (_approved.contains(id) ? 'approved (unsaved)' : e.reviewState),
                key: ValueKey('caption-status-$id'),
                style: muted,
              ),
            ),
          if (widget.onNever != null)
            SizedBox(
              width: _neverCol,
              child: Align(
                alignment: Alignment.topLeft,
                child: Checkbox(
                  key: ValueKey('caption-never-$id'),
                  value: e.never,
                  // A captioned segment is training data; clear its caption first.
                  onChanged: (e.captioned || _saving) ? null : (v) => _setNever([id], v == true),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
