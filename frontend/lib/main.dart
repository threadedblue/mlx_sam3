// DROP-IN replacement for your file (Flutter Web-safe: no dart:io, no File/Image.file)
//
// What changed vs your version:
// - Refactored the main canvas into a layered architecture using a new `LayeredSegmentationCanvas` widget.
// - The old `SegmentationCanvas` is replaced with a new version that composes the layers and interaction controls.
// - The old `SegmentationPainter` is removed and its logic is split into `_updateSegmentsFromResult` (for data) and a new `PromptPainter` (for display).
// - State management is updated to handle `ui.Image` and `List<Segment>` for the new canvas.
// - The "Segment Layers" card now includes a toggle for the "Original" image layer.
//
// If your current ApiService only accepts File, update it to accept bytes, or create an overload.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:http/http.dart' as http;

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:aa_preview_table/aa_preview_table.dart';

import 'services/api_service.dart';
import 'services/sf_aa_preview_adapter.dart';
import 'segment_layers_card.dart';
import 'package:provider/provider.dart';
import 'layered_segmentation_canvas.dart';
import 'layer_state.dart';
import 'models/result_datum.dart';
import 'widgets/result_cell.dart';
import 'widgets/balloon_scrub_card.dart';
import 'widgets/panels_card.dart';
import 'widgets/box_select_card.dart';
import 'widgets/lbs_card.dart';
import 'widgets/objects_selected_card.dart';
import 'widgets/point_select_card.dart';
import 'widgets/prompt_card.dart';
import 'widgets/save_card.dart';
import 'launch_config.dart';
import 'sf_mode.dart';

void main() {
  runApp(const SamApp());
}

/// Which card currently owns canvas interaction.
///
/// The radio buttons on the Prompt, Box Select, Point Select and Panels cards
/// form one mutually exclusive group over this enum; `null` means no mode is
/// active and the canvas ignores taps and drags.
///
/// [panel] is the odd one out on purpose: its drag authors a LAYOUT REGION
/// (v4 §4) via /panel/add and never reaches SAM3, so it creates no mask and
/// nothing in Objects Selected. It is in this enum only because canvas
/// interaction is genuinely exclusive — a drag cannot mean both "box prompt"
/// and "panel bounds".
enum SelectionMode { prompt, box, point, panel }

/// Whether the Prompt card's submit button should be enabled — shared by
/// both of its faces: "Select" (a standalone text-prompt search) and "Save
/// Caption" (committing a caption to the already-focused mask).
///
/// sf-display-and-workflow-v3-spec.md §4: **Select** is disabled while Box,
/// Point or Panel mode is active — mode-exclusive, not a permanent disable, since
/// Select is specifically the trigger for a standalone text-prompt search
/// and isn't meant to run alongside box/point. **Save Caption is
/// unaffected by this**: it commits a caption to whatever mask box/point
/// selection just focused, which is a normal step immediately after a
/// box/point selection, not a competing search — disabling it on the same
/// condition would break that workflow rather than fix anything §4 asks
/// for.
bool promptCardCanSubmit({
  required bool hasSession,
  required bool promptEmpty,
  required bool isLoading,
  required bool captionMode,
  required SelectionMode? selectedMode,
}) {
  if (!hasSession || promptEmpty || isLoading) return false;
  if (captionMode) return true;
  return selectedMode != SelectionMode.box &&
      selectedMode != SelectionMode.point &&
      selectedMode != SelectionMode.panel;
}

/// A Target/Avoid prompt with nothing recorded (`mask_ids` empty) draws its
/// marker exactly like one that found something — a small dot, accumulating
/// with every click — which is easy to mistake for "a mask should have
/// appeared here" (the investigation's Case F/G: three Avoid clicks with no
/// Target, landing on unintended objects, giving no feedback that nothing
/// was ever selected). This distinguishes the two in the same status line
/// the "Done" label already occupies, rather than adding a new UI element.
String promptStatusLabel(Map<String, dynamic>? result) {
  final maskIds = result?['mask_ids'] as List?;
  return (maskIds == null || maskIds.isEmpty) ? '0 objects found' : 'Done';
}

/// Box/point status, aware of what THIS call itself found — not just
/// whether the current pass has ever selected anything ([promptStatusLabel]
/// alone).
///
/// sf-display-and-workflow-v3-spec.md §3 Fix 2: a positive click that finds
/// no containing instance no longer silently falls back to an unrelated
/// object — the backend clears `selected_mask_id` to null specifically for
/// that case (see sf_engine.py's `_select_single`). So on a POSITIVE click,
/// a null `selected_mask_id` means this click itself found nothing, even
/// though other selections may already exist from earlier in the pass. On
/// a NEGATIVE (Avoid) click `selected_mask_id` is left untouched by design
/// (an Avoid call doesn't set focus — see `_select_single`'s own comment),
/// so it carries no such meaning there and [promptStatusLabel]'s original
/// whole-pass check is what applies.
String positiveClickStatusLabel(Map<String, dynamic> response, bool isPositive) {
  if (isPositive && response['selected_mask_id'] == null) {
    return '0 objects found';
  }
  return promptStatusLabel(response['results'] as Map<String, dynamic>?);
}

/// The Image URL field's display value from a `/loadSession` response —
/// `session['image_url']`, sourced from registry.parquet's `image_url`
/// column (services.py's `load_session_from_disk`), trimmed, with an
/// absent/empty/whitespace-only value normalized to null (the field's own
/// "(no image URL)" placeholder condition) rather than shown as an empty
/// string.
String? sessionImageUrlFrom(Map<String, dynamic> session) {
  final url = (session['image_url'] as String?)?.trim();
  return (url != null && url.isNotEmpty) ? url : null;
}

class SamApp extends StatelessWidget {
  const SamApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (context) => LayerState()),
      ],
      child: MaterialApp(
        title: 'SegForge Studio',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.indigo,
            brightness: Brightness.light,
          ),
          useMaterial3: true,
          cardTheme: const CardThemeData(elevation: 2, margin: EdgeInsets.zero),
        ),
        darkTheme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.indigo,
            brightness: Brightness.dark,
          ),
          useMaterial3: true,
          cardTheme: const CardThemeData(elevation: 2, margin: EdgeInsets.zero),
        ),
        themeMode: ThemeMode.dark,
        // The picker is purely the fallback for "nothing told SF what to do
        // yet" — launched with args (session id and/or image url), DN's Open
        // Forge flow always supplies at least one, so this only fires on a
        // bare standalone launch.
        home: (LaunchConfig.sessionId == null && LaunchConfig.imageUrl == null)
            ? const SessionPickerScreen()
            : const HomeScreen(),
      ),
    );
  }
}

/// Standalone session open/create screen, shown only when SF is launched
/// with no session id or image URL at all (see [SamApp.build]). Lists
/// sessions SF has actually saved (reading the same registry.parquet-backed
/// listing DoubleNaught's picklist reads) and offers "+ New" — nothing is
/// written to disk for a new session until its first Save, same rule as
/// everywhere else in this flow.
class SessionPickerScreen extends StatefulWidget {
  const SessionPickerScreen({super.key});

  @override
  State<SessionPickerScreen> createState() => _SessionPickerScreenState();
}

class _SessionPickerScreenState extends State<SessionPickerScreen> {
  final ApiService _api = ApiService();
  List<Map<String, dynamic>> _sessions = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final sessions = await _api.listSavedSessions();
      if (!mounted) return;
      setState(() {
        _sessions = sessions;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _openExisting(String sessionId) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => HomeScreen(initialSessionId: sessionId)),
    );
  }

  Future<void> _createNew() async {
    final nameCtrl = TextEditingController();
    final descCtrl = TextEditingController();

    final create = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('New SegForge Session'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: nameCtrl,
              decoration: const InputDecoration(labelText: 'Name'),
            ),
            TextField(
              controller: descCtrl,
              decoration: const InputDecoration(labelText: 'Description'),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Create'),
          ),
        ],
      ),
    );
    if (create != true || !mounted) return;

    try {
      final sessionId = await _api.newSession();
      if (sessionId == null) return;
      // Same convention DN's own "+ New Session" form uses: metadata is
      // registered on the backend, but nothing is written to
      // storage/sf/sessions/ until the first explicit Save.
      await _api.initSessionMetadata(
        sessionId: sessionId,
        name: nameCtrl.text.trim(),
        description: descCtrl.text.trim(),
      );
      if (!mounted) return;
      _openExisting(sessionId);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Centered title at top
          Padding(
            padding: const EdgeInsets.only(top: 32, bottom: 24),
            child: Text(
              'SegForge Sessions',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
          ),
          // Error banner
          if (_error != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Text(
                _error!,
                style: const TextStyle(color: Colors.red, fontSize: 12),
              ),
            ),
          // Session list or loading/empty state
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : _sessions.isEmpty
                    ? const Center(child: Text('No saved sessions yet.'))
                    : ListView.builder(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        itemCount: _sessions.length,
                        itemBuilder: (context, i) {
                          final s = _sessions[i];
                          final name = (s['name'] as String?)?.trim() ?? '';
                          final description = (s['description'] as String?)?.trim() ?? '';
                          final sessionId = s['session_id'] as String;

                          final displayName = name.isNotEmpty ? name : 'Untitled session';
                          final displayDesc = description.isNotEmpty ? description : 'No description';

                          return Padding(
                            padding: const EdgeInsets.only(bottom: 12),
                            child: GestureDetector(
                              onTap: () => _openExisting(sessionId),
                              child: Container(
                                decoration: BoxDecoration(
                                  border: Border.all(
                                    color: Colors.white,
                                    width: 1.5,
                                  ),
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                child: Padding(
                                  padding: const EdgeInsets.all(16),
                                  child: Row(
                                    children: [
                                      Expanded(
                                        child: Column(
                                          crossAxisAlignment: CrossAxisAlignment.start,
                                          children: [
                                            // Name (prominent)
                                            Text(
                                              displayName,
                                              style: Theme.of(context)
                                                  .textTheme
                                                  .titleMedium
                                                  ?.copyWith(
                                                    fontWeight: FontWeight.w600,
                                                  ),
                                            ),
                                            const SizedBox(height: 4),
                                            // Description (smaller, dimmer)
                                            Text(
                                              displayDesc,
                                              style: Theme.of(context)
                                                  .textTheme
                                                  .bodySmall
                                                  ?.copyWith(
                                                    color: Colors.grey[400],
                                                  ),
                                            ),
                                            const SizedBox(height: 8),
                                            // Session ID (faint)
                                            Text(
                                              sessionId,
                                              style: Theme.of(context)
                                                  .textTheme
                                                  .labelSmall
                                                  ?.copyWith(
                                                    color: Colors.grey[500],
                                                  ),
                                              maxLines: 1,
                                              overflow: TextOverflow.ellipsis,
                                            ),
                                          ],
                                        ),
                                      ),
                                      // Chevron (right-aligned)
                                      const Padding(
                                        padding: EdgeInsets.only(left: 12),
                                        child: Icon(
                                          Icons.chevron_right,
                                          color: Colors.white,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          );
                        },
                      ),
          ),
          // "+ New Session" button at bottom
          Padding(
            padding: const EdgeInsets.all(16),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _createNew,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                ),
                child: const Text('+ New Session'),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class HomeScreen extends StatefulWidget {
  /// Overrides [LaunchConfig.sessionId] — set when [SessionPickerScreen]
  /// hands off a chosen or newly created session. Null means fall back to
  /// the launch-time value, the ordinary (DN-launched) path.
  final String? initialSessionId;

  const HomeScreen({super.key, this.initialSessionId});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService _api = ApiService();
  final TextEditingController _textController = TextEditingController();

  // State (supplied at launch, not user-entered)
  String? _sessionId;
  String? _imageUrl; // Supplied at launch, non-editable

  /// The in-flight (or finished) launch-session load. The auto image download
  /// waits for it: a saved session brings its own bytes, and racing it with a
  /// download of the (possibly stale) launch URL produced a spurious 404.
  Future<void>? _launchSessionLoad;

  /// Whether SF was launched standalone (no launch args). Determines whether
  /// the Image URL field shows a file picker affordance.
  late final bool _isStandalone;
  String? _pickedFilePath; // For standalone mode: display the picked file path

  /// Session metadata, read back from the backend at startup. Whoever created
  /// the session named it (DoubleNaught's Seg Forge node does, over
  /// `/initSession`), and that name is more use to the reader than the id.
  String? _sessionName;
  String? _sessionDescription;

  Uint8List? _imageBytes; // Web-safe image data (current pass's working image)
  ui.Image? _uiImage; // Decoded image for canvas (current pass's working image)
  Size? _imageSize; // Current pass's image size

  /// Pass 0 — the true original image, retained separately from
  /// [_imageBytes]/[_uiImage]/[_imageSize] above so it survives a scrub.
  /// Set ONCE, at the same two points those fields are first set (launch
  /// session load, upload completion) — see [_loadLaunchSession] and
  /// [_loadImageFromUrl] — and never touched anywhere else, specifically
  /// not in [_scrubLamaBackground]. Confirmed live: before this slot
  /// existed, `_imageBytes`/`_uiImage` was a single slot unconditionally
  /// overwritten on every scrub, so "Original" silently became whatever
  /// the current pass was for any session with a scrub behind it.
  Uint8List? _originalBytes;
  ui.Image? _originalUiImage;
  Size? _originalImageSize;

  Map<String, dynamic>? _result;
  // Session-wide mask ledger (every pass, not just the current one) for
  // the AA Preview tab — see backend services.py's
  // serialize_sf_masks_all_passes docstring for why this has to be a
  // SEPARATE field from `_result`: `_result` is current-pass-only
  // (matches what's paintable on the canvas right now), so a captioned/
  // kept mask silently vanished from the AA Preview the moment its pass
  // was scrubbed past, when `_result` reset to null — confirmed live, the
  // exact case reported ("I did the scrub. AA preview shows nothing.").
  Map<String, dynamic>? _aaPreviewData;
  List<Segment> _segments = [];
  // Dirty tracking for the "Unsaved Changes" prompt (_hasUnsavedWork
  // below). _mutationCount increments once per successful call to every
  // handler that changes session state /saveSession would persist —
  // text/box/point selection, hold/include toggle, caption commit, scrub,
  // Clear Prompts. _savedAtMutationCount is _mutationCount's value as of
  // the last successful save; work is unsaved iff they differ. NOT a plain
  // bool: a failed save must leave existing dirt in place, and a second
  // mutation after a save must not be silently absorbed by a bool that's
  // already true. View-layer toggles (_saveLayerState) deliberately don't
  // touch this — they persist themselves immediately on every change, so
  // there's nothing pending to lose.
  int _mutationCount = 0;
  int _savedAtMutationCount = 0;
  bool _isLoading = false;
  String? _error;
  String _backendStatus = "checking";
  SelectionMode? _selectedMode;
  String _boxMode = "positive"; // "positive" or "negative"
  String _pointMode = "positive"; // "positive" or "negative"

  /// Guards the one-shot auto-load of the launch-supplied image URL.
  bool _autoLoadStarted = false;

  // Per-card result state
  bool _imageSourceRunning = false;
  bool _textPromptRunning = false;
  List<ResultDatum> _textPromptResult = [];
  bool _boxRunning = false;
  List<ResultDatum> _boxResult = [];
  bool _pointRunning = false;
  List<ResultDatum> _pointResult = [];
  bool _isScrubbing = false;
  /// Which pass a balloon scrub produced, from the backend's
  /// `wb_scrub_pass`. Null until one has run — which is also exactly
  /// what a zero-balloon page looks like, so the WB-scrub layer stays
  /// absent rather than mislabeling an ordinary pass.
  int? _wbScrubPass;
  /// Manually authored panel bounds (v4 §4), in authoring order — raw maps
  /// straight from the backend (`panel_id`, `x0`, `y0`, `x1`, `y1`).
  ///
  /// Deliberately separate from `_result`/`_segments`: a panel is a layout
  /// region, never a mask, so it must not appear in Objects Selected and
  /// must not affect `hasSelection`/the Scrub button. Keeping it in its own
  /// field is what makes that structural rather than a filter someone can
  /// later forget to apply.
  List<Map<String, dynamic>> _panels = const [];
  /// Which toolset to show. Seeded from SEGFORGE_MODE at launch and
  /// replaced by the session's OWN stored mode on resume, since DN sends
  /// nothing on a resume. Null means legacy/unfiltered — never Comix.
  SfMode? _mode = SfMode.parse(LaunchConfig.mode);
  bool _isDetectingBalloons = false;
  /// Null until the first detection run this session — distinct from 0.
  int? _lastDetectedCount;
  String? _lastScrubLabel;

  /// Whether a scrub exists to step back from, and the last undo's refusal
  /// reason (shown under the Undo button).
  bool _canUndo = false;
  bool _isUndoing = false;
  String? _undoMessage;
  /// True once [_loadLaunchSession] has successfully loaded a session from
  /// the backend (not just attempted — a session with nothing stored yet
  /// leaves this false, same as a brand-new upload). [BalloonScrubCard]
  /// needs this: a resumed session with `_wbScrubPass == null` genuinely
  /// cannot tell "balloon scrub never ran" from "ran and found nothing" (
  /// `MaskRecord.source` isn't persisted — see load_session_from_disk's own
  /// docstring), so it must show a distinct "Not run in this session" text
  /// rather than the fresh-session "No detection run yet" default, which
  /// would otherwise read as "never looked" when it might have.
  bool _isResumedSession = false;

  /// The mask a box/point selection (or the scrub-batch checkbox list)
  /// most recently touched — what the Prompt card's caption mode and
  /// [_setMaskHeld] act on. Cleared on scrub (the pass changes; whatever
  /// was focused no longer has a live geometry in the new pass's result
  /// set).
  String? _focusedMaskId;

  /// Guards the caption-draft pre-fill (templating v1) against a response
  /// that arrives after focus has already moved on. Bumped on every request;
  /// a reply whose generation no longer matches is dropped.
  int _draftGen = 0;
  /// The mask a pre-fill has already been requested for, so re-rendering or
  /// a second hold toggle on the SAME mask does not re-request or stomp on
  /// text the user has since typed.
  String? _draftRequestedFor;

  Segment? get _focusedSegment {
    final id = _focusedMaskId;
    if (id == null) return null;
    for (final s in _segments) {
      if (s.maskId == id) return s;
    }
    return null;
  }

  /// design doc's exact trigger: held AND still unassigned. A captioned
  /// (`keep`) mask stays in grounding mode even if re-focused — captioning
  /// again would just overwrite the existing caption via the same call,
  /// which isn't the flow this mode is for.
  bool get _isCaptionMode {
    final seg = _focusedSegment;
    return seg != null && seg.held && seg.datasetStatus == 'unassigned';
  }
  bool _resultsRunning = false;
  List<ResultDatum> _resultsResult = [];
  Timer? _healthCheckTimer;

  // Layer State
  LayerState? _layerState;

  @override
  void initState() {
    super.initState();
    // Session id and image URL are launch-time inputs, not user-entered.
    // A null session id is fine: /upload allocates one and returns it.
    _sessionId = widget.initialSessionId ?? LaunchConfig.sessionId;
    _imageUrl = LaunchConfig.imageUrl;
    // Standalone mode: no launch image URL and either no session ID or session
    // came from the picker (initialSessionId != null). This gates the file
    // picker affordance on the Image URL field.
    _isStandalone = _imageUrl == null && (LaunchConfig.sessionId == null || widget.initialSessionId != null);
    // Whatever the backend already holds for this session goes on screen at
    // once — it needs no model, so the window is never blank while SAM loads.
    _launchSessionLoad = _loadLaunchSession();
    // The Select button is disabled while the prompt is empty, so the field has
    // to trigger a rebuild as it is typed into.
    _textController.addListener(_onPromptTextChanged);
    _checkHealth();
    _healthCheckTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => _checkHealth(),
    );
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final newLayerState = Provider.of<LayerState>(context, listen: false);
    if (_layerState != newLayerState) {
      _layerState?.removeListener(_onLayerStateChanged);
      _layerState = newLayerState;
      _layerState?.addListener(_onLayerStateChanged);
    }
  }

  void _onLayerStateChanged() {
    _saveLayerState();
  }

  void _onPromptTextChanged() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _healthCheckTimer?.cancel();
    _textController.removeListener(_onPromptTextChanged);
    _textController.dispose();
    _layerState?.removeListener(_onLayerStateChanged);
    super.dispose();
  }


  Future<void> _checkHealth() async {
    try {
      final health = await _api.checkHealth();
      if (!mounted) return;
      setState(() {
        _backendStatus = health['model_loaded'] == true ? "online" : "offline";
      });
    } catch (e) {
      debugPrint("Health check error: $e");
      if (!mounted) return;
      setState(() {
        _backendStatus = "offline";
        _error ??= "Health check failed: $e";
      });
    }
    _maybeAutoLoadImage();
  }

  /// Puts a launch-supplied session on screen: its name, its description and
  /// its image, straight from the backend.
  ///
  /// Deliberately independent of the model: `/loadSession` reads the
  /// persisted AAs directly, so none of this waits on SAM — but it ALSO now
  /// fully restores the backend's in-memory session and SFSession as a side
  /// effect (services.py's `load_session_from_disk`), so by the time this
  /// returns, `/segment/*` already has everything it needs for this
  /// session. [_maybeAutoLoadImage] still fires afterward, but for a
  /// resumed session (bytes already in hand here) it no longer re-uploads —
  /// see `_loadImageFromUrl`'s early return — since doing so used to rebuild
  /// the session from scratch and silently discard what this just restored.
  Future<void> _loadLaunchSession() async {
    final sessionId = _sessionId;
    if (sessionId == null || sessionId.isEmpty) return;

    final Map<String, dynamic>? loaded;
    try {
      loaded = await _api.loadSession(sessionId);
    } catch (e) {
      // A session with nothing stored yet is ordinary, not an error worth
      // showing: the URL auto-load below is the other way in.
      debugPrint('Could not load launch session $sessionId: $e');
      return;
    }
    if (loaded == null || !mounted) return;
    final session = loaded;

    final b64 = session['image_b64'] as String?;
    Uint8List? bytes;
    ui.Image? decoded;
    if (b64 != null && b64.isNotEmpty) {
      try {
        bytes = base64Decode(b64);
        decoded = await _decodeImage(bytes);
      } catch (e) {
        debugPrint('Could not decode session image: $e');
        bytes = null;
        decoded = null;
      }
    }
    if (!mounted) return;

    // Wiring gap (v3): /loadSession's response has carried `image_url` from
    // registry.parquet all along (services.py's load_session_from_disk),
    // but nothing here ever read it — the Image URL field showed its
    // "(no image URL)" placeholder even for a session that genuinely has
    // one on disk, same shape as the earlier scrub/image_b64 and
    // caption-color wiring gaps.
    final sessionImageUrl = sessionImageUrlFrom(session);

    setState(() {
      _sessionName = (session['name'] as String?)?.trim();
      _sessionDescription = (session['description'] as String?)?.trim();
      if (sessionImageUrl != null) {
        _imageUrl = sessionImageUrl;
      }
      if (bytes != null && decoded != null) {
        _imageBytes = bytes;
        _uiImage = decoded;
        _imageSize = Size(
          (session['width'] as num?)?.toDouble() ?? decoded.width.toDouble(),
          (session['height'] as num?)?.toDouble() ?? decoded.height.toDouble(),
        );
        _originalBytes = bytes;
        _originalUiImage = decoded;
        _originalImageSize = _imageSize;
      }
      // Same field, same shape/purpose as every live /segment/*, /mask/*
      // and /lama/scrub response's `all_masks` — a resumed session's AA
      // Preview tab needs this too, not just a live in-progress one (see
      // load_session_from_disk's own docstring). Confirmed live: this was
      // present in the response all along but never read here, leaving
      // the tab in its "Empty associative array" state on every resume
      // regardless of how many masks the session actually has.
      _aaPreviewData = session['all_masks'] as Map<String, dynamic>?;
      // Same two lines every live /segment/*, /mask/* and /lama/scrub
      // response already uses to populate the canvas. Confirmed live: on
      // resume this was simply never called, so `_segments` stayed empty
      // forever — ObjectsSelectedCard showed no checkboxes and LBSCard's
      // `hasSelection` stayed false (Scrub permanently disabled) no matter
      // how many masks the session actually had. `results` is filtered to
      // the session's CURRENT pass (same as every live response) — a
      // session whose masks sit at an earlier pass than its current one
      // correctly gets nothing here rather than something misleading; that
      // is a real, separate resume-reconciliation gap, not fixed by this.
      _result = session['results'] as Map<String, dynamic>?;
      _updateSegmentsFromResult();
      _wbScrubPass = (session['wb_scrub_pass'] as num?)?.toInt();
      // Panel bounds (v4 §4) restored from the same /loadSession response
      // every other resumed field comes from. A session saved before panels
      // existed sends no `panels` key, which correctly reads as none.
      _panels = _panelsFrom(session);
      // A resumed session is authoritative about its own mode: DN only
      // supplies one when launching a NEW session. A session saved before
      // modes existed sends null here, which correctly leaves the sidebar
      // unfiltered rather than narrowing it.
      _mode = SfMode.parse(session['mode'] as String?);
      // Reaching this point means the backend actually returned a session
      // (the early-return catch above handles "nothing stored yet"), so
      // this IS a resume, independent of what it carries — a resumed
      // session with wb_scrub_pass == null still needs to be told apart
      // from a brand-new one (see BalloonScrubCard's text, _buildBalloon-
      // ScrubCard below).
      _isResumedSession = true;
      // /loadSession carries the session's current pass (services.py's
      // load_session_from_disk, bug-3 audit finding #3) — LBSCard showed
      // "No scrub yet" on every resumed session regardless of scrub
      // history, because _lastScrubLabel is otherwise only ever set by a
      // live scrub response. Same format as that live label
      // (_formatScrubLabel), so the two can't read differently for the
      // same transition. Only the LAST transition is ever shown, same as
      // live: a session scrubbed twice shows "Pass 1 -> Pass 2", not its
      // full history. At pass 0 there is no transition to show.
      final pass = (session['pass'] as num?)?.toInt() ?? 0;
      _lastScrubLabel = pass > 0 ? _formatScrubLabel(pass - 1, pass) : null;
      // Optimistic on resume (history is not persisted); the backend answers
      // honestly if there turns out to be nothing to undo.
      _canUndo = pass > 0;
      // A resumed session starts clean: everything it carries just came
      // FROM the last save, not from unsaved work in this run. Explicit
      // rather than relying on both counters defaulting to 0 on a fresh
      // State — this is the specific moment "clean" has to hold.
      _savedAtMutationCount = _mutationCount;
    });
    // Outside setState: _applyViewLayers mutates LayerState (a separate
    // ChangeNotifier), not this State's own fields, and already notifies
    // its own listeners (Provider's dependents rebuild on that, same as a
    // live checkbox toggle would). Detaches _onLayerStateChanged for its
    // duration so this restoration can never itself trigger a save.
    _applyViewLayers(session['view_layers'] as Map<String, dynamic>?);
  }

  /// Gets a genuinely new launch image registered with the backend once the
  /// model is ready — a resumed session (bytes already in hand from
  /// [_loadLaunchSession]) needs no registration here at all; see
  /// `_loadImageFromUrl`'s early return.
  ///
  /// /upload answers 503 until the model finishes loading, so this waits for
  /// the first "online" health check rather than firing from initState.
  void _maybeAutoLoadImage() {
    if (_autoLoadStarted) return;
    if (_backendStatus != "online") return;
    final haveBytes = _imageBytes != null;
    if (!haveBytes && (_imageUrl == null || _imageUrl!.isEmpty)) return;
    _autoLoadStarted = true;
    // A resumed session must finish loading first: once its bytes are in,
    // _loadImageFromUrl returns early and never touches the URL.
    (_launchSessionLoad ?? Future<void>.value()).whenComplete(() {
      if (mounted) _loadImageFromUrl();
    });
  }

  /// Open file picker to select an image file (standalone mode only).
  /// Converts the selected file path to a proper file:// URI and loads it through
  /// the same path as DN-launched images (_loadImageFromUrl), ensuring consistent
  /// behavior across both launch modes.
  Future<void> _pickImageFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: false,
    );
    if (result == null || result.files.isEmpty || !mounted) return;

    final file = result.files.single;
    final path = file.path;
    if (path == null || path.isEmpty) {
      debugPrint('File picker: no path available');
      setState(() => _error = 'Could not read file path');
      return;
    }

    // Convert bare filesystem path to proper file:// URI, with percent-encoding
    // for spaces and non-ASCII characters. This matches how DN supplies paths.
    final fileUri = Uri.file(path).toString();

    // Display the picked file name and load the image
    setState(() {
      _pickedFilePath = file.name;
      _imageUrl = fileUri;
      _error = null;
    });

    // Use the existing _loadImageFromUrl path, which already handles file:// URIs
    // correctly. This ensures standalone and DN-launched images flow through the
    // same code path.
    await _loadImageFromUrl();
  }

  Future<ui.Image> _decodeImage(Uint8List bytes) {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, (ui.Image img) {
      return completer.complete(img);
    });
    return completer.future;
  }

  void _updateSegmentsFromResult() {
    // This method is called inside setState(), so we update _segments directly.
    debugPrint('Updating segments from result...');
    if (_result == null) {
      _segments = [];
      return;
    }

    final List<Segment> newSegments = [];
    // Parallel to masks/boxes/scores (see services.serialize_sf_masks) —
    // looked up by index, defensively: absent/short arrays just leave a
    // segment's identity fields null rather than throwing.
    final maskIds = _result!['mask_ids'] as List?;
    final datasetStatuses = _result!['dataset_statuses'] as List?;
    final heldFlags = _result!['held_flags'] as List?;
    String? idAt(int i) => (maskIds != null && i < maskIds.length) ? maskIds[i] as String? : null;
    String? statusAt(int i) => (datasetStatuses != null && i < datasetStatuses.length) ? datasetStatuses[i] as String? : null;
    bool heldAt(int i) => (heldFlags != null && i < heldFlags.length) ? heldFlags[i] as bool : false;

    try {
      // 1. Try to load Masks (RLE)
      final masks = _result!['masks'] as List?;
      if (masks != null && masks.isNotEmpty && masks[0] is Map) {
        for (var i = 0; i < masks.length; i++) {
          try {
            final rle = masks[i] as Map;
            final counts = (rle['counts'] as List).cast<int>();
            final size = (rle['size'] as List).cast<int>(); // [H, W]
            final w = size[1];

            final path = Path();
            int p = 0;
            bool isForeground = false; // First run is always background (0) per backend logic

            for (final count in counts) {
              if (isForeground) {
                // Add rects for this run of 1s
                int start = p;
                int end = p + count;
                int curr = start;
                while (curr < end) {
                  int y = curr ~/ w;
                  int x = curr % w;
                  int endOfRow = (y + 1) * w;
                  int runEnd = (end < endOfRow) ? end : endOfRow;
                  int len = runEnd - curr;
                  path.addRect(Rect.fromLTWH(x.toDouble(), y.toDouble(), len.toDouble(), 1.0));
                  curr = runEnd;
                }
              }
              p += count;
              isForeground = !isForeground;
            }
            newSegments.add(Segment(
              path: path,
              maskId: idAt(i),
              datasetStatus: statusAt(i),
              held: heldAt(i),
            ));
          } catch (e, st) {
            debugPrint('Error parsing RLE mask: $e\n$st');
          }
        }
      }
      // 2. Fallback to Boxes if no RLE masks found
      else {
        final boxes = _result!['boxes'] as List? ?? _result!['masks'] as List?;
        if (boxes != null) {
          for (var i = 0; i < boxes.length; i++) {
            final maskData = boxes[i];
            if (maskData is List && maskData.length == 4) {
              final list = maskData.map((e) => (e as num).toDouble()).toList();
              final rect = Rect.fromLTRB(list[0], list[1], list[2], list[3]);
              final path = Path()..addRect(rect);
              newSegments.add(Segment(
                path: path,
                maskId: idAt(i),
                datasetStatus: statusAt(i),
                held: heldAt(i),
              ));
            }
          }
        }
      }
    } catch (e, st) {
      debugPrint('Error updating segments from result: $e\n$st');
    }
    
    _segments = newSegments;
    debugPrint('Finished updating segments. Found ${newSegments.length} segments.');
  }

  Future<void> _loadImageFromUrl() async {
    // Bytes from the session load are the same bytes the URL serves, so the
    // download is skipped when they are already here.
    final preloaded = _imageBytes;
    final url = _imageUrl;
    if (preloaded == null && (url == null || url.isEmpty)) return;

    setState(() {
      _isLoading = true;
      _imageSourceRunning = true;
      _error = null;
    });

    try {
      if (preloaded != null) {
        // Resuming an existing session: these bytes came from /loadSession
        // (_loadLaunchSession), which already fully restored the backend's
        // in-memory session AND reconstructed its SFSession with every
        // persisted mask (services.py's load_session_from_disk now warms
        // that cache as a side effect). There is nothing left for /upload
        // to legitimately do here — it exists to (re-)initialize SAM3's
        // state for a genuinely NEW image, which this isn't. Calling it
        // anyway used to rebuild the session from scratch and silently
        // discard everything /loadSession just restored (confirmed live:
        // name, description, prompts and masks all gone the instant the
        // app finished launching, before the user touched anything).
        // Every field the old upload-response handler set here
        // (_imageBytes/_uiImage/_imageSize/_originalBytes/etc.) is already
        // in place too — _loadLaunchSession set all of it from the same
        // /loadSession response these bytes came from.
        return;
      }

      // preloaded == null here (the resume case returned above), so §807's
      // guard guarantees url is non-null and non-empty — a genuinely new
      // image, from a URL: standalone file-pick (file://) or DN launching
      // straight into a fresh, never-saved session (http(s)://).
      Uint8List imageBytes;
      ui.Image decodedImage;
      // Handle file:// URIs (from standalone file picker) by reading directly
      // from the filesystem. HTTP URLs use http.get as before.
      if (url!.startsWith('file://')) {
        // Convert file:// URI to filesystem path and read bytes directly.
        // Use toFilePath() instead of .path to properly decode percent-encoding
        // (e.g. %20 → space) and handle platform-specific path conversion.
        final filePath = Uri.parse(url).toFilePath();
        debugPrint('Loading image from file:// URI: $url -> path: $filePath');
        try {
          imageBytes = await File(filePath).readAsBytes();
          debugPrint('Successfully read ${imageBytes.length} bytes from file');
          decodedImage = await _decodeImage(imageBytes);
        } catch (e) {
          debugPrint('Error reading file from file:// URI: $e');
          rethrow;
        }
      } else {
        final response = await http.get(Uri.parse(url));
        if (response.statusCode != 200) {
          throw Exception("Failed to download image: ${response.statusCode}");
        }
        imageBytes = response.bodyBytes;
        decodedImage = await _decodeImage(imageBytes);
      }

      final filename = url.split('/').last.isEmpty ? "image.png" : url.split('/').last;

      // Upload to backend. This is what gives /segment/* something to work
      // from for a genuinely new image — a resumed session's in-memory
      // state comes from /loadSession instead (see the early return above).
      debugPrint('Uploading ${imageBytes.length} bytes with filename=$filename, sessionId=$_sessionId');
      final uploadResponse = await _api.uploadImageBytes(
        imageBytes,
        filename: filename,
        sessionId: _sessionId,
        imageUrl: url,
      );
      debugPrint('Upload response: $uploadResponse');

      if (uploadResponse != null && mounted) {
        setState(() {
          _sessionId ??= uploadResponse['session_id'] as String?;
          _imageBytes = imageBytes;
          _uiImage = decodedImage;
          _imageSize = Size(
            (uploadResponse['width'] as num).toDouble(),
            (uploadResponse['height'] as num).toDouble(),
          );
          _originalBytes = imageBytes;
          _originalUiImage = decodedImage;
          _originalImageSize = _imageSize;
          _result = null;
          _segments = [];
          _aaPreviewData = null; // fresh upload — new session/pass 0, nothing selected yet
        });
        debugPrint('Upload complete. sessionId=$_sessionId, imageSize=$_imageSize');
      }
    } catch (e) {
      if (_imageBytes != null) {
        // The page is already on screen (a saved session's own bytes), so a
        // failed re-download of the launch URL is not worth an error banner.
        debugPrint('Image URL load failed; keeping the loaded image: $e');
      } else if (mounted) {
        setState(() => _error = e.toString());
      }
    } finally {
      if (mounted) setState(() { _isLoading = false; _imageSourceRunning = false; });
    }
  }

  /// Prompt card dual mode: grounding (default) vs. caption (see
  /// [_isCaptionMode]) — submitting text calls a completely different
  /// endpoint depending on which. [_buildTextPromptCard]'s header/label
  /// changes with the same condition, so the mode is always visible on
  /// screen rather than something to infer from what happens after submit
  /// (this is the deliberate mitigation for the IncludeExcludeToggle
  /// ambiguity bug from the earlier investigation).
  Future<void> _sendTextPrompt() async {
    if (_sessionId == null || _textController.text.isEmpty) return;
    if (_isCaptionMode) {
      await _attachCaptionToFocusedMask();
      return;
    }
    setState(() { _isLoading = true; _textPromptRunning = true; _textPromptResult = []; });
    try {
      final response = await _api.segmentWithText(_sessionId!, _textController.text);
      if (!mounted) return;
      if (response != null) {
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
          _updateSegmentsFromResult();
          _textPromptResult = [const ResultDatum(label: 'Status', value: 'Done')];
          _mutationCount++;
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _textPromptRunning = false; });
    }
  }

  /// The Prompt card's caption-mode submit: attaches the typed text as a
  /// caption on the focused mask (`/mask/caption`) instead of running a
  /// SAM3 text-grounding call. Patches the one changed record locally
  /// rather than re-deriving segments — this endpoint returns just that
  /// record, not a full results payload like /segment/* does.
  /// Pre-fills the caption box from this session's prompt template when the
  /// focused mask enters caption mode (templating v1).
  ///
  /// Deliberately conservative — it only ever writes into an UNTOUCHED box:
  ///
  ///  * No template (`draft` empty) means the controller is not touched at
  ///    all, so a session without one behaves exactly as it did before this
  ///    existed. Note what that baseline actually is: typed text is NOT
  ///    cleared when focus moves (only a successful Save Caption and Clear
  ///    Prompts clear it), so "don't touch" genuinely means leave it.
  ///  * A late reply is dropped (`gen != _draftGen`), as is one for a mask
  ///    that is no longer focused.
  ///  * If the text changed between request and reply, the user typed while
  ///    it was in flight and their text wins.
  ///
  /// Does NOT bump `_mutationCount`: nothing persistent changed. The draft
  /// becomes real work only when Save Caption sends it, which already counts.
  /// Schedules a pre-fill once the current `setState` has been applied.
  ///
  /// The box/point/balloon handlers set `_focusedMaskId` INSIDE their
  /// `setState` callback, where `_isCaptionMode` would still read the old
  /// focus; deferring to after the frame lets the check see the new one.
  void _prefillAfterFrame() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) _maybePrefillCaptionDraft();
    });
  }

  Future<void> _maybePrefillCaptionDraft() async {
    final sessionId = _sessionId;
    final maskId = _focusedMaskId;
    if (sessionId == null || maskId == null) return;
    if (!_isCaptionMode) return;
    // Already handled this mask; re-requesting would fight the user's edits.
    if (_draftRequestedFor == maskId) return;
    _draftRequestedFor = maskId;

    final gen = ++_draftGen;
    final textAtRequest = _textController.text;
    final draft = await _api.promptDraft(sessionId, maskId);
    if (!mounted) return;
    if (gen != _draftGen) return;                       // a newer request won
    if (_focusedMaskId != maskId) return;               // focus moved on
    if (draft == null || draft.isEmpty) return;         // no template: hands off
    if (_textController.text != textAtRequest) return;  // user typed; theirs wins
    _textController.text = draft;
  }

  Future<void> _attachCaptionToFocusedMask() async {
    final maskId = _focusedMaskId;
    if (_sessionId == null || maskId == null) return;
    setState(() { _isLoading = true; _textPromptRunning = true; _textPromptResult = []; });
    try {
      final response = await _api.attachCaption(_sessionId!, maskId, _textController.text);
      if (!mounted) return;
      if (response != null) {
        _patchMaskLocally(maskId, datasetStatus: response['dataset_status'] as String?);
        _textController.clear();
        _draftRequestedFor = null;
        setState(() {
          _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
          _textPromptResult = [const ResultDatum(label: 'Status', value: 'Captioned')];
          _mutationCount++;
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _textPromptRunning = false; });
    }
  }

  /// Patches one mask's fields in both [_segments] (what the canvas
  /// renders) and `_result`'s parallel arrays (what a later
  /// [_updateSegmentsFromResult] call would rebuild [_segments] from) —
  /// needed because /mask/hold and /mask/caption return only the one
  /// changed record, not a full results payload, so without this the two
  /// would silently drift apart on the next unrelated rebuild.
  void _patchMaskLocally(String maskId, {String? datasetStatus, bool? held}) {
    setState(() {
      _segments = [
        for (final s in _segments)
          if (s.maskId == maskId)
            Segment(
              path: s.path,
              maskId: s.maskId,
              datasetStatus: datasetStatus ?? s.datasetStatus,
              held: held ?? s.held,
            )
          else
            s,
      ];
      final maskIds = _result?['mask_ids'] as List?;
      if (maskIds != null) {
        final idx = maskIds.indexOf(maskId);
        if (idx != -1) {
          if (datasetStatus != null) (_result!['dataset_statuses'] as List)[idx] = datasetStatus;
          if (held != null) (_result!['held_flags'] as List)[idx] = held;
        }
      }
    });
  }

  /// Sets `held` on any mask by id — not gated on `_focusedMaskId`.
  ///
  /// A text prompt returns several masks at once with none of them
  /// focused (`_sendTextPrompt` never sets `_focusedMaskId` — only
  /// box/point selection does, since text has no single "the" instance to
  /// point at). The Hold checkbox used to be gated on `_focusedSegment`,
  /// which meant a multi-object text selection had NO way to hold any of
  /// its results without individually re-clicking each one via box/point
  /// first — confirmed as a real gap, not a display bug, when investigated.
  /// `_buildResultsCard` now renders one checkbox per selected mask, all
  /// routed through here, instead of one checkbox for whichever mask
  /// happened to be focused.
  ///
  /// sf-display-and-workflow-v3-spec.md §5: the backend now auto-holds a
  /// mask the instant it's selected (every `/segment/*` call holds its own
  /// new records), so in practice this is called with `held: false` far
  /// more often now — unchecking a row to opt it OUT of the next scrub
  /// batch — rather than `held: true` to opt one in. The call itself is
  /// unchanged either way; only which direction gets used more changed.
  Future<void> _setMaskHeld(String maskId, bool held) async {
    if (_sessionId == null) return;
    try {
      final response = await _api.setMaskHeld(_sessionId!, maskId, held);
      if (!mounted || response == null) return;
      _patchMaskLocally(maskId, held: response['held'] as bool?);
      // Whichever mask the user just held is a reasonable thing to also
      // focus — it's what makes the Prompt card's caption mode target the
      // mask just interacted with, the same way box/point selection
      // already sets focus as a side effect of touching a mask.
      setState(() {
        _focusedMaskId = maskId;
        _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
        _mutationCount++;
      });
      // Holding a mask is exactly the transition into caption mode
      // (held && unassigned), so this is where a template pre-fills the box.
      await _maybePrefillCaptionDraft();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    }
  }

  Future<void> _sendBoxPrompt(List<double> box) async {
    if (_sessionId == null) return;
    setState(() { _isLoading = true; _boxRunning = true; _boxResult = []; });
    try {
      final response = await _api.segmentWithBox(_sessionId!, box, _boxMode == "positive");
      if (!mounted) return;
      if (response != null) {
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
          _updateSegmentsFromResult();
          _focusedMaskId = response['selected_mask_id'] as String? ?? _focusedMaskId;
          // ignore: unawaited_futures
          _prefillAfterFrame();
          _boxResult = [ResultDatum(label: 'Status', value: positiveClickStatusLabel(response, _boxMode == "positive"))];
          _mutationCount++;
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _boxRunning = false; });
    }
  }

  Future<void> _sendPointPrompt(List<double> point) async {
    if (_sessionId == null) return;
    setState(() { _isLoading = true; _pointRunning = true; _pointResult = []; });
    try {
      final response = await _api.segmentWithPoint(_sessionId!, point, _pointMode == "positive");
      if (!mounted) return;
      if (response != null) {
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
          _updateSegmentsFromResult();
          _focusedMaskId = response['selected_mask_id'] as String? ?? _focusedMaskId;
          // ignore: unawaited_futures
          _prefillAfterFrame();
          _pointResult = [ResultDatum(label: 'Status', value: positiveClickStatusLabel(response, _pointMode == "positive"))];
          _mutationCount++;
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _pointRunning = false; });
    }
  }

  Future<void> _reset() async {
    if (_sessionId == null) return;
    setState(() { _isLoading = true; _resultsRunning = true; _resultsResult = []; });
    try {
      final response = await _api.resetPrompts(_sessionId!);
      if (!mounted) return;
      if (response != null) {
        setState(() {
          _result = response['results'] as Map<String, dynamic>?;
          _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
          _textController.clear();
          _updateSegmentsFromResult();
          // Deliberately NOT the `?? _focusedMaskId` fallback the box/point
          // handlers use (there, a null selected_mask_id means "recorded
          // nothing new, keep whatever was focused" — see add_box_selection's
          // negative-label case). A reset discards this pass's records
          // outright, so focus has to actually clear, not just skip an
          // update: reset now backend-clears sf_session.last_touched_mask_id
          // too, but that alone can't fix this side — this method never
          // read selected_mask_id at all before, so _focusedMaskId would
          // otherwise keep pointing at a mask reset just deleted.
          _focusedMaskId = null;
          _draftRequestedFor = null;
          _resultsResult = [const ResultDatum(label: 'Status', value: 'Done')];
          _mutationCount++;
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; _resultsRunning = false; });
    }
  }

  /// Formats a scrub transition the one way it's shown anywhere: the live
  /// post-scrub update below and a resumed session's restored label in
  /// [_loadLaunchSession]. One function so the two strings can't drift
  /// apart — resume only has a single current pass to work from, not a
  /// from_pass/to_pass pair, so it calls this with (pass - 1, pass).
  String _formatScrubLabel(Object? from, Object? to) =>
      "Pass $from → Pass $to";

  Future<void> _scrubLamaBackground() async {
    if (_sessionId == null) return;
    setState(() { _isScrubbing = true; });
    try {
      final response = await _api.scrubLamaBackground(_sessionId!);
      if (!mounted) return;
      if (response != null) {
        // /lama/scrub returns the new pass's actual image (LaMa applied to
        // the held set) as image_b64 -- decode it now, the same way
        // _loadLaunchSession/the upload path do, so the canvas shows the
        // scrubbed result instead of the pre-scrub image with its masks
        // silently cleared out from under it.
        final b64 = response['image_b64'] as String?;
        Uint8List? bytes;
        ui.Image? decoded;
        if (b64 != null && b64.isNotEmpty) {
          try {
            bytes = base64Decode(b64);
            decoded = await _decodeImage(bytes);
          } catch (e) {
            debugPrint('Could not decode scrubbed image: $e');
            bytes = null;
            decoded = null;
          }
        }
        if (!mounted) return;
        setState(() {
          _lastScrubLabel =
              _formatScrubLabel(response['from_pass'], response['to_pass']);
          if (bytes != null && decoded != null) {
            // Deliberately NOT touching _originalBytes/_originalUiImage/
            // _originalImageSize here — those are pass 0 and must survive
            // every scrub unchanged; only the current-pass slot moves.
            _imageBytes = bytes;
            _uiImage = decoded;
            _imageSize = Size(
              (response['width'] as num?)?.toDouble() ?? decoded.width.toDouble(),
              (response['height'] as num?)?.toDouble() ?? decoded.height.toDouble(),
            );
          }
          // The pass just changed; whatever was focused/shown belonged to
          // the pass before the scrub and has no live geometry in the new
          // one's result set (serialize_sf_masks only returns the current
          // pass — see its docstring).
          _focusedMaskId = null;
          _draftRequestedFor = null;
          _result = null;
          _segments = [];
          // NOT nulled like _result — a scrub is exactly the moment a
          // captioned/kept mask stops being "the current pass", so this
          // has to keep reflecting the session-wide ledger, not reset with
          // it. Confirmed live: without this, the AA Preview tab went
          // empty after every scrub regardless of whether anything was
          // captioned, defeating its purpose (previewing exportable data)
          // precisely when the data became exportable.
          _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
          // Only a balloon scrub sets this; an ordinary S&S scrub leaves
          // whatever it already was (null, or an earlier WB pass).
          _wbScrubPass = (response['wb_scrub_pass'] as num?)?.toInt() ?? _wbScrubPass;
          _canUndo = response['can_undo'] as bool? ?? true;
          _undoMessage = null;
          _mutationCount++;
        });
        // Pass 0 still contains everything just scrubbed. If it stays on, an
        // object that was scrubbed shows again whenever Current is hidden to
        // look at masks. So a scrub switches the reference layer off and the
        // working image on; Original can be turned back on to compare.
        _layerState?.setOriginal(false);
        _layerState?.setCurrent(true);
      } else {
        // scrubLamaBackground doesn't rethrow (see ApiService) — a null
        // response means the call itself failed (network error, or the
        // backend doesn't have this endpoint) — v2 has no "already
        // scrubbed" state to distinguish here, unlike the old two-pass cap.
        setState(() => _error = "LaMa scrub failed.");
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isScrubbing = false; });
    }
  }

  /// Deletes one current-pass selection ("masked by mistake"). The backend
  /// answers with the pass's remaining selections, which replace what is
  /// drawn, so the screen can never disagree with the session.
  Future<void> _removeMask(String maskId) async {
    if (_sessionId == null) return;
    final response = await _api.removeMask(_sessionId!, maskId);
    if (!mounted) return;
    if (response == null) {
      setState(() => _error = 'Could not remove that selection.');
      return;
    }
    setState(() {
      _result = response['result'] as Map<String, dynamic>?;
      _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
      _updateSegmentsFromResult();
      if (_focusedMaskId == maskId) {
        _focusedMaskId = null;
        _draftRequestedFor = null;
      }
      _mutationCount++;
    });
  }

  /// Steps back one scrub: the previous pass's image returns, the masks that
  /// were scrubbed are selected and queued again. Refused (with the reason
  /// shown) if it would lose captioned work.
  Future<void> _undoScrub() async {
    if (_sessionId == null || _isUndoing) return;
    setState(() { _isUndoing = true; _undoMessage = null; });
    try {
      final response = await _api.undoScrub(_sessionId!);
      if (!mounted) return;
      if (response == null) {
        setState(() => _error = 'Undo failed.');
        return;
      }
      if (response['ok'] != true) {
        setState(() => _undoMessage = response['detail'] as String?);
        return;
      }
      final b64 = response['image_b64'] as String?;
      Uint8List? bytes;
      ui.Image? decoded;
      if (b64 != null && b64.isNotEmpty) {
        try {
          bytes = base64Decode(b64);
          decoded = await _decodeImage(bytes);
        } catch (e) {
          debugPrint('Could not decode restored image: $e');
        }
      }
      if (!mounted) return;
      final pass = (response['pass'] as num?)?.toInt() ?? 0;
      setState(() {
        if (bytes != null && decoded != null) {
          _imageBytes = bytes;
          _uiImage = decoded;
          _imageSize = Size(
            (response['width'] as num?)?.toDouble() ?? decoded.width.toDouble(),
            (response['height'] as num?)?.toDouble() ?? decoded.height.toDouble(),
          );
        }
        _result = response['result'] as Map<String, dynamic>?;
        _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
        _updateSegmentsFromResult();
        _focusedMaskId = null;
        _draftRequestedFor = null;
        _lastScrubLabel = pass > 0 ? _formatScrubLabel(pass - 1, pass) : null;
        _wbScrubPass = (response['wb_scrub_pass'] as num?)?.toInt();
        _canUndo = response['can_undo'] as bool? ?? false;
        _mutationCount++;
      });
    } finally {
      if (mounted) setState(() { _isUndoing = false; });
    }
  }

  Future<void> _detectBalloons() async {
    if (_sessionId == null) return;
    setState(() { _isDetectingBalloons = true; });
    try {
      final response = await _api.detectBalloons(_sessionId!);
      if (!mounted) return;
      if (response != null) {
        setState(() {
          // Deliberately the same result handling as _sendBoxPrompt: this is
          // an ordinary selection response, so it renders through the existing
          // Objects Selected / checkbox list / Masks layer path.
          _result = response['results'] as Map<String, dynamic>?;
          _aaPreviewData = response['all_masks'] as Map<String, dynamic>?;
          _updateSegmentsFromResult();
          _focusedMaskId = response['selected_mask_id'] as String? ?? _focusedMaskId;
          // ignore: unawaited_futures
          _prefillAfterFrame();
          _lastDetectedCount = (response['detection_count'] as num?)?.toInt() ?? 0;
          _mutationCount++;
        });
      } else {
        setState(() => _error = "Balloon detection failed.");
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isDetectingBalloons = false; });
    }
  }

  /// Authors a panel from a canvas drag (v4 §4).
  ///
  /// Never calls SAM3 and never creates a mask — see [SelectionMode.panel].
  /// A rejected drag (too small; the backend's MIN_PANEL_SIDE floor) returns
  /// null and is a silent no-op, matching how an undersized box prompt is
  /// already dropped on the canvas side rather than raised to the user.
  Future<void> _addPanel(int x0, int y0, int x1, int y1) async {
    final sessionId = _sessionId;
    if (sessionId == null) return;
    try {
      final response = await _api.addPanel(sessionId, x0, y0, x1, y1);
      if (!mounted || response == null) return;
      setState(() {
        _panels = _panelsFrom(response);
        // Panels are saved state, so authoring one is unsaved work until
        // Save runs — same contract as every other mutation handler.
        _mutationCount++;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    }
  }

  Future<void> _deletePanel(String panelId) async {
    final sessionId = _sessionId;
    if (sessionId == null) return;
    try {
      final response = await _api.deletePanel(sessionId, panelId);
      if (!mounted || response == null) return;
      setState(() {
        _panels = _panelsFrom(response);
        _mutationCount++;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    }
  }

  /// `panels` out of any response that carries it, normalised to the shape
  /// [PanelsCard] and [PanelOutlinePainter] read. One place, so the add,
  /// delete and resume paths cannot disagree about it.
  static List<Map<String, dynamic>> _panelsFrom(Map<String, dynamic> response) {
    final raw = response['panels'] as List?;
    if (raw == null) return const [];
    return raw
        .whereType<Map>()
        .map((p) => Map<String, dynamic>.from(p))
        .toList(growable: false);
  }

  Future<void> _saveSession() async {
    if (_sessionId == null) return;
    setState(() { _isLoading = true; });
    try {
      final response = await _api.saveSession(_sessionId!);
      if (!mounted) return;
      if (response != null) {
        // Only a genuine 200 reaches here (saveSession throws otherwise —
        // see ApiService), so this is exactly "the save succeeded": record
        // the mutation count as of right now. A failed save falls into the
        // catch block below and never touches this, so unsaved work stays
        // unsaved until a save actually goes through.
        setState(() => _savedAtMutationCount = _mutationCount);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Session saved successfully")),
        );
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() { _isLoading = false; });
    }
  }


  /// Applies a resumed session's persisted `view_layers` (the same four
  /// keys [_saveLayerState] writes) to [LayerState] — wiring gap (bug-3
  /// audit finding #5): the backend has sent this on every /loadSession
  /// response all along, but nothing here ever read it, so a session saved
  /// with e.g. Raw unchecked silently came back with Raw checked again on
  /// every resume (LayerState's own field defaults).
  ///
  /// Missing keys keep whatever LayerState already has; unknown keys (a
  /// session saved before the Final -> Current rename carries a "final"
  /// row) are ignored by construction — only the four named keys are ever
  /// read.
  ///
  /// [_onLayerStateChanged] is detached for the duration: loading a session
  /// must not immediately re-save the very state it just loaded (an
  /// ordinary user toggle still saves normally once this returns).
  void _applyViewLayers(Map<String, dynamic>? viewLayers) {
    final layerState = _layerState;
    if (viewLayers == null || layerState == null) return;
    layerState.removeListener(_onLayerStateChanged);
    try {
      final original = viewLayers['original'];
      if (original is bool) layerState.setOriginal(original);
      final masks = viewLayers['masks'];
      if (masks is bool) layerState.setMasks(masks);
      // `viewLayers['raw']` may still be present in sessions saved before the
      // Raw layer was removed; it is deliberately ignored.
      final current = viewLayers['current'];
      if (current is bool) layerState.setCurrent(current);
    } finally {
      layerState.addListener(_onLayerStateChanged);
    }
  }

  Future<void> _saveLayerState() async {
    if (_sessionId == null || _layerState == null) return;
    try {
      // Note: Ensure ApiService has saveSessionSettings(String, Map)
      await _api.saveSessionSettings(_sessionId!, {
        'view_layers': {
          'original': _layerState!.showOriginal,
          'masks': _layerState!.showMasks,
          'current': _layerState!.showCurrent,
        }
      });
    } catch (e) {
      // Ignore errors for background saves
      debugPrint("Failed to save layer state: $e");
    }
  }

  /// Whether any mutation since the last successful save is still
  /// unsaved. Was `_result != null`, which is true the instant anything is
  /// selected and stays true forever after — including right after a
  /// successful Save, since Save never touched `_result`. Confirmed live:
  /// that fired the "Unsaved Changes" dialog on every session switch after
  /// the first selection, saved or not.
  bool get _hasUnsavedWork => _mutationCount != _savedAtMutationCount;

  /// Switch to a different session (standalone mode only).
  /// Warns if there are unsaved changes before discarding them.
  Future<void> _switchSession() async {
    if (!_isStandalone) return;

    if (_hasUnsavedWork) {
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Unsaved Changes'),
          content: const Text(
            'You have unsaved segmentation results. Switch to a different session anyway?',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Keep Working'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(ctx, true),
              style: TextButton.styleFrom(foregroundColor: Colors.red),
              child: const Text('Discard & Switch'),
            ),
          ],
        ),
      );
      if (confirmed != true) return;
    }

    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const SessionPickerScreen()),
    );
  }


  @override
  Widget build(BuildContext context) {
    // If you later want a responsive layout, you can use isWide.
    // final bool isWide = MediaQuery.of(context).size.width > 900;

    return Scaffold(
      appBar: AppBar(
        title: const Row(
          children: [
            Icon(Icons.auto_awesome, color: Colors.indigo),
            SizedBox(width: 10),
            Text('SegForge Studio', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        actions: [
          _buildStatusBadge(),
          const SizedBox(width: 16),
        ],
      ),
      body: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Sidebar + middle result column (scroll together)
          SizedBox(
            width: 520,
            // One group spanning the Prompt, Box Select and Point Select cards,
            // so their radios are mutually exclusive. `toggleable: true` on each
            // radio reports null when the selected one is tapped again, which is
            // what clears the mode.
            child: RadioGroup<SelectionMode>(
              groupValue: _selectedMode,
              onChanged: (mode) => setState(() => _selectedMode = mode),
              child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (_showsCard(SfCard.session))
                    _buildCardRow(_buildSessionCard(),       const ResultCell(isRunning: false, data: [])),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.upload))
                    _buildCardRow(_buildUploadCard(),        ResultCell(isRunning: _imageSourceRunning, data: const [])),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.prompt))
                    _buildCardRow(_buildTextPromptCard(),    ResultCell(isRunning: _textPromptRunning, data: _textPromptResult)),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.boxSelect))
                    _buildCardRow(_buildBoxPromptCard(),     ResultCell(isRunning: _boxRunning, data: _boxResult)),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.pointSelect))
                    _buildCardRow(_buildPointPromptCard(),   ResultCell(isRunning: _pointRunning, data: _pointResult)),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.panels))
                    _buildCardRow(_buildPanelsCard(),        const ResultCell(isRunning: false, data: [])),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.objectsSelected))
                    _buildCardRow(_buildResultsCard(),       ResultCell(isRunning: _resultsRunning, data: _resultsResult)),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.balloonScrub))
                    _buildCardRow(_buildBalloonScrubCard(), ResultCell(isRunning: _isDetectingBalloons, data: const [])),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.lamaBackgroundScrub))
                    _buildCardRow(_buildLBSCard(),           const ResultCell(isRunning: false, data: [])),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.displayLayers))
                    _buildCardRow(_buildSegmentLayersCard(), const ResultCell(isRunning: false, data: [])),
                  const SizedBox(height: 16),
                  if (_showsCard(SfCard.save))
                    _buildCardRow(_buildSaveCard(),          const ResultCell(isRunning: false, data: [])),
                  if (_error != null) ...[
                    const SizedBox(height: 16),
                    SizedBox(
                      width: 340,
                      child: Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.errorContainer,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Theme.of(context).colorScheme.error),
                        ),
                        child: Text(
                          _error!,
                          style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer, fontSize: 12),
                        ),
                      ),
                    ),
                  ],
                ],
              ),
              ),
            ),
          ),

          // Main display area: Canvas / AA Preview tabs
          Expanded(
            child: Container(
              margin: const EdgeInsets.fromLTRB(0, 16, 16, 16),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
              ),
              clipBehavior: Clip.antiAlias,
              child: DisplayAreaTabs(
                imageBytes: _imageBytes,
                uiImage: _uiImage,
                originalUiImage: _originalUiImage,
                segments: _segments,
                result: _result,
                aaPreviewData: _aaPreviewData,
                isLoading: _isLoading,
                mode: _selectedMode,
                panels: _panels,
                onBoxDrawn: _sendBoxPrompt,
                onPointDrawn: _sendPointPrompt,
                onPanelDrawn: (b) => _addPanel(b[0], b[1], b[2], b[3]),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCardRow(Widget card, Widget cell) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(width: 340, child: card),
        const SizedBox(width: 8),
        SizedBox(width: 140, child: cell),
      ],
    );
  }

  Widget _buildStatusBadge() {
    Color color;
    IconData icon;
    String text;

    switch (_backendStatus) {
      case "online":
        color = Colors.green;
        icon = Icons.check_circle;
        text = "Model Ready";
        break;
      case "offline":
        color = Colors.red;
        icon = Icons.error;
        text = "Backend Offline";
        break;
      default:
        color = Colors.orange;
        icon = Icons.hourglass_empty;
        text = "Connecting...";
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 6),
          Text(text, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  /// Whether [card] belongs in this session's sidebar (v4 §3).
  bool _showsCard(SfCard card) => cardsFor(_mode).contains(card);

  Widget _buildBorderedCard(Widget child) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.6),
          width: 1,
        ),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: child,
      ),
    );
  }

  Widget _buildSessionCard() {
    final name = _sessionName;
    final description = _sessionDescription;

    return _buildBorderedCard(
      Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Session", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
            const SizedBox(height: 8),
            // Name and description are only known once the backend has been
            // asked, and only when whoever created the session supplied them,
            // so each row appears only if there is something in it.
            if (name != null && name.isNotEmpty) ...[
              _sessionField("Name", name),
              const SizedBox(height: 8),
            ],
            if (description != null && description.isNotEmpty) ...[
              _sessionField("Description", description),
              const SizedBox(height: 8),
            ],
            _sessionField("ID", _sessionId ?? "(none)", dim: _sessionId == null),
            if (_isStandalone) ...[
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  icon: const Icon(Icons.folder_open, size: 16),
                  label: const Text("Switch Session"),
                  onPressed: _switchSession,
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                    side: const BorderSide(color: Colors.white),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(24),
                    ),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _sessionField(String label, String value, {bool dim = false}) {
    final scheme = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: TextStyle(fontSize: 11, color: scheme.onSurfaceVariant)),
        const SizedBox(height: 4),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          decoration: BoxDecoration(
            border: Border.all(color: scheme.outlineVariant),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Text(
            value,
            style: TextStyle(
              fontSize: 13,
              color: dim ? scheme.onSurfaceVariant : scheme.onSurface,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildUploadCard() {
    const greenColor = Color(0xFF007F00);

    return _buildBorderedCard(
      Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Image URL", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
            const SizedBox(height: 8),
            if (_imageUrl != null && _imageUrl!.isNotEmpty)
              GestureDetector(
                onTap: _isLoading ? null : _loadImageFromUrl,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    border: Border.all(color: greenColor, width: 1.5),
                    borderRadius: BorderRadius.circular(24),
                    color: Colors.transparent,
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.link, size: 14, color: greenColor),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          _imageUrl!,
                          style: const TextStyle(fontSize: 12, color: greenColor),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else if (_isStandalone && _pickedFilePath != null)
              GestureDetector(
                onTap: _isLoading ? null : _loadImageFromUrl,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    border: Border.all(color: greenColor, width: 1.5),
                    borderRadius: BorderRadius.circular(24),
                    color: Colors.transparent,
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.image, size: 14, color: greenColor),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          _pickedFilePath!,
                          style: const TextStyle(fontSize: 12, color: greenColor),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else
              Row(
                children: [
                  Expanded(
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                      decoration: BoxDecoration(
                        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        "(no image URL)",
                        style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.onSurfaceVariant),
                      ),
                    ),
                  ),
                  if (_isStandalone && !_isLoading)
                    Padding(
                      padding: const EdgeInsets.only(left: 8),
                      child: IconButton(
                        icon: const Icon(Icons.folder_open),
                        tooltip: 'Browse for image',
                        iconSize: 18,
                        padding: EdgeInsets.zero,
                        constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                        onPressed: _pickImageFile,
                      ),
                    ),
                ],
              ),
            if (_imageSize != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  "${_imageSize!.width.toInt()} × ${_imageSize!.height.toInt()} px",
                  style: TextStyle(fontSize: 11, color: Theme.of(context).colorScheme.onSurfaceVariant),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTextPromptCard() {
    return PromptCard(
      // REQUIRED, not cosmetic: the mode must be visible on screen at all
      // times, never something to infer from what happens after submit —
      // see _sendTextPrompt's docstring for why.
      captionMode: _isCaptionMode,
      focusedMaskId: _focusedSegment?.maskId,
      promptController: _textController,
      hasSession: _sessionId != null,
      isLoading: _isLoading,
      selectedMode: _selectedMode,
      onSubmit: _sendTextPrompt,
    );
  }

  Widget _buildBoxPromptCard() {
    return BoxSelectCard(
      isPositive: _boxMode == "positive",
      onModeChanged: (isPositive) => setState(() => _boxMode = isPositive ? "positive" : "negative"),
    );
  }

  Widget _buildPointPromptCard() {
    return PointSelectCard(
      isPositive: _pointMode == "positive",
      onModeChanged: (isPositive) => setState(() => _pointMode = isPositive ? "positive" : "negative"),
    );
  }

  Widget _buildResultsCard() {
    return ObjectsSelectedCard(
      maskCount: (_result?['masks'] as List?)?.length ?? 0,
      segments: _segments,
      enabled: _sessionId != null && !_isLoading,
      onSetHeld: _setMaskHeld,
      onClearPrompts: _reset,
      onRemove: _removeMask,
    );
  }


  Widget _buildSegmentLayersCard() {
    // Layer visibility is its own Consumer<LayerState>; whether the
    // WB-scrub layer EXISTS is session state, so it comes from here.
    return SegmentLayersCard(hasWbScrubLayer: _wbScrubPass != null);
  }

  Widget _buildLBSCard() {
    // Now that /segment/* reports held state per mask (v2), "pending" can
    // finally mean what LBSCard's label says — regions actually queued for
    // the next scrub — rather than the total-selected-count proxy used
    // before that data existed.
    final heldFlags = _result?['held_flags'] as List?;
    final pendingCount = heldFlags?.where((h) => h == true).length ?? 0;
    // v3 spec §5: Scrub's own enabled state is gated on SELECTION existing
    // at all, not on held count — same holdable filter ObjectsSelectedCard
    // uses for its checkbox list (a bbox-fallback segment with no maskId
    // can't be tracked as held/scrubbed, so it doesn't count as a
    // selection either).
    final hasSelection = _segments.any((s) => s.maskId != null);
    return LBSCard(
      pendingCount: pendingCount,
      hasSelection: hasSelection,
      lastScrubLabel: _lastScrubLabel,
      isScrubbing: _isScrubbing,
      onScrub: _scrubLamaBackground,
      canUndo: _canUndo,
      isUndoing: _isUndoing,
      onUndo: _undoScrub,
      undoMessage: _undoMessage,
    );
  }

  Widget _buildPanelsCard() {
    return PanelsCard(
      panels: _panels,
      hasImage: _sessionId != null && _uiImage != null,
      onDeletePanel: _deletePanel,
    );
  }

  Widget _buildBalloonScrubCard() {
    return BalloonScrubCard(
      hasImage: _sessionId != null && _uiImage != null,
      lastDetectedCount: _lastDetectedCount,
      isResumedSession: _isResumedSession,
      resumedWbScrubPass: _wbScrubPass,
      isDetecting: _isDetectingBalloons,
      onDetect: _detectBalloons,
    );
  }

  Widget _buildSaveCard() {
    return SaveCard(
      hasSession: _sessionId != null,
      isLoading: _isLoading,
      onSave: _saveSession,
    );
  }


}

/// The main display area: a "Canvas" tab (unchanged
/// [LayeredSegmentationCanvas] behaviour via [SegmentationCanvas]) and an
/// "AA Preview" tab (the shared `aa_preview_table` package, via
/// [sfResultToAaPayload]).
///
/// A standalone widget rather than inline in [_HomeScreenState.build] for
/// two reasons: it owns its own [TabController] (a ticker, needing its own
/// `SingleTickerProviderStateMixin`/dispose — no reason to put that on the
/// much bigger [_HomeScreenState]), and it's directly testable the same
/// way [SegmentationCanvas] already is, by rebuilding it with new [result]
/// values rather than driving all of [HomeScreen]'s network/timer setup.
class DisplayAreaTabs extends StatefulWidget {
  final Uint8List? imageBytes;
  final ui.Image? uiImage;
  /// Pass 0 — the true original image, separate from [uiImage] (the
  /// current-pass working image). See main.dart's `_originalUiImage` field
  /// doc comment.
  final ui.Image? originalUiImage;
  final List<Segment> segments;
  final Map<String, dynamic>? result;
  // Session-wide mask ledger (every pass), separate from `result` (current
  // pass only, what the canvas paints) — see main.dart's `_aaPreviewData`
  // field doc comment for why these can't be the same value: `result`
  // resets to null on every scrub, which silently dropped captioned/kept
  // masks from the AA Preview the moment their pass was scrubbed past.
  final Map<String, dynamic>? aaPreviewData;
  final bool isLoading;
  final SelectionMode? mode;
  /// Panel bounds to outline on the canvas (v4 §4) — page-pixel maps, the
  /// same raw shape main.dart's `_panels` holds. Display + authoring only;
  /// these are NOT segments and never reach [LayeredSegmentationCanvas].
  final List<Map<String, dynamic>> panels;
  final void Function(List<double>) onBoxDrawn;
  final void Function(List<double>) onPointDrawn;
  /// A completed panel drag, already in integer page pixels
  /// `[x0, y0, x1, y1]` — not the normalised cx/cy/w/h [onBoxDrawn] uses,
  /// because /panel/add stores page pixels verbatim.
  final void Function(List<int>) onPanelDrawn;

  const DisplayAreaTabs({
    super.key,
    required this.imageBytes,
    required this.uiImage,
    required this.originalUiImage,
    required this.segments,
    required this.result,
    required this.aaPreviewData,
    required this.isLoading,
    required this.mode,
    this.panels = const [],
    required this.onBoxDrawn,
    required this.onPointDrawn,
    required this.onPanelDrawn,
  });

  @override
  State<DisplayAreaTabs> createState() => _DisplayAreaTabsState();
}

class _DisplayAreaTabsState extends State<DisplayAreaTabs> with SingleTickerProviderStateMixin {
  late final TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: "Canvas"),
            Tab(text: "AA Preview"),
          ],
        ),
        Expanded(
          child: TabBarView(
            controller: _tabController,
            children: [
              _buildCanvasTab(context),
              // Built directly from `widget.aaPreviewData` on every
              // rebuild of THIS widget — not computed once and cached —
              // so every mutation the parent already calls setState() for
              // (select, hold, caption, scrub) reaches this tab via the
              // ordinary "new widget, new build()" path, no separate
              // refresh mechanism. Deliberately NOT `widget.result`: that
              // field is current-pass-only (what the canvas paints) and
              // resets to null on every scrub, which silently dropped
              // captioned/kept masks from this tab the moment their pass
              // was scrubbed past — confirmed live, see
              // serialize_sf_masks_all_passes' docstring (backend
              // services.py) for the full story. TabBarView builds all of
              // its children up front (it's a PageView under the hood, not
              // lazy per active tab), so this recomputes whether or not
              // the tab is currently visible; for SF's per-session mask
              // counts that's negligible, and simpler than gating it on
              // `_tabController.index`, which would need its own listener
              // for no real benefit.
              AaPreviewTableWidget(
                aa: sfResultToAaPayload(widget.aaPreviewData),
                imageColumn: 'crop_png_bytes',
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildCanvasTab(BuildContext context) {
    if (widget.imageBytes == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.image_outlined, size: 64, color: Theme.of(context).colorScheme.onSurfaceVariant),
            const SizedBox(height: 16),
            Text("Enter an Image URL to start", style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant)),
          ],
        ),
      );
    }
    if (widget.uiImage == null) {
      return const Center(child: CircularProgressIndicator());
    }
    return SegmentationCanvas(
      uiImage: widget.uiImage!,
      // Falls back to the current-pass image only if the original slot
      // somehow hasn't been populated yet — in practice both are always
      // set together (see main.dart's `_originalUiImage` field doc
      // comment), so this is defensive, not the normal path.
      originalImage: widget.originalUiImage ?? widget.uiImage!,
      segments: widget.segments,
      result: widget.result,
      isLoading: widget.isLoading,
      mode: widget.mode,
      panels: widget.panels,
      onBoxDrawn: widget.onBoxDrawn,
      onPointDrawn: widget.onPointDrawn,
      onPanelDrawn: widget.onPanelDrawn,
    );
  }
}

class SegmentationCanvas extends StatefulWidget {
  final ui.Image uiImage;
  /// Pass 0 — the true original image, distinct from [uiImage] (the
  /// current-pass working image). Fed straight through to
  /// [LayeredSegmentationCanvas]'s Original layer.
  final ui.Image originalImage;
  final List<Segment> segments;
  final Map<String, dynamic>? result;
  final bool isLoading;

  /// Active selection mode. Only [SelectionMode.box] accepts drags and only
  /// [SelectionMode.point] accepts taps; anything else leaves the canvas inert.
  final SelectionMode? mode;

  /// Panel bounds to outline, in page pixels (v4 §4).
  final List<Map<String, dynamic>> panels;

  final Function(List<double>) onBoxDrawn;
  final Function(List<double>) onPointDrawn;
  final void Function(List<int>) onPanelDrawn;

  const SegmentationCanvas({
    super.key,
    required this.uiImage,
    required this.originalImage,
    required this.segments,
    this.result,
    required this.isLoading,
    required this.mode,
    this.panels = const [],
    required this.onBoxDrawn,
    required this.onPointDrawn,
    required this.onPanelDrawn,
  });

  @override
  State<SegmentationCanvas> createState() => _SegmentationCanvasState();
}

class _SegmentationCanvasState extends State<SegmentationCanvas> {
  Offset? _startDrag;
  Offset? _currentDrag;

  bool get _boxing => widget.mode == SelectionMode.box;
  bool get _pointing => widget.mode == SelectionMode.point;
  /// Panel authoring (v4 §4). Shares the rubber-band drag with box mode —
  /// the gesture is identical, only what it produces differs: a box prompt
  /// goes to SAM3, a panel goes to /panel/add and never touches it.
  bool get _panelling => widget.mode == SelectionMode.panel;
  bool get _dragging => _boxing || _panelling;

  @override
  void didUpdateWidget(SegmentationCanvas oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Leaving box mode mid-drag would otherwise leave the rubber-band rect
    // painted with no way to finish or cancel it.
    if (!_dragging && (_startDrag != null || _currentDrag != null)) {
      _startDrag = null;
      _currentDrag = null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // The core display layers
        LayeredSegmentationCanvas(
          originalImage: widget.originalImage,
          currentImage: widget.uiImage,
          segments: widget.segments,
        ),

        // The interaction and prompt overlay
        _buildInteractionOverlay(),

        // Loading indicator on top of everything
        if (widget.isLoading)
          Container(
            color: Colors.black12,
            child: const Center(child: CircularProgressIndicator()),
          ),
      ],
    );
  }

  Widget _buildInteractionOverlay() {
    // This overlay needs to scale and position itself exactly like the
    // content of LayeredSegmentationCanvas. We can achieve this by
    // wrapping it in an identical FittedBox/SizedBox structure.
    return FittedBox(
      fit: BoxFit.contain,
      child: SizedBox(
        width: widget.uiImage.width.toDouble(),
        height: widget.uiImage.height.toDouble(),
        child: LayoutBuilder(builder: (context, constraints) {
          // Inside this LayoutBuilder, the coordinate system matches the original image.
          return Stack(
            children: [
              // Painter for showing existing prompts (the blue/red boxes)
              CustomPaint(
                size: Size.infinite,
                painter: PromptPainter(result: widget.result),
              ),

              // Panel outlines (v4 §4), in the SAME interaction overlay as
              // the prompt markers and in the same image-pixel coordinate
              // space — deliberately NOT a LayeredSegmentationCanvas layer:
              // the layer set stays at four, and a panel is authoring
              // furniture like a prompt marker, not display content like a
              // mask. Painted after PromptPainter so an outline stays
              // visible over a prompt marker it overlaps.
              CustomPaint(
                size: Size.infinite,
                painter: PanelOutlinePainter(panels: widget.panels),
              ),

              // Gesture detector for prompting. Only the handlers belonging to
              // the active mode are attached: leaving the pan recognizer live
              // in point mode would let it claim taps before onTapUp fires.
              GestureDetector(
                onTapUp: _pointing
                    ? (details) {
                        final local = details.localPosition;
                        final nx = local.dx / widget.uiImage.width;
                        final ny = local.dy / widget.uiImage.height;
                        // Simple bounds check to ensure we clicked inside
                        if (nx >= 0 && nx <= 1 && ny >= 0 && ny <= 1) {
                          widget.onPointDrawn([nx, ny]);
                        }
                      }
                    : null,
                onPanStart: _dragging
                    ? (details) => setState(() {
                          _startDrag = details.localPosition;
                          _currentDrag = details.localPosition;
                        })
                    : null,
                onPanUpdate: _dragging
                    ? (details) => setState(() => _currentDrag = details.localPosition)
                    : null,
                onPanEnd: _dragging
                    ? (details) {
                        if (_startDrag != null && _currentDrag != null) {
                          final rect = Rect.fromPoints(_startDrag!, _currentDrag!);

                          if (_panelling) {
                            // Page pixels, verbatim — a panel is stored as
                            // drawn (v4 §4), so unlike a box prompt there is
                            // nothing to normalise. This LayoutBuilder's
                            // coordinate space already IS image pixels (see
                            // _buildInteractionOverlay), so the only work is
                            // rounding. The backend clamps and enforces its
                            // own minimum size, so no floor is applied here:
                            // duplicating MIN_PANEL_SIDE on this side would
                            // be a second threshold to keep in sync, and a
                            // too-small drag is already a silent no-op.
                            widget.onPanelDrawn([
                              rect.left.round(), rect.top.round(),
                              rect.right.round(), rect.bottom.round(),
                            ]);
                          } else {
                            // Normalize coordinates for the API call.
                            final double nx = rect.center.dx / widget.uiImage.width;
                            final double ny = rect.center.dy / widget.uiImage.height;
                            final double nw = rect.width / widget.uiImage.width;
                            final double nh = rect.height / widget.uiImage.height;

                            if (nw > 0.005 && nh > 0.005) { // Avoid tiny boxes
                              widget.onBoxDrawn([nx, ny, nw, nh]);
                            }
                          }
                        }
                        setState(() {
                          _startDrag = null;
                          _currentDrag = null;
                        });
                      }
                    : null,
                child: MouseRegion(
                  cursor: (_pointing || _dragging)
                      ? SystemMouseCursors.precise
                      : MouseCursor.defer,
                  child: Container(color: Colors.transparent),
                ),
              ),

              // Painter for the box being currently drawn
              if (_startDrag != null && _currentDrag != null)
                CustomPaint(
                  size: Size.infinite,
                  painter: DragBoxPainter(
                    rect: Rect.fromPoints(_startDrag!, _currentDrag!),
                  ),
                ),
            ],
          );
        }),
      ),
    );
  }
}

/// Outlines every authored panel (v4 §4) in the interaction overlay.
///
/// Draws in IMAGE-PIXEL coordinates: the overlay this sits in is already
/// sized to the image inside a FittedBox (see
/// `_SegmentationCanvasState._buildInteractionOverlay`), so panel bounds —
/// which are stored as page pixels — need no conversion at all.
///
/// This is not a display layer. The layer set stays at four
/// (Original/Masks/Raw/Current) and has no panel checkbox; panels are
/// authoring furniture, shown wherever prompt markers are shown.
class PanelOutlinePainter extends CustomPainter {
  final List<Map<String, dynamic>> panels;

  PanelOutlinePainter({required this.panels});

  @override
  void paint(Canvas canvas, Size size) {
    if (panels.isEmpty) return;
    // Amber, distinct from the prompt markers' blue/red (Target/Avoid) and
    // from the mask overlays, so a panel boundary can never be misread as a
    // selection. 3px because these are page-pixel coordinates on a ~1800px
    // page scaled down to fit, where a 1px line all but disappears.
    final stroke = Paint()
      ..color = const Color(0xFFFFB300)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;
    for (final panel in panels) {
      final x0 = (panel['x0'] as num?)?.toDouble();
      final y0 = (panel['y0'] as num?)?.toDouble();
      final x1 = (panel['x1'] as num?)?.toDouble();
      final y1 = (panel['y1'] as num?)?.toDouble();
      if (x0 == null || y0 == null || x1 == null || y1 == null) continue;
      canvas.drawRect(Rect.fromLTRB(x0, y0, x1, y1), stroke);
    }
  }

  @override
  bool shouldRepaint(covariant PanelOutlinePainter oldDelegate) =>
      oldDelegate.panels != panels;
}

/// A painter for drawing the user's input prompts (positive/negative boxes).
class PromptPainter extends CustomPainter {
  final Map<String, dynamic>? result;

  PromptPainter({required this.result});

  @override
  void paint(Canvas canvas, Size size) {
    if (result == null) return;

    final promptedPoints = result!['prompted_points'] as List?;
    if (promptedPoints != null) {
      for (var pp in promptedPoints) {
        final point = (pp['point'] as List).map((e) => (e as num).toDouble()).toList();
        final label = pp['label'] as int; // 1 or 0

        final paint = Paint()
          ..color = (label == 1) ? Colors.green : Colors.red
          ..style = PaintingStyle.fill;

        // Draw a dot for the point
        canvas.drawCircle(Offset(point[0], point[1]), 5.0, paint);
      }
    }

    final promptedBoxes = result!['prompted_boxes'] as List?;
    if (promptedBoxes != null) {
      for (var pb in promptedBoxes) {
        final box = (pb['box'] as List).map((e) => (e as num).toDouble()).toList();
        final label = pb['label'] as bool;

        final paint = Paint()
          ..color = label ? Colors.blue : Colors.red
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0;

        // Coordinates are already in image space, no scaling needed.
        final rect = Rect.fromLTRB(box[0], box[1], box[2], box[3]);
        canvas.drawRect(rect, paint);

        final iconPaint = Paint()..color = label ? Colors.blue : Colors.red;
        canvas.drawCircle(rect.topLeft, 4, iconPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant PromptPainter oldDelegate) => result != oldDelegate.result;
}

class DragBoxPainter extends CustomPainter {
  final Rect rect;
  DragBoxPainter({required this.rect});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.blue
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawRect(rect, paint);

    final fillPaint = Paint()
      ..color = Colors.blue.withValues(alpha: 0.1)
      ..style = PaintingStyle.fill;
    canvas.drawRect(rect, fillPaint);
  }

  @override
  bool shouldRepaint(covariant DragBoxPainter oldDelegate) => rect != oldDelegate.rect;
}

/*
========================
ApiService NOTE (required)
========================

Your existing ApiService probably has something like:
  Future<Map<String,dynamic>?> uploadImage(File file)

For Flutter Web, you need an overload like:

  Future<Map<String,dynamic>?> uploadImageBytes(Uint8List bytes, {required String filename})

Implementation idea (using package:http):
- POST multipart/form-data
- add a MultipartFile.fromBytes('file', bytes, filename: filename)

If you paste your current ApiService, I’ll provide the exact drop-in update for it too.
*/