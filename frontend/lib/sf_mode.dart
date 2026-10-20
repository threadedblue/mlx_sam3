/// Which toolset SegForge is running (sf-v4-greyscale-layer-design.md §3).
///
/// One shared engine — SFSession, the pass chain, selection/hold/caption/scrub,
/// save/resume — is unchanged regardless of mode. A mode is purely which
/// sidebar cards are shown, so this file is a mapping and nothing more; no
/// mode may acquire behaviour of its own without that being a deliberate,
/// separate decision.
library;

/// A sidebar card, named independently of its widget class so the mapping can
/// be declared without importing the whole widget layer (and so tests can
/// assert on the mapping without building anything).
enum SfCard {
  session,
  upload,
  prompt,
  boxSelect,
  pointSelect,
  objectsSelected,
  displayLayers,
  balloonScrub,
  lamaBackgroundScrub,
  panels,
  captions,
  save,
}

/// Every card, in the order the sidebar lays them out today. This is also the
/// legacy/unfiltered list — what a launch with no mode at all must keep
/// showing, unchanged from pre-mode behaviour.
const List<SfCard> kAllCards = [
  SfCard.session,
  SfCard.upload,
  SfCard.prompt,
  SfCard.boxSelect,
  SfCard.pointSelect,
  SfCard.objectsSelected,
  SfCard.balloonScrub,
  SfCard.lamaBackgroundScrub,
  SfCard.displayLayers,
  SfCard.captions,
  SfCard.save,
];

/// Comix's toolset: everything in [kAllCards] plus the Panels card, placed
/// next to the other canvas-drawing cards rather than at the end — a panel is
/// authored by dragging on the canvas, like Box Select, not by a terminal
/// action like Save.
const List<SfCard> kComixCards = [
  SfCard.session,
  SfCard.upload,
  SfCard.prompt,
  SfCard.boxSelect,
  SfCard.pointSelect,
  SfCard.panels,
  SfCard.objectsSelected,
  SfCard.balloonScrub,
  SfCard.lamaBackgroundScrub,
  SfCard.displayLayers,
  SfCard.captions,
  SfCard.save,
];

enum SfMode {
  comix,
  poster,
  cine;

  /// Parse a wire/env value. Unknown or absent -> null, which callers must
  /// treat as "unfiltered", never as a default mode. Case-insensitive because
  /// the value crosses an env var and a Parquet column before arriving.
  static SfMode? parse(String? raw) {
    if (raw == null) return null;
    final v = raw.trim().toLowerCase();
    for (final m in SfMode.values) {
      if (m.name == v) return m;
    }
    return null;
  }

  /// The wire form — what rides in SEGFORGE_MODE and registry.parquet.
  String get wireValue => name;

  /// Whether this mode is actually built yet. Poster and Cine are visible,
  /// deliberately-disabled stubs on the DoubleNaught side; nothing should
  /// treat them as working toolsets until their card lists are real.
  bool get isImplemented => this == SfMode.comix;
}

/// Cards for [mode], or the full unfiltered set when [mode] is null.
///
/// Comix is the general-purpose cards plus Balloon Scrub plus Panels. It is
/// no longer identical to "unfiltered": [SfCard.panels] (v4 §4) is the first
/// genuinely mode-specific card, which is exactly the divergence the two were
/// kept as separate concepts for. Panel cutouts are a comics-layout notion,
/// so a no-mode launch — defined as "unchanged from pre-mode behaviour" —
/// does not get it.
///
/// Poster and Cine return empty: their toolsets are not scoped yet (v4 §3
/// sequences Comix first, then Poster, with Cine needing its own design pass).
/// Empty is the honest answer — better than quietly showing Comix's cards and
/// implying a mode works when it does not.
List<SfCard> cardsFor(SfMode? mode) {
  switch (mode) {
    case null:
      return kAllCards;
    case SfMode.comix:
      return kComixCards;
    case SfMode.poster:
      return const [];
    case SfMode.cine:
      return const [];
  }
}
