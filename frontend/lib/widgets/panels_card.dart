import 'package:flutter/material.dart';

import '../main.dart' show SelectionMode;

/// "Panels" card: manually authored panel bounds (v4 §4).
///
/// A panel is a LAYOUT REGION, not an object. Drawing one calls
/// `/panel/add` and nothing else — no SAM3, no MaskRecord, nothing in
/// Objects Selected. That separation is the whole point of the card: panel
/// detection was validated zero-shot first and none of the candidates were
/// good enough to build on, so these bounds are authored, not detected.
///
/// Stateless by design, like [LBSCard]/[ResultCell]/`IncludeExcludeToggle` —
/// the caller (main.dart's state) owns [panels] and the actual API calls,
/// the same split already used for every other per-card widget here. Chrome
/// is inlined (Container + border, Card, 16px padding) rather than factored
/// out, matching LBSCard exactly; nothing in this app shares a card shell
/// widget yet and this is not the place to introduce one.
class PanelsCard extends StatelessWidget {
  /// Every panel on this session, in authoring order — the order the
  /// backend returns and persistence preserves.
  ///
  /// Raw maps rather than a model class, matching how [LBSCard] and
  /// `ObjectsSelectedCard` take their backend data: there is no Panel type
  /// in the frontend, and inventing one for a list of four ints and an id
  /// would add a conversion layer with nothing to convert.
  final List<Map<String, dynamic>> panels;

  /// Whether the canvas has an image to draw on. Gates the hint text the
  /// same way [LBSCard.hasSelection] gates its button: with nothing on
  /// screen there is nothing to draw a panel over.
  final bool hasImage;

  final void Function(String panelId) onDeletePanel;

  const PanelsCard({
    super.key,
    required this.panels,
    required this.hasImage,
    required this.onDeletePanel,
  });

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.white.withValues(alpha: 0.6), width: 1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  // Plain text title, same style as LBSCard's — not the
                  // icon + bold Row that BoxSelectCard uses.
                  Text('Panels', style: textTheme.titleMedium),
                  const Spacer(),
                  // Joins main.dart's one RadioGroup<SelectionMode>, so
                  // selecting Panel deselects Box/Point and vice versa.
                  // `toggleable` matches the other three: tapping the
                  // selected radio again clears the mode.
                  const Radio<SelectionMode>(
                    value: SelectionMode.panel,
                    toggleable: true,
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                '${panels.length} panel${panels.length == 1 ? '' : 's'}',
                style: textTheme.bodySmall,
              ),
              const SizedBox(height: 8),
              Text(
                hasImage
                    ? 'Select Panel, then drag on the canvas.'
                    : 'Load an image to draw panels.',
                style: textTheme.bodySmall,
              ),
              if (panels.isNotEmpty) ...[
                const SizedBox(height: 12),
                // One row per panel. Listed in authoring order and numbered
                // from 1 for display only — the id is what the delete call
                // routes by, never the index, so deleting row 2 of 5 cannot
                // shift which panel a later tap removes.
                for (var i = 0; i < panels.length; i++)
                  _PanelRow(
                    label: 'Panel ${i + 1}',
                    bounds: _boundsLabel(panels[i]),
                    onDelete: () => onDeletePanel(panels[i]['panel_id'] as String),
                  ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  /// Page-pixel extent, shown so a mis-drawn panel is identifiable in the
  /// list without hunting for it on the canvas.
  static String _boundsLabel(Map<String, dynamic> panel) {
    final x0 = (panel['x0'] as num?)?.toInt() ?? 0;
    final y0 = (panel['y0'] as num?)?.toInt() ?? 0;
    final x1 = (panel['x1'] as num?)?.toInt() ?? 0;
    final y1 = (panel['y1'] as num?)?.toInt() ?? 0;
    return '${x1 - x0}x${y1 - y0} at $x0,$y0';
  }
}

class _PanelRow extends StatelessWidget {
  final String label;
  final String bounds;
  final VoidCallback onDelete;

  const _PanelRow({
    required this.label,
    required this.bounds,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          Expanded(
            child: Text('$label  —  $bounds', style: textTheme.bodySmall),
          ),
          IconButton(
            icon: const Icon(Icons.delete_outline, size: 18),
            // Named so the delete control is findable per row in tests and
            // readable to a screen reader, which a bare icon is not.
            tooltip: 'Delete $label',
            visualDensity: VisualDensity.compact,
            onPressed: onDelete,
          ),
        ],
      ),
    );
  }
}
