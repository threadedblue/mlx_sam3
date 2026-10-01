import 'package:flutter/material.dart';

import '../main.dart' show SelectionMode;
import 'include_exclude_toggle.dart';

/// "Box Select" card: the mode radio and the Target/Avoid polarity toggle
/// for box selections. The box itself is drawn on the canvas, not here.
///
/// Stateless by design, like [ObjectsSelectedCard]/`LBSCard` — the caller
/// (main.dart's state) owns [isPositive] and the `_boxMode` mutation behind
/// [onModeChanged].
class BoxSelectCard extends StatelessWidget {
  final bool isPositive;
  final ValueChanged<bool> onModeChanged;

  const BoxSelectCard({
    super.key,
    required this.isPositive,
    required this.onModeChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.white.withValues(alpha: 0.6), width: 1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.crop_free, size: 16),
                  const SizedBox(width: 8),
                  const Text("Box Select", style: TextStyle(fontWeight: FontWeight.bold)),
                  const Spacer(),
                  const Radio<SelectionMode>(
                    value: SelectionMode.box,
                    toggleable: true,
                  ),
                ],
              ),
              const SizedBox(height: 12),
              IncludeExcludeToggle(
                value: isPositive,
                onChanged: onModeChanged,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
