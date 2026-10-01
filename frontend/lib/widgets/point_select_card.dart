import 'package:flutter/material.dart';

import '../main.dart' show SelectionMode;
import 'include_exclude_toggle.dart';

/// "Point Select" card: the mode radio and the Target/Avoid polarity toggle
/// for point selections. The point itself is placed on the canvas, not here.
///
/// Stateless by design, like [ObjectsSelectedCard]/`LBSCard` — the caller
/// (main.dart's state) owns [isPositive] and the `_pointMode` mutation
/// behind [onModeChanged].
class PointSelectCard extends StatelessWidget {
  final bool isPositive;
  final ValueChanged<bool> onModeChanged;

  const PointSelectCard({
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
                  const Icon(Icons.touch_app, size: 16),
                  const SizedBox(width: 8),
                  const Text("Point Select", style: TextStyle(fontWeight: FontWeight.bold)),
                  const Spacer(),
                  const Radio<SelectionMode>(
                    value: SelectionMode.point,
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
