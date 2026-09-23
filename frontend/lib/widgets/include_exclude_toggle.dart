import 'package:flutter/material.dart';

/// SAM3 grounding polarity for a box/point prompt — the `label: bool` the
/// backend forwards to `add_geometric_prompt`/`add_point_prompt`: "this
/// region IS part of what I'm describing" vs "it ISN'T".
///
/// Labelled Target/Avoid rather than Include/Exclude because it is NOT
/// what decides whether an object gets scrubbed or captioned — those are
/// `held`/`dataset_status` (see ObjectsSelectedCard's "Include ... in next
/// scrub" checkbox list and the Prompt card's caption mode), a separate
/// axis entirely. The old Include/Exclude wording read as a duplicate of
/// that axis and was the likeliest reason users expected this control to
/// drive scrubbing.
class IncludeExcludeToggle extends StatelessWidget {
  final bool value; // true = Target (positive), false = Avoid (negative)
  final ValueChanged<bool> onChanged;

  const IncludeExcludeToggle({
    super.key,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    const greenColor = Color(0xFF007F00);
    const greenFill = Color(0xFF007F00);
    const redColor = Color(0xFFFF0000);
    const height = 40.0;

    return Container(
      height: height,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(height / 2),
        border: Border.all(
          color: value ? greenColor : redColor,
          width: 1.5,
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Row(
        children: [
          // Target / positive (left half)
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: value ? greenFill.withValues(alpha: 0.25) : Colors.transparent,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(height / 2),
                  bottomLeft: Radius.circular(height / 2),
                ),
              ),
              constraints: const BoxConstraints.expand(),
              child: Material(
                color: Colors.transparent,
                child: InkWell(
                  onTap: () => onChanged(true),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.check,
                        size: 18,
                        color: value ? greenColor : Colors.grey,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        'Target',
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                          color: value ? greenColor : Colors.grey,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          // Divider
          Container(
            width: 1,
            color: value ? greenColor : redColor,
          ),
          // Avoid / negative (right half)
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: !value ? redColor.withValues(alpha: 0.25) : Colors.transparent,
                borderRadius: const BorderRadius.only(
                  topRight: Radius.circular(height / 2),
                  bottomRight: Radius.circular(height / 2),
                ),
              ),
              constraints: const BoxConstraints.expand(),
              child: Material(
                color: Colors.transparent,
                child: InkWell(
                  onTap: () => onChanged(false),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.indeterminate_check_box_outlined,
                        size: 18,
                        color: !value ? redColor : Colors.grey,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        'Avoid',
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                          color: !value ? redColor : Colors.grey,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
