import 'package:flutter/material.dart';

class IncludeExcludeToggle extends StatelessWidget {
  final bool value; // true = Include, false = Exclude
  final ValueChanged<bool> onChanged;

  const IncludeExcludeToggle({
    super.key,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    const greenColor = Color(0xFF007F00);
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
      child: Row(
        children: [
          // Include (left half)
          Expanded(
            child: Material(
              color: value ? greenColor : Colors.transparent,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(height / 2),
                bottomLeft: Radius.circular(height / 2),
              ),
              child: InkWell(
                onTap: () => onChanged(true),
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(height / 2),
                  bottomLeft: Radius.circular(height / 2),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.check,
                      size: 18,
                      color: value ? Colors.black : Colors.grey,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      'Include',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                        color: value ? Colors.black : Colors.grey,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          // Divider
          Container(
            width: 1,
            color: value ? greenColor : redColor,
          ),
          // Exclude (right half)
          Expanded(
            child: Material(
              color: !value ? Colors.transparent : Colors.transparent,
              borderRadius: const BorderRadius.only(
                topRight: Radius.circular(height / 2),
                bottomRight: Radius.circular(height / 2),
              ),
              child: InkWell(
                onTap: () => onChanged(false),
                borderRadius: const BorderRadius.only(
                  topRight: Radius.circular(height / 2),
                  bottomRight: Radius.circular(height / 2),
                ),
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
                      'Exclude',
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
        ],
      ),
    );
  }
}
