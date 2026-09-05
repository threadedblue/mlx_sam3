import 'package:flutter/material.dart';
import '../models/result_datum.dart';

class ResultCell extends StatelessWidget {
  final bool isRunning;
  final List<ResultDatum> data;

  const ResultCell({
    super.key,
    required this.isRunning,
    required this.data,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    if (isRunning) {
      return const Center(
        child: SizedBox(
          width: 20,
          height: 20,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
      );
    }

    if (data.isEmpty) return const SizedBox.shrink();

    return Align(
      alignment: Alignment.topLeft,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (int i = 0; i < data.length; i++) ...[
              if (i > 0) const SizedBox(height: 4),
              Text(
                data[i].label,
                style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
              ),
              Text(
                data[i].value,
                style: const TextStyle(fontSize: 11),
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ],
        ),
      ),
    );
  }
}
