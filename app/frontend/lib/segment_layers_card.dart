import 'package:flutter/material.dart';

class SegmentLayersCard extends StatelessWidget {
  final bool showMasks;
  final bool showRaw;
  final bool showFinal;
  final ValueChanged<bool> onMasksToggle;
  final ValueChanged<bool> onRawToggle;
  final ValueChanged<bool> onFinalToggle;

  const SegmentLayersCard({
    super.key,
    required this.showMasks,
    required this.showRaw,
    required this.showFinal,
    required this.onMasksToggle,
    required this.onRawToggle,
    required this.onFinalToggle,
  });

  Widget _buildLayerCheckbox(
      String label, bool value, ValueChanged<bool> onChanged) {
    return CheckboxListTile(
      title: Text(label),
      value: value,
      onChanged: (v) => onChanged(v ?? false),
      controlAffinity: ListTileControlAffinity.leading,
      contentPadding: EdgeInsets.zero,
      dense: true,
    );
  }

  Widget _buildLayers(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'Segment Layers',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 8),
        _buildLayerCheckbox('Masks', showMasks, onMasksToggle),
        _buildLayerCheckbox('Raw', showRaw, onRawToggle),
        _buildLayerCheckbox('Final', showFinal, onFinalToggle),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: _buildLayers(context),
      ),
    );
  }
}