import 'package:flutter/material.dart';
import 'layer_state.dart';
import 'package:provider/provider.dart';

class SegmentLayersCard extends StatelessWidget {
  const SegmentLayersCard({super.key});

  Widget _buildLayerCheckbox(
      String label, bool value, ValueChanged<bool> onChanged) {
    return CheckboxListTile(
      title: Text(label),
      value: value,
      onChanged: (newValue) => onChanged(newValue ?? false),
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
        Consumer<LayerState>(builder: (context, layerState, child) {
          return Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildLayerCheckbox('Original', layerState.showOriginal, layerState.setOriginal),
              _buildLayerCheckbox('Masks', layerState.showMasks, layerState.setMasks),
              _buildLayerCheckbox('Raw', layerState.showRaw, layerState.setRaw),
              _buildLayerCheckbox('Final', layerState.showFinal, layerState.setFinal),
            ],
          );
        }),
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