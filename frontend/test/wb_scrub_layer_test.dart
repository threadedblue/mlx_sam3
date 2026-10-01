// v4 §2: the Display Layers card shows "WB-scrub result" as its own layer,
// and only when one actually exists.
//
// The card can't infer this from the pass count — a balloon scrub may land at
// pass 1, pass 2, or never — so whether the layer exists is passed in from
// session state (backend's `wb_scrub_pass`), not derived here.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:frontend/layer_state.dart';
import 'package:frontend/segment_layers_card.dart';

Widget _wrap(Widget child) => ChangeNotifierProvider<LayerState>(
      create: (_) => LayerState(),
      child: MaterialApp(home: Scaffold(body: child)),
    );

void main() {
  group('SegmentLayersCard WB-scrub layer', () {
    testWidgets('no WB layer when a balloon scrub has never run', (tester) async {
      await tester.pumpWidget(_wrap(const SegmentLayersCard(hasWbScrubLayer: false)));

      expect(find.text('WB-scrub result'), findsNothing);
      // The always-present layers are unaffected.
      expect(find.text('Original'), findsOneWidget);
      expect(find.text('Current'), findsOneWidget);
    });

    testWidgets('the WB layer appears once a balloon scrub has produced one',
        (tester) async {
      await tester.pumpWidget(_wrap(const SegmentLayersCard(hasWbScrubLayer: true)));

      expect(find.text('WB-scrub result'), findsOneWidget);
    });

    testWidgets('WB-scrub result is distinct from Current, not a rename',
        (tester) async {
      await tester.pumpWidget(_wrap(const SegmentLayersCard(hasWbScrubLayer: true)));

      expect(find.text('WB-scrub result'), findsOneWidget);
      expect(find.text('Current'), findsOneWidget);
    });

    testWidgets('the WB layer toggles independently of the others',
        (tester) async {
      await tester.pumpWidget(_wrap(const SegmentLayersCard(hasWbScrubLayer: true)));

      final state = Provider.of<LayerState>(
          tester.element(find.byType(SegmentLayersCard)), listen: false);
      expect(state.showWbScrub, isFalse, reason: 'off until asked for');
      final before = (state.showOriginal, state.showMasks, state.showCurrent);

      await tester.tap(find.text('WB-scrub result'));
      await tester.pump();

      expect(state.showWbScrub, isTrue);
      expect((state.showOriginal, state.showMasks, state.showCurrent), before,
          reason: 'multi-select: toggling one layer must not disturb others');
    });
  });
}
