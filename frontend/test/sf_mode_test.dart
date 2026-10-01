// v4 §3: mode -> card list. Pure mapping; no widgets, no network.
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/sf_mode.dart';

void main() {
  group('SfMode.parse', () {
    test('parses the three modes case-insensitively', () {
      expect(SfMode.parse('comix'), SfMode.comix);
      expect(SfMode.parse('COMIX'), SfMode.comix);
      expect(SfMode.parse(' Comix '), SfMode.comix);
      expect(SfMode.parse('poster'), SfMode.poster);
      expect(SfMode.parse('cine'), SfMode.cine);
    });

    test('absent or unrecognised is null, never a default mode', () {
      expect(SfMode.parse(null), isNull);
      expect(SfMode.parse(''), isNull);
      expect(SfMode.parse('nonsense'), isNull);
    });

    test('wireValue round-trips through parse', () {
      for (final m in SfMode.values) {
        expect(SfMode.parse(m.wireValue), m);
      }
    });
  });

  group('cardsFor', () {
    test('no mode shows every card — the legacy/unfiltered contract', () {
      // Regression guard on today's behaviour: a standalone launch, an old
      // DoubleNaught, or a pre-mode session must lose nothing.
      expect(cardsFor(null), kAllCards);
      expect(cardsFor(null).length, SfCard.values.length);
    });

    test('comix shows every card, including Balloon Scrub', () {
      final cards = cardsFor(SfMode.comix);
      expect(cards, contains(SfCard.balloonScrub));
      expect(cards, contains(SfCard.prompt));
      expect(cards, contains(SfCard.boxSelect));
      expect(cards, contains(SfCard.pointSelect));
      expect(cards, contains(SfCard.objectsSelected));
      expect(cards, contains(SfCard.lamaBackgroundScrub));
      expect(cards, contains(SfCard.save));
      expect(cards, contains(SfCard.displayLayers));
    });

    test('unbuilt modes are empty, not silently Comix', () {
      // Showing Comix's cards here would imply Poster/Cine work.
      expect(cardsFor(SfMode.poster), isEmpty);
      expect(cardsFor(SfMode.cine), isEmpty);
    });

    test('only comix reports itself implemented', () {
      expect(SfMode.comix.isImplemented, isTrue);
      expect(SfMode.poster.isImplemented, isFalse);
      expect(SfMode.cine.isImplemented, isFalse);
    });

    test('the mapping structure takes a new mode without rework', () {
      // Every mode resolves to a list today, so adding Poster's real cards
      // later is editing one case arm, not changing the shape.
      for (final m in SfMode.values) {
        expect(cardsFor(m), isA<List<SfCard>>());
      }
    });
  });
}
