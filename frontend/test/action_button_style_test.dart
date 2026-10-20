// Every action button: green with white text; orange with white text while busy.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:frontend/widgets/action_button_style.dart';
import 'package:frontend/widgets/captions_card.dart';

Color? _bg(ButtonStyle s, Set<WidgetState> states) => s.backgroundColor!.resolve(states);
Color? _fg(ButtonStyle s, Set<WidgetState> states) => s.foregroundColor!.resolve(states);

void main() {
  test('idle: green with white text', () {
    final s = actionButtonStyle();
    expect(_bg(s, {}), actionButtonGreen);
    expect(_fg(s, {}), Colors.white);
  });

  test('busy: orange with white text, even though the button is disabled', () {
    final s = actionButtonStyle(busy: true);
    expect(_bg(s, {WidgetState.disabled}), actionButtonOrange);
    expect(_fg(s, {WidgetState.disabled}), Colors.white);
  });

  test('unavailable but not busy: muted, not orange', () {
    final s = actionButtonStyle();
    expect(_bg(s, {WidgetState.disabled}), isNot(actionButtonOrange));
    expect(_fg(s, {WidgetState.disabled}), isNot(Colors.white));
  });

  testWidgets('the Run captioner button turns orange while it runs', (t) async {
    Widget card({required bool running}) => MaterialApp(
          home: Scaffold(
            body: CaptionsCard(hasSession: true, blank: 3, onRun: () {}, running: running, progress: 'Captioning 1 of 3'),
          ),
        );
    Color? colour(WidgetTester t) => t
        .widget<FilledButton>(find.byKey(const ValueKey('captions-card-run')))
        .style!
        .backgroundColor!
        .resolve(<WidgetState>{WidgetState.disabled});

    await t.pumpWidget(card(running: false));
    expect(t.widget<FilledButton>(find.byKey(const ValueKey('captions-card-run'))).onPressed, isNotNull);
    await t.pumpWidget(card(running: true));
    expect(t.widget<FilledButton>(find.byKey(const ValueKey('captions-card-run'))).onPressed, isNull);
    expect(colour(t), actionButtonOrange);
  });
}
