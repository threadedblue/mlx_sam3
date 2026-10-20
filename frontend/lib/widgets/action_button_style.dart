import 'package:flutter/material.dart';

/// The look every action button shares: green with white text; orange with white
/// text while the action it starts is running ([busy]); a muted grey when it is
/// simply unavailable.
///
/// A [ButtonStyle] rather than a widget, so each button keeps its own type
/// (ElevatedButton, OutlinedButton, FilledButton) and its callers and tests keep
/// finding it. A busy button is normally also disabled (`onPressed: null`), so
/// the orange is applied to the disabled state when [busy] is set.
const Color actionButtonGreen = Color(0xFF007F00);
const Color actionButtonOrange = Color(0xFFE67E00);

ButtonStyle actionButtonStyle({bool busy = false}) => ButtonStyle(
      backgroundColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.disabled)) {
          return busy ? actionButtonOrange : const Color(0xFF3A3A3A);
        }
        return actionButtonGreen;
      }),
      foregroundColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.disabled)) {
          return busy ? Colors.white : Colors.white38;
        }
        return Colors.white;
      }),
      side: const WidgetStatePropertyAll(BorderSide.none),
      shape: WidgetStatePropertyAll(RoundedRectangleBorder(borderRadius: BorderRadius.circular(24))),
    );
