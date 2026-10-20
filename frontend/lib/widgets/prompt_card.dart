import 'action_button_style.dart';
import 'package:flutter/material.dart';

import '../main.dart' show SelectionMode, promptCardCanSubmit;

/// "Prompt" card: a standalone text-prompt search (title "Prompt"), or, when
/// a held-and-unassigned mask is focused, the same field repurposed to
/// caption that mask (title "Caption: `<id>`", green accent, no mode radio).
///
/// Stateless by design, like [ObjectsSelectedCard]/`LBSCard` — the caller
/// (main.dart's state) owns [promptController], [captionMode],
/// [focusedMaskId], the selection mode, and the actual `/segment/text` /
/// `/mask/caption` call behind [onSubmit].
///
/// `captionMode`/`focusedMaskId` are passed in already-computed, not derived
/// here — main.dart's `_isCaptionMode`/`_focusedSegment` getters are the
/// single source of truth for what "captioning" means (held AND still
/// unassigned); duplicating that logic here would risk the two drifting.
class PromptCard extends StatelessWidget {
  final bool captionMode;
  final String? focusedMaskId;
  final TextEditingController promptController;
  final bool hasSession;
  final bool isLoading;
  final SelectionMode? selectedMode;
  final VoidCallback onSubmit;

  const PromptCard({
    super.key,
    required this.captionMode,
    required this.focusedMaskId,
    required this.promptController,
    required this.hasSession,
    required this.isLoading,
    required this.selectedMode,
    required this.onSubmit,
  });

  @override
  Widget build(BuildContext context) {
    // Not a bare .substring(0, 8): real ids are f"{pass}:{uuid4()}"
    // (backend/sf_engine.py), always well over 8 chars here, but nothing
    // guarantees that for every possible maskId — same clamp as
    // ObjectsSelectedCard's identical truncation.
    final focusedIdTail = focusedMaskId?.split(':').last;
    final headerLabel = captionMode
        ? "Caption: ${focusedIdTail == null ? '' : (focusedIdTail.length <= 8 ? focusedIdTail : focusedIdTail.substring(0, 8))}"
        : "Prompt";

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
                  Icon(captionMode ? Icons.local_offer : Icons.text_fields, size: 16),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      headerLabel,
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: captionMode ? const Color(0xFF007F00) : null,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  if (!captionMode)
                    const Radio<SelectionMode>(
                      value: SelectionMode.prompt,
                      toggleable: true,
                    ),
                ],
              ),
              const SizedBox(height: 12),
              TextField(
                controller: promptController,
                maxLines: null,
                decoration: InputDecoration(
                  hintText: captionMode ? 'Describe this object for training' : 'e.g. "cat", "wheel"',
                  isDense: true,
                  border: const OutlineInputBorder(),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
                ),
                enabled: hasSession && !isLoading,
                onSubmitted: (_) => onSubmit(),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton(
                  onPressed: promptCardCanSubmit(
                    hasSession: hasSession,
                    promptEmpty: promptController.text.isEmpty,
                    isLoading: isLoading,
                    captionMode: captionMode,
                    selectedMode: selectedMode,
                  ) ? onSubmit : null,
                  style: actionButtonStyle(busy: isLoading),
                  child: Text(captionMode ? "Save Caption" : "Select"),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
