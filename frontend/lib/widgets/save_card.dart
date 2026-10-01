import 'package:flutter/material.dart';

/// "Save" card: writes the current session to storage on an explicit click
/// (this app has no autosave — see LBSCard/BalloonScrubCard's own notes on
/// the same convention).
///
/// Stateless by design, like [ObjectsSelectedCard]/`LBSCard` — the caller
/// (main.dart's state) owns [hasSession]/[isLoading] and the actual
/// `/saveSession` call behind [onSave].
class SaveCard extends StatelessWidget {
  final bool hasSession;
  final bool isLoading;
  final VoidCallback onSave;

  const SaveCard({
    super.key,
    required this.hasSession,
    required this.isLoading,
    required this.onSave,
  });

  @override
  Widget build(BuildContext context) {
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
              const Row(
                children: [
                  Icon(Icons.save, size: 16),
                  SizedBox(width: 8),
                  Text("Save", style: TextStyle(fontWeight: FontWeight.bold)),
                ],
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: (!hasSession || isLoading) ? null : onSave,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF007F00),
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(24),
                    ),
                  ),
                  child: const Text("Save"),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
