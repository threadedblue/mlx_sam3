import 'package:flutter/foundation.dart';

class LayerState with ChangeNotifier {
  bool _showOriginal = true;
  bool _showMasks = true;
  bool _showCurrent = true;
  /// The WB-scrub result layer (v4 §2). Off by default: it only exists
  /// once a balloon scrub has actually run, and the card hides the
  /// checkbox entirely until then.
  bool _showWbScrub = false;

  bool get showOriginal => _showOriginal;
  bool get showMasks => _showMasks;
  bool get showCurrent => _showCurrent;
  bool get showWbScrub => _showWbScrub;

  void setOriginal(bool value) {
    if (_showOriginal == value) return;
    _showOriginal = value;
    notifyListeners();
  }

  void setWbScrub(bool value) {
    if (_showWbScrub == value) return;
    _showWbScrub = value;
    notifyListeners();
  }

  void setMasks(bool value) {
    if (_showMasks == value) return;
    _showMasks = value;
    notifyListeners();
  }

  void setCurrent(bool value) {
    if (_showCurrent == value) return;
    _showCurrent = value;
    notifyListeners();
  }
}
