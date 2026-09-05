import 'package:flutter/foundation.dart';

class LayerState with ChangeNotifier {
  bool _showOriginal = true;
  bool _showMasks = true;
  bool _showRaw = true;
  bool _showFinal = true;

  bool get showOriginal => _showOriginal;
  bool get showMasks => _showMasks;
  bool get showRaw => _showRaw;
  bool get showFinal => _showFinal;

  void setOriginal(bool value) {
    if (_showOriginal == value) return;
    _showOriginal = value;
    notifyListeners();
  }

  void setMasks(bool value) {
    if (_showMasks == value) return;
    _showMasks = value;
    notifyListeners();
  }

  void setRaw(bool value) {
    if (_showRaw == value) return;
    _showRaw = value;
    notifyListeners();
  }

  void setFinal(bool value) {
    if (_showFinal == value) return;
    _showFinal = value;
    notifyListeners();
  }
}