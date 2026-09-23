import 'package:flutter/foundation.dart';

class LayerState with ChangeNotifier {
  bool _showOriginal = true;
  bool _showMasks = true;
  bool _showRaw = true;
  bool _showCurrent = true;

  bool get showOriginal => _showOriginal;
  bool get showMasks => _showMasks;
  bool get showRaw => _showRaw;
  bool get showCurrent => _showCurrent;

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

  void setCurrent(bool value) {
    if (_showCurrent == value) return;
    _showCurrent = value;
    notifyListeners();
  }
}
