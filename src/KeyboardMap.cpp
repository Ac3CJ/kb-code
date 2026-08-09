#include "KeyboardMap.h"

namespace cv_keyboard {

KeyboardMap::KeyboardMap() {
    // Start empty. You can call loadUKLayout() during setup.
}

KeyboardMap::~KeyboardMap() = default;

void KeyboardMap::addRow(float start_x, float y, const std::vector<std::pair<std::string, float>>& row_data) {
    float current_x = start_x;
    for (const auto& key : row_data) {
        keys_.push_back({
            key.first, 
            current_x, 
            y, 
            key.second, // width
            1.0f        // height is always 1u for these keys
        });
        current_x += key.second; // Move cursor right by the width of the key
    }
}

void KeyboardMap::loadUKLayout() {
    keys_.clear();
    markers_.clear();

    // ---------------------------------------------------------
    // 1. Define Keys (Origin 0,0 is top-left of the first key)
    // ---------------------------------------------------------
    
    // Row 1: Numbers (Y = 0.0)
    addRow(0.0f, 0.0f, {
        {"~", 1.0f}, {"1", 1.0f}, {"2", 1.0f}, {"3", 1.0f}, {"4", 1.0f}, 
        {"5", 1.0f}, {"6", 1.0f}, {"7", 1.0f}, {"8", 1.0f}, {"9", 1.0f}, 
        {"0", 1.0f}, {"-", 1.0f}, {"=", 1.0f}, {"Backspace", 2.0f}
    }); // Total width: 15u

    // Row 2: QWERTY (Y = 1.0)
    addRow(0.0f, 1.0f, {
        {"Tab", 1.5f}, {"Q", 1.0f}, {"W", 1.0f}, {"E", 1.0f}, {"R", 1.0f}, 
        {"T", 1.0f}, {"Y", 1.0f}, {"U", 1.0f}, {"I", 1.0f}, {"O", 1.0f}, 
        {"P", 1.0f}, {"[", 1.0f}, {"]", 1.0f}, {"Enter", 1.5f} // Top half of Enter
    }); // Total width: 15u

    // Row 3: ASDF (Y = 2.0)
    addRow(0.0f, 2.0f, {
        {"Caps", 1.75f}, {"A", 1.0f}, {"S", 1.0f}, {"D", 1.0f}, {"F", 1.0f}, 
        {"G", 1.0f}, {"H", 1.0f}, {"J", 1.0f}, {"K", 1.0f}, {"L", 1.0f}, 
        {";", 1.0f}, {"'", 1.0f}, {"#", 1.0f}, {"Enter", 1.25f} // Bottom half of Enter
    }); // Total width: 15u

    // Row 4: ZXCV (Y = 3.0)
    addRow(0.0f, 3.0f, {
        {"L_Shift", 1.25f}, {"\\", 1.0f}, {"Z", 1.0f}, {"X", 1.0f}, {"C", 1.0f}, 
        {"V", 1.0f}, {"B", 1.0f}, {"N", 1.0f}, {"M", 1.0f}, {",", 1.0f}, 
        {".", 1.0f}, {"/", 1.0f}, {"R_Shift", 2.75f}
    }); // Total width: 15u

    // Row 5: Modifiers (Y = 4.0)
    addRow(0.0f, 4.0f, {
        {"L_Ctrl", 1.25f}, {"L_Win", 1.25f}, {"L_Alt", 1.25f}, {"Space", 6.25f}, 
        {"R_AltGr", 1.25f}, {"Fn", 1.25f}, {"Option", 1.25f}, {"R_Ctrl", 1.25f}
    }); // Total width: 15u


    // ---------------------------------------------------------
    // 2. Define ArUco Markers
    // ---------------------------------------------------------
    // Note: These coordinates estimate their placement based on the image provided.
    // They are placed outside the 15x5u main key grid.
    
    // Top Row (above the keys)
    markers_.push_back({0,  0.0f, -1.0f}); // Above '~' (x=0)
    markers_.push_back({6,  4.0f, -1.0f}); // Above '4' (x=4)
    markers_.push_back({8,  8.0f, -1.0f}); // Above '8' (x=8)
    markers_.push_back({1, 14.0f, -1.0f}); // Above right-edge of Backspace (x=14)

    // Middle Row (Sides)
    markers_.push_back({4, -1.0f, 2.0f});  // Left of Caps Lock (x=-1)
    markers_.push_back({5, 15.0f, 2.0f});  // Right of Enter (x=15)

    // Bottom Row (below the keys)
    markers_.push_back({2,  0.0f, 5.0f});  // Below left Ctrl (x=0)
    markers_.push_back({7,  4.0f, 5.0f});  // Below left side of Space (x=4)
    markers_.push_back({9,  8.0f, 5.0f});  // Below right side of Space (x=8)
    markers_.push_back({3, 14.0f, 5.0f});  // Below right Ctrl (x=14)
}

std::string KeyboardMap::getKeyAt(float x_cm, float y_cm) const {
    // Simple AABB (Axis-Aligned Bounding Box) collision check
    for (const auto& key : keys_) {
        float kx = key.x_cm();
        float ky = key.y_cm();
        float kw = key.width_cm();
        float kh = key.height_cm();

        if (x_cm >= kx && x_cm <= (kx + kw) &&
            y_cm >= ky && y_cm <= (ky + kh)) {
            return key.id;
        }
    }
    return ""; // No key found
}

} // namespace cv_keyboard