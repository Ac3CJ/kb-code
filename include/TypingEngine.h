#ifndef CV_KEYBOARD_TYPING_ENGINE_H
#define CV_KEYBOARD_TYPING_ENGINE_H

#include <string>
#include <unordered_set>
#include <iostream>
#include <cctype>
#include <deque>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace cv_keyboard {

struct Keystroke {
    std::string display_name;
    int64_t timestamp_us;
};

class TypingEngine {
public:
    TypingEngine() = default;
    ~TypingEngine() = default;

    /// Evaluates the currently clicked keys against the previous frame to detect new strikes.
    void processClicks(const std::unordered_set<std::string>& current_clicks, int64_t timestamp_us = 0);

    /// Gets the current buffer of typed text
    const std::string& getText() const { return text_buffer_; }
    
    /// Clears the text buffer
    void clearText() { text_buffer_.clear(); }

    /// Resets the internal state (useful when scrubbing backwards in the tester)
    void reset();

private:
    void handleKeyStrike(const std::string& key_id, bool is_shifted);
    
    // --- UI Rendering Helpers ---
    void render() const;
    std::vector<std::string> splitTextBuffer() const;
    std::string formatTime(int64_t timestamp_us) const;

    std::unordered_set<std::string> prev_clicks_;
    std::string text_buffer_;

    // --- History Buffer State ---
    size_t max_history_size_ = 10;
    std::deque<Keystroke> history_buffer_;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_TYPING_ENGINE_H