#include "TypingEngine.h"

namespace cv_keyboard {

void TypingEngine::reset() {
    prev_clicks_.clear();
    text_buffer_.clear();
    history_buffer_.clear(); // Added clearing of the UI buffer
}

void TypingEngine::processClicks(const std::unordered_set<std::string>& current_clicks, int64_t timestamp_us) {
    // Check Shift
    bool is_shifted = current_clicks.count("L_Shift") || current_clicks.count("R_Shift");
    std::vector<std::string> new_strikes;

    // Find the new clicks that weren't present before
    for (const auto& key_id : current_clicks) {
        if (prev_clicks_.find(key_id) == prev_clicks_.end()) {
            new_strikes.push_back(key_id);
            handleKeyStrike(key_id, is_shifted);
        }
    }

    // 3. Update visualiser for new strikes
    if (!new_strikes.empty()) {
        std::string history_str = "[";
        
        // If Shift is held down, but wasn't pressed this frame, add it for visual clarity
        if (is_shifted && 
            std::find(new_strikes.begin(), new_strikes.end(), "L_Shift") == new_strikes.end() && 
            std::find(new_strikes.begin(), new_strikes.end(), "R_Shift") == new_strikes.end()) {
            history_str += "Shift + ";
        }

        // Combine all simultaneous clicks into one string (e.g., "A + B")
        for (size_t i = 0; i < new_strikes.size(); ++i) {
            history_str += new_strikes[i];
            if (i < new_strikes.size() - 1) history_str += " + ";
        }
        history_str += "]";

        history_buffer_.push_back({history_str, timestamp_us});
        if (history_buffer_.size() > max_history_size_) {
            history_buffer_.pop_front();
        }

        render();
    }

    // Update previous state for the next frame
    prev_clicks_ = current_clicks;
}

void TypingEngine::handleKeyStrike(const std::string& key_id, bool is_shifted) {
    // Ignore modifier keys as direct text inputs
    if (key_id == "L_Shift" || key_id == "R_Shift" || 
        key_id == "L_Ctrl" || key_id == "R_Ctrl" || 
        key_id == "L_Alt" || key_id == "R_AltGr" || 
        key_id == "L_Win" || key_id == "Fn" || key_id == "Option") {
        return;
    }

    if (key_id == "Space") {
        text_buffer_ += " ";
    } else if (key_id == "Backspace") {
        if (!text_buffer_.empty()) {
            text_buffer_.pop_back();
        }
    } else if (key_id == "Enter") {
        // Chat-box style Enter logic
        if (is_shifted) {
            text_buffer_ += "\n";
        } else {
            text_buffer_.clear(); // Send message / flush buffer
        }
    } else if (key_id == "Tab") {
        text_buffer_ += "    ";
    } else if (key_id.length() == 1) {
        // Handle standard alphabetical / numeric characters
        char c = key_id[0];
        
        // Basic UK layout shift mapping
        if (is_shifted) {
            if (c >= 'a' && c <= 'z') c = std::toupper(c);
            else if (c >= 'A' && c <= 'Z') c = c; 
            else if (c == '1') c = '!';
            else if (c == '2') c = '"';
            else if (c == '3') c = '$'; 
            else if (c == '4') c = '$';
            else if (c == '5') c = '%';
            else if (c == '6') c = '^';
            else if (c == '7') c = '&';
            else if (c == '8') c = '*';
            else if (c == '9') c = '(';
            else if (c == '0') c = ')';
            else if (c == '-') c = '_';
            else if (c == '=') c = '+';
            else if (c == '[') c = '{';
            else if (c == ']') c = '}';
            else if (c == ';') c = ':';
            else if (c == '\'') c = '@'; 
            else if (c == '#') c = '~';  
            else if (c == ',') c = '<';
            else if (c == '.') c = '>';
            else if (c == '/') c = '?';
        } else {
            if (c >= 'A' && c <= 'Z') {
                c = std::tolower(c);
            }
        }
        text_buffer_ += c;
    }
}

// ============================================================================
// UI RENDERING LOGIC
// ============================================================================

std::vector<std::string> TypingEngine::splitTextBuffer() const {
    std::vector<std::string> lines;
    std::istringstream stream(text_buffer_);
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }
    // Handle empty buffers or trailing newlines smoothly
    if (text_buffer_.empty() || text_buffer_.back() == '\n') {
        lines.push_back("");
    }
    return lines;
}

std::string TypingEngine::formatTime(int64_t timestamp_us) const {
    double seconds = timestamp_us / 1000000.0;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << seconds << "s";
    return ss.str();
}

void TypingEngine::render() const {
    // Clear terminal and move cursor to top-left (ANSI escape code)
    std::cout << "\033[H\033[J"; 

    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(40) << " KEY HISTORY (TEST MODE)" 
              << "| TEXT BUFFER\n";
    std::cout << "----------------------------------------+---------------------------------------\n";

    std::vector<std::string> right_lines = splitTextBuffer();
    
    size_t total_rows = std::max(history_buffer_.size(), right_lines.size());
    total_rows = std::max(total_rows, max_history_size_);

    for (size_t i = 0; i < total_rows; ++i) {
        // Render Left Side (History)
        std::string left_col = "";
        if (i < history_buffer_.size()) {
            const auto& stroke = history_buffer_[i];
            left_col = "[" + formatTime(stroke.timestamp_us) + "] " + stroke.display_name;
        }
        std::cout << " " << std::left << std::setw(39) << left_col << "| ";

        // Render Right Side (Text Buffer)
        if (i < right_lines.size()) {
            std::cout << right_lines[i];
        }
        std::cout << "\n";
    }
    
    std::cout << "================================================================================\n";
    std::cout << std::flush;
}

} // namespace cv_keyboard