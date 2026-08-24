#include "Application.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    cv_keyboard::Settings settings;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tracker" && i + 1 < argc) {
            settings.tracker_name = argv[++i];
        } else if (arg == "--processor" && i + 1 < argc) {
            settings.processor_name = argv[++i];
        }
    }

    std::cout << "CV Keyboard v0.1.0\n";
    std::cout << "Tracker: " << settings.tracker_name << ", Processor: " << settings.processor_name << "\n";
    std::cout << "Controls:\n";
    std::cout << "  ESC  — Quit\n";
    std::cout << "  C    — Toggle camera source (laptop / phone)\n";
    std::cout << "  S    — Toggle skeleton mode (finger tips / full 21-point)\n";
    std::cout << "  D    — Toggle debug coordinate overlay\n";
    std::cout << "  G    — Toggle grid overlay\n";
    std::cout << "  K    — Toggle virtual keyboard overlay\n";

    cv_keyboard::Application app(settings);

    if (!app.isRunning()) {
        std::cerr << "Application failed to initialise.\n";
        return 1;
    }

    app.run();

    std::cout << "CV Keyboard shut down cleanly.\n";
    return 0;
}
