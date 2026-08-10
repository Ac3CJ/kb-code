#include "Application.h"

#include <iostream>

int main() {
    std::cout << "CV Keyboard v0.1.0\n";
    std::cout << "Controls:\n";
    std::cout << "  ESC  — Quit\n";
    std::cout << "  C    — Toggle camera source (laptop / phone)\n";
    std::cout << "  S    — Toggle skeleton mode (finger tips / full 21-point)\n";
    std::cout << "  D    — Toggle debug coordinate overlay\n";
    std::cout << "  G    — Toggle grid overlay\n";
    std::cout << "  K    — Toggle virtual keyboard overlay\n";

    cv_keyboard::Application app;

    if (!app.isRunning()) {
        std::cerr << "Application failed to initialise.\n";
        return 1;
    }

    app.run();

    std::cout << "CV Keyboard shut down cleanly.\n";
    return 0;
}
