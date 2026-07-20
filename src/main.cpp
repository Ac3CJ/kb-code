#include "Application.h"

#include <iostream>

int main() {
    std::cout << "CV Keyboard v0.1.0\n";
    std::cout << "Controls: ESC = quit, C = toggle camera source\n";

    cv_keyboard::Application app;

    if (!app.isRunning()) {
        std::cerr << "Application failed to initialise.\n";
        return 1;
    }

    app.run();

    std::cout << "CV Keyboard shut down cleanly.\n";
    return 0;
}