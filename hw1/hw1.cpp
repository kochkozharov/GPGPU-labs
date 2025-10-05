#include <iostream>
#include <cmath>
#include <iomanip>

void log_incorrect_coeffs() {
    std::cout << "incorrect\n";
}

template <typename T>
int sign(T x, T epsilon = std::numeric_limits<T>::epsilon()) {
    return (std::fabs(x) < epsilon) ? 0 : (x > 0 ? 1 : -1);
}

int main() {
    float a, b, c;
    if (! (std::cin >> a >> b >> c)) {
        log_incorrect_coeffs();
        return 0;
    }

    if (sign(a) == 0) {
        if (sign(b) == 0) {
            if (sign(c) == 0) {
                std::cout << "any\n";
            } else {
                log_incorrect_coeffs();
            }
        } else {
            float x = -c / b;
            std::cout << std::fixed << std::setprecision(6) << x << '\n';
        }
    } else {
        float D = b*b - 4.0f*a*c;

        if (sign(D) > 0) {
            float sd = sqrt(D);
            float x1 = (-b + sd) / (2.0f * a);
            float x2 = (-b - sd) / (2.0f * a);
            std::cout << std::fixed << std::setprecision(6) << x1 << ' ' << x2 << '\n';
        } else if (sign(D) == 0) {
            float x = -b / (2.0f * a);
            std::cout << std::fixed << std::setprecision(6) << x << '\n';
        } else {
            std::cout << "imaginary\n";
        }
    }

    return 0;
}