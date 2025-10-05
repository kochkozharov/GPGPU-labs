#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

int main() {
    int n;
    std::cin >> n;
    std::vector<float> a(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> a[i];
    }
    std::sort(a.begin(), a.end());
    for (int i = 0; i < n; ++i) {
        std::cout << std::scientific << std::setprecision(6) << a[i] << " ";
    }
    return 0;
}