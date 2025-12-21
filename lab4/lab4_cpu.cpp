#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "matrix_generator.h"

constexpr double EPS = 1e-7;

// CPU версия алгоритма Гаусса-Жордана для вычисления обратной матрицы
std::vector<double> computeInverseCPU(const std::vector<double>& matrix, int n) {
    int width = 2 * n;
    std::vector<double> augmented(n * width, 0.0);
    
    // Создаем расширенную матрицу [A | I]
    // Транспонируем исходную матрицу (как в CUDA версии)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[j * n + i] = matrix[i * n + j];
        }
    }
    
    // Добавляем единичную матрицу справа
    for (int i = 0; i < n; i++) {
        augmented[(n + i) * n + i] = 1.0;
    }
    
    // Метод Гаусса-Жордана
    for (int k = 0; k < n; k++) {
        // Поиск максимального элемента в столбце k (начиная с строки k)
        int maxIdx = k;
        double maxVal = std::fabs(augmented[k * n + k]);
        
        for (int i = k + 1; i < n; i++) {
            double val = std::fabs(augmented[k * n + i]);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        
        // Проверка на сингулярность
        if (maxVal < EPS) {
            std::cerr << "Matrix is singular or nearly singular\n";
            return std::vector<double>();
        }
        
        // Перестановка строк
        if (maxIdx != k) {
            for (int col = 0; col < width; col++) {
                std::swap(augmented[col * n + k], augmented[col * n + maxIdx]);
            }
        }
        
        double pivotVal = augmented[k * n + k];
        
        // Деление pivot-строки на pivot-элемент
        for (int col = k; col < width; col++) {
            augmented[col * n + k] /= pivotVal;
        }
        
        // Исключение
        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            
            double factor = augmented[k * n + i];
            for (int col = k + 1; col < width; col++) {
                augmented[col * n + i] -= factor * augmented[col * n + k];
            }
        }
    }
    
    // Извлечение обратной матрицы
    std::vector<double> inverse(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[j * n + i] = augmented[(n + j) * n + i];
        }
    }
    
    return inverse;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> [seed]\n";
        std::cerr << "Example: " << argv[0] << " 100 42\n";
        return EXIT_FAILURE;
    }
    
    int n = std::stoi(argv[1]);
    unsigned int seed = (argc > 2) ? std::stoul(argv[2]) : 42;
    
    if (n <= 0) {
        std::cerr << "Invalid matrix size\n";
        return EXIT_FAILURE;
    }
    
    MatrixGenerator gen(seed);
    std::vector<double> matrix = gen.generateWellConditionedMatrix(n);
    
    // Замер времени
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> inverse = computeInverseCPU(matrix, n);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (inverse.empty()) {
        return EXIT_FAILURE;
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    
    // Вывод результата
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU Time: " << time_ms << " ms\n";
    
    // Опционально: вывод матрицы (закомментировано для бенчмарка)
    /*
    std::cout << std::scientific << std::setprecision(10);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > 0) std::cout << " ";
            std::cout << inverse[j * n + i];
        }
        std::cout << "\n";
    }
    */
    
    return 0;
}

