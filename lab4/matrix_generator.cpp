#include "matrix_generator.h"
#include <cmath>
#include <algorithm>

MatrixGenerator::MatrixGenerator(unsigned int seed) 
    : gen(seed), dist(-10.0, 10.0) {
}

std::vector<double> MatrixGenerator::generateMatrix(int n) {
    std::vector<double> matrix(n * n);
    for (int i = 0; i < n * n; i++) {
        matrix[i] = dist(gen);
    }
    return matrix;
}

std::vector<double> MatrixGenerator::generateWellConditionedMatrix(int n) {
    // Генерируем матрицу, которая с большой вероятностью будет невырожденной
    // Используем диагонально доминирующую матрицу
    std::vector<double> matrix(n * n, 0.0);
    
    for (int i = 0; i < n; i++) {
        double rowSum = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                matrix[i * n + j] = dist(gen) * 0.1; // Малые недиагональные элементы
                rowSum += std::abs(matrix[i * n + j]);
            }
        }
        // Диагональный элемент больше суммы остальных
        matrix[i * n + i] = rowSum + dist(gen) + 1.0;
    }
    
    return matrix;
}

