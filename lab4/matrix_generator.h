#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <vector>
#include <random>

class MatrixGenerator {
public:
    MatrixGenerator(unsigned int seed = 42);
    
    // Генерирует случайную матрицу n x n
    std::vector<double> generateMatrix(int n);
    
    // Генерирует хорошо обусловленную матрицу (для избежания сингулярности)
    std::vector<double> generateWellConditionedMatrix(int n);

private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;
};

#endif // MATRIX_GENERATOR_H

