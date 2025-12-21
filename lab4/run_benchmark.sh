#!/bin/bash

# Простой скрипт для запуска бенчмарков с nvprof
# Просто запускает nvprof для различных конфигураций

# Проверка наличия необходимых файлов
if [ ! -f "./target/lab4_cpu" ] || [ ! -f "./target/lab4_benchmark" ]; then
    echo "Error: Executables not found. Please run 'make' first."
    exit 1
fi

# Проверка наличия nvprof
if ! command -v nvprof &> /dev/null; then
    echo "Error: nvprof not found. Please install CUDA Toolkit."
    exit 1
fi

echo "=== CUDA Benchmark for Matrix Inversion ==="
echo ""

# Размеры матриц для тестирования
SIZES=(10 100 1000)

# Конфигурации grid/block для тестирования
# Формат: block_size_x grid_size_x block_size_y grid_size_y
CONFIGS=(
    "1 1 1 1"
    "1 32 1 1"
    "8 8 8 8"
    "16 16 16 16"
    "16 32 32 32"
    "32 32 32 32"
    "32 64 32 32"
    "64 64 32 32"
)

SEED=42

# Создаем директорию для результатов
RESULTS_DIR="benchmark_results"
mkdir -p "$RESULTS_DIR"

# Запуск бенчмарков для каждого размера матрицы
for SIZE in "${SIZES[@]}"; do
    echo "========================================"
    echo "Matrix size: ${SIZE}x${SIZE}"
    echo "========================================"
    echo ""
    
    # CPU бенчмарк
    echo "--- CPU Benchmark ---"
    ./target/lab4_cpu $SIZE $SEED 2>&1 | tee "$RESULTS_DIR/cpu_${SIZE}.txt"
    echo ""
    
    # CUDA бенчмарки с различными конфигурациями
    for CONFIG in "${CONFIGS[@]}"; do
        read -r BLOCK_SIZE_X GRID_SIZE_X BLOCK_SIZE_Y GRID_SIZE_Y <<< "$CONFIG"
        
        # Проверяем, что конфигурация не превышает лимиты CUDA
        TOTAL_BLOCK=$((BLOCK_SIZE_X * BLOCK_SIZE_Y))
        if [ $TOTAL_BLOCK -gt 1024 ]; then
            echo "Skipping config: block=($BLOCK_SIZE_X,$BLOCK_SIZE_Y) grid=($GRID_SIZE_X,$GRID_SIZE_Y) (block size too large: $TOTAL_BLOCK)"
            continue
        fi
        
        echo "--- CUDA Config: block=($BLOCK_SIZE_X,$BLOCK_SIZE_Y) grid=($GRID_SIZE_X,$GRID_SIZE_Y) ---"
        
        # Запускаем nvprof с дефолтным выводом
        OUTPUT_FILE="$RESULTS_DIR/nvprof_${SIZE}_${BLOCK_SIZE_X}_${GRID_SIZE_X}_${GRID_SIZE_Y}.txt"
        
        nvprof ./target/lab4_benchmark $SIZE $BLOCK_SIZE_X $GRID_SIZE_X $BLOCK_SIZE_Y $GRID_SIZE_Y $SEED 2>&1 | tee "$OUTPUT_FILE"
        
        echo ""
    done
    
    echo ""
done

echo "=== Benchmark completed ==="
echo "Results saved in: $RESULTS_DIR/"
