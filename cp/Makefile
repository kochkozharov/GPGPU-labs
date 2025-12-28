# Makefile для Ray Tracing на CUDA
# Вариант 3: Тетраэдр, Гексаэдр, Икосаэдр

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11

TARGET = raytracing
SRC = raytracing.cu

.PHONY: all clean run run-cpu run-gpu default test
.PHONY: run1 run2 run3 run4 run5 run-all

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
	rm -rf output

# Создать директории для выходных данных
output:
	mkdir -p output

output/result_1:
	mkdir -p output/result_1

output/result_2:
	mkdir -p output/result_2

output/result_3:
	mkdir -p output/result_3

output/result_4:
	mkdir -p output/result_4

output/result_5:
	mkdir -p output/result_5

# Запуск с GPU (по умолчанию, старый тест)
run: $(TARGET) output
	./$(TARGET) --gpu < input/test

# Запуск с CPU
run-cpu: $(TARGET) output
	./$(TARGET) --cpu < input/test

# Запуск с GPU
run-gpu: $(TARGET) output
	./$(TARGET) --gpu < input/test

# Вывод конфигурации по умолчанию
default: $(TARGET)
	./$(TARGET) --default

# Тестовый запуск с небольшим количеством кадров
test: $(TARGET) output
	./$(TARGET) --gpu < input/test_small

# Конфигурация 1: 1280x720, 120 кадров, SSAA 4x4
run1: $(TARGET) output/result_1
	./$(TARGET) --gpu < input/config1

# Конфигурация 2: 1920x1080, 100 кадров, SSAA 4x4 (самая тяжелая)
run2: $(TARGET) output/result_2
	./$(TARGET) --gpu < input/config2

# Конфигурация 3: 1280x720, 150 кадров, SSAA 4x4
run3: $(TARGET) output/result_3
	./$(TARGET) --gpu < input/config3

# Конфигурация 4: 1600x900, 80 кадров, SSAA 4x4
run4: $(TARGET) output/result_4
	./$(TARGET) --gpu < input/config4

# Конфигурация 5: 1280x720, 200 кадров, SSAA 4x4
run5: $(TARGET) output/result_5
	./$(TARGET) --gpu < input/config5

# Запуск всех конфигураций
run-all: run1 run2 run3 run4 run5
	@echo "All configurations completed!"

# CPU версии для сравнения
run1-cpu: $(TARGET) output/result_1
	./$(TARGET) --cpu < input/config1

run2-cpu: $(TARGET) output/result_2
	./$(TARGET) --cpu < input/config2
