# Makefile для Ray Tracing на CUDA
# Вариант 3: Тетраэдр, Гексаэдр, Икосаэдр

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11

TARGET = raytracing
SRC = raytracing.cu

.PHONY: all clean run run-cpu run-gpu default test

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
	rm -rf output

# Создать директорию для выходных данных
output:
	mkdir -p output

# Запуск с GPU (по умолчанию)
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

