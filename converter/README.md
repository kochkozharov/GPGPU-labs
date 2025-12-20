# Image Converter

Конвертер между кастомным бинарным форматом изображений и PNG.

## Формат бинарного файла

- Первые 4 байта: ширина изображения (int, little-endian)
- Следующие 4 байта: высота изображения (int, little-endian)
- Далее построчно данные пикселей
- Каждый пиксель занимает 4 байта в формате RGBA (R, G, B, A)

## Установка

```bash
cd converter
uv sync
```

## Использование

### Конвертация из бинарного формата в PNG

```bash
uv run image-converter to-png input.data output.png
```

### Конвертация из PNG в бинарный формат

```bash
uv run image-converter to-binary input.png output.data
```

### Игнорирование альфа-канала

Флаг `--ignore-alpha` устанавливает альфа-канал в максимальное значение (полная непрозрачность) для всех пикселей:

```bash
uv run image-converter to-png input.data output.png --ignore-alpha
uv run image-converter to-binary input.png output.data --ignore-alpha
```

## Примеры

```bash
# Базовое использование
uv run image-converter to-png lab3/input/09.data output.png
uv run image-converter to-binary output.png converted_back.data

# С игнорированием альфа-канала
uv run image-converter to-png lab3/input/09.data output.png --ignore-alpha
```
