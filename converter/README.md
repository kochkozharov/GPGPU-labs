# Image Converter

Конвертер между бинарным форматом изображений (используемым в лабораторных работах по GPGPU) и стандартными форматами (PNG, MP4).

## Установка

```bash
cd converter
uv sync
```

Или с pip:

```bash
pip install -e .
```

## Формат бинарных изображений

Бинарный формат `.data`:
- Первые 4 байта: ширина (int32, little-endian)
- Следующие 4 байта: высота (int32, little-endian)
- Остальные байты: пиксели построчно, каждый пиксель — 4 байта (R, G, B, A)

## Использование

### Конвертация изображений

```bash
# Бинарный формат -> PNG
image-converter to-png input.data output.png

# PNG -> бинарный формат
image-converter to-binary input.png output.data

# С игнорированием alpha-канала (устанавливается в 255)
image-converter to-png input.data output.png --ignore-alpha
image-converter to-binary input.png output.data --ignore-alpha
```

### Создание видео из кадров

```bash
# Из последовательности кадров (с паттерном %d)
image-converter to-video "frames/frame_%d.data" output.mp4 --fps 30

# Из всех .data файлов в директории
image-converter to-video-dir ./frames output.mp4 --fps 30
```

#### Опции для видео

- `--fps N` — частота кадров (по умолчанию: 30)
- `--start-frame N` — начальный номер кадра (по умолчанию: 0)
- `--end-frame N` — конечный номер кадра (по умолчанию: автоопределение)
- `--codec CODEC` — видеокодек (по умолчанию: libx264)
- `--crf N` — качество видео, 0-51, меньше = лучше (по умолчанию: 23)
- `--pattern GLOB` — шаблон для поиска файлов в директории (по умолчанию: *.data)
- `--no-ignore-alpha` — сохранить оригинальный alpha-канал

### Требования для создания видео

Для создания видео требуется установленный **ffmpeg**:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg
```

## Примеры

```bash
# Конвертировать результаты ray tracing в видео
image-converter to-video "./output/frame_%d.data" animation.mp4 --fps 24

# Высокое качество видео
image-converter to-video "./output/frame_%d.data" animation_hq.mp4 --fps 30 --crf 18

# Из директории с произвольными именами файлов
image-converter to-video-dir ./rendered_frames result.mp4 --pattern "*.data" --fps 60
```
