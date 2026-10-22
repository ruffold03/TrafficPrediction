# Прогноз загруженности участков дорожной сети

Система из расчётно-пояснительной записки: YOLO и ByteTrack считают транспорт на видео, календарь и погода Open-Meteo дополняют ряд, STGCN с gating прогнозирует скорость на 15, 30 и 60 минут.

## Подготовка

```text
docker compose up -d
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m traffic_prediction.forecast.train
.venv\Scripts\python -m traffic_prediction.video.cli
.venv\Scripts\python -m uvicorn traffic_prediction.api.main:app --app-dir src
```

Интерфейс открывается на http://127.0.0.1:8000 . PostgreSQL публикуется на порту 5433, чтобы не пересечься с локальным сервером на 5432. Если база недоступна, приложение сохраняет данные в `data/traffic.db`.

Обучение идёт на подграфе METR-LA (размер задаётся в `configs/default.yaml`, поле `max_nodes`). Видео AIC21 используется для подсчёта, а не как многонедельный обучающий ряд: для ролика прогноз строится через нормировку интенсивности в скорость первого датчика графа.
