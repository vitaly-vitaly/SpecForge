## План исправлений под Qwen2.5-VL-32B

### 1. Конфиг драфт-модели для Qwen2.5-VL-32B

- [ ] Создать файл по аналогии с `configs/qwen2-5-vl-7b-eagle3.json`:
  - `configs/qwen2_5_vl_32b_eagle3.json`
- [ ] В конфиге указать:
  - `hidden_size`: `5120`  
    (совпадает с размерностью скрытого слоя у Qwen2.5-VL-32B)
- [ ] Проверить, что словарь (vocab) драфта согласован с таргетом через `vocab_mapping`  
  (см. `t2d` / `d2t`; при первом шаге можно сделать их тождественными, а потом оптимизировать).

---

### 2. Обобщение `QwenVLOnlineEagle3Model` на 32B

Файл: `specforge/core/eagle3.py`

- [ ] Убедиться, что класс `QwenVLOnlineEagle3Model`:
  - не жёстко привязан к `"Qwen/Qwen2.5-VL-7B-Instruct"`;
  - принимает путь к таргет-модели из аргумента, например `target_model_path`.
- [ ] Прокинуть `target_model_path`:
  - из `scripts/train_eagle3_online.py` (CLI-аргумент `--target-model-path`);
  - в конструктор `QwenVLOnlineEagle3Model`.
- [ ] Использовать общую фабрику загрузки моделей SpecForge:
  - та, что уже умеет работать с tensor parallelism (TP — tensor parallelism, тензорный параллелизм) и FSDP (Fully Sharded Data Parallel — полностью фрагментированный дата-параллелизм) для текстовых моделей;
  - обернуть загрузку `Qwen2.5-VL` такой же логикой, но с учётом vision-части.
- [ ] Проверить обработку мультимодальных входов:
  - корректная работа с `pixel_values`, `image_grid_thw`, `input_ids`, `attention_mask`;
  - формирование `input_embeds` (текст + визуальные эмбеддинги) не должно зависеть от размера модели (7B vs 32B).

---

### 3. Доработка `train_eagle3_online.py` под Qwen2.5-VL-32B

Файл: `scripts/train_eagle3_online.py`

- [ ] Добавить/проверить аргументы CLI:
  - `--target-model-path` (строка, путь/имя таргет-модели);
  - `--draft-model-config` (путь к `qwen2_5_vl_32b_eagle3.json`);
  - `--chat-template qwen2-vl` (или эквивалент для Qwen-VL).
- [ ] При инициализации:
  - создать экземпляр `QwenVLOnlineEagle3Model` с указанными путями и конфигами;
  - передавать туда параметры параллелизма (`tp_size`, world size и т.п.).
- [ ] Проверить, что:
  - датасет-лоадер отдаёт `pixel_values`, `image_grid_thw`, `input_ids`, `loss_mask`, `attention_mask`;
  - `plosses` (per-step losses — лоссы по шагам) корректно агрегируются в итоговый токен-лосс (cross entropy, кросс-энтропия);
  - логирование acceptance-метрик (`acces`) работает и не привязано к размеру модели.

---

### 4. Shell-скрипт запуска под 32B

Файл: `examples/run_qwen2_5_vl_32b_eagle3_online.sh`

- [ ] Создать новый скрипт на основе `run_qwen2_5_vl_eagle3_online.sh` (для 7B):
  - заменить:
    - `--target-model-path` → `Qwen/Qwen2.5-VL-32B-Instruct`;
    - `--draft-model-config` → `configs/qwen2_5_vl_32b_eagle3.json`;
  - уменьшить `--batch-size` до 1–2 для начала;
  - добавить параметры параллелизма:
    - `NUM_GPUS` (число GPU);
    - `TP_SIZE` (размер tensor parallelism, TP — tensor parallelism, тензорный параллелизм) — начально 1.
- [ ] Протестировать запуск:
  - сначала `NUM_GPUS=1, TP_SIZE=1` (без тензорного параллелизма),
  - убедиться, что модель 32B влезает в память A100 80GB при `batch_size=1` и типе данных BF16 (bfloat16 — формат числа с плавающей точкой).

---

### 5. Поддержка TP > 1 для Qwen2.5-VL-32B

Если на одном GPU не хватает памяти или нужен больший throughput:

- [ ] Реализовать поддержку TP > 1 по аналогии с текстовыми моделями:
  - использовать существующий механизм инициализации TP/FSDP в SpecForge;
  - настроить группы процессов для tensor parallelism;
  - убедиться, что:
    - текстовый блок Qwen-VL шардируется корректно по TP;
    - vision-энкодер и fusion-часть совместимы с таким режимом.
- [ ] В `run_qwen2_5_vl_32b_eagle3_online.sh`:
  - разрешить значения `TP_SIZE=2/4/8` (в зависимость от числа A100);
  - при необходимости добавить отдельный флаг `--fsdp` для FSDP (Fully Sharded Data Parallel — полностью фрагментированный дата-параллелизм).

---

### 6. Минимальный sanity-check после правок

- [ ] Локальный прогон на маленьком подмножестве данных (1000 примеров `allava4v`):
  - `NUM_GPUS=1`, `TP_SIZE=1`, `batch_size=1`, `num_epochs=1`;
  - убедиться, что:
    - `generate_vocab_mapping_file` отрабатывает до конца;
    - лосс убывает хотя бы на маленьком датасете;
    - checkpoint и `eagle3-config.json` сохраняются.
- [ ] Тестовый запуск через SGLang:
  - поднять сервер с `Qwen/Qwen2.5-VL-32B-Instruct` + новым EAGLE3-драфтом;
  - проверить визуальный запрос/ответ (VQA) и измерить ускорение по tokens/s.

