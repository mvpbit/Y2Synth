# Magenta DrumsRNN — проверка применимости к Y2Synth

**Дата исследования:** 2026-11-16  
**Статус:** подтверждено первичными источниками / эксперимент на Mac ещё не выполнен / на Y2 не проверено.

## Задача

Проверить, можем ли мы использовать уже обученную Magenta DrumsRNN для **локальной генерации барабанных MIDI-паттернов**. Обучение модели на самом Y2 не предполагается. Нейросеть не должна выполнять вычисления в аудиопотоке.

## Установлено по исходникам Magenta

1. В официальном README DrumsRNN опубликован [готовый bundle `drum_kit_rnn.mag`](https://download.magenta.tensorflow.org/models/drum_kit_rnn.mag); указан пример генерации с `drums_rnn_generate --config=drum_kit --bundle_file=...`.
2. Конфигурация `drum_kit` сводит ударные к **9 классам**; комбинации ударов на одном шаге представлены как **512 состояний** (2^9). Кроме того, вход включает двоичные счётчики.
3. В [исходной конфигурации модели](https://github.com/magenta/magenta/blob/main/magenta/models/drums_rnn/drums_rnn_model.py) заданы `rnn_layer_sizes=[256,256,256]`, `attn_length=32` и `batch_size=128` для обучения. **Это параметры конфигурации исходников; параметры конкретного скачанного bundle ещё не сверены.** Уменьшение слоёв до `[64,64]` в документации является примером **обучения новой модели**, не способом уменьшить готовые веса.
4. Формат `.mag` — сериализованный `GeneratorBundle`: он содержит checkpoint модели, metagraph и метаданные. В [`unpack_bundle.py`](https://github.com/magenta/magenta/blob/main/magenta/scripts/unpack_bundle.py) разработчики показывают извлечение `checkpoint_file[0]` и `metagraph_file` посредством TensorFlow. Это **не** самостоятельный файл весов для прямой загрузки в Rust.
5. Для генерации в Magenta используется **TensorFlow в режиме совместимости с TF1**. Перенос только весов требует корректной реализации вычислительного графа, кодирования входа, рекуррентного состояния и логики декодирования.
6. Исходный [репозиторий Magenta](https://github.com/magenta/magenta) архивирован 2026-01-06. Код распространяется под [Apache-2.0](https://github.com/magenta/magenta/blob/main/LICENSE). **Применимость этой лицензии к распространяемому отдельно обученному bundle требует отдельной проверки**, не следует автоматически приравнивать лицензии кода и весов.

Источники: [официальный README DrumsRNN](https://github.com/magenta/magenta/blob/main/magenta/models/drums_rnn/README.md), [конфигурация модели](https://github.com/magenta/magenta/blob/main/magenta/models/drums_rnn/drums_rnn_model.py), [генератор](https://github.com/magenta/magenta/blob/main/magenta/models/drums_rnn/drums_rnn_generate.py), [распаковщик bundle](https://github.com/magenta/magenta/blob/main/magenta/scripts/unpack_bundle.py).

## Проверка файла bundle

Выполнить в **корне локального клона Y2Synth**:

```bash
bash experiments/drumsrnn/download-model.sh
```

После ошибки TLS у старого хоста (сертификат не соответствует `download.magenta.tensorflow.org`) скрипт переключён на **официальный конвертированный Magenta.js checkpoint**, размещённый на Google Cloud Storage. Он скачивает `config.json`, `weights_manifest.json` и перечисленные в манифесте бинарные файлы весов в `experiments/drumsrnn/models/drum_kit_rnn_tfjs/` (папка игнорируется Git), проверяет базовую структуру манифеста и выводит размеры и SHA-256. Это **не файл `.mag`**. SHA-256 пока не сопоставляется с официальным эталоном. Успешная загрузка не доказывает совместимость с Rust или Y2.

**Фактический результат на дату исследования:** первоначальная попытка загрузить `.mag` на Mac завершилась ошибкой `curl: (60) SSL: no alternative certificate subject name matches target host name`; отключать TLS-проверку нельзя. Ветку обновили для использования официальных конвертированных весов из [каталога Magenta.js](https://github.com/magenta/magenta-js/blob/master/music/checkpoints/checkpoints.json), где `drum_kit_rnn` указан как MusicRNN, конвертированная из оригинального `.mag`, приблизительно 11.9 МБ. Новая загрузка на Mac ещё не подтверждена.

После успешной загрузки Magenta.js: прочитать `config.json`, `weights_manifest.json`, перечислить тензоры, их формы, типы, суммарный объём и параметры attention. Сравнить список переменных с [реализацией MusicRNN на TypeScript](https://github.com/magenta/magenta-js/blob/master/music/src/music_rnn/model.ts). Это удобный исходный формат для экспериментального Rust-парсера, но **не готовый Rust runtime**. Распаковка оригинального `.mag` остаётся отдельным необязательным исследованием.

## Гипотезы интеграции

**A. Конвертация TF1-графа.** Извлечь checkpoint+metagraph, восстановить inference-граф, проверить экспорт в переносимый формат; пробовать `tract` лишь после проверки фактического набора операторов, формы входов/выходов и обработки LSTM/attention. [Документация tract](https://github.com/sonos/tract) описывает поддержку некоторых TF1 frozen graph, но **не гарантирует поддержку DrumsRNN**.

**B. Специализированный inference на Rust.** Извлечь конкретные массивы весов и реализовать модель, операции, последовательность входных событий и декодер; сравнить распределения/выходы с оригинальным TensorFlow на фиксированных тестах. Предпочтительно исследовать, если конвертация графа слишком сложна или runtime слишком велик для 32-битной ARM-платформы.

**Не выбран ни один вариант:** нужны сведения из bundle и результат генерации-эталона на Mac.

## Критерии следующего эксперимента на Mac

- [ ] Успешно скачать официальный Magenta.js checkpoint (`config.json`, `weights_manifest.json`, бинарные shard-файлы), зафиксировать SHA-256, точные размеры и источник. Оригинальный `.mag` пока недоступен по HTTPS.
- [ ] Проверить JSON-манифест, реальные тензоры и конфигурацию **фактически сохранённой модели**.
- [ ] Воспроизвести генерацию Magenta.js MusicRNN на Mac, сохранить MIDI и версии зависимостей.
- [ ] Получить эталонные входы и результаты inference (фиксировать seed, когда возможно).
- [ ] Проверить корректность результатов Rust/conversion по сравнению с эталоном.
- [ ] Измерить пиковое потребление памяти и время генерации с описанием окружения.
- [ ] Отдельно подтвердить возможность запуска на Y2Linux (CPU ABI, runtime, память).
- [ ] На Y2 измерить выполнение AI параллельно DSP, без выпадений аудио.

## Ограничение архитектуры

Планируемый путь: `Application pattern -> AI worker -> candidate MIDI events -> validation -> Application -> DSP sequencer`. Модель генерирует **заранее**, а аудиопоток только воспроизводит подготовленные события. AI не входит в realtime callback; недоступность модели не должна останавливать синтезатор.

**Вывод:** существует официальный конвертированный checkpoint той же pretrained DrumsRNN, пригодный для дальнейшего исследования. **Запуск на Innioasis Y2 пока не доказан**; его выполнимость зависит от конкретных весов, runtime, CPU/RAM и совместимости Y2Linux. Никакие прошивки или аппаратные изменения в ходе исследования не производились.


## Эксперимент 1 — загрузка на Mac (2026-11-16)

**Статус:** скачивание подтверждено выводом команды пользователя на Mac, анализ тензоров ещё не выполнен.

```text
config.json            237 bytes       SHA256 29da526d5830344bb6b9cbad6a506671fc01cf183fda353116bec0ba209fa041
weights_manifest.json  1,758 bytes     SHA256 3f64e0300de500d59b74cb33d7c009dc6630491885d90dbaf93fa422e0798764
group1-shard1of3       4,194,304 bytes SHA256 b822cab0c4913db6f0e6a69e33bd82f0a832a0e18e5d2e22d8a0f676b3c668f0
group1-shard2of3       4,194,304 bytes SHA256 2e211975ea0d68b07c63faffd0e990bd9fb0f57c3260a1c32f19935826708c82
group1-shard3of3       3,482,792 bytes SHA256 024cd22e7f0ed9777599027b399ce330607c920c4e22057586f3a9b2f5706ee9
```

**Всего бинарных весов:** 11,871,400 bytes (11.32 MiB). Эти SHA-256 — измерения локальных файлов, **не подтверждённые производителем эталонные значения**. `curl` завершился без ошибки, но сами тензоры пока не исследованы.

### Эксперимент 2 — разбор весов без TensorFlow

Добавлен `experiments/drumsrnn/inspect_checkpoint.py` на стандартной библиотеке Python. Для запуска **из корня локального репозитория**:

```bash
git pull --ff-only
python3 experiments/drumsrnn/inspect_checkpoint.py
```

Программа выведет `config.json`, имена, типы и формы тензоров из манифеста, суммарное число элементов, SHA-256 shard-файлов и сравнит размеры с ожидаемыми по dtype/shape. **Скрипт не выполняет inference, не сверяет численные значения и не подтверждает происхождение весов**; если размер не совпадает, завершится с ошибкой.

После получения вывода: сопоставить формы ядер LSTM и attention с исходниками `music_rnn/model.ts`, оценить стоимость inference и только после этого выбрать вариант Rust runtime.

Источники для структуры файлов и модели:
- [Magenta.js — pre-trained checkpoints](https://github.com/magenta/magenta-js/blob/master/music/checkpoints/README.md)
- [Magenta.js MusicRNN implementation](https://github.com/magenta/magenta-js/blob/master/music/src/music_rnn/model.ts)
- [TensorFlow.js weight manifest writer](https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/python/tensorflowjs/write_weights.py)


## Эксперимент 2 — проверка манифеста на Mac (2026-11-16)

**Статус: выполнено на Mac, по выводу скрипта пользователя.** Контрольным хешам бинарников соответствуют те же файлы, которые были скачаны на предыдущем шаге.

- `config.json`: `MusicRNN`, `DrumsOneHotConverter`, `BinaryCounter(numBits=6)`, `attentionLength=32`.
- 16 тензоров в 1 группе, только `float32`.
- 2,967,850 числовых элементов × 4 байта = **11,871,400 байт** (11.32 MiB).
- Три LSTM-слоя: ядра `[774, 1024]`, `[512, 1024]`, `[512, 1024]`; bias каждого слоя `[1024]`. Четыре gate-вектора по 256 значений на слой.
- Выходное отображение: `fully_connected/weights [256,512]`, bias `[512]`.
- Веса attention присутствуют, длина памяти 32 шага.
- Проверка `manifest expected bytes == sum(shard file bytes)`: **OK**.

**Вывод:** размеры и структура скачанных checkpoint файлов согласованы. Их численная корректность, реальное исполнение inference и производительность на Y2 **не проверены**. Размер весов не равен пиковому расходу RAM в приложении.

### Эксперимент 3 — эталонный inference в браузере на Mac

Добавлен `experiments/drumsrnn/reference.html` (пока **не протестирован в браузере**): оригинальная `@magenta/music@1.23.1` загружается с CDN; модель **читает уже скачанные локальные веса** с простого HTTP-сервера и продолжает фиксированный 16-шаговый барабанный паттерн ещё на 16 шагов. Sampling **argmax** (без случайности); экспортирует JSON, включая распределения вероятностей 16×512, длительность загрузки и генерации.

В **корне Y2Synth** на Mac:

```bash
git pull --ff-only
python3 -m http.server 8765 --bind 127.0.0.1
```

Затем открыть:

```text
http://127.0.0.1:8765/experiments/drumsrnn/reference.html
```

Нажать **Generate reference**, затем **Save JSON**. Записать версии браузера и macOS, вывод, время генерации и ошибки, если возникнут. Файл JSON по умолчанию окажется в папке загрузок браузера; в Git его не добавлять, пока мы не решим, какие данные действительно нужны для тестов. Для работы требуется Интернет **для JS-библиотеки с CDN**, но **не для весов**. Тесты браузера не отражают время выполнения на слабом ARM CPU и не подтверждают совместимость с Y2Linux.

Основной API: [MusicRNN.continueSequenceAndReturnProbabilities](https://magenta.github.io/magenta-js/music/classes/_music_rnn_model_.musicrnn.html); [исходный код](https://github.com/magenta/magenta-js/blob/master/music/src/music_rnn/model.ts). 

## Эксперимент 3 — первый браузерный inference (отчёт пользователя, 2026-11-17)

**Статус: выполнено на Mac в локальном браузере.** Результат получен через `reference.html` и `@magenta/music@1.23.1`, загрузка checkpoint из локальной директории.

- Время инициализации: **29.0 ms**.
- Генерация продолжения **16 шагов:** **10,801.0 ms** (~675 ms на шаг).
- Выходная форма вероятностей: **16 × 512**.
- 12 барабанных ударов в продолжении, полностью совпавших с исходным `primer`:
  `(0,36),(0,42),(2,42),(4,38),(4,42),(6,42),(8,36),(8,42),(10,42),(12,38),(12,42),(14,42)`.
- Поскольку оригинальный `continueSequenceAndReturnProbabilities` декодирует только `samples`, а не объединяет `primer + samples`, повторяющийся паттерн **может быть настоящим предсказанием модели** при детерминированном argmax; сам по себе он не указывает на ошибку отображения. Для уверенности требуется контрольный запуск с другим входом и анализ вероятностей.

**Ограничения измерений:** браузерная реализация и время копирования распределений вероятностей; неизвестны browser/tfjs backend, состояние кэша и аппаратные параметры запуска. 10.8 сек нельзя использовать как оценку native Rust или 32-bit ARM Y2. Хэши весов проверены на совпадение размеров, но не на эталонное происхождение.

**Следующий контрольный тест:** обновлена `reference.html` с выбором `Regular groove` и `Syncopated groove`, сравнением продолжения с входом и средней максимальной вероятностью. После `git pull --ff-only` перезагрузить страницу, сгенерировать обе последовательности и сравнить результаты. Сохранить JSON отдельно для последующего сравнения с Rust.

## Эксперимент 4 — другой primer, повторный запуск (отчёт пользователя, 2026-11-17)

**Статус:** успешный запуск на Mac через браузер, вторая последовательность `syncopated`.

- Время загрузки модели: **32.0 мс**.
- Генерация 16 шагов: **361.0 мс** (примерно 22.6 мс/шаг по общему времени вызова).
- Primer: 13 ударов; continuation: 13 ударов.
- **Точное совпадение continuation с primer:** `YES`.
- Средняя максимальная вероятность класса по шагам: **0.9532**; высокая уверенность именно в выборе наиболее вероятного класса.
- В предыдущем тесте (regular primer) измерено **10,801.0 мс** на 16 шагов. Разницу в ~30 раз нельзя пока интерпретировать как устойчивое ускорение: без повторений и диагностики браузерного backend вероятны прогрев, компиляция шейдеров и/или кэширование. Время инициализации отдельно от генерации **не гарантирует** исключения всех расходов первого вычисления.

По [исходникам `MusicRNN.continueSequenceImpl`](https://github.com/magenta/magenta-js/blob/master/music/src/music_rnn/model.ts) возвращается именно `samples` (генерация), а не склейка с `primer`; это подтверждает, что точное совпадение **само по себе не указывает на ошибку отображения**. Но пока нельзя исключить другие особенности модели или конфигурации.

**Следующий диагностический тест:** в `reference.html` добавлен выбор sampling (`argmax`, `temperature 1.0`, `temperature 1.2`). Запустить тот же syncopated primer при температуре 1.0 и 1.2, желательно по 2–3 раза, и сравнить возвращаемые паттерны. Значение `temperature` передаётся третьим аргументом в оригинальный API; при отсутствии аргумента выбирается `argmax`. При `temperature>0` модель применяет multinomial sampling. Эти проверки должны установить, выдаёт ли модель вариации, но сами по себе не докажут её пригодность для Y2.

Новые результаты также потребуют измерения серии повторных прогонов для оценки медианной задержки и cold-start стоимости.

## Эксперимент 5 — стохастическая вариация при temperature=1.2 (отчёт пользователя, 2026-11-17)

**Статус:** успешный inference на Mac через Magenta.js, вывод пользователя. Это первый зарегистрированный пример **отличного от primer** варианта.

- Model initialization: **14.0 ms**.
- Generation, 16 steps: **243.0 ms** (среднее по вызову **15.2 ms/step**, но это не задержка каждого шага отдельно).
- Primer: `syncopated`, 13 MIDI hits, сэмплирование с `temperature=1.2`.
- Continuation: **13 MIDI hits**, флаг сравнения с primer **NO**.
- Mean maximum class probability: **0.8118** (среднее по максимальным вероятностям классов, не «общая уверенность» в сгенерированной последовательности).
- Вход (step,pitch): `(0,36),(1,42),(3,42),(4,38),(5,42),(7,36),(7,42),(9,42),(10,36),(11,42),(12,38),(13,42),(15,42)`.
- Выход: `(0,42),(1,42),(3,42),(4,38),(5,42),(7,36),(7,42),(9,36),(9,42),(11,42),(12,38),(13,42),(15,42)`.
- Отличия: удалены `(0,36)` и `(10,36)`, добавлены `(0,42)` и `(9,36)` — четыре изменения в бинарной матрице событий.

**Интерпретация:** веса и runtime способны выдавать вариации исходного паттерна. Вывод **не доказывает** качество ритмов или стабильную производительность. Модель с вероятностным сэмплированием может выдавать разные результаты на повторных запусках. Данное измерение 243 ms нельзя переносить на Rust и Innioasis Y2.

**Следующие действия:** сохранить экспортированный JSON с вероятностями одного удачного запуска (пока без добавления автоматически скачанного файла в Git); выполнить ещё несколько прогретых измерений и записать браузер/TF.js backend; затем сделать верифицируемый эталон для будущего Rust inference.

## Product requirement update — autonomous generation (2026-11-17)

**Explicit user requirement:** AI must **generate complete new drum patterns from scratch**, **not** require a human pattern and **not** be built around human-pattern continuation or variations. Prior primer experiments only established that the weights and runtime work; they are not the intended product interaction.

### Technical adaptation

The official Magenta Python CLI starts with an **implicit single bass-drum hit** when no primer is supplied (see `drums_rnn_generate.py`). The Magenta.js `MusicRNN.continueSequence` API requires an input `NoteSequence`; its implementation calls `DrumsOneHotConverter.toTensor`. This converter maps an empty step to index 0 (the no-drum class) as long as `totalQuantizedSteps` is a positive number. To preserve the product's no-user-input behavior without injecting any programmed drum groove, use exactly **one silent bootstrap step** internally:

```js
const silentSeed = {
  notes: [],
  quantizationInfo: {stepsPerQuarter: 4},
  tempos: [{time: 0, qpm: 120}],
  timeSignatures: [{time: 0, numerator: 4, denominator: 4}],
  totalQuantizedSteps: 1,
};
const continuation = await model.continueSequenceAndReturnProbabilities(
  silentSeed, 16, 1.0
);
```

**This is musically from scratch, but is not mathematically zero-input unconditional sampling:** RNN needs an initial event/state to compute its first output. Here the bootstrap event is silence and is **not** included in the resulting 16-step pattern. The actual output from this mode on Mac **has not yet been tested**. Possible failure modes include an all-silent result or suboptimal groove quality. If it fails, investigate the official one-kick internal seed and/or a different pretrained model. Even in the one-kick fallback, no user groove is required.

### Mac smoke test

The experimental `reference.html` now defaults to **From scratch — no user input** and **Temperature 1.0**. Older regular/syncopated patterns remain as explicit diagnostic modes only.

```bash
git pull --ff-only
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/experiments/drumsrnn/reference.html`, choose **From scratch — no user input**, then **Generate pattern**. Record **Generated hits**, **Input MIDI hits** (must be 0), **Generation time**, and drum events. Re-run 2-3 times. Save reference JSON locally when working; do not commit downloaded weights or large debug outputs.

Source:
- [Magenta.js MusicRNN internals](https://github.com/magenta/magenta-js/blob/master/music/src/music_rnn/model.ts)
- [Magenta.js DrumsOneHotConverter](https://github.com/magenta/magenta-js/blob/master/music/src/core/data.ts)
- [Original Magenta drums_rnn_generate.py default primer](https://github.com/magenta/magenta/blob/main/magenta/models/drums_rnn/drums_rnn_generate.py)

### Target user interaction

`Generate -> model produces a candidate 16/32/64-step drum pattern -> preview -> accept to drum sequencer or regenerate`. Length, BPM, density/style and temperature can be user controls **without any requirement to enter musical notes**. Note: the current pretrained DrumsRNN does not automatically support explicit density/style conditioning; these would require additional research or post-filtering, and are not implemented.

## Эксперимент 6 — первая генерация с нуля (отчёт пользователя, 2026-11-17)

**Статус: выполнено в браузере на Mac.** Пользователь выбрал **FROM SCRATCH**, передавая модели один пустой служебный шаг (0 ударов) вместо человеческого музыкального primer.

- Инициализация: **29.0 ms**.
- Генерация 16 шагов: **440.0 ms** (~27.5 ms/step в среднем за весь вызов).
- Sampling temperature: **1.2**.
- Входных MIDI hits: **0**.
- Сгенерированных MIDI hits: **28**.
- Матрица вероятностей: **16 × 512**.
- Mean maximum class probability: **0.4171**.
- Сгенерированная последовательность (step, MIDI pitch):

```text
(1,38),(1,46),(2,42),(3,36),(3,45),(5,42),(5,45),(6,36),
(7,46),(7,49),(7,50),(7,51),(9,42),(9,45),(10,36),(10,50),
(11,36),(11,38),(11,42),(11,46),(11,50),(13,42),(13,49),
(14,42),(15,36),(15,38),(15,46),(15,49)
```

Это подтверждает **техническую возможность автономной генерации** событий с внутренним пустым seed на Mac. При этом результат необычно плотный: на шагах 7 и 11 звучит сразу 4 и 5 барабанов, соответственно; не следует считать музыкальное качество подтверждённым без прослушивания. Высокая температура может увеличивать вариативность и хаотичность паттерна.

### Следующий тест — прослушивание

В `reference.html` добавлена кнопка **Preview groove**, включающаяся после генерации непустого паттерна. Для звука используются **синтезированные браузером тестовые барабаны** (Web Audio), без внешних сэмплов; это не финальный аудиодвижок проекта. Аудиопредпросмотр браузера **ещё не проверен на Mac**.

```bash
git pull --ff-only
# Существующий локальный HTTP сервер можно оставить работающим
```

Обновить страницу `reference.html`, выбрать `From scratch`, `Temperature 1.0` или `1.2`, нажать `Generate pattern` и затем `Preview groove`. Сравнить музыкальность/плотность нескольких генераций, не меняя исходный паттерн пользователя (его нет). Контрольные измерения времени генерации требуют повторений и отдельного учёта прогрева.

## Эксперимент 7 — первый одобренный на слух ритм, созданный с нуля (отчёт пользователя, 2026-11-17)

**Статус: автономная генерация и пользовательское прослушивание на Mac подтверждены по отчёту пользователя.** Пользователь написал: «Мне понравилось как звучит». Это качественная субъективная оценка одного сгенерированного грува, не слепое аудиотестирование и не гарантия стабильного музыкального качества.

- Model initialization: **10.0 ms**.
- Generation (16 steps): **195.0 ms** (12.2 ms/step *в среднем за весь вызов*, не latency отдельного шага).
- Generation mode: **FROM SCRATCH** (один внутренний пустой bootstrap step; 0 исходных MIDI-ударов).
- Temperature: **1.2**.
- Generated hits: **9**.
- Mean maximum class probability: **0.6141**.
- Вероятностная матрица: **16 × 512**.
- Результат (step, MIDI pitch): `(2,42),(4,46),(8,42),(12,36),(12,42),(12,45),(13,45),(14,42),(14,45)`.

По сравнению с прошлой автономной генерацией из 28 ударов этот паттерн менее плотный. На данном этапе **не стоит вводить жёсткое ограничение числа ударов или менять модель**: сначала надо получить несколько прослушанных примеров.

Следующий практический шаг: экспортировать `drumsrnn-reference.json` для этого удачного результата в качестве эталонного примера; сохранить текущую версию и хеши весов, условия браузера и timing. Для сравнения Rust-реализации вероятности лучше оценивать до стадии стохастического сэмплирования, так как случайные результаты при `temperature=1.2` невоспроизводимы без контроля RNG.

## Эксперимент 8 — первый Rust-компонент: загрузчик весов (2026-11-17)

**Статус: исходники написаны в исследовательской ветке GitHub, сборка и выполнение на Mac ещё не проверены.** Это **не inference** и не генератор паттернов. Файлы находятся в `experiments/drumsrnn/rust/`.

- Чтение `config.json` и TF.js `weights_manifest.json` (через `serde`/`serde_json`).
- Валидация типа MusicRNN, имён тензоров и файлов, форм, размеров и неподдерживаемой квантизации.
- Чтение float32 little-endian в порядке, заданном манифестом; streaming через несколько shard-файлов, в том числе через границы тензоров.
- Отклонение неверного количества байт, повторов имён и нечисловых (`NaN`/`Inf`) значений.
- Автотесты на синтетических файлах: граница shard внутри float32, несколько тензоров, ошибка размера, traversal path, NaN.
- Для целевого checkpoint CLI дополнительно проверяет 16 тензоров и 2,967,850 параметров, печатает образец декодированных значений.
- Реальные веса лежат локально вне Git, никаких скачиваний загрузчик не выполняет.

Проверить из корня локального клона:

```bash
git pull --ff-only
cargo test --manifest-path experiments/drumsrnn/rust/Cargo.toml
cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml
```

**Важно:** Rust toolchain отсутствует в текущем инструментальном окружении ассистента, поэтому исходники пока не были скомпилированы; любые ошибки компиляции/тестов будут исправлены после первого локального прогона. Измерения производительности, анализ пикового RAM, LSTM, attention, сэмплирование и генерация с нуля в Rust **не реализованы**. Библиотека пока предназначена для Mac, не для запуска на Y2. `Cargo.lock` будет создан при первой локальной сборке Cargo, после чего его следует коммитить для воспроизводимости.

## Эксперимент 9 — Rust загрузка весов подтверждена, добавлен BasicLSTMCell (2026-11-17)

**Уже проверено на Mac (по выводу пользователя):**
- `cargo test --manifest-path experiments/drumsrnn/rust/Cargo.toml`: **5/5 tests passed** для первоначального checkpoint loader.
- `cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml`: успешно.
- Реальные веса: `attentionLength=32`, **16 тензоров**, **2,967,850 float32**, **11,871,400 байт**.
- Фактические значения из `fully_connected/biases`: `[4.8072643, -1.462866, -1.1874784, -2.1494598]`.
- Это тестировало **разбор весов**, но ещё не вычисления нейросети.

**Добавлено в ветку после этих проверок (код ещё не тестировался на Mac):**
- `experiments/drumsrnn/rust/src/lstm.rs` — отдельное вычисление **одного шага** `BasicLSTMCell` с матрицей TF1 `[input+hidden, 4*hidden]`, порядком вентилей `i, j, f, o`, `forget_bias=1.0`, `tanh` и `sigmoid`. Реализация специально простая: скалярные вычисления без аппаратной оптимизации, переиспользование временных буферов без динамических аллокаций внутри шага.
- Автотесты с искусственными весами проверяют размерности, порядок gate-векторов, runtime forget bias, влияние предыдущего hidden-state, отсутствие `NaN` во входе. Эти тесты **не доказывают численную эквивалентность с TensorFlow**.
- Консольный `cargo run --release` после чтения реальных весов дополнительно создаёт каждый из 3 pretrained LSTM layers и исполняет по одному шагу на **искусственном нулевом входе**. Это позволяет проверить действительные матрицы/размерности/конечность результата, но не является генерацией MIDI.
- Зафиксирован источник формул: [TensorFlow 1 BasicLSTMCell source](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py).

Запуск из корня репозитория:

```bash
git pull --ff-only
cargo test --manifest-path experiments/drumsrnn/rust/Cargo.toml
cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml
```

**Следующий рубеж:** реализовать точную структуру `AttentionCellWrapper`, контекстные входы, шаги рекуррентного состояния 3 слоёв, output projection и сопоставить распределения классов с Magenta.js. Без этого Rust-движок **не генерирует** барабанные паттерны и не является готовым inference.

## Эксперимент 10 — шаг полной attention-сети на Rust (2026-11-17)

**Подтверждённый предыдущий результат на Mac:** 10 Rust tests passed, release loader успешно обработал 16 тензоров/2,967,850 параметров, три pretrained LSTM слоя завершили пробный шаг на нулевых входах без NaN/Inf. Пользователь прислал полный лог этих проверок.

**Добавлено в ветку, но ещё не проверено сборкой/сравнением:**
- Новый `experiments/drumsrnn/rust/src/attention.rs`: скалярная реализация одного шага в точном порядке из Magenta.js (ввод класса + 6-битный счётчик `-1/+1`, input projection, 3 LSTM, query из `c0,h0,c1,h1,c2,h2`, 32-slot attention, выходной projection и 512 logits).
- Специально не добавляется `attention/bias` к query: он есть в weights, но в оригинальном [attention.ts](https://github.com/magenta/magenta-js/blob/master/music/src/music_rnn/attention.ts) `attnY = tf.matMul(attnQuery, this.attnMatrix)` без bias. Это может быть важно для численной эквивалентности.
- В CLI после загрузки реальных весов исполняется **один** шаг с внутренней тишиной (класс `0`, step `0`), после чего выводятся топ-5 распределения вероятностей. Это **ещё не полный генератор**, и полученные значения должны быть сверены с Magenta.js.
- В `reference.html` добавлен вывод топ-5 классов **первого сгенерированного шага**; это позволяет сравнить с Rust **без сохранения JSON**, чего пользователь не хочет. Для сравнения выбрать From scratch, Temperature 1.0.
- Добавлены синтетические тесты внимания, бинарного счётчика и softmax; их прохождение пока не подтверждено.

**Следующий шаг на Mac:**

```bash
git pull --ff-only
cargo test --manifest-path experiments/drumsrnn/rust/Cargo.toml
cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml
```

Затем в открытой странице `reference.html` обновить страницу, выбрать `From scratch` и `Temperature 1.0`, нажать `Generate pattern`, скопировать только 5 пар `[class, probability]` первого шага и сравнить с Rust CLI. Не требуется сохранять MIDI/JSON. При расхождении сначала анализировать LSTM bias/matmul, порядок строк и self-attention state. Совпадение топ-5 **само по себе не доказывает** полной численной корректности, но полезно для первичной диагностики.

Source of model sequence and binary counter: [music_rnn/model.ts](https://github.com/magenta/magenta-js/blob/master/music/src/music_rnn/model.ts), [attention.ts](https://github.com/magenta/magenta-js/blob/master/music/src/music_rnn/attention.ts), [aux_inputs.ts](https://github.com/magenta/magenta-js/blob/master/music/src/core/aux_inputs.ts).

## Эксперимент 11 — первый шаг Magenta.js, эталон для Rust (отчёт пользователя, 2026-11-17)

**Условия:** From scratch, один беззвучный служебный стартовый шаг, 0 исходных MIDI-ударов, temperature=1.0, 16 генерируемых шагов. Сгенерировано 3 MIDI-удара: `(11,46),(15,42),(15,46)`. На Mac в браузере измерено: инициализация 14.0 ms, генерация 195.0 ms; средняя максимальная вероятность класса по 16 шагам 0.8311.

**Первое предсказание: top-5 класс/вероятность (Magenta.js):**

```text
class 0:   0.954430
class 4:   0.020489
class 2:   0.007357
class 1:   0.002394
class 256: 0.002259
```

Это эталон для **первого** forward-step с пустым внутренним шагом на Rust. Результат ещё **не сравнен с Rust**, т.к. пользователь пока прислал только вывод браузера. Class 0 представляет отсутствие ударов, что согласуется с разреженным получившимся паттерном. Поскольку здесь temperature=1.0, температуры в softmax не должно быть дополнительного масштабирующего эффекта.

**Следующее действие:** получить вывод `cargo test --manifest-path experiments/drumsrnn/rust/Cargo.toml` и `cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml`; сравнить Rust top-5 с приведённым. При совпадении top-5 всё равно необходимы точная проверка всех 512 вероятностей и дальнейших шагов с учётом float32/операций attention. JSON пользователь сохранять не хочет.

## Эксперимент 12 — совпадение первого forward-step и автономный генератор на Rust (2026-11-17)

**Проверенный результат на Mac (вывод пользователя):**
- `cargo test`: 14/14 тестов прошли.
- `cargo run --release`: загрузка 16 тензоров, 2,967,850 параметров; расчёт трёх LSTM-слоёв и одного шага attention выдал конечные значения.
- Топ-5 вероятностей первого шага после одного пустого входа **совпали с Magenta.js до 6 знаков**: `(0,0.954430), (4,0.020489), (2,0.007357), (1,0.002394), (256,0.002259)`.
- Это подтверждает корректность первой итерации, **но ещё не всей последовательности**, и не проверяет обработку состояния после нескольких шагов.

**Следующий код добавлен в ветку, ещё не проверен компиляцией/запуском на Mac:**
- `src/generation.rs`: генерация из одной служебной тишины, последовательное сэмплирование распределения 512 классов (temperature или argmax), обратная подача выбранного класса в рекуррентное состояние, перевод битов в девять MIDI-инструментов `[36,38,42,46,45,48,50,49,51]`.
- `src/bin/generate.rs`: CLI с `--steps`, `--temperature`, `--argmax`, `--seed`, `--weights`. Текстовый грид и события печатаются в консоль, никаких JSON-файлов не требуется.
- Синтетические модульные тесты битовой декодировки, псевдослучайности, выбора классов и топ-5.
- Страница `reference.html` в браузере показывает выбор классов и топ-5 на шагах 0, 1, 7 и 15.

**Команды для проверки из корня Y2Synth:**

```bash
git pull --ff-only
cargo test --manifest-path experiments/drumsrnn/rust/Cargo.toml
cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml --bin generate -- --steps 16 --temperature 1.2
cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml --bin generate -- --steps 16 --argmax
```

При проверке открыть `reference.html` после обновления, выбрать `From scratch` и `Argmax`, сравнить 16 class ID и топ-5 вероятностей на шагах 0/1/7/15 с CLI. При temperature>0 класс случайно выбирается по вероятностям и Rust RNG не обязан давать те же MIDI-события, что TensorFlow.js. Веса и бинарные артефакты остаются локальными, основная ветка `main` не менялась.

## Эксперимент 13 — первый автономный Rust-грув на Mac (проверено пользователем, 2026-11-17)

**УСПЕХ: код собран и реально исполнился на Mac.** Запуск:

```bash
cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml --bin generate -- --steps 16 --temperature 1.2
```

- Загрузка весов: **15.2 ms**; генерация 16 шагов: **40.2 ms** (замер одного вызова, не стабильный бенчмарк).
- Температура: 1.2; seed **1794863657202886824** (для внутренней воспроизводимости Rust).
- **0 пользовательских MIDI-ударов**, только один внутренний тихий bootstrap.
- 9 ударов; 16 полученных классов: `[0,8,0,0,0,9,0,0,10,0,72,0,64,0,0,8]`.
- События `(step, pitch)`: `(1,46),(5,36),(5,46),(8,38),(8,46),(10,46),(10,50),(12,50),(15,46)`.
- Топ-5 вероятностей класса на шаге 0 (temperature=1.2): `(0,0.888006),(4,0.036160),(2,0.015400),(1,0.006042),(256,0.005757)`.
- На шагах 1,7,15 были выведены конечные вероятности. Таким образом, многошаговая рекуррентная генерация **выполняется**, но **ещё не сверена численно с Magenta.js**.
- Время браузерного опыта (~195ms) и Rust (40.2ms) нельзя считать корректным контролируемым бенчмарком на данном этапе.

**Следующий тест:** детерминированный argmax на Rust и Magenta.js, сравнение 16 выбранных классов и top-5 вероятностей на шагах 0/1/7/15. Поскольку при температуре 1.2 RNG разный, повторение случайных паттернов браузера и Rust не должно совпадать. Команда:

```bash
cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml --bin generate -- --steps 16 --argmax
```

В браузере обновить `reference.html` и выбрать `From scratch` + `Argmax`. Файлы JSON сохранять не нужно. Первые пять вероятностей на шаге 0 при temperature=1.0 уже проверены и совпадают до 6 знаков.

## Эксперимент 14 — многошаговая сверка Rust и Magenta.js (2026-11-17)

**Реально проверено на Mac пользователем:** браузер `reference.html` после исправления синтаксической ошибки снова выполняет генерацию; в режиме `From scratch + Argmax` получил 16 шагов тишины (все классы 0), как и Rust CLI с `--steps 16 --argmax`.

- **Выбранные классы полностью совпали:** `[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]`.
- **Top-5 ID и вероятности** на всех четырёх проверенных шагах совпали до примерно `1e-6` (из вывода с шестью знаками):
  - Шаг 0, класс 0: браузер `0.954430`, Rust `0.954430`.
  - Шаг 1, класс 0: браузер `0.901594`, Rust `0.901594`.
  - Шаг 7, класс 0: браузер `0.631096`, Rust `0.631097`.
  - Шаг 15, класс 0: браузер `0.214144`, Rust `0.214145`.
- Остальные классы в top-5 также совпадают по ID и отличаются не более округления на последних знаках; например, шаг 15: классы `0,1,129,5,2` в обеих реализациях.
- Браузер: инициализация `34.0ms`, генерация `298.0ms`, средняя максимальная вероятность по 16 шагам `0.7982`.
- Rust: загрузка `14.9ms`, генерация `43.6ms` (одно неконтролируемое измерение). Из этих двух замеров нельзя утверждать, что Rust ускоряет вычисление в фиксированное число раз.
- Отсутствие звука в Argmax **не ошибка**: модель предпочитает класс 0 (тишина), особенно в начале с внутренним пустым seed. Для создания музыкального грува использовать температуру `1.0` или `1.2`, а Argmax — для детерминированной проверки математики.

**Вывод:** подтверждён успешный full-sequence recurrent inference и очень близкие распределения на четырёх шагах. Это **не** проверка всех 512 вероятностей на каждом из 16 шагов, и не доказательство битовой идентичности.

## Эксперимент 15 — офлайн прослушивание сгенерированного Rust-грува (код добавлен, ещё не проверен)

В экспериментальную ветку добавлен `src/audio.rs`: чистый Rust-рендерер базовых синтетических ударных в **WAV PCM16 mono 44.1kHz** с 120 BPM по умолчанию (можно изменить `--bpm`). Не использует браузер, TF.js, внешние звуковые библиотеки или сторонние сэмплы. Нейросеть выбирает события, а WAV-код превращает их в тестовый звук; это **не** будущий DSP для Y2.

CLI получил **необязательный** аргумент `--wav FILE`; без него работает по-старому. Добавлены синтетические unit tests на WAV-заголовок, длительность тишины, формирование ненулевого сигнала, позицию удара, ошибочный темп. Код **пока не был собран или прослушан на Mac**, ожидается следующий пользовательский прогон.

```bash
git pull --ff-only
cargo test --manifest-path experiments/drumsrnn/rust/Cargo.toml
cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml --bin generate -- --steps 16 --temperature 1.2 --wav /tmp/y2synth-groove.wav --bpm 120
afplay /tmp/y2synth-groove.wav
```

Это не экспорт JSON. WAV сохраняется локально в `/tmp` и не коммитится в Git. При случайной полной тишине следует снова запустить генератор с температурой `1.2`.

## Эксперимент — полная численная валидация всех 512 классов на 16 шагах (подготовлено 2026-11-17)

Сравниваем **8192 значения**, а не 512 временных шагов. В Rust добавлен `generate_with_observer`, который отдаёт полные softmax-вектора на каждом шаге того же алгоритма `generate`, и независимый компаратор `verification.rs` с CLI `--bin verify_reference`.

Для эталона используем уже существующую страницу `experiments/drumsrnn/reference.html` с зафиксированной версией **@magenta/music@1.23.1**. В браузере выбрать **From scratch** + **Argmax**, сгенерировать и сохранить **JSON** (поле `probabilities` содержит 16 массивов по 512 значений). В корне репозитория запустить:

```bash
cargo test --locked --manifest-path experiments/drumsrnn/rust/Cargo.toml
cargo run --release --locked --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --bin verify_reference -- --reference "$HOME/Downloads/drumsrnn-reference.json"
```

Компаратор проверяет 16 выбранных классов, форму 16×512, нормировку распределений и **каждое** значение вероятности с порогом по умолчанию `1e-5`; выводит per-step maximum absolute error, global mean/RMS, worst step/class, count превышений и PASS/FAIL. Это тест одной фиксированной Argmax-траектории без пользовательского primer; он не устанавливает равенство распределений на всех возможных музыкальных паттернах.

**Статус: ПОДТВЕРЖДЕНО на Mac Studio (2026-11-17): полная сверка одной Argmax-траектории 16 × 512 прошла — PASS.** Результаты и ограничения ниже. Ранее проверенная top-5 сверка теперь дополнена полной проверкой всех 8192 значений.

## Эксперимент — результат полной сверки 8192 вероятностей (Mac Studio, 2026-11-17)

**Статус: PASS, подтверждён пользовательским запуском** `cargo run --release --locked --manifest-path experiments/drumsrnn/rust/Cargo.toml --bin verify_reference -- --reference ~/Downloads/drumsrnn-reference.json`.

| Метрика | Результат |
|---|---:|
| Эталон | Magenta.js `@magenta/music@1.23.1`, TF.js backend `webgl` |
| Модель | `drum_kit_rnn_tfjs` |
| Режим | From scratch, один внутренний тихий шаг, `Argmax` |
| Проверено значений | 16 × 512 = **8192** |
| Максимальная абсолютная ошибка | **1.192092896 × 10⁻⁶** (шаг 2, класс 0) |
| Средняя абсолютная ошибка | **1.529578996 × 10⁻⁹** |
| RMS абсолютной ошибки | **2.754894988 × 10⁻⁸** |
| Порог | **1 × 10⁻⁵** |
| Превышений порога | **0 / 8192** |
| Несовпадений выбранных Argmax-классов | **0 / 16** |

Все 16 классов в обоих движках равнялись нулю, то есть тест выполнен на **одной детерминированной рекуррентной траектории тишины**. Несмотря на это, на каждом шаге сравнивались *все 512 вероятностей*, а не только вероятность класса 0. Следовательно, полнота проверки относится к выходным распределениям **на этой траектории**, а не ко всем возможным траекториям и состояниям модели. Различия не равны нулю побитово и укладываются в численную погрешность двух реализаций (Rust CPU и Magenta.js WebGL).

**Следующий уровень уверенности:** принудительно подать в обе реализации одну и ту же **ненулевую** последовательность предыдущих барабанных классов (teacher-forcing) и сравнить все 512 вероятностей по шагам. Это не меняет требование к пользовательскому режиму — музыка по-прежнему должна генерироваться автономно; ненулевая последовательность нужна исключительно для диагностического теста.

