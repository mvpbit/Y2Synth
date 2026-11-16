# Архитектура Y2Synth

Это схема **текущей экспериментальной реализации и планируемого разделения ответственности**, а не описание работающей прошивки.

## Пайплайн

```text
   Pretrained Magenta DrumsRNN checkpoint (локальные файлы, не в Git)
                   |
                   v
   Rust loader -> BasicLSTMCell x3 + attention -> sampling
                   |
                   v
       GeneratedPattern (16 x class 0..511)
                   |
              MIDI drum hits
                   |
         +---------+-----------+
         |                     |
         v                     v
  WAV renderer              Sequencer / transport (код добавлен)
  44.1 kHz mono                  |      play / stop / BPM / regenerate
  PCM16 file                     v
                           Prepared audio loop
                                 |
                                 v
                           CPAL audio backend (код добавлен)
                            (Mac, затем Y2Linux)
```

**Проверено пользователем на Mac:** checkpoint → Rust-генерация → MIDI-подобные события → тестовый WAV. **Первый live-запуск предыдущей CPAL-версии подтверждён пользователем.** После этого добавлен lock-free triple buffering и отдельный `realtime.rs`; **переработанная версия ещё не проверена на Mac**. Задержки/xrun и реальная RT-безопасность не измерены.

## Реальные модули (`experiments/drumsrnn/rust/src/`)

| Файл | Ответственность | Статус |
|---|---|---|
| `lib.rs` | Чтение, проверка и декодирование checkpoint | Работает на Mac |
| `lstm.rs` | Совместимый с Magenta TF BasicLSTMCell | Проверен в составе inference |
| `attention.rs` | Прямой проход attention-wrapper, logits 512 классов | Проверен на Mac |
| `generation.rs` | Автономная генерация, temperature / argmax, MIDI-удары | Работает на Mac |
| `audio.rs` | Тестовый синтез ударных, PCM16 WAV; рендер при заданном sample rate | WAV прослушан на Mac, последующие изменения требуют регрессии |
| `sequencer.rs` | Подготовка зацикленного PCM-такта на control thread | В `main`, требует новой Mac-регрессии |
| `realtime.rs` | Triple-buffered чтение PCM, атомарный play/stop, crossfade, смена на границе такта | Новый код, требует сборки/прослушивания на Mac |
| `bin/generate.rs` | CLI автономной генерации и необязательной записи WAV | Использован на Mac |
| `bin/live.rs` | CPAL/CoreAudio интерактивный CLI: play/stop/BPM/regenerate | Пользователь подтвердил работу на Mac; сценарии команд требуют отдельной проверки |

`Cargo.toml` содержит `serde`, `serde_json`, опциональные **CPAL 0.18.2** и **triple_buffer 9.0.0** (`--features live-audio`). В `src/bin/live.rs` есть CLI и CPAL output callback; **пользователь подтвердил, что прототип работает на Mac**. При следующем Cargo build нужно обновить `Cargo.lock`.

## Принципы будущего live-аудио

1. **Control/inference thread:** загрузка весов, AI-генерация, выбор нового паттерна, подготовка аудиобуфера и публикация через `triple_buffer::Input`; все замены/удаления старых `Vec` выполняются здесь.
2. **Audio callback:** `RealtimePlayer` использует `triple_buffer::Output` и `AtomicBool`; читает готовые семплы, делает 15мс fade и не владеет/не удаляет `Vec`. Не запускает модель или файловый ввод-вывод.
3. **Bar-boundary swap:** новый паттерн/темп становится активным в начале следующего такта. Не допускать внезапных обрывов и щелчков; хвосты звуков и crossfade требуют тестирования.
4. **Platform isolation:** отделить общий код генерации и секвенсора от CoreAudio/macOS и от будущего Linux backend.
5. **Measurements:** оценивать xrun, latency, память и загрузку CPU на реальных устройствах, а не по одному Mac-бенчмарку.

В новой реализации нет явного выделения или освобождения PCM-буферов в callback и нет `std::sync::mpsc`. Однако отсутствие блокировок внутри всех зависимостей, системного backend и отсутствие xrun ещё необходимо проверить на практике.

## UI и музыкальное состояние (план)

Утверждена как **редактируемый черновик** иерархия единственного главного действия PLAY: настройки USER (отдельные страницы двух синтезаторов и барабанов) и AI (Generate All либо Generate Track для отдельной дорожки). UI должен обращаться к управляющему слою, а не выполнять генерацию или файловые операции в audio callback. Музыкальное состояние должно содержать три независимые дорожки и выбранный режим; отдельная генерация не заменяет остальные дорожки. Никакая часть этого экранного UI пока не реализована. См. [UI MENU](UI_MENU.md).

## Программная архитектура меню и музыкального проекта (переносимое ядро)

На базе [UI MENU](UI_MENU.md) добавлены модули **`src/ui.rs`** и **`src/project.rs`** в текущий экспериментальный Rust-крейт. Это **начальная архитектура**, не UI-приложение для дисплея Y2.

```text
Wheel / buttons [future]
       |
       v
 InputEvent ---(pure Rust)---> MenuController
                                 |
                                 v
                    Command: PlayLast | EditUserTrack
                             | GenerateAll | GenerateTrack
                                 |
                                 v
                   App Coordinator [future]
                     /           |            \
            MusicSession     AI Generators     Audio control
            3 track slots   DrumsRNN ready     CPAL / triple buffer
                            Synths pending
```

- `MenuController` не выполняет реальную генерацию, запись/чтение файлов или аудио: выдаёт только **намерение/команду** из абстрактных событий колеса. Пользовательские страницы USER, AI Generate All и AI Generate Track представлены в переходах состояний и проверяются unit-тестами.
- `MusicProject<Synth1, Synth2, Drums>` оставляет у каждой дорожки **собственный тип музыкального содержимого**, не подменяя отсутствующие синтезаторы барабанным форматом. `Track.source` хранит User/Ai даже в смешанном проекте.
- `GeneratedAll` — объект для цельного применения результата после завершения всех трёх AI-генераторов; `replace_drums` и `replace_synth1/2` не меняют остальные дорожки.
- `MusicSession::last_saved` — **только снимок в RAM**. PLAY должен обратиться к этому снимку через будущий App Coordinator; файловое хранение, синхронизация с аудио и правила автосохранения ещё не реализованы.
- `live.rs` остаётся **отдельным работающим CLI для барабанов**. Он пока не связан с новым контроллером меню: это намеренно, чтобы не ломать подтверждённое воспроизведение.
- Проверка физического InputEvent-кодирования, экрана и сохранения между перезагрузками возможна после исследования Y2.

## Application Controller: интеграция меню с реальным DrumsRNN

Модуль `src/application.rs` реализует `ApplicationController`, который выполняется на контрольном потоке. Он связывает:
- чистые события `ui::MenuController` → `ui::Command`;
- `MusicSession` с текущим проектом и последним сохранённым снимком;
- `DrumGenerator` — порт для действующей DrumsRNN (`generate`);
- `AudioPort` — публикацию готовых `PreparedLoop` в triple buffer и запуск транспорта.

```text
User wheel / CLI simulator → MenuController → Command
                                          |
                                   ApplicationController
                           +--------------+----------------+
                           |              |                |
                  MusicSession       DrumGenerator       AudioPort
                  saved in RAM     Magenta Rust AI      triple_buffer/Input
                           |                               |
                       mode/BPM                        RealtimePlayer
                       3 tracks                        CoreAudio callback
```

**Фактическое подключение Mac:** `src/bin/live.rs` использует `ApplicationController` для `play`, `regenerate`, `bpm`, настройки кита и команд `center/next/prev/back/wheel-play`. В `live.rs` реализован адаптер `LiveAudio` к `triple_buffer::Input` и `AtomicBool`, а AI-порт представляет собой closure над `AttentionRnn`. Сам аудиопоток не менялся.

**Поддерживаемый вертикальный сценарий:** AI → Generate Track → Drums вызывает настоящую DrumsRNN; возвращение на главный экран → PLAY читает и воспроизводит последнее состояние без генерации. Недоступные синтезаторы представлены `UnavailableSynth` (ненаселяемый Rust enum), а не фиктивными мелодиями. Generate All возвращает `NotImplemented`. Ошибка генерации, подготовки или публикации не изменяет сохранённое состояние проекта. Изменения кита отмечаются пользовательским режимом USER; tempo сохраняется в общем проекте.

**Не реализованы:** AI/редактирование Synth1/Synth2, настоящая совместная генерация трёх дорожек, файловое/энергонезависимое хранение, экран и аппаратный ввод Y2. Новая интеграция добавлена в `main`, но сборка и аудиорегрессия на Mac ещё нужны.

## Неизвестные ограничения Y2

Архитектура CPU, доступная память, поддерживаемый audio I/O, ОС, схема управления и физические возможности устройства требуют проверки. Наличие файлов модели не означает, что нейросеть реально запустится на Y2 с приемлемой скоростью.

См. [PROJECT](PROJECT.md), [ROADMAP](ROADMAP.md), [DEVELOPMENT](DEVELOPMENT.md) и [исследование](research/magenta-drumsrnn.md).

## Ограничения текущей live-реализации

- Нейросеть и подготовка 16-шагового аудиобуфера выполняются на CLI/control thread. Audio callback выдаёт подготовленные семплы в формате CoreAudio и опрашивает очередь команд.
- `Cue` сменяет паттерн на границе такта во время воспроизведения, либо сразу в состоянии stop.
- `bpm` пересчитывает подготовленный аудиобуфер на control thread; не выполняет динамическое время-растяжение на аудиопотоке.
- Использование `std::sync::mpsc` в callback и уничтожение предыдущих буферов **пока не гарантируют strict real-time safety**. Это следующий архитектурный долг.
- Текущий backend использует устройство вывода macOS по умолчанию и сводит все голоса в моно (дублирует сигнал на выходные каналы). Нет контроля выбора устройства и ошибок route change.

**Первый работающий live-прототип на Mac подтверждён пользователем; расширенная проверка команд и строгая real-time безопасность ещё впереди.**

### RT-связь с управляющим потоком

```text
CLI + DrumsRNN + offline render (контрольный поток)
       |
       +-- Input<PreparedLoop> -- publish / latest wins
       |                              |
       |                          TripleBuffer (3 слота)
       |                              |
       +-- Arc<AtomicBool> -----+     v
                                 RealtimePlayer (CoreAudio callback)
                                     |
                            15мс crossfade + PCM samples
```

Каждый буфер содержит целый 16-шаговый такт. Смена активной версии происходит на границе такта, а при stop новые паттерны доступны сразу. Пересоздание `Vec` происходит исключительно на control thread. Scratch для crossfade выделяется заранее, на этапе конструирования RealtimePlayer.

**Оговорки:** модель пока не измерена на слабом железе, реальный вывод на macOS после этих изменений не проверялся; некоторые системные процессы CoreAudio и остановку stream нельзя объявить RT-safe без замеров.
