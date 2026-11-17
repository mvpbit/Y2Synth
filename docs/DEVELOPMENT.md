# Y2Synth — актуальное состояние разработки

**Дата последней ревизии:** 2026-11-17. Этот документ создан для передачи контекста между чатами и разработчиками. Актуальные факты всегда сверяйте с кодом `main`.

## Git и принятые решения

- Репозиторий: `mvpbit/Y2Synth`, основная рабочая ветка **`main`**.
- Исследовательская ветка `research/magenta-drumsrnn-feasibility` ранее смержена и удалена. По просьбе пользователя новые экспериментальные изменения идут сразу в `main` без PR.
- Весовые файлы checkpoint остаются локальными; **не коммитить**. CPAL уже зафиксирован в `Cargo.lock`, но после добавления нового `triple_buffer` lockfile нужно снова обновить и закоммитить.
- Концепция: два синтезатора и драм-машина; текущая реализация фокусируется только на автономных AI-барабанах.
- AI должен генерировать с нуля, без исходного ритма пользователя; один *внутренний* тихий bootstrap-шаг допустим.

## Проверенные результаты

- Checkpoint Magenta DrumsRNN: 16 тензоров, 2 967 850 float32-параметров (~11,32 MiB).
- Rust: собственный loader, три LSTM-слоя, attention, 512-классовый выход, temperature / argmax и декодирование девяти барабанных классов.
- Пример успешной генерации 16 шагов при temperature 1.2: 9 MIDI-ударов за ~40,2 мс на Mac (разовый замер, **не** бенчмарк).
- Проверка Rust vs Magenta.js в режиме `From scratch / Argmax`: все 16 классов идентичны (нули, тишина); top-5 вероятности на шагах 0/1/7/15 отличаются примерно на 10⁻⁶. Это не доказывает совпадения всех 512 вероятностей на каждом шаге.
- Пользователь подтвердил, что Rust WAV-рендерер создал файл, который удалось прослушать на Mac.
- **2026-11-17:** пользователь сообщил «Работает» после команд проверки сборки и запуска интерактивного live-прототипа с CPAL. Это подтверждает первый рабочий live-запуск на Mac; результаты отдельного тестирования `regenerate`, `bpm`, `stop` и длительной работы пока не записаны.

## Что сейчас есть в Git, но требует работы

- `experiments/drumsrnn/rust/src/audio.rs`: после успешного WAV-прогона добавлен рендер при заданной sample rate. Необходимо повторно прогнать тесты.
- `experiments/drumsrnn/rust/src/sequencer.rs`: теперь содержит **только подготовку** PCM-такта в управляющем потоке. Состояние аудио и переходы вынесены в новый `src/realtime.rs` за фичей `live-audio`. Добавлены unit-тесты передачи, старта/остановки, смены баров и BPM. Новая версия ещё **не собрана и не прослушана на Mac**.
- В `Cargo.toml` есть опциональные **CPAL 0.18.2** и **triple_buffer 9.0.0** (фича `live-audio`). `src/bin/live.rs` поддерживает `play`, `stop`, `bpm`, `regenerate`, `status`, `quit`; теперь передаёт подготовленный PCM через неблокирующий triple buffer и управляет транспортом через `AtomicBool`. Пользователь подтверждал работу **предыдущей** CPAL-версии; **последняя RT-переработка ещё не проверена на Mac**.
- На Innioasis Y2 ничего не запускалось, производительность и совместимость устройства неизвестны.

## Команды из корня проекта на Mac

При необходимости получить модель (веса не включены в Git):

```bash
bash experiments/drumsrnn/download-model.sh
```

Тесты — после синхронизации локального checkout:

```bash
git switch main
git pull --ff-only
cargo test --manifest-path experiments/drumsrnn/rust/Cargo.toml
```

Создать AI-грув, сохранить и прослушать тестовый WAV:

```bash
cargo run --release \
  --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --bin generate -- \
  --steps 16 --temperature 1.2 --bpm 120 \
  --wav /tmp/y2synth-groove.wav

afplay /tmp/y2synth-groove.wav
```

Воспроизвести ранее подтверждённый пример (9 ударов):

```bash
cargo run --release --locked \
  --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --bin generate -- --steps 16 --temperature 1.2 \
  --seed 1794863657202886824 --wav /tmp/y2synth-groove.wav
```

Детерминированная математическая сверка (ожидаются все нулевые классы — это нормально):

```bash
cargo run --release --locked \
  --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --bin generate -- --steps 16 --argmax
```

Оригинальная проверочная страница Magenta.js при необходимости запускается из корня репозитория через `python3 -m http.server 8765 --bind 127.0.0.1`, затем `http://127.0.0.1:8765/experiments/drumsrnn/reference.html`.

## Ближайшая задача

**Проверить после RT-переработки и измерить стабильность live-секвенсора:**

1. **На Mac** выполнить `cargo test --features live-audio` с новым triple buffer; исправить возможные ошибки компиляции. Добавить обновлённый `Cargo.lock` в Git.
2. Запустить `cargo run --release --features live-audio --bin live` (с `--manifest-path` из корня проекта).
3. Протестировать `play`, `stop`, `bpm 90`, `bpm 160`, многократный `regenerate`, `quit` при непрерывном звуке, без щелчков/заиканий.
4. Измерить xrun/latency/CPU и длительную стабильность, проверив также влияние переключения устройства.
5. Оценить поддержку динамических буферов при пересоздании потока и резервную стратегию для слабого CPU Y2.

После каждого изменения обновлять эту страницу. Более длинные исследовательские протоколы: [Magenta DrumsRNN](research/magenta-drumsrnn.md).

## Live CLI: запланированная проверка на Mac

```bash
git switch main
git pull --ff-only

# Без --locked при первом запуске: новая зависимость triple_buffer ещё не внесена в Cargo.lock.
cargo test --manifest-path experiments/drumsrnn/rust/Cargo.toml --features live-audio

cargo run --release --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --features live-audio --bin live -- --bpm 120 --temperature 1.2
```

В интерактивном терминале:

```text
play
regenerate
bpm 140
stop
play
quit
```

Текущая реализация готовит новый AI-паттерн и звуковой цикл на основном потоке CLI; пока этот поток рассчитывает ритм, аудио-CPAL callback читает старый подготовленный буфер. Команда `regenerate` не требует исходного грува пользователя. Не путать отсутствие промежуточного WAV с отсутствием *заранее подготовленных в памяти* PCM-буферов.

После успешной сборки необходимо зафиксировать обновлённый `Cargo.lock` в `main`; затем можно вернуть `--locked`. В активном callback больше не используются `std::sync::mpsc` и операции владения/освобождения `Vec`; вместо них `triple_buffer::Output` и `AtomicBool`. Это пока не доказательство полного отсутствия xrun: нужны реальные замеры и мониторинг CoreAudio.

## Эксперимент — lock-free аудиопоток после первого работающего live-прототипа (2026-11-17)

**Коммиты в `main` внесены, Mac-регрессия ещё не выполнена.**

- `src/sequencer.rs` содержит только подготовку звукового такта. Это контрольный поток, допускающий аллокации.
- Добавлен `src/realtime.rs` и опциональный `triple_buffer = 9.0.0`.
- В `live.rs` убран `std::sync::mpsc`; управление `play/stop` осуществляется атомарным флагом `AtomicBool`, новый ритм/темп передаётся через single-producer/single-consumer triple buffer.
- Новый аудиообработчик использует `Output::update()` только на границе такта или при остановленном транспорте и `output_buffer()` для чтения. При публикации нового `PreparedLoop` старый `Vec` освобождается на управляющем потоке, а не в callback.
- Для 15-мс crossfade выделена память при создании проигрывателя. При переключении выполняется ограниченное копирование головы предыдущего такта в уже выделенный буфер.
- Unit-тесты: публикация при остановке, смена только на границе такта, перезапись старых неполученных паттернов, плавный stop/start, изменение длины такта, неверная частота дискретизации. Также сохранены тесты offline-подготовки.
- **Предыдущее прослушивание на Mac относится к старой CPAL-реализации**. Новую версию нужно скомпилировать, прослушать и проверить при быстрых сменах ритма/темпа.
- Синтезированный audio loop по-прежнему предрассчитывается в RAM, без WAV. На сильных CPU это удобно; на Y2 будущие требования могут заставить перейти к синтезу отдельных голосов в callback при сохранении RT-безопасности.

## Проверка lock-free live-аудио на Mac (2026-11-17)

**Подтверждено пользователем:** после перехода на triple buffering и атомарное управление transport пользователь повторно собрал/запустил прототип и сообщил: «Всё работает отлично». Это подтверждает практическую работоспособность обновлённой live-драм-машины в его среде. Точные логи отдельных тестов и измерения задержек/xrun не предоставлены; отсутствие xrun в долгом прогоне не доказано.

**Проверено через GitHub:** обновлённый `experiments/drumsrnn/rust/Cargo.lock` уже содержит `triple_buffer` и находится в `main`. Для последующих воспроизводимых сборок можно вернуться к флагу `--locked`.

Следующий этап: стресс-тесты многократной регенерации/смены темпа, оценка нагрузки CPU/памяти и буферизации, затем улучшение качества барабанного синтеза. Для Y2Linux сохраняется отдельная задача проверки реальной аппаратной платформы.

## Архитектурный каркас меню PLAY (2026-11-17)

Добавлены `experiments/drumsrnn/rust/src/ui.rs` и `src/project.rs`, экспортированы из `lib.rs`. Это чистый переносимый Rust, без новых зависимостей и без изменений работающего `live.rs`.

- `MenuController`: конечный автомат, принимает `PlayPressed`, `CenterPressed`, `BackPressed`, `Next`, `Previous`; охватывает Home → PlaySettings → USER/Synth1/Synth2/Drums и AI/Generate All/Generate Track/Synth1/Synth2/Drums. Возвращает `Command` для будущего app coordinator, ничего не генерирует сам.
- `MusicProject<S1,S2,D>`: три независимые типизированные дорожки, `Track.source` и последний способ редактирования `CreationMode`. `GeneratedAll` заменяет все три разом; отдельные `replace_*` изменяют только целевую.
- `MusicSession`: рабочее состояние и `last_saved` с явным `save_current`. **Только RAM**, ещё нет хранения между перезапусками.
- Unit-тесты покрывают навигацию, BACK, единичную/все AI-команды, PLAY без генерации, независимые треки, сохранённый снимок.
- Отсутствуют реальные AI-генераторы синтезаторов, экран Y2, аппаратная раскладка кнопок, сериализация, интеграция с CPAL, real-time transport и обработчики всех UI-команд.
- **Сборка нового каркаса на Mac пока не подтверждена.** Проверить:

```bash
cargo test --locked --manifest-path experiments/drumsrnn/rust/Cargo.toml
cargo test --locked --manifest-path experiments/drumsrnn/rust/Cargo.toml --features live-audio
```

## Application Controller и подключение консольного меню (2026-11-17)

Заложен реальный **Application Controller** — `src/application.rs`. В `src/project.rs` добавлен общий `bpm` со валидацией и сохранением в RAM-снимке. `src/bin/live.rs` подключён к нему через абстракции `DrumGenerator` (closure с настоящей DrumsRNN) и `AudioPort` (готовый PCM в triple buffer + атомарный старт). Audio callback и зависимости Cargo **не изменены**.

Фактически реализовано:
- AI → Generate Track → Drums генерирует новый паттерн, готовит PCM вне RT, публикует на границе такта, после успешного вывода сохраняет его в RAM.
- PLAY использует `last_saved`: без генерации восстанавливает паттерн, настройки `DrumKit`, BPM и режим, передаёт PCM и запускает аудио.
- Изменения `bpm/level/tone/decay/reset-kit` теперь проходят через coordinator и сохраняются; старые команды `play` и `regenerate` также маршрутизированы через него.
- CLI симулирует колесо: `menu`, `center`, `next`, `prev`, `back`, `wheel-play`.
- `GenerateAll` и генерация Synth1/Synth2 возвращают `NotImplemented` и **не меняют проект**. `EditUserTrack(Drums)` направляет к командам управления китом; полноценного UI редактора нет.
- Добавлены unit-тесты на отсутствие повторного инференса при PLAY, целостность проекта при ошибке, сохранение BPM/кита, отсутствие фальшивых Synth и сквозной путь `MenuController → ApplicationController`.
- В начале `live` готовится и автоматически сохраняется один сгенерированный AI-паттерн, но playback начинается только по PLAY; при завершении процесса снимок **теряется**. Никакой файловой сериализации пока нет.

**Проверка на Mac ещё необходима** (новый код не запускался в окружении ассистента):

```bash
git pull --ff-only
cargo test --locked --manifest-path experiments/drumsrnn/rust/Cargo.toml --features live-audio
cargo run --release --locked --manifest-path experiments/drumsrnn/rust/Cargo.toml --features live-audio --bin live -- --bpm 120 --temperature 1.2
```

Интерактивный сценарий:
```text
menu
center
next
center
next
center
next
next
center
back
back
back
wheel-play
bpm 90
tone snare 0.8
stop
play
quit
```

**Что дальше:** собрать и прослушать на Mac, проверить, что `wheel-play` не регенерирует новый pattern, затем реализовать долговременное хранение музыкального проекта и позднее добавить реальные synth-generators. Не объявлять этот этап проверенным до фактического прогона.




## Первые результаты сборки Application Controller на Mac Studio (2026-11-17)

Пользователь запустил `cargo test --locked --features live-audio` после установки Rust (cargo 1.100.0). Все зависимости скачались, но **тестовая сборка завершилась E0502** в `src/application.rs`: две тестовые closure захватывали локальный счётчик `calls` по `&mut`, а затем тест проверял `calls` до прекращения mutable borrow.

**Исправление внесено в `main`:** в тестах вместо `let mut calls` используется `std::cell::Cell<usize>` и методы `get/set`. Это изменение касается только тестового кода; движок, модель и аудиообработчик не менялись. **Повторный прогон cargo test после исправления ещё не подтверждён**.

Повторить:

```bash
git pull --ff-only
cargo test --locked --manifest-path experiments/drumsrnn/rust/Cargo.toml --features live-audio
```
