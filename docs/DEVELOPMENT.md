# Y2Synth — актуальное состояние разработки

**Дата последней ревизии:** 2026-11-17. Этот документ создан для передачи контекста между чатами и разработчиками. Актуальные факты всегда сверяйте с кодом `main`.

## Git и принятые решения

- Репозиторий: `mvpbit/Y2Synth`, основная рабочая ветка **`main`**.
- Исследовательская ветка `research/magenta-drumsrnn-feasibility` ранее смержена и удалена. По просьбе пользователя новые экспериментальные изменения идут сразу в `main` без PR.
- Весовые файлы checkpoint остаются локальными; **не коммитить**. После добавления CPAL `Cargo.lock` требует пересоздания и нового коммита: существующая версия lockfile **пока устарела**.
- Концепция: два синтезатора и драм-машина; текущая реализация фокусируется только на автономных AI-барабанах.
- AI должен генерировать с нуля, без исходного ритма пользователя; один *внутренний* тихий bootstrap-шаг допустим.

## Проверенные результаты

- Checkpoint Magenta DrumsRNN: 16 тензоров, 2 967 850 float32-параметров (~11,32 MiB).
- Rust: собственный loader, три LSTM-слоя, attention, 512-классовый выход, temperature / argmax и декодирование девяти барабанных классов.
- Пример успешной генерации 16 шагов при temperature 1.2: 9 MIDI-ударов за ~40,2 мс на Mac (разовый замер, **не** бенчмарк).
- Проверка Rust vs Magenta.js в режиме `From scratch / Argmax`: все 16 классов идентичны (нули, тишина); top-5 вероятности на шагах 0/1/7/15 отличаются примерно на 10⁻⁶. Это не доказывает совпадения всех 512 вероятностей на каждом шаге.
- Пользователь подтвердил, что Rust WAV-рендерер создал файл, который удалось прослушать на Mac.

## Что сейчас есть в Git, но требует работы

- `experiments/drumsrnn/rust/src/audio.rs`: после успешного WAV-прогона добавлен рендер при заданной sample rate. Необходимо повторно прогнать тесты.
- `experiments/drumsrnn/rust/src/sequencer.rs`: подключён через `pub mod sequencer;` в `lib.rs`; содержит unit-тесты loop-транспорта и смены на границе такта. **В текущем окружении нет Rust-компилятора, поэтому новые тесты ещё не выполнялись.**
- В `Cargo.toml` есть **опциональный CPAL 0.18.2** (фича `live-audio`) и `src/bin/live.rs`: пробная программа с `play`, `stop`, `bpm`, `regenerate`, `status`, `quit`. Звук выводится из предварительно подготовленных PCM-буферов в callback CoreAudio. **Ни сборка, ни реальное воспроизведение на Mac пока не проверены.**
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

**Довести live-секвенсор до первой проверенной сборки**, прежде чем добавлять дополнительные функции:

1. **На Mac** выполнить `cargo test --features live-audio` и исправить возможные ошибки. Сохранить обновлённый `Cargo.lock` в Git после подтверждённой сборки.
2. Запустить `cargo run --release --features live-audio --bin live` (с `--manifest-path`, если запуск из корня репозитория).
3. Проверить `play / stop / bpm / regenerate / quit`: темп, смену на границе такта, звук, щелчки и стабильность.
4. Заменить потенциально real-time-небезопасные `std::mpsc` и освобождение старых буферов в audio callback на неблокирующий, контролируемый механизм.
5. Измерить задержку, CPU и xrun на реальном Mac.

После каждого изменения обновлять эту страницу. Более длинные исследовательские протоколы: [Magenta DrumsRNN](research/magenta-drumsrnn.md).

## Live CLI: запланированная проверка на Mac

```bash
git switch main
git pull --ff-only

# Без --locked при первом запуске: CPAL ещё не внесён в Cargo.lock.
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

После первой успешной сборки и прослушивания необходимо зафиксировать обновлённый `Cargo.lock` в `main`; только затем вернуться к `--locked` для воспроизводимых проверок. Это не production realtime: `std::sync::mpsc` и освобождение объектов в callback ещё требуют переработки.
