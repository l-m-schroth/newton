# Chrono ground-truth CLI (`newton_chrono_gt`)

This directory contains a tiny C++ executable that links against the local Chrono build and exposes selected Chrono
functions as a JSON-in / JSON-out CLI for Python unit tests.

## Build

From the repository root:

```bash
cmake -S newton/newton/tests/tires/chrono_gt -B newton/newton/tests/tires/chrono_gt/build -DChrono_DIR="$(pwd)/chrono/build/cmake"
cmake --build newton/newton/tests/tires/chrono_gt/build -j
```

Python tests locate the binary either via `NEWTON_CHRONO_GT_BIN` or by default at:

`newton/newton/tests/tires/chrono_gt/build/newton_chrono_gt`
