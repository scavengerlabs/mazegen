## Build

```bash
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web
```

## Serve

```bash
python -m http.server
```

navigate to localhost:8000/index.html

## Test

```bash
cargo test
```

## Run Fortune's algorithm

```bash
cargo run --bin fortunes
```
