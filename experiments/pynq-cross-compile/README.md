# PYNQ-Z1 Zig Cross-Compile Smoke

Builds two ARM Linux hard-float binaries for the PYNQ-Z1 Cortex-A9:

- `penzai-pynq-cross-smoke`: target/ABI/pointer-size checksum probe.
- `penzai-pynq-stdlib-smoke`: file I/O, clock/sleep, thread, and TCP loopback probe.

Build reproducibly:

```sh
nix build
```

The binaries land in `result/bin/`.

Run both on the board:

```sh
nix develop -c zig build run-board -Dboard=xilinx@pynq
```

Default Zig target: `arm-linux-gnueabihf -mcpu=cortex_a9`.
