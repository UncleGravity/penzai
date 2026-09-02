# Deployment

The qualified board environment is Certified Ubuntu for Xilinx Devices 24.04,
kernel `6.8.0-1015-xilinx`, and XRT userspace 2.18.0. Deployment helpers default
to `ubuntu@kria`; set `BOARD` to override it.

## Board setup

Grant the login access to the accelerator devices, then log out and back in:

```sh
sudo usermod -aG render,video "$USER"
```

Build ZOCL from the XRT 2024.2 source that matches userspace 2.18:

```sh
git clone --depth 1 -b 2024.2 https://github.com/Xilinx/XRT
cd XRT/src/runtime_src/core/edge/drm/zocl
make
sudo make modules_install
sudo depmod -a
sudo modprobe zocl
echo zocl | sudo tee /etc/modules-load.d/zocl.conf
```

The module is kernel-specific and must be rebuilt after a kernel change. Keep
the qualified kernel installed manually and remove the incompatible distro DKMS
package:

```sh
sudo apt-get purge -y xrt-dkms
sudo apt-mark manual \
  linux-image-6.8.0-1015-xilinx \
  linux-headers-6.8.0-1015-xilinx
```

## CMA

The resident model image and session state use XRT buffers backed by CMA. The
validated reservation is 1536 MiB:

```sh
sudo sed -i 's/cma="800M"/cma="1536M"/' \
  /etc/flash-kernel/bootscript/bootscr.zynqmp.kria
sudo flash-kernel
sudo reboot
grep CmaTotal /proc/meminfo
```

Do not request 2048 MiB: CMA must share the low 2 GiB DDR window with the
kernel. The development daemon helper defaults to 1500 MiB, leaving 36 MiB of
the configured CMA reservation outside its allocator. Override
`PENZAI_HEAP_MIB` when other CMA users need headroom or when reproducing the
smaller qualification heaps below.

Use the heap size qualified for the selected model:

| Model | `PENZAI_HEAP_MIB` |
| --- | ---: |
| 1.7B Q1/Q2 | 768 |
| 4B Q1/Q2 | 1280 |
| 8B Q1 | 1392 |

The 8B Q1 run was qualified with context 513; after daemon initialization it
left about 138,048 KiB of CMA free and the model planner reported 69,758,976
bytes of heap slack. The engine also accepts the 8B Q2 file format, but its
2.338 GB resident image cannot fit this CMA configuration and must fail during
allocation rather than publish a model.

## Bitstream

Create the local FPGA configuration, verify the closed production source set,
and build the qualified `f225` target:

```sh
nix develop -c zig build regmap
nix develop -c zig build verify-rtl
cd fpga/build
cp config.env.example config.env
# Set VM, VM_DIR, BOARD, and BOARD_TMP.
./build.sh f225
./deploy.sh f225
```

The build promotes only a complete timing-passing run bundle. Deployment
verifies its hashes, installs `/lib/firmware/xilinx/penzai`, loads it with
`xmutil`, and writes the receipt consumed by `penzaid`.

## Daemon

From the repository root, deploy the binary once:

```sh
nix run .#deploy-penzaid
```

Terminal 1 then owns both the board daemon and the SSH local forward:

```sh
nix run .#serve-penzaid
```

The first command copies the cross-compiled daemon to `/tmp/penzai/penzaid`.
The second keeps one SSH session attached, binds the daemon to
`tcp:127.0.0.1:29092` on the board, and forwards it to
`tcp:127.0.0.1:29092` on the host. It enables fail-closed forwarding and SSH
keepalives. Its other defaults are XRT memory and a 1500 MiB heap; override them
with `PENZAI_LOCAL_PORT`, `PENZAI_REMOTE_PORT`, `PENZAI_MEM`, and
`PENZAI_HEAP_MIB`. `PENZAI_PORT` remains a compatibility alias for the local
port. A non-default local port must also be passed to the client. Automated
qualification sets `PENZAI_FORWARD=0` because it owns a separate evidence-bound
forward on local port 39092; normal use should keep the default value of `1`.

In terminal 2, verify the live engine and deployment identity through the
default tunnel:

```sh
nix run .#penzai -- inspect device
```

Require a loaded receipt, `bitstream_hash_verified=true`, wire ABI 18, metrics
schema 1, engine ID `0xB05A4000`, interface `0x00010007`, and the expected
`f225` run identity. See [verification.md](verification.md) for the model
qualification gate.

### Optional direct TCP

Direct board TCP is not part of the default flow. It requires a separately
started daemon bound to a non-loopback board address plus working host routing
and firewall rules. Only after those prerequisites are in place should a client
use an endpoint such as `--device tcp:kria:29092`; `serve-penzaid` intentionally
does not expose that listener.
