{
  description = "penzai";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    zig-flake.url = "github:mitchellh/zig-overlay";
    llama-cpp-src = {
      # First upstream CPU release of Q2_0 (GGUF type 42, group 64).
      # url = "github:ggml-org/llama.cpp/bec4772f6a2527d371557b5d2032641e5ff7619c";
      # Latest upstream CPU release of Q2_0 (GGUF type 42, group 64).
      url = "github:ggml-org/llama.cpp/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2";
      flake = false;
    };

    # ensure zls <-> zig match versions
    zls-flake = {
      url = "github:zigtools/zls?ref=0.16.0";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ flake-parts, nixpkgs, zig-flake, zls-flake, llama-cpp-src, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      perSystem = { system, ... }:
        let
          pkgs = import nixpkgs {
            inherit system;

            overlays = [
              (final: prev: {
                zig = zig-flake.packages.${system}."0.16.0";
                zls = zls-flake.packages.${system}.default.overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ])
                    ++ [ final.zig ];
                });
              })
            ];
          };

          # Build with clang/libc++ (not the default gcc/libstdc++). The Zig
          # host compiles host/llama/prompt.cpp with Zig's bundled libc++, so the
          # common_chat_* C++ API (which takes std::string by ref) must be
          # mangled/ABI-matched under libc++ (std::__1) too — otherwise the host
          # link fails with undefined std::__1::basic_string symbols. This
          # mirrors darwin, where the default stdenv is already clang/libc++.
          #
          # The two llama.cpp variants below share this stdenv, src, and the
          # bulk of the cmake flags — they differ only in extraCmakeFlags +
          # build/install phases. Factoring the shared flags into one list keeps
          # the variants from silently drifting apart.
          llama-ui = pkgs.stdenvNoCC.mkDerivation {
            pname = "penzai-llama-ui";
            version = "pinned";
            src = "${llama-cpp-src}/tools/ui";

            nativeBuildInputs = [
              pkgs.nodejs
              pkgs.importNpmLock.linkNodeModulesHook
            ];

            npmDeps = pkgs.importNpmLock.buildNodeModules {
              npmRoot = "${llama-cpp-src}/tools/ui";
              inherit (pkgs) nodejs;
            };

            installPhase = ''
              runHook preInstall
              LLAMA_UI_OUT_DIR="$out" npm run build --offline
              runHook postInstall
            '';
          };

          mkLlamaCpp =
            { pname
            , patches ? [ ./patches/llama-inference-v1.patch ]
            , withUi ? false
            , extraCmakeFlags
            , buildPhase
            , installPhase
            }:
            pkgs.libcxxStdenv.mkDerivation {
              inherit pname patches buildPhase installPhase;
              version = "pinned";
              src = llama-cpp-src;

              nativeBuildInputs = [
                pkgs.cmake
                pkgs.ninja
              ];

              cmakeFlags = [
                "-DBUILD_SHARED_LIBS=ON"
                # We install by copying the .so's out of build/bin (not `cmake
                # --install`), so they keep their build-tree RPATH. Emit those
                # self-references as $ORIGIN-relative instead of absolute
                # /build/... paths — NixOS's fixup audit forbids /build/ refs in
                # store outputs. All libs are co-located (build/bin -> $out/lib),
                # so $ORIGIN resolves correctly; absolute nix-store rpaths (e.g.
                # libstdc++) are left untouched.
                "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
                "-DLLAMA_BUILD_COMMON=ON"
                "-DLLAMA_BUILD_TESTS=OFF"
                "-DLLAMA_BUILD_EXAMPLES=OFF"
                "-DLLAMA_CURL=OFF"
                "-DGGML_METAL=OFF"
                "-DGGML_ACCELERATE=OFF"
                "-DGGML_BLAS=OFF"
              ] ++ extraCmakeFlags;

              postPatch = pkgs.lib.optionalString withUi ''
                cp -R "${llama-ui}" tools/ui/dist
                chmod -R u+w tools/ui/dist
              '';
            };

          llama-cpp = mkLlamaCpp {
            pname = "penzai-llama-cpp";

            extraCmakeFlags = [
              "-DGGML_BACKEND_DL=OFF"
              "-DLLAMA_BUILD_TOOLS=OFF"
              "-DLLAMA_BUILD_SERVER=OFF"
            ];

            buildPhase = ''
              runHook preBuild
              cmake --build . --target llama llama-common
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p "$out/lib"
              find . \( -type f -o -type l \) \
                \( -name 'lib*.so*' -o -name 'lib*.dylib' -o -name 'lib*.a' \) \
                -exec cp -P {} "$out/lib/" \;
              runHook postInstall
            '';
          };

          # DL-enabled llama.cpp tools with the versioned whole-token executor patch.
          llama-cpp-dl = mkLlamaCpp {
            pname = "penzai-llama-cpp-dl";
            withUi = true;

            extraCmakeFlags = [
              "-DGGML_BACKEND_DL=ON"
              "-DLLAMA_BUILD_TOOLS=ON"
              "-DLLAMA_BUILD_SERVER=ON"
              "-DLLAMA_TOOLS_INSTALL=OFF"
            ];

            buildPhase = ''
              runHook preBuild
              cmake --build .
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p "$out/bin" "$out/lib" "$out/include/ggml"
              cp bin/llama-cli "$out/bin/"
              cp bin/llama-server "$out/bin/"
              # The in-tree CPU backend must sit next to llama-cli for the DL loader
              # to find it; the rest of the shared libs go to lib/.
              if [ -e bin/libggml-cpu.so ]; then cp -P bin/libggml-cpu.so "$out/bin/"; fi
              find bin -maxdepth 1 \( -name 'lib*.so*' -o -name 'lib*.dylib' \) \
                ! -name 'libggml-cpu.so' -exec cp -P {} "$out/lib/" \;
              cp -R "$src/ggml/include/." "$out/include/ggml/"
              runHook postInstall
            '';
          };

          dynamicLibraryPathVar =
            if pkgs.stdenv.hostPlatform.isDarwin then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH";
          soExt = if pkgs.stdenv.hostPlatform.isDarwin then "dylib" else "so";

          # Zig's native libc detection fails inside the Nix build sandbox and
          # falls back to a musl ABI (wrong on glibc NixOS). For the libc++ host
          # build that breaks two ways: libc++ takes its musl <bits/alltypes.h>
          # mbstate_t branch (compile error), and any binary it did emit would
          # carry a musl loader (/lib/ld-musl-*) that can't run here. Force the
          # glibc ABI for the host target so libc++ uses glibc; autoPatchelfHook
          # then repoints the interpreter/rpath at the Nix glibc. null on darwin
          # (no musl/glibc split there) — keep native.
          hostGnuTarget = {
            "x86_64-linux" = "x86_64-linux-gnu";
            "aarch64-linux" = "aarch64-linux-gnu";
          }.${system} or null;

          zigCacheSetup = ''
            export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global-cache"
            export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local-cache"
            mkdir -p "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR"
          '';

          # Source tree for the Zig builds, restricted to what `zig build`
          # actually reads (per the b.path()/addCosim()/sby references in
          # build.zig). Using ./. verbatim would make unrelated routed artifacts
          # invalidate every Zig package. Local models are untracked and remain
          # outside the flake source.
          zigSrc = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [
              ./build.zig
              ./device
              ./host
              ./shared
              ./test
              ./fpga/build
              ./fpga/regmap
              ./fpga/rtl
              ./fpga/verify
            ];
          };

          mkZigPackage =
            { pname
            , step
            , optimize
            , withLlama ? false
            , llamaLib ? llama-cpp
            }:
            let
              # Only the llama host build pulls in libc++ and a dynamic loader,
              # so the glibc-ABI workaround (see hostGnuTarget) applies there —
              # and only on Linux (hostGnuTarget is null on darwin).
              forceGnu = withLlama && hostGnuTarget != null;
            in
            pkgs.stdenv.mkDerivation {
              inherit pname;
              version = "0.1.0";
              src = zigSrc;

              nativeBuildInputs = [
                pkgs.zig
              ] ++ pkgs.lib.optional forceGnu pkgs.autoPatchelfHook;

              # autoPatchelfHook repoints the host binary's interpreter to the
              # Nix glibc and resolves the llama .so DT_NEEDED entries.
              buildInputs = pkgs.lib.optional forceGnu llamaLib;

              dontConfigure = true;

              buildPhase = ''
                runHook preBuild
                ${zigCacheSetup}
                zig build ${step} \
                  -Doptimize=${optimize} \
                  ${pkgs.lib.optionalString forceGnu "-Dtarget=${hostGnuTarget} "}${pkgs.lib.optionalString withLlama ''
                    -Dllama-src=${llama-cpp-src} \
                    -Dllama-lib=${llamaLib} \
                  ''}--prefix "$out"
                runHook postBuild
              '';

              installPhase = ''
                runHook preInstall
                runHook postInstall
              '';

              passthru = {
                inherit optimize;
              } // pkgs.lib.optionalAttrs withLlama {
                inherit llama-cpp;
              };
            };

          mkZigCheck =
            { pname
            , step
            , optimize ? "Debug"
            , extraNativeBuildInputs ? [ ]
            }:
            pkgs.stdenv.mkDerivation {
              inherit pname;
              version = "0.1.0";
              src = zigSrc;

              nativeBuildInputs = [
                pkgs.zig
              ] ++ extraNativeBuildInputs;

              dontConfigure = true;

              buildPhase = ''
                runHook preBuild
                ${zigCacheSetup}
                zig build ${step} -Doptimize=${optimize}
                runHook postBuild
              '';

              installPhase = ''
                runHook preInstall
                mkdir -p "$out"
                touch "$out/passed"
                runHook postInstall
              '';
            };

          mkPenzai = name: optimize: mkZigPackage {
            pname = "penzai-${name}";
            step = "install-penzai";
            inherit optimize;
            withLlama = true;
          };

          mkPenzaid = name: optimize: mkZigPackage {
            pname = "penzaid-${name}";
            step = "install-penzaid";
            inherit optimize;
          };

          # The out-of-tree backend .so, built by Zig against the DL llama's ggml-base
          # so it shares one libggml-base with the llama-cli that dlopens it.
          penzai-backend-so = mkZigPackage {
            pname = "penzai-backend-so";
            step = "backend-so";
            optimize = "ReleaseFast";
            withLlama = true;
            llamaLib = llama-cpp-dl;
          };

          penzai-core = mkPenzai "releasefast" "ReleaseFast";
          penzai-app = pkgs.runCommand "penzai-releasefast" {
            nativeBuildInputs = [ pkgs.makeWrapper ];
          } ''
            mkdir -p "$out/bin"
            makeWrapper ${penzai-core}/bin/penzai "$out/bin/penzai" \
              --set-default PENZAI_LLAMA_SERVER ${llama-cpp-dl}/bin/llama-server \
              --set-default PENZAI_GGML_BACKEND_PATH ${penzai-backend-so}/lib/libggml-penzai.${soExt} \
              --set-default GGML_BACKEND_PATH ${llama-cpp-dl}/bin/libggml-cpu.so \
              --prefix ${dynamicLibraryPathVar} : ${llama-cpp-dl}/lib:${llama-cpp-dl}/bin:${penzai-backend-so}/lib
          '';

          mkApp = package: program: description: {
            type = "app";
            program = "${package}/bin/${program}";
            meta.description = description;
          };

          mkDeploy = name: label: penzaidPackage:
            pkgs.writeShellApplication {
              inherit name;
              runtimeInputs = [
                pkgs.openssh
              ];
              text = ''
                set -euo pipefail

                BOARD="''${BOARD:-ubuntu@kria}"
                BOARD_TMP="''${BOARD_TMP:-/tmp/penzai}"
                SSH_BIN="''${PENZAI_SSH_BIN:-/usr/bin/ssh}"
                SCP_BIN="''${PENZAI_SCP_BIN:-/usr/bin/scp}"
                REMOTE_BIN="$BOARD_TMP/penzaid"
                REMOTE_KILL_PATTERN="$BOARD_TMP/[p]enzaid"

                [ -x "$SSH_BIN" ] || SSH_BIN=ssh
                [ -x "$SCP_BIN" ] || SCP_BIN=scp

                SSH_ARGS=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10)
                if [ -n "''${PENZAI_SSH_IDENTITY:-}" ]; then
                  [ -r "$PENZAI_SSH_IDENTITY" ] || {
                    echo "ERROR: PENZAI_SSH_IDENTITY is not readable: $PENZAI_SSH_IDENTITY" >&2
                    exit 1
                  }
                  SSH_ARGS+=(-o IdentityAgent=none -o IdentitiesOnly=yes -i "$PENZAI_SSH_IDENTITY")
                fi

                echo "== deploy ${label} penzaid -> $BOARD:$REMOTE_BIN =="
                echo "== stop existing penzaid on $BOARD =="
                # shellcheck disable=SC2029
                "$SSH_BIN" "''${SSH_ARGS[@]}" "$BOARD" "sudo pkill -f '$REMOTE_KILL_PATTERN' 2>/dev/null || true"
                # shellcheck disable=SC2029
                "$SSH_BIN" "''${SSH_ARGS[@]}" "$BOARD" "mkdir -p '$BOARD_TMP' && rm -f '$REMOTE_BIN'"

                echo "== copying penzaid to $BOARD:$REMOTE_BIN =="
                "$SCP_BIN" "''${SSH_ARGS[@]}" "${penzaidPackage}/bin/penzaid" "$BOARD:$REMOTE_BIN"
                # shellcheck disable=SC2029
                "$SSH_BIN" "''${SSH_ARGS[@]}" "$BOARD" "chmod u+w,+x '$REMOTE_BIN'"
                echo "== deployed $REMOTE_BIN =="
              '';
            };

          mkServe = name:
            pkgs.writeShellApplication {
              inherit name;
              runtimeInputs = [
                pkgs.openssh
              ];
              text = ''
                set -euo pipefail

                BOARD="''${BOARD:-ubuntu@kria}"
                BOARD_TMP="''${BOARD_TMP:-/tmp/penzai}"
                PENZAI_LOCAL_PORT="''${PENZAI_LOCAL_PORT:-''${PENZAI_PORT:-29092}}"
                PENZAI_REMOTE_PORT="''${PENZAI_REMOTE_PORT:-29092}"
                PENZAI_FORWARD="''${PENZAI_FORWARD:-1}"
                PENZAI_MEM="''${PENZAI_MEM:-xrt}"
                PENZAI_HEAP_MIB="''${PENZAI_HEAP_MIB:-1500}"
                SSH_BIN="''${PENZAI_SSH_BIN:-/usr/bin/ssh}"
                REMOTE_BIN="$BOARD_TMP/penzaid"

                [ -x "$SSH_BIN" ] || SSH_BIN=ssh

                SSH_ARGS=(
                  -o BatchMode=yes
                  -o StrictHostKeyChecking=yes
                  -o ConnectTimeout=10
                  -o ServerAliveInterval=15
                  -o ServerAliveCountMax=3
                  -o TCPKeepAlive=yes
                )
                case "$PENZAI_FORWARD" in
                  1)
                    SSH_ARGS+=(
                      -o ExitOnForwardFailure=yes
                      -L "127.0.0.1:$PENZAI_LOCAL_PORT:127.0.0.1:$PENZAI_REMOTE_PORT"
                    )
                    ;;
                  0) ;;
                  *)
                    echo "ERROR: PENZAI_FORWARD must be 0 or 1" >&2
                    exit 1
                    ;;
                esac
                if [ -n "''${PENZAI_SSH_IDENTITY:-}" ]; then
                  [ -r "$PENZAI_SSH_IDENTITY" ] || {
                    echo "ERROR: PENZAI_SSH_IDENTITY is not readable: $PENZAI_SSH_IDENTITY" >&2
                    exit 1
                  }
                  SSH_ARGS+=(-o IdentityAgent=none -o IdentitiesOnly=yes -i "$PENZAI_SSH_IDENTITY")
                fi

                if [ "$PENZAI_FORWARD" = 1 ]; then
                  echo "== forward tcp:127.0.0.1:$PENZAI_LOCAL_PORT -> $BOARD tcp:127.0.0.1:$PENZAI_REMOTE_PORT =="
                else
                  echo "== use independently managed tunnel to $BOARD tcp:127.0.0.1:$PENZAI_REMOTE_PORT =="
                fi
                echo "== serve penzaid mem=$PENZAI_MEM heap_mib=$PENZAI_HEAP_MIB =="
                # shellcheck disable=SC2029
                exec "$SSH_BIN" "''${SSH_ARGS[@]}" "$BOARD" "sudo '$REMOTE_BIN' serve --device 'tcp:127.0.0.1:$PENZAI_REMOTE_PORT' --mem '$PENZAI_MEM' --heap-mib '$PENZAI_HEAP_MIB'"
              '';
            };

          mkHello = penzaiPackage:
            pkgs.writeShellApplication {
              name = "hello";
              text = ''
                set -euo pipefail

                "${penzaiPackage}/bin/penzai" run \
                  -m ./models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
                  --prompt "hello" \
                  --max-tokens 7 \
                  --metrics summary
              '';
            };

          # Patched llama tools with the Penzai executor selected by default.
          llama-cli-penzai = pkgs.writeShellScriptBin "llama-cli-penzai" ''
            set -e
            export GGML_BACKEND_PATH="${penzai-backend-so}/lib/libggml-penzai.${soExt}"
            export PENZAI_METRICS=none
            export ${dynamicLibraryPathVar}="${llama-cpp-dl}/lib:${llama-cpp-dl}/bin:${penzai-backend-so}/lib''${${dynamicLibraryPathVar}:+:''${${dynamicLibraryPathVar}}}"
            exec "${llama-cpp-dl}/bin/llama-cli" \
              --device penzai -ngl 999 --no-op-offload -fa on \
              --no-context-shift -c 4096 -b 32 -ub 16 --temp 0 --no-warmup "$@"
          '';

          llama-server-penzai = pkgs.writeShellScriptBin "llama-server-penzai" ''
            set -e
            export GGML_BACKEND_PATH="${penzai-backend-so}/lib/libggml-penzai.${soExt}"
            export PENZAI_METRICS=none
            export ${dynamicLibraryPathVar}="${llama-cpp-dl}/lib:${llama-cpp-dl}/bin:${penzai-backend-so}/lib''${${dynamicLibraryPathVar}:+:''${${dynamicLibraryPathVar}}}"
            exec "${llama-cpp-dl}/bin/llama-server" \
              --device penzai -ngl 999 --no-op-offload -fa on \
              --parallel 1 --no-context-shift \
              --ctx-size 4096 --batch-size 32 --ubatch-size 16 \
              --temp 0 --no-warmup "$@"
          '';
        in
        rec {
          packages = rec {
            inherit llama-cpp llama-cpp-dl penzai-backend-so llama-cli-penzai llama-server-penzai;

            penzai = penzai-app;
            penzaid = mkPenzaid "releasefast" "ReleaseFast";
            # Native penzaid for protocol and memory testing without a board.
            penzaid-native = mkZigPackage {
              pname = "penzaid-native";
              step = "install-penzaid-native";
              optimize = "ReleaseFast";
            };

            deploy-penzaid = mkDeploy "deploy-penzaid" "ReleaseFast" penzaid;
            serve-penzaid = mkServe "serve-penzaid";
            hello = mkHello penzai;

            default = penzai;
          };

          apps = rec {
            default = penzai;

            penzai = mkApp packages.penzai "penzai" "Run the releasefast penzai host CLI";

            deploy-penzaid = mkApp packages.deploy-penzaid "deploy-penzaid" "Deploy the releasefast KR260 penzaid daemon";
            serve-penzaid = mkApp packages.serve-penzaid "serve-penzaid" "Run the deployed KR260 penzaid daemon over SSH";
            penzaid-native = mkApp packages.penzaid-native "penzaid-native" "Run a local native penzaid daemon";
            hello = mkApp packages.hello "hello" "Run the Bonsai hello inference path";
            llama-cli-penzai = mkApp packages.llama-cli-penzai "llama-cli-penzai" "Stock llama-cli with the out-of-tree penzai backend";
            llama-server-penzai = mkApp packages.llama-server-penzai "llama-server-penzai" "Stock llama-server (HTTP) with the out-of-tree penzai backend";
          };

          checks = rec {
            default = zig-tests;
            zig-tests = mkZigCheck {
              pname = "penzai-zig-tests";
              step = "test";
            };
            rtl-lint = mkZigCheck {
              pname = "penzai-rtl-lint";
              step = "lint-rtl";
              extraNativeBuildInputs = [
                pkgs.bash
                pkgs.verilator
              ];
            };
            rtl-sim = mkZigCheck {
              pname = "penzai-rtl-sim";
              step = "test-rtl";
              extraNativeBuildInputs = [
                pkgs.bash
                pkgs.verilator
              ];
            };
            rtl-synth = mkZigCheck {
              pname = "penzai-rtl-synth";
              step = "synth-rtl";
              extraNativeBuildInputs = [
                pkgs.bash
                pkgs.yosys
              ];
            };
            rtl-formal = mkZigCheck {
              pname = "penzai-rtl-formal";
              step = "formal";
              extraNativeBuildInputs = [
                pkgs.yosys
                pkgs.sby
                pkgs.boolector
              ];
            };
            llama-inference = pkgs.runCommand "penzai-llama-inference-smoke" {
              nativeBuildInputs = [
                pkgs.python3
                pkgs.libcxxStdenv.cc
              ];
            } ''
              cc -std=c11 -Wall -Wextra -Werror \
                -I${llama-cpp-src}/include \
                -I${llama-cpp-dl}/include/ggml \
                -L${llama-cpp-dl}/lib \
                -Wl,-rpath,${llama-cpp-dl}/lib \
                ${zigSrc}/test/llama_logits_probe.c \
                -lllama -lggml \
                -o llama-logits-probe
              export ${dynamicLibraryPathVar}="${llama-cpp-dl}/lib:${llama-cpp-dl}/bin:${penzai-backend-so}/lib"
              python3 ${zigSrc}/test/llama_inference_smoke.py \
                --llama-cli ${llama-cpp-dl}/bin/llama-cli \
                --llama-server ${llama-cpp-dl}/bin/llama-server \
                --penzai ${penzai-app}/bin/penzai \
                --backend ${penzai-backend-so}/lib/libggml-penzai.${soExt} \
                --cpu-backend ${llama-cpp-dl}/bin/libggml-cpu.so \
                --logits-probe ./llama-logits-probe
              mkdir -p "$out"
              touch "$out/passed"
            '';
            inherit (packages) penzai penzaid;
          };

          devShells.default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.zig
              pkgs.zls
              pkgs.verilator
              pkgs.python3
              pkgs.yosys
              pkgs.sby
              pkgs.boolector
              pkgs.cmake
              pkgs.ninja
              pkgs.openssh
            ];

            LLAMA_CPP_SRC = llama-cpp-src;
            LLAMA_CPP_LIB = llama-cpp;
          };
        };
    };
}
