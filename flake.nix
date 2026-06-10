{
  description = "penzai";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    zig-flake.url = "github:mitchellh/zig-overlay";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/a95a11e5b834057e684712963f90bbb730f4745c";
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

          tiny-gguf = pkgs.fetchurl {
            url = "https://huggingface.co/ggml-org/tiny-llamas/resolve/def3e2dd70df35ecbf6403ea347de4c5977220c1/stories260K.gguf";
            hash = "sha256-BHv0ZFWlRJMc/2/vFNeRAVTFavvCOrHF5Wpy5pkSwEs=";
          };

          llama-cpp = pkgs.stdenv.mkDerivation {
            pname = "penzai-llama-cpp";
            version = "pinned";
            src = llama-cpp-src;

            nativeBuildInputs = [
              pkgs.cmake
              pkgs.ninja
            ];

            cmakeFlags = [
              "-DBUILD_SHARED_LIBS=ON"
              "-DGGML_BACKEND_DL=OFF"
              "-DLLAMA_BUILD_COMMON=ON"
              "-DLLAMA_BUILD_TOOLS=OFF"
              "-DLLAMA_BUILD_TESTS=OFF"
              "-DLLAMA_BUILD_EXAMPLES=OFF"
              "-DLLAMA_BUILD_SERVER=OFF"
              "-DLLAMA_CURL=OFF"
              "-DGGML_METAL=OFF"
              "-DGGML_ACCELERATE=OFF"
              "-DGGML_BLAS=OFF"
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

          zigCacheSetup = ''
            export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global-cache"
            export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local-cache"
            mkdir -p "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR"
          '';

          mkZigPackage =
            { pname
            , step
            , optimize
            , withLlama ? false
            }:
            pkgs.stdenv.mkDerivation {
              inherit pname;
              version = "0.1.0";
              src = ./.;

              nativeBuildInputs = [
                pkgs.zig
              ];

              dontConfigure = true;

              buildPhase = ''
                runHook preBuild
                ${zigCacheSetup}
                zig build ${step} \
                  -Doptimize=${optimize} \
                  ${pkgs.lib.optionalString withLlama ''
                    -Dllama-src=${llama-cpp-src} \
                    -Dllama-lib=${llama-cpp} \
                    -Dmodel=${tiny-gguf} \
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
                inherit llama-cpp tiny-gguf;
              };
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
                REMOTE_BIN="$BOARD_TMP/penzaid"
                REMOTE_KILL_PATTERN="$BOARD_TMP/[p]enzaid"

                echo "== deploy ${label} penzaid -> $BOARD:$REMOTE_BIN =="
                echo "== stop existing penzaid on $BOARD =="
                # shellcheck disable=SC2029
                ssh "$BOARD" "pkill -f '$REMOTE_KILL_PATTERN' 2>/dev/null || true"
                # shellcheck disable=SC2029
                ssh "$BOARD" "mkdir -p '$BOARD_TMP'"
                scp "${penzaidPackage}/bin/penzaid" "$BOARD:$REMOTE_BIN"
                # shellcheck disable=SC2029
                ssh "$BOARD" "chmod +x '$REMOTE_BIN'"
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
                PENZAI_PORT="''${PENZAI_PORT:-29092}"
                PENZAI_MEM="''${PENZAI_MEM:-xrt}"
                PENZAI_HEAP_MIB="''${PENZAI_HEAP_MIB:-768}"
                REMOTE_BIN="$BOARD_TMP/penzaid"

                echo "== serve penzaid on $BOARD tcp:0.0.0.0:$PENZAI_PORT mem=$PENZAI_MEM heap_mib=$PENZAI_HEAP_MIB =="
                # shellcheck disable=SC2029
                exec ssh "$BOARD" "'$REMOTE_BIN' serve --device 'tcp:0.0.0.0:$PENZAI_PORT' --mem '$PENZAI_MEM' --heap-mib '$PENZAI_HEAP_MIB'"
              '';
            };

          mkHello = penzaiPackage:
            pkgs.writeShellApplication {
              name = "hello";
              text = ''
                set -euo pipefail

                "${penzaiPackage}/bin/penzai" run \
                  -m ./models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
                  --device tcp:kria:29092 \
                  --prompt "hello" \
                  --max-tokens 7 \
                  --prof
                say hello
              '';
            };
        in
        rec {
          packages = rec {
            inherit llama-cpp tiny-gguf;

            penzai = penzai-releasefast;
            penzai-debug = mkPenzai "debug" "Debug";
            penzai-releasefast = mkPenzai "releasefast" "ReleaseFast";

            penzaid = penzaid-releasefast;
            penzaid-debug = mkPenzaid "debug" "Debug";
            penzaid-releasefast = mkPenzaid "releasefast" "ReleaseFast";

            deploy-penzaid = deploy-penzaid-releasefast;
            deploy-penzaid-debug = mkDeploy "deploy-penzaid-debug" "Debug" penzaid-debug;
            deploy-penzaid-releasefast = mkDeploy "deploy-penzaid-releasefast" "ReleaseFast" penzaid-releasefast;
            serve-penzaid = mkServe "serve-penzaid";
            hello = mkHello penzai;

            default = penzai-releasefast;
          };

          apps = rec {
            default = penzai;

            penzai = mkApp packages.penzai-releasefast "penzai" "Run the releasefast penzai host CLI";
            penzai-debug = mkApp packages.penzai-debug "penzai" "Run the debug penzai host CLI";
            penzai-releasefast = mkApp packages.penzai-releasefast "penzai" "Run the releasefast penzai host CLI";

            deploy-penzaid = mkApp packages.deploy-penzaid "deploy-penzaid-releasefast" "Deploy the releasefast KR260 penzaid daemon";
            deploy-penzaid-debug = mkApp packages.deploy-penzaid-debug "deploy-penzaid-debug" "Deploy the debug KR260 penzaid daemon";
            deploy-penzaid-releasefast = mkApp packages.deploy-penzaid-releasefast "deploy-penzaid-releasefast" "Deploy the releasefast KR260 penzaid daemon";
            serve-penzaid = mkApp packages.serve-penzaid "serve-penzaid" "Run the deployed KR260 penzaid daemon over SSH";
            hello = mkApp packages.hello "hello" "Run the Bonsai hello inference path";
          };

          checks = {
            default = packages.penzai-releasefast;
            inherit (packages) penzai-debug penzai-releasefast penzaid-debug penzaid-releasefast;
          };

          devShells.default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.zig
              pkgs.zls
              pkgs.verilator
              pkgs.cmake
              pkgs.ninja
              pkgs.openssh
            ];

            LLAMA_CPP_SRC = llama-cpp-src;
            LLAMA_CPP_LIB = llama-cpp;
            TINY_GGUF = tiny-gguf;
          };
        };
    };
}
