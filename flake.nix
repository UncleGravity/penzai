{
  description = "YOUR DESCRIPTION HERE";

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

            # --------------- Zig
            overlays = [
              (final: prev: {
                zig = zig-flake.packages.${system}."0.16.0";
                zls = zls-flake.packages.${system}.default.overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ])
                    ++ [ final.zig ];
                });
              })
            ];
            # ---------------
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
              "-DLLAMA_BUILD_COMMON=OFF"
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
              cmake --build . --target llama
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

          penzai = pkgs.stdenv.mkDerivation {
            pname = "penzai";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = [
              pkgs.zig
            ];

            buildPhase = ''
              runHook preBuild
              export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global-cache"
              export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local-cache"
              mkdir -p "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR"
              zig build \
                -Doptimize=ReleaseSafe \
                -Dllama-src=${llama-cpp-src} \
                -Dllama-lib=${llama-cpp} \
                -Dmodel=${tiny-gguf}
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p "$out/bin"
              cp zig-out/bin/* "$out/bin/"
              runHook postInstall
            '';

            passthru = {
              inherit llama-cpp tiny-gguf;
            };
          };
        in {
          packages = {
            inherit llama-cpp tiny-gguf penzai;
            default = penzai;
          };

          apps = {
            default = {
              type = "app";
              program = "${penzai}/bin/penzai";
            };
            penzai = {
              type = "app";
              program = "${penzai}/bin/penzai";
            };
          };

          checks = {
            default = penzai;
            inherit penzai;
          };

          devShells.default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.zig
              pkgs.zls
              pkgs.verilator
              pkgs.cmake
              pkgs.ninja
            ];
            LLAMA_CPP_SRC = llama-cpp-src;
            LLAMA_CPP_LIB = llama-cpp;
            TINY_GGUF = tiny-gguf;
          };
        };
    };
}
