{
  description = "penzai llama.cpp backend experiments";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    zig-flake.url = "github:mitchellh/zig-overlay";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/a95a11e5b834057e684712963f90bbb730f4745c";
      flake = false;
    };
  };

  outputs = inputs@{ flake-parts, nixpkgs, zig-flake, llama-cpp-src, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      perSystem = { self', system, ... }:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              (_final: _prev: {
                zig = zig-flake.packages.${system}."0.16.0";
              })
            ];
          };

          tiny-gguf = pkgs.fetchurl {
            url = "https://huggingface.co/ggml-org/tiny-llamas/resolve/def3e2dd70df35ecbf6403ea347de4c5977220c1/stories260K.gguf";
            hash = "sha256-BHv0ZFWlRJMc/2/vFNeRAVTFavvCOrHF5Wpy5pkSwEs=";
          };

          llama-cpp = pkgs.stdenv.mkDerivation {
            pname = "penzai-e2e-llama-cpp";
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

          llama-experiments = pkgs.stdenv.mkDerivation {
            pname = "penzai-llama-experiments";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = [
              pkgs.zig
            ];

            buildPhase = ''
              runHook preBuild
              export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global-cache"
              export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local-cache"
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
            inherit llama-cpp tiny-gguf;
            all = llama-experiments;
            default = llama-experiments;
            backend-e2e = llama-experiments;
            remote-buffer-e2e = llama-experiments;
            partial-offload = llama-experiments;
            op-census = llama-experiments;
            binding-lower-dryrun = llama-experiments;
          };

          apps = {
            default = self'.apps.backend-e2e;
            backend-e2e = {
              type = "app";
              program = "${self'.packages.all}/bin/llama-backend-e2e";
            };
            remote-buffer-e2e = {
              type = "app";
              program = "${self'.packages.all}/bin/llama-remote-buffer-e2e";
            };
            partial-offload = {
              type = "app";
              program = "${self'.packages.all}/bin/llama-partial-offload";
            };
            op-census = {
              type = "app";
              program = "${self'.packages.all}/bin/llama-op-census";
            };
            binding-lower-dryrun = {
              type = "app";
              program = "${self'.packages.all}/bin/llama-binding-lower-dryrun";
            };
          };

          checks = {
            backend-e2e = self'.packages.backend-e2e;
            remote-buffer-e2e = self'.packages.remote-buffer-e2e;
            partial-offload = self'.packages.partial-offload;
            op-census = self'.packages.op-census;
            binding-lower-dryrun = self'.packages.binding-lower-dryrun;
          };

          devShells.default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.zig
              pkgs.cmake
              pkgs.ninja
            ];
            shellHook = ''
              echo "llama experiments:"
              echo "  nix run .#backend-e2e"
              echo "  nix run .#remote-buffer-e2e"
              echo "  nix run .#partial-offload"
              echo "  nix run .#op-census"
              echo "  nix run .#binding-lower-dryrun"
            '';
          };
        };
    };
}
