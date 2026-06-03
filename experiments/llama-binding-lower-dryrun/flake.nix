{
  description = "penzai llama.cpp remote binding and dry-run lowering experiment";

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
                zig = zig-flake.packages.${system}."master";
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

          e2e = pkgs.stdenv.mkDerivation {
            pname = "penzai-llama-binding-lower-dryrun";
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
              cp zig-out/bin/llama-binding-lower-dryrun "$out/bin/"
              runHook postInstall
            '';

            passthru = {
              inherit llama-cpp tiny-gguf;
            };
          };
        in {
          packages = {
            inherit e2e llama-cpp tiny-gguf;
            default = e2e;
          };

          apps = {
            default = {
              type = "app";
              program = "${self'.packages.default}/bin/llama-binding-lower-dryrun";
            };
            e2e = {
              type = "app";
              program = "${self'.packages.e2e}/bin/llama-binding-lower-dryrun";
            };
          };

          checks = {
            e2e = self'.packages.e2e;
          };

          devShells.default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.zig
              pkgs.cmake
              pkgs.ninja
            ];
            shellHook = ''
              echo "llama binding lower dryrun:"
              echo "  nix run .#e2e"
            '';
          };
        };
    };
}
