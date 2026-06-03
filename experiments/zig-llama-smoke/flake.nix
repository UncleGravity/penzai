{
  description = "penzai Zig + llama.cpp/ggml smoke experiment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    zig-flake.url = "github:mitchellh/zig-overlay";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/a95a11e5b834057e684712963f90bbb730f4745c";
      flake = false;
    };
  };

  outputs = inputs@{ nixpkgs, zig-flake, llama-cpp-src, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              (_final: _prev: {
                zig = zig-flake.packages.${system}."master";
              })
            ];
          };

          llama-cpp = pkgs.stdenv.mkDerivation {
            pname = "penzai-smoke-llama-cpp";
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

          smoke = pkgs.stdenv.mkDerivation {
            pname = "penzai-ggml-smoke";
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
                -Dllama-lib=${llama-cpp}
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p "$out/bin"
              cp zig-out/bin/penzai-ggml-smoke "$out/bin/"
              runHook postInstall
            '';

            passthru = {
              inherit llama-cpp;
            };
          };
        in {
          inherit llama-cpp smoke;
          default = smoke;
        });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${inputs.self.packages.${system}.default}/bin/penzai-ggml-smoke";
        };
        smoke = {
          type = "app";
          program = "${inputs.self.packages.${system}.smoke}/bin/penzai-ggml-smoke";
        };
      });

      checks = forAllSystems (system: {
        smoke = inputs.self.packages.${system}.smoke;
      });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              (_final: _prev: {
                zig = zig-flake.packages.${system}."master";
              })
            ];
          };
        in {
          default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.zig
              pkgs.cmake
              pkgs.ninja
            ];
            shellHook = ''
              echo "zig llama smoke:"
              echo "  nix run .#smoke"
            '';
          };
        });
    };
}
