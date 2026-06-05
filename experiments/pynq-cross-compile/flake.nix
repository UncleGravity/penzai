{
  description = "penzai PYNQ-Z1 Zig cross-compile smoke experiment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    zig-flake.url = "github:mitchellh/zig-overlay";
  };

  outputs = inputs@{ nixpkgs, zig-flake, ... }:
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
                zig = zig-flake.packages.${system}."0.16.0";
              })
            ];
          };

          smoke = pkgs.stdenv.mkDerivation {
            pname = "penzai-pynq-cross-compile-smoke";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = [
              pkgs.zig
            ];

            buildPhase = ''
              runHook preBuild
              export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global-cache"
              export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local-cache"
              zig build -Doptimize=ReleaseSafe
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p "$out/bin"
              cp zig-out/bin/penzai-pynq-cross-smoke "$out/bin/"
              cp zig-out/bin/penzai-pynq-stdlib-smoke "$out/bin/"
              runHook postInstall
            '';
          };
        in {
          default = smoke;
        });

      checks = forAllSystems (system: {
        default = inputs.self.packages.${system}.default;
      });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              (_final: _prev: {
                zig = zig-flake.packages.${system}."0.16.0";
              })
            ];
          };
        in {
          default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.zig
              pkgs.openssh
            ];
            shellHook = ''
              echo "pynq cross-compile smoke:"
              echo "  nix build"
              echo "  zig build run-board -Dboard=xilinx@pynq"
            '';
          };
        });
    };
}
