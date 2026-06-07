{
  description = "KR260 Q1A8 matmul bring-up: reference oracle, packer, RTL sim, hardware differential";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    zig-flake.url = "github:mitchellh/zig-overlay";
  };

  outputs = inputs@{ nixpkgs, zig-flake, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      pkgsFor = system: import nixpkgs {
        inherit system;
        overlays = [ (_final: _prev: { zig = zig-flake.packages.${system}."0.16.0"; }) ];
      };
    in {
      packages = forAllSystems (system:
        let pkgs = pkgsFor system; in {
          default = pkgs.stdenv.mkDerivation {
            pname = "kr260-q1a8-matmul-bringup";
            version = "0.1.0";
            src = ./.;
            nativeBuildInputs = [ pkgs.zig ];
            buildPhase = ''
              runHook preBuild
              export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global-cache"
              export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local-cache"
              zig build test
              zig build -Doptimize=ReleaseSafe
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              mkdir -p "$out/bin"
              cp zig-out/bin/kr260-q1a8-selftest "$out/bin/"
              runHook postInstall
            '';
          };
        });

      checks = forAllSystems (system: { default = inputs.self.packages.${system}.default; });

      devShells = forAllSystems (system:
        let pkgs = pkgsFor system; in {
          default = pkgs.mkShell {
            nativeBuildInputs = [ pkgs.zig pkgs.verilator pkgs.openssh pkgs.dtc ];
            shellHook = ''
              echo "kr260-q1a8-matmul-bringup:"
              echo "  zig build test       # M0 oracle + M1 packer unit tests"
              echo "  zig build run        # laptop self-test (pack roundtrip + reference)"
              echo "  zig build lint-rtl   # M2 entry: Verilator lint of ported RTL"
            '';
          };
        });
    };
}
