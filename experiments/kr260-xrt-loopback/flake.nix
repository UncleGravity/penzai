{
  description = "KR260 XRT BO + AXI DMA loopback verifier in Zig";

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
            pname = "kr260-xrt-loopback";
            version = "0.1.0";
            src = ./.;
            nativeBuildInputs = [ pkgs.zig ];
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
              cp zig-out/bin/kr260-xrt-loopback "$out/bin/"
              runHook postInstall
            '';
          };
        });

      checks = forAllSystems (system: {
        default = inputs.self.packages.${system}.default;
      });

      devShells = forAllSystems (system:
        let pkgs = pkgsFor system; in {
          default = pkgs.mkShell {
            nativeBuildInputs = [ pkgs.zig pkgs.openssh ];
            shellHook = ''
              echo "kr260-xrt-loopback:"
              echo "  zig build bitstream  # build loopback.bit.bin on the Vivado VM"
              echo "  zig build deploy     # install/load the XRT app on the KR260"
              echo "  zig build verify     # copy/run the Zig verifier"
              echo "  zig build all        # bitstream + deploy + verify"
            '';
          };
        });
    };
}
