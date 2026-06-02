{
  description = "YOUR DESCRIPTION HERE";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    zig-flake.url = "github:mitchellh/zig-overlay";

    # ensure zls <-> zig match versions
    zls-flake = {
      url = "github:zigtools/zls?ref=master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ flake-parts, nixpkgs, zig-flake, zls-flake, ... }:
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
                zig = zig-flake.packages.${system}."master";
                zls = zls-flake.packages.${system}.default.overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ])
                    ++ [ final.zig ];
                });
              })
            ];
            # ---------------
          };
        in {
          devShells.default = pkgs.mkShell {
            nativeBuildInputs = [ pkgs.zig pkgs.zls ];
          };
        };
    };
}
