{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  outputs = { nixpkgs, ... } @ inputs:
    let
      inherit (nixpkgs) lib;
      withSystem = f:
        lib.fold lib.recursiveUpdate { }
          (map f [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ]);
    in
    withSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        inherit (pkgs) stdenv;
        # Use impure cc on Darwin so that python packages build correctly
        mkShell =
          if stdenv.isLinux
          then pkgs.mkShell
          else pkgs.mkShell.override { stdenv = pkgs.stdenvNoCC; };
      in
      {
        devShells.${system}.default =
          mkShell
            {
              packages = with pkgs; [
                (python312.withPackages (p: with p; [
                  jax
                  jaxtyping
                  optax
                  numpy
                  marimo
                  anywidget
                  safetensors
                ]))
              ];

              # Use ipdb
              PYTHONBREAKPOINT = "ipdb.set_trace";
            };
      }
    );
}
