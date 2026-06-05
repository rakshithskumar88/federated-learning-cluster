{
  description = "Federated Learning - Multi-Node RPC Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;   # Required for NVIDIA binaries
            cudaSupport = true;   # Enforce CUDA acceleration explicitly
          };
        };
        
        # We explicitly define our Python ecosystem here.
        # Note the use of `torch-bin`: this pulls pre-compiled PyTorch binaries,
        # preventing a 4-hour local compilation loop on your machine.
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          torch-bin       
          grpcio
          grpcio-tools    # Contains the protoc compiler for Python
          protobuf
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.protobuf  # The core C++ protocol buffer compiler
          ];

          shellHook = ''
            echo "========================================================"
            echo " Federated Learning Research Environment Initialized"
            echo " Compute Target: NVIDIA CUDA Runtime Bound"
            echo " Protocol Buffer RPC Engine: Active"
            echo "========================================================"
            
            # Ensure our source directory structure exists
            mkdir -p src/generated
          '';
        };
      }
    );
}
