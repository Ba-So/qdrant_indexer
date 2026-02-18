{
  description = "Qdrant Indexer - Index documentation into Qdrant vector database";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  nixConfig = {
    extra-substituters = [ "https://cuda-maintainers.cachix.org" ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;

      # CUDA is only supported on x86_64-linux
      defaultSystems = lib.systems.flakeExposed;
      cudaSystems = [ "x86_64-linux" ];

      forAllSystems = lib.genAttrs defaultSystems;
      forCudaSystems = lib.genAttrs cudaSystems;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      # Overlay to patch pymupdf packages with mupdf libraries
      # pymupdf-layout expects libmupdf.so.26.11 but nixpkgs has 1.26.10
      # We ignore these missing deps since pymupdf bundles compatible libraries at runtime
      mkMupdfOverlay = pkgs: final: prev: {
        pymupdf = prev.pymupdf.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [
            pkgs.mupdf
          ];
          autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [
            "libmupdf.so.26.11"
            "libmupdfcpp.so.26.11"
          ];
        });
        pymupdf-layout = prev.pymupdf-layout.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [
            pkgs.mupdf
            final.pymupdf  # Get libraries from pymupdf
          ];
          autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [
            "libmupdf.so.26.11"
            "libmupdfcpp.so.26.11"
          ];
        });
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      # Standard Python sets (CPU only)
      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python312;
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              (mkMupdfOverlay pkgs)
            ]
          )
      );

      # CUDA-enabled Python sets (only for supported systems)
      cudaPythonSets = forCudaSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          };
          python = pkgs.python312;

          # Overlay to patch onnxruntime-gpu with CUDA libraries
          # and replace onnxruntime with onnxruntime-gpu to avoid conflicts
          cudaOverlay = final: prev: {
            onnxruntime-gpu = prev.onnxruntime-gpu.overrideAttrs (old: {
              buildInputs = (old.buildInputs or []) ++ [
                pkgs.cudaPackages.cuda_cudart
                pkgs.cudaPackages.cudnn
                pkgs.cudaPackages.libcublas
                pkgs.cudaPackages.libcurand
                pkgs.cudaPackages.libcufft
              ];
              autoPatchelfIgnoreMissingDeps = [
                "libnvinfer.so.10"
                "libnvonnxparser.so.10"
              ];
            });
            # Replace onnxruntime with onnxruntime-gpu to avoid file collisions
            onnxruntime = final.onnxruntime-gpu;
          };
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              (mkMupdfOverlay pkgs)
              cudaOverlay
            ]
          )
      );

      # CUDA library paths for LD_LIBRARY_PATH
      makeCudaLibPath = pkgs: with pkgs.cudaPackages; lib.makeLibraryPath [
        cuda_cudart
        cudnn
        libcublas
        libcurand
        libcusolver
        libcusparse
        cuda_nvrtc
      ];

    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "qdrant-indexer-dev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              echo "Qdrant Indexer development environment"
              echo "Run 'qdrant-indexer --help' to see CLI options"
            '';
          };
        }
        # Add CUDA devShell for supported systems
        // (if builtins.elem system cudaSystems then
          let
            cudaPkgs = import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
                cudaSupport = true;
              };
            };
            cudaPythonSet = cudaPythonSets.${system}.overrideScope editableOverlay;
            cudaVirtualenv = cudaPythonSet.mkVirtualEnv "qdrant-indexer-cuda-dev-env" workspace.deps.all;
            cudaLibPath = makeCudaLibPath cudaPkgs;
          in
          {
            cuda = cudaPkgs.mkShell {
              packages = [
                cudaVirtualenv
                cudaPkgs.uv
                # CUDA packages
                cudaPkgs.cudaPackages.cuda_cudart
                cudaPkgs.cudaPackages.cudnn
                cudaPkgs.cudaPackages.libcublas
                cudaPkgs.cudaPackages.cuda_nvrtc
              ];
              env = {
                UV_NO_SYNC = "1";
                UV_PYTHON = cudaPythonSet.python.interpreter;
                UV_PYTHON_DOWNLOADS = "never";
                CUDA_PATH = "${cudaPkgs.cudaPackages.cuda_cudart}";
                QDRANT_INDEXER_USE_CUDA = "1";
              };
              shellHook = ''
                unset PYTHONPATH
                export REPO_ROOT=$(git rev-parse --show-toplevel)
                export LD_LIBRARY_PATH="${cudaLibPath}:$LD_LIBRARY_PATH"
                echo "Qdrant Indexer CUDA development environment"
                echo "CUDA support enabled - GPU acceleration available"
                echo "Run 'qdrant-indexer --help' to see CLI options"
                echo "Use --gpu flag to enable GPU acceleration"
              '';
            };
          }
        else {})
      );

      packages = forAllSystems (system:
        {
          default = pythonSets.${system}.mkVirtualEnv "qdrant-indexer-env" workspace.deps.default;
        }
        # Add CUDA package for supported systems
        // (if builtins.elem system cudaSystems then
          let
            cudaPkgs = import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
                cudaSupport = true;
              };
            };
            cudaPythonSet = cudaPythonSets.${system};
            cudaLibPath = makeCudaLibPath cudaPkgs;
            # Use workspace.deps.optionals to include the 'gpu' extra (fastembed-gpu)
            cudaVenv = cudaPythonSet.mkVirtualEnv "qdrant-indexer-cuda-env" workspace.deps.optionals;
            # ONNX Runtime needs its capi directory in LD_LIBRARY_PATH for dynamic provider loading
            onnxrtLibPath = "${cudaVenv}/lib/python3.12/site-packages/onnxruntime/capi";
          in
          {
            # CUDA-enabled package: nix build .#cuda
            cuda = cudaPkgs.stdenv.mkDerivation {
              name = "qdrant-indexer-cuda";
              src = ./.;
              nativeBuildInputs = [ cudaPkgs.makeWrapper ];
              buildInputs = [
                cudaVenv
                cudaPkgs.cudaPackages.cuda_cudart
                cudaPkgs.cudaPackages.cudnn
                cudaPkgs.cudaPackages.libcublas
                cudaPkgs.cudaPackages.cuda_nvrtc
              ];
              installPhase = ''
                mkdir -p $out/bin
                makeWrapper ${cudaVenv}/bin/qdrant-indexer $out/bin/qdrant-indexer \
                  --set LD_LIBRARY_PATH "${cudaLibPath}:${onnxrtLibPath}" \
                  --set CUDA_PATH "${cudaPkgs.cudaPackages.cuda_cudart}" \
                  --set QDRANT_INDEXER_USE_CUDA "1" \
                  --unset PYTHONPATH
              '';
              meta = {
                description = "Qdrant Indexer with CUDA/GPU support";
                mainProgram = "qdrant-indexer";
              };
            };
          }
        else {})
      );
    };
}
