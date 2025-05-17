let
  pkgs = import <nixpkgs> {};
in

pkgs.mkShell {
  buildInputs = [
    pkgs.python311
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.joblib
    pkgs.python311Packages.pandas
    pkgs.python311Packages.scikit-learn
    pkgs.python311Packages.numpy
    pkgs.python311Packages.seaborn
    pkgs.cope
  ];
}