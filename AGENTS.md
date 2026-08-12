Never use `path:`/`builtins.path`; they copy (sometimes large) untracked files.
Keep `src = zigSrc;`. Reclaim space: `nix store gc && nix store optimise`.
