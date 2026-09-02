Never use `path:`/`builtins.path`; they copy (sometimes large) untracked files.
Keep `src = zigSrc;`.

Reclaim space. ONLY run when space is tight.
```sh
nix store gc && nix store optimise
```
