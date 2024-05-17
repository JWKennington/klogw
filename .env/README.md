# Environment Control for KLogW

Base environment files stored with `base` in name

Locked environment files stored with `lock` in name


## Locking envs

```bash
conda-lock lock -f klogw-base.yml -p osx-arm64 -p linux-64 -p osx-64 -p linux-aarch64 --filename-template "conda-klogw-{platform}.conda.lock" --lockfile "conda-lock-klogw.yml"
```

## Rendering envs

```bash
conda-lock render -p linux-64 -p osx-64 -p osx-arm64 -p linux-aarch64 --filename-template "conda-klogw-{platform}.lock" "conda-lock-klogw.yml"
```

## Creating envs from locks

Note: you must deactivate your current env before creating a new one if you're in `klogw-dev` env, which you can do with `conda deactivate`.

```bash
conda create -n klogw-dev --file conda-klogw-osx-arm64.lock --force
```