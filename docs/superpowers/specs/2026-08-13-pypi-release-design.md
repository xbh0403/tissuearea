# Publishing `tissuearea` to PyPI

**Date:** 2026-08-13
**Status:** approved

## Goal

`pip install tissuearea` installs a working package. Today it does not: the
project depends on `openslide-python`, which is only a binding — without a
native OpenSlide library present, `import tissuearea` raises

```
ModuleNotFoundError: Couldn't locate OpenSlide shared library.
Try `pip install openslide-bin`.
```

Both the library import and the `tissuearea` console script fail this way on a
clean install. So "publish to PyPI" and "make `pip install` work" are two
separate pieces of work, and only doing the first would ship a package that is
broken on arrival.

## Non-goals

No changelog file, no documentation site, no version-bump automation, no
coverage or badge tooling. None of it serves the goal, and each is a thing to
maintain.

## A. Self-sufficient install

Add `openslide-bin>=4.0` to `[project.dependencies]`.

Verified wheel coverage for openslide-bin 4.0.1.2:

| platform | wheel |
|---|---|
| Linux x86-64 / aarch64 | `manylinux_2_28` |
| macOS 11+ | `universal2` |
| Windows | `amd64` |

The full dependency set resolves on 3.11, 3.12 and 3.13.

**Accepted limitation.** There is no wheel for glibc < 2.28 (RHEL/CentOS 7) or
Windows ARM. On those platforms pip falls back to openslide-bin's sdist and will
almost certainly fail to compile. This package has an HPC audience, so some
users will hit it. The conda path (`environment.yml`, or a system/module
OpenSlide) stays documented as the supported fallback rather than pretending
the pip path is universal.

`environment.yml` drops its explicit `openslide-bin` line, which is now implied.

## B. Metadata

- `license = { text = "MIT" }` → `license = "MIT"` plus
  `license-files = ["LICENSE"]`, and drop the `License :: OSI Approved` trove
  classifier, which PEP 639 forbids alongside a license expression.
  Test-built: `Metadata-Version: 2.5`, `License-Expression: MIT`,
  `License-File: LICENSE`, `twine check` passes.
- Add an `Issues` project URL.

Version stays **0.1.0** — it has never been published.

## C. CI — `.github/workflows/ci.yml`

On push and pull request:

- Ubuntu × Python 3.11, 3.12, 3.13
- Windows and macOS on 3.12

The Windows job is not padding. `bc22c4e` fixed a Windows-only font-resolution
bug that was found by hand; nothing currently stops the next one. The matrix
also continuously proves the `requires-python = ">=3.11"` floor.

Because `openslide-bin` is now a real dependency, a plain `pip install -e .[dev]`
on a runner exercises the same install path an end user gets — CI validates the
"pip install works" promise rather than just the test suite.

## D. Release — `.github/workflows/release.yml`

Trusted Publishing over OIDC (`permissions: id-token: write`). No API token is
stored in the repository, in GitHub secrets, or on any developer machine.

```
test (gate) → build (python -m build + twine check, upload artifact)
                ├─ workflow_dispatch → TestPyPI   [environment: testpypi]
                └─ tag v*            → PyPI       [environment: pypi]
```

Publishing uses `pypa/gh-action-pypi-publish`. Manual dispatch is the rehearsal
lever; pushing a `v*` tag is the real release.

## E. Sequence

1. Maintainer configures Trusted Publishers on TestPyPI and PyPI (web UI only —
   cannot be automated).
2. Dispatch the workflow → 0.1.0 to TestPyPI.
3. Verify: install from TestPyPI into a clean virtualenv, confirm `import
   tissuearea`, the CLI, and a real run against a `.svs` slide.
4. Tag `v0.1.0` → PyPI.

A version number is single-use *per index*. If the rehearsal exposes a problem,
0.1.0 is burned on TestPyPI only and the retry bumps there; real PyPI 0.1.0
stays clean.

## Risks

- **Publishing is effectively irreversible.** A yanked release cannot have its
  version reused, and the name `tissuearea` becomes permanently owned. The name
  was confirmed unclaimed on 2026-08-13.
- **The `openslide-bin` dependency is the one real judgement call.** It is what
  makes `pip install` honest, at the cost of shipping a ~30 MB native library to
  users who already have OpenSlide, where it will take precedence over a system
  build. Verification runs for the `--resume` fix already used exactly this
  bundled library and reproduced the stored reference outputs byte-identically,
  so correctness is not a concern.

## Verification

- Clean-virtualenv install of the built wheel, with no OpenSlide on the library
  path, must import and run the CLI.
- `python -m build` + `twine check` on both sdist and wheel.
- Full test suite green.
- Workflow YAML parsed and linted before commit.
