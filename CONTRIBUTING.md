# Contributing to OpenLanguageModel

Thanks for helping make OpenLanguageModel better. OLM is built for people who
want to learn, inspect, and modify language models without losing the ordinary
PyTorch underneath, so contributions that make the code clearer, safer, better
tested, or easier to teach from are especially welcome.

## Ways To Contribute

- Fix docs, examples, typos, broken links, or confusing explanations.
- Add focused tests for model families, training behavior, datasets, or public
  APIs.
- Improve generated API docstrings, shapes, return types, and examples.
- Report bugs with a small reproduction.
- Propose model, training, or documentation improvements through an issue before
  opening a large implementation PR.

## Development Setup

Use Python 3.10, 3.11, or 3.12.

Unless you have write access to this repository, fork it first — you will not be
able to push a branch to the official repository, and the pull request flow below
assumes you are pushing to your own fork.

```bash
# Fork https://github.com/openlanguagemodel/openlanguagemodel on GitHub, then:
git clone https://github.com/<your-username>/openlanguagemodel.git
cd openlanguagemodel
git remote add upstream https://github.com/openlanguagemodel/openlanguagemodel.git
git fetch upstream

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

That leaves `origin` pointing at your fork (where you push) and `upstream` at the
official repository (where you pull from and open pull requests against).

If you do have write access, you can clone the official repository directly and
skip the `upstream` remote.

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

Run the Python checks:

```bash
python -m compileall -q src tests scripts
pytest -q tests
```

For website work:

```bash
cd website
npm ci
npm run lint
npm run build
```

## Pull Request Flow

1. Sync with `dev` and branch from it:

   ```bash
   git fetch upstream
   git checkout -b my-change upstream/dev
   ```

2. Keep the change focused. Small PRs are easier to review and merge.
3. Add or update tests when behavior changes.
4. Update docs when public APIs, examples, installation, or training behavior
   changes.
5. Push the branch to your fork and open a pull request into `dev`:

   ```bash
   git push -u origin my-change
   ```

6. Fill out the pull request template and link related issues.

`main` is reserved for stable release rollouts. Maintainers merge `dev` into
`main` when preparing a versioned release.

For branch names, use a readable prefix such as:

```bash
git checkout -b tavish/fix-tokenizer-streaming
git checkout -b username/docs-first-model
```

## Code Style

- Follow the style already present in the surrounding file.
- Prefer explicit, readable PyTorch over hidden framework behavior.
- Keep abstractions small and only add them when they remove real repetition or
  clarify a public path.
- Public classes and methods should have useful docstrings, including expected
  tensor shapes and return values where relevant.
- Avoid unrelated refactors inside bug-fix PRs.

## Tests

Please run the relevant subset locally before opening a PR. For broad or public
API changes, run the full suite:

```bash
pytest -q tests
```

For model-family changes, include at least a constructor/config check and a tiny
forward/backward or one-batch training smoke test when practical.

## Documentation

Docs live in `docs/` and are rendered into the website. If you add or change a
public component, update the relevant guide or API docstring. If you add a
notebook, also update `docs/colab-notebooks.md`.

## Adding A Model Family

Before adding a new model family, open an issue describing:

- the model family and reference source
- the architecture pieces needed
- which parts are exact, approximate, or intentionally omitted
- the tests you plan to add

Model implementations should be readable worked examples assembled from OLM's
public components, not hidden configuration blobs.

## Community Expectations

All contributors are expected to follow the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Please report security issues using
[`SECURITY.md`](SECURITY.md), not public issues.
