# Releasing DTGeoSOM

Releases are automatic (`.github/workflows/publish.yml`): **pushing a new version number to `main` publishes it.**
When the version in `src/mt/dtgeosom/__init__.py` has no `vX.Y.Z` tag yet, the workflow builds the sdist and wheel,
tests the wheel in a clean environment, uploads to PyPI with *trusted publishing* (no API token), then tags the commit
and creates the GitHub release, with the `CHANGELOG.md` notes and the built files attached. Pushes that leave the
version alone publish nothing.

## One-time set-up

1. Create the GitHub repository `takatsuka/DTGeoSOM` and push `main`.
2. On PyPI, add a *pending* trusted publisher (Account settings → Publishing → Add a new pending publisher):
   project `dtgeosom`, owner `takatsuka`, repository `DTGeoSOM`, workflow `publish.yml`, environment `pypi`.
   Do the same on TestPyPI with environment `testpypi` if you want rehearsals.
3. In the GitHub repository: Settings → Environments → create `pypi` and `testpypi`
   (add yourself as a required reviewer on `pypi` if you want to approve each upload).

## Each release

1. Set the same version in `src/mt/dtgeosom/__init__.py` (`__version__`) and `CITATION.cff` (`version:`).
2. Rename the *Unreleased* section of `CHANGELOG.md` to `## X.Y.Z — YYYY-MM-DD` (the workflow refuses to release
   a version without its section, and uses it as the release notes).
3. Optional rehearsal: Actions → publish → *Run workflow* builds, tests and uploads to TestPyPI only.
4. Commit and push to `main`.  Watch Actions → publish; the release appears on PyPI and GitHub when it finishes.

A version can be uploaded to PyPI only once: to fix a broken release, bump the version (e.g. 0.1.1).
