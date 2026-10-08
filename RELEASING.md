# Releasing DTGeoSOM

**Publishing a GitHub release publishes the package to PyPI** (`.github/workflows/publish.yml`). Pushing to `main`
never publishes anything; it only runs the tests.

When a release is published, the workflow checks that its tag matches the version numbers, builds the sdist and
wheel, tests the wheel in a clean environment, and uploads to PyPI with *trusted publishing* (no API token).

## One-time set-up

1. Create the GitHub repository `takatsuka/DTGeoSOM` and push `main`.
2. On PyPI, add a *pending* trusted publisher (Account settings → Publishing → Add a new pending publisher):
   project `dtgeosom`, owner `takatsuka`, repository `DTGeoSOM`, workflow `publish.yml`, environment `pypi`.
   Do the same on TestPyPI with environment `testpypi` if you want rehearsals.
3. In the GitHub repository: Settings → Environments → create `pypi` and `testpypi`
   (add yourself as a required reviewer on `pypi` if you want to approve each upload).

## Each release

1. Set the same version in `src/mt/dtgeosom/__init__.py` (`__version__`) and `CITATION.cff` (`version:`), and
   move the *Unreleased* notes in `CHANGELOG.md` under `## X.Y.Z — YYYY-MM-DD`.  Commit and push to `main`.
2. Optional rehearsal: Actions → publish → *Run workflow* builds, tests and uploads to TestPyPI only.
3. On GitHub: Releases → *Draft a new release* → new tag `vX.Y.Z` on `main` (e.g. `v0.1.0`), paste the changelog
   notes, *Publish release*.  Watch Actions → publish; the package appears on PyPI when it finishes.

If the tag does not match both version numbers the workflow stops before uploading: fix the versions, delete the
release and its tag, and publish it again.  A version can be uploaded to PyPI only once: to fix a broken release,
bump the version (e.g. 0.1.1).
