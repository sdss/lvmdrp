# LVM DRP Decision Log

This file records workflow decisions and repository changes.
Meeting decisions and implementation details are listed separately.

## 2026-10-29 — Development workflow

The LVM-Data meeting established the following workflow:

- Use task- or issue-specific branches. Multiple developers may
  collaborate on the same branch.
- Rename the existing `master` branch to `dev`, preserving its history.
- Use `dev` for integration and regression testing.
- Create a stable `main` branch from DRP tag `1.2.1`.
- Review, test, and document changes before promoting them to `main`.
- Maintain a chronological record of decisions in the repository.

The initial stable baseline is `1.2.1`; subsequent development remains
available in `dev` for evaluation before promotion.

## 2026-11-04 — Repository reorganization

### Completed changes

- Created and verified a full Git backup before reorganizing branches.
- Renamed `master` to `dev`.
- Created `main` from tag `1.2.1`, at commit
  `e3517d340cf89db1a494cbc2d8fb04275c39dffe`.
- Kept `dev` as the default branch.
- Configured separate protection rulesets for `main` and `dev`.
- Verified that all three existing open pull requests target `dev`.

### Initial protection settings

These settings implement the workflow and are recorded separately
from the meeting decisions:

- Changes to `main` and `dev` require a pull request.
- Pull requests require one approving review.
- New reviewable commits dismiss previous approvals.
- Review conversations must be resolved before merging.
- Branch deletion and force pushes are restricted.
- No bypass entries were added.
- Required status checks are not yet configured.

Automated checks remain enabled. Their results must be reviewed
before merging, even while they are not enforced by the rulesets.

### CI follow-up

The initial CI run on `main` reported a lint failure:

- Check: `Build and Test / build (ubuntu-latest, true, 3.12) (push)`
- Reported findings: 1,085 lint errors.
- Exit code: 1.

This result does not, by itself, establish a scientific regression.
The lint configuration, tool versions, and remaining build and test
results need investigation before selecting required status checks.

### Pending work

- Resolve or characterize the CI failures.
- Define and enable appropriate required status checks.
- Document regression tests and criteria for stable promotion.
- Keep this log updated as further decisions are adopted.
