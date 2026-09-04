# Repository Workflow

This repository uses GitHub issues and pull requests for every change.

## Required Change Flow

For any code, test, documentation, configuration, or repository metadata change:

1. Sync the local `main` branch with `origin/main`.
2. Create or update a GitHub issue describing the change.
3. Create a dedicated feature branch from the latest `main`.
4. Commit only the files related to that issue.
5. Open a GitHub pull request to `main`.
6. Verify tests/checks that are available for the change.
7. Merge the pull request into `main`.
8. Fetch `origin/main` and update local `main` so GitHub and local `main` stay synchronized.

Do not commit directly to `main`.

## Dirty Worktree

If unrelated local changes already exist, leave them untouched and keep them out of the issue, commit, and pull request.
