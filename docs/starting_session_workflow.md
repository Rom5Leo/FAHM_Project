# Daily Development Workflow

This project is developed as a professional ML engineering portfolio project.

## Start of every work session

```bash
cd /home/leoadmin/Documents/GitHub/FAHM_Project

git status
git pull --rebase

poetry install
poetry run python --version
poetry run pytest
```

## Working rule

Each session should focus on one small task.

Good task examples:

* Add a simulator function
* Add one feature engineering module
* Add one model baseline
* Improve one evaluation metric
* Document one project decision
* Add or update one test

Avoid mixing many unrelated changes in the same commit.

## Before committing

```bash
poetry run pytest
git status
git diff
```

Review the changed files and make sure the commit contains only intentional work.

## Commit and push

```bash
git add .
git commit -m "Describe the change clearly"
git push
```

## Commit message examples

Good:

```bash
git commit -m "Add initial leak simulation scenario"
git commit -m "Add rolling sensor features"
git commit -m "Add baseline anomaly detection pipeline"
git commit -m "Document simulator design decision"
```

Bad:

```bash
git commit -m "stuff"
git commit -m "update"
git commit -m "changes"
```

## Project principle

Notebooks are for exploration.

Reusable logic belongs in `src/`.

Reasoning belongs in `docs/`.

Tests belong in `tests/`.

GitHub should show steady, professional progress.
