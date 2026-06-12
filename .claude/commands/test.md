---
description: Run the test suite and summarize results by category
allowed-tools: Bash(pytest:*)
---
Run: pytest tests/ -v --tb=short --no-header 2>&1
Then summarize: total passed/failed/errors/skipped + duration. For each failure: test name, one-line cause, the failing assertion. If failures exist, name the most likely root cause and the minimal fix. Do not edit test files unless I ask.