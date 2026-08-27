# Non-scored Hermes harness smoke

This is a bounded infrastructure smoke task, not a benchmark score.

1. Inspect the repository and its public test.
2. Change `smoke_project.py` so `STATUS` is exactly `"READY"`.
3. Create an untracked text file named `SMOKE_NOTES.txt` containing exactly:

       Hermes smoke completed.

   Include the trailing newline.
4. Run `python3 -m unittest discover -s tests -v` and make the deterministic
   public tests pass.
5. Inspect `git status --short`, `git diff`, and `git diff --check`.
6. Report what changed and the test result accurately.

Do not modify tests, add package-marker files, or change unrelated files. Do
not commit the candidate changes.
