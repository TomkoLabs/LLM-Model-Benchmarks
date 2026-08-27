# Coding evaluation task

Harden this archive extraction library against unsafe or malformed member
paths. Complete the implementation rather than merely describing it.

## Required public behavior

Preserve these public APIs:

    normalize_archive_member(raw: str, *, max_length: int = 255) -> str

    destination_for(
        root,
        member_name,
        *,
        policy: ExtractionPolicy | None = None
    ) -> Path

    extract_member(
        root,
        member_name,
        data,
        *,
        policy: ExtractionPolicy | None = None
    ) -> Path

## Path normalization requirements

1. `raw` must be a string; otherwise raise TypeError.
2. `max_length` must be a positive integer; otherwise raise ValueError.
3. Every percent sign must be followed by exactly two hexadecimal digits.
4. Percent-decode exactly once.
5. Convert backslashes to forward slashes.
6. Reject literal and percent-decoded NUL characters.
7. Reject POSIX absolute paths.
8. Reject UNC paths.
9. Reject Windows drive-qualified paths, including both `C:/x` and `C:x`.
10. Collapse repeated separators.
11. Remove `.` path components.
12. Reject every `..` component after decoding and normalization.
13. Reject paths with no remaining components.
14. Reject a normalized path longer than `max_length`.
15. Preserve case, spaces, and ordinary valid component content.

Use ValueError for invalid-path conditions.

## Extraction requirements

16. `destination_for` must pass `policy.max_member_length` to the
    normalization function.
17. The returned destination must remain under the resolved extraction root.
18. Existing symlinks inside the root must not permit writing outside it.
19. Preserve the existing overwrite policy.
20. `extract_member` must write only to the validated destination.
21. Existing valid behavior must continue to work.

## Working process

Before editing:

1. Inspect the repository structure and every relevant source and test file.
2. Restate the outcome as explicit acceptance criteria.
3. Create a checklist of every required change.
4. Identify every affected file and call site.
5. Do not begin editing until that inspection is complete.

During implementation:

6. Keep the checklist current.
7. Update or add tests for all changed behavior.
8. Preserve behavior outside the requested scope.
9. Do not silently omit difficult or ambiguous requirements.
10. Do not inspect sibling evaluation directories.

Before claiming completion:

11. Review the complete diff against every acceptance criterion.
12. Run the complete public test suite.
13. Run any other appropriate validation.
14. Explicitly list anything not verified.
15. Do not claim completion while a checklist item remains unresolved.

Create `MODEL_EVAL_REPORT.md` containing:

- Restated acceptance criteria
- Files inspected
- Files changed
- Commands and tests run
- Results
- Unresolved concerns
- Final pass/fail checklist
