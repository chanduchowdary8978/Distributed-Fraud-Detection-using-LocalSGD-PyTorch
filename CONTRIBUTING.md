# Contributing

This project is designed to be implemented incrementally, across multiple
phases, potentially by different AI coding assistants (Claude, GPT, Gemini,
etc.) working at different times. To keep the codebase coherent and
predictable across contributors, every contribution must follow the rules
below.

## Engineering Rules

- **Never rename public interfaces.** Class names, function names, and
  module paths documented as part of the public interface are stable.
  Renaming breaks contracts other phases depend on.
- **Never move files.** The repository structure established in Phase 0 is
  fixed. New functionality goes into existing files or new files within
  the existing directories, not into a reorganized layout.
- **Never duplicate functionality.** Before adding a new function or class,
  check whether an existing module already owns that responsibility.
- **Reuse existing modules.** Prefer importing and extending an existing
  module over writing a parallel implementation.
- **Keep one responsibility per file.** Each file should do one thing, in
  line with its documented purpose.
- **Stop if a dependency is missing.** Do not silently work around a
  missing dependency by vendoring code or switching libraries. Flag it
  instead.
- **Use type hints.** All new functions and methods should be fully typed.
- **Use docstrings.** Every module, class, and public function should have
  a docstring describing what it does.
- **Keep architecture stable.** Do not introduce new architectural
  patterns (e.g. new communication protocols, new storage layers) without
  explicit sign-off; implement within the existing design.
- **Never redesign previous phases.** Once a phase is complete, its public
  interfaces and structure are considered final unless explicitly revised
  as part of a dedicated task.
- **Every new feature belongs to its own phase.** Do not bundle unrelated
  functionality into a single change. Keep phases scoped and reviewable.
