# Project Workflow

## Guiding Principles

1. **The Plan is the Source of Truth:** All work must be tracked in `plan.md`
2. **The Tech Stack is Deliberate:** Changes to the tech stack must be documented in `tech-stack.md` *before* implementation
3. **Test-Driven Development:** Write unit tests before implementing functionality
4. **High Code Coverage:** Aim for >80% code coverage for all Go modules. Lower coverage is acceptable for C/C++/Metal native code where testing is more complex.
5. **User Experience First:** Every decision should prioritize user experience
6. **Non-Interactive & CI-Aware:** Prefer non-interactive commands. Use `CI=true` for watch-mode tools (tests, linters) to ensure single execution.

## Task Workflow

All tasks follow a strict lifecycle:

### Standard Task Workflow

1. **Select Task:** Choose the next available task from `plan.md` in sequential order

2. **Mark In Progress:** Before beginning work, edit `plan.md` and change the task from `[ ]` to `[~]`

3. **Write Failing Tests (Red Phase):**
   - Create a new test file for the feature or bug fix.
   - Write one or more unit tests that clearly define the expected behavior and acceptance criteria for the task.
   - **CRITICAL:** Run the tests and confirm that they fail as expected. This is the "Red" phase of TDD. Do not proceed until you have failing tests.

4. **Implement to Pass Tests (Green Phase):**
   - Write the minimum amount of application code necessary to make the failing tests pass.
   - Run the test suite again and confirm that all tests now pass. This is the "Green" phase.

5. **Refactor (Optional but Recommended):**
   - With the safety of passing tests, refactor the implementation code and the test code to improve clarity, remove duplication, and enhance performance without changing the external behavior.
   - Rerun tests to ensure they still pass after refactoring.

6. **Verify Coverage:** Run coverage reports using the project's chosen tools. For example, in a Go project, this might look like:
   ```bash
   go test -coverprofile=coverage.out ./...
   go tool cover -func=coverage.out
   ```
   Target: >80% coverage for Go code.

7. **Document Deviations:** If implementation differs from tech stack:
   - **STOP** implementation
   - Update `tech-stack.md` with new design
   - Add dated note explaining the change
   - Resume implementation

8. **Update Plan Progress:**
    - Find the line for the in-progress task in `plan.md`, update its status from `[~]` to `[x]`.
    - If this task completes a phase, proceed to the Phase Completion Protocol. Otherwise, continue to the next task in the phase.

### Phase Completion Verification and Checkpointing Protocol

**Trigger:** This protocol is executed immediately after all tasks in a phase in `plan.md` are completed.

1.  **Announce Protocol Start:** Inform the user that the phase is complete and the verification and checkpointing protocol has begun.

2.  **Ensure Test Coverage for Phase Changes:**
    -   **Step 2.1: Determine Phase Scope:** To identify the files changed in this phase, you must first find the starting point. Read `plan.md` to find the Git commit SHA of the *previous* phase's checkpoint. If no previous checkpoint exists, the scope is all changes since the first commit.
    -   **Step 2.2: List Changed Files:** Execute `git diff --name-only <previous_checkpoint_sha> HEAD` (or `git status` if no previous commit) to get a list of all files modified during this phase.
    -   **Step 2.3: Verify and Create Tests:** For each code file, verify a corresponding test file exists.

3.  **Execute Automated Tests with Proactive Debugging:**
    -   Execute the automated test suite (e.g., `go test ./...`).
    -   If tests fail, you must inform the user and begin debugging.

4.  **Propose a Detailed, Actionable Manual Verification Plan:**
    -   Analyze `product.md`, `product-guidelines.md`, and `plan.md` to determine the user-facing goals of the completed phase.
    -   Generate a step-by-step plan for manual verification.

5.  **Await Explicit User Feedback:**
    -   Ask the user for confirmation: "**Does this meet your expectations? Please confirm with yes or provide feedback on what needs to be changed.**"
    -   **PAUSE** and await the user's response.

6.  **Commit Phase Changes:**
    -   Stage all code and plan changes.
    -   Perform the commit with a clear and concise message (e.g., `feat(auth): Complete Phase 1 - User Authentication Flow`).

7.  **Attach Auditable Verification Report using Git Notes:**
    -   **Step 7.1: Get Commit Hash:** Obtain the hash of the phase completion commit.
    -   **Step 7.2: Draft Note Content:** Create a detailed summary of the phase, including tasks completed, test results, and verification steps.
    -   **Step 7.3: Attach Note:** Use the `git notes` command to attach the report to the commit.
     ```bash
     git notes add -m "<note content>" <commit_hash>
     ```

8.  **Get and Record Phase Checkpoint SHA:**
    -   **Step 8.1: Get Commit Hash:** Obtain the hash of the checkpoint commit.
    -   **Step 8.2: Update Plan:** Read `plan.md`, find the heading for the completed phase, and append the first 7 characters of the commit hash in the format `[checkpoint: <sha>]`.

9. **Commit Plan Update:**
    - Stage the modified `plan.md` file.
    - Commit this change: `conductor(plan): Mark phase '<PHASE NAME>' as complete`.

10.  **Announce Completion:** Inform the user that the phase is complete and the checkpoint has been created.

## Quality Gates

Before marking any phase complete, verify:

- [ ] All tests pass
- [ ] Code coverage meets requirements (>80% for Go)
- [ ] Code follows project's code style guidelines
- [ ] All public functions/methods are documented
- [ ] Type safety is enforced
- [ ] No linting or static analysis errors
- [ ] Documentation updated if needed

## Development Commands

### Setup
```bash
go mod tidy
```

### Daily Development
```bash
go test ./...
go fmt ./...
```

### Before Committing Phase
```bash
make check # if applicable, or go test ./... && go vet ./...
```

## Commit Guidelines

### Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Maintenance tasks

## Definition of Done

A phase is complete when:

1. All code implemented to specification
2. Unit tests written and passing
3. Code coverage meets project requirements
4. Documentation complete (if applicable)
5. Changes committed with proper message
6. Git note with phase summary attached to the completion commit
7. Phase marked as complete in `plan.md` with checkpoint SHA
