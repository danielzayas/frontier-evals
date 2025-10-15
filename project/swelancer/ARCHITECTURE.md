# SWE-Lancer Architecture Summary

This repo contains the dataset and code for the paper ["SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?"](https://arxiv.org/pdf/2502.12115). As of 2025/07/17, this repo contains 463 tasks (a subset of the original 237 problems mentioned in the paper) that were adjusted and verified to run successfully offline.

**Github repository:** https://github.com/openai/frontier-evals/tree/main/project/swelancer  
**Dataset:** https://github.com/openai/frontier-evals/blob/main/project/swelancer/all_swelancer_tasks.csv

## Overview

Uses 463 real freelance tasks posted by Expensify to Upwork.com. Measures performance in monetary value (total reward across all tasks). Each Upwork task corresponds to exactly one GitHub issue posted by the Expensify GitHub organization:

1. **198 `ic_swe` tasks** (e.g., [Issue 29916](https://github.com/Expensify/App/issues/29916) → $500 job → [example fix](https://github.com/Expensify/App/pull/29969)). Candidate solutions are evaluated by running automated Playwright tests that verify the bug is fixed.

2. **265 `swe_manager` tasks**, which are multiple-choice questions where the AI must select the correct implementation proposal from several options (e.g., [Issue 15193](https://github.com/Expensify/App/issues/15193) → $4000 job → [proposal](https://github.com/Expensify/App/issues/15193#issuecomment-1503309296) → fix via [PR-513](https://github.com/Expensify/App/pull/513) and [PR-16437](https://github.com/Expensify/App/pull/16437)). Binary scoring: 1.0 if the correct proposal is selected, 0.0 otherwise. The AI does not need to implement the fix, just identify the best proposal.

> **Note:** In real-world Upwork, freelancers must both write and implement proposals. In this benchmark, `ic_swe` tasks require implementation but not proposal writing, while `swe_manager` tasks require proposal selection but not implementation.

## Example Tasks

Real freelance tasks posted by Expensify to Upwork.com:

1. **`ic_swe` task `29916_609`**: "Update all selectors to use a new format for selected participant" ([Issue 29916](https://github.com/Expensify/App/issues/29916)) valued at $500. Candidate solutions are evaluated by running specific automated tests.

2. **`ic_swe` task `28565_1001`**: "Menu Item shows pointer cursor when not clickable" ([Issue 28565](https://github.com/Expensify/App/issues/28565)) valued at $500. Tests verify the cursor style changes correctly.

3. **`swe_manager` task `15193-manager-0`**: "Bug causes text to be incorrectly bolded during chat" ([Issue 15193](https://github.com/Expensify/App/issues/15193)) valued at $4000. Pick the golden proposal out of four options.

4. **`swe_manager` task `28565-manager-0`**: "Show the default cursor when the menu item is not clickable" ([Issue 28565](https://github.com/Expensify/App/issues/28565)) valued at $500. Pick the golden proposal from multiple options.

## Example Evaluation Command

The evaluation is designed for x86 architecture. It may not work on ARM Macbooks with x86 emulation (setup phase can timeout). Example command:

```bash
cd /path/to/swelancer && uv run python swelancer/run_swelancer.py \
  swelancer.split=diamond \
  swelancer.task_type=ic_swe \
  swelancer.taskset="['29916_609']" \
  swelancer.solver=swelancer.solvers.dummy.solver:DummySolver \
  swelancer.solver.test_user_tool=False \
  swelancer.solver.apply_gold_solution=False \
  swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  swelancer.solver.computer_runtime.env.pull_from_registry=True \
  swelancer.docker_image_prefix=swelancer/swelancer_x86 \
  swelancer.docker_image_tag=releasev1 \
  swelancer.use_single_image=True \
  swelancer.disable_internet=False \
  runner.concurrency=1 \
  runner.experimental_use_multiprocessing=False \
  runner.enable_slackbot=False \
  runner.recorder=nanoeval.recorder:dummy_recorder \
  runner.max_retries=0
```

## Repository Architecture for Orchestrating Evaluations

The system orchestrates the full evaluation lifecycle: setting up buggy environments, obtaining candidate solutions from AI systems, and grading fixes.

### Execution Flow for Task `28565_1001`

**Step 1: Docker Container Setup**
- Docker container starts with Expensify app at specific commit (from `commit_id.txt`)
- Applies `bug_reintroduce.patch` to recreate the bug
- Sets up test environment (Xvfb display server, Pusher-fake for websockets, NGINX proxy, etc.)
- In `eval.py`, `SWELancerTask._setup()` (line 120-172)

**Step 2: AI Agent Attempts Fix**
- AI receives problem description from `issue_data.json`
- Explores codebase via Python code execution
- Generates and applies fix
- In `solver.py`, `SimpleAgentSolver.run()` (line 204-356)

**Step 3: Grade the Solution**
- Runs `test.py` to verify if bug is fixed
- For `ic_swe`: Runs automated Playwright tests up to 3 times (configurable via `n_test_runs`)
- For `swe_manager`: Checks if selected proposal ID matches correct answer
- Records success/failure, patch, and metrics (tokens, cost, etc.)
- In `eval.py`, `SWELancerTask.grade()` (line 174-203) → `_grade_swe_ic()` (line 255-375) or `_grade_swe_manager()` (line 205-253)

**Step 4: Report Results**
- Aggregates results across all tasks
- Calculates accuracy, total earnings, token usage, etc.
- Outputs final evaluation report and CSV
- In `eval.py`, `get_full_summary()` (line 580-646)

### Key Components

#### 1. Entry Point & Runner
- **`/project/swelancer/swelancer/run_swelancer.py`**: Main entry point
  - Initializes evaluation with configuration (uses `chz` for config management)
  - Calls `nanoeval.run()` to execute the evaluation
  - Sets up logging infrastructure

#### 2. Evaluation Orchestrator
- **`/project/swelancer/swelancer/eval.py`**: Core evaluation logic
  - `SWELancerEval` class (line 379): Main evaluation configuration, inherits from `PythonCodingEval`
  - `SWELancerTask` class (line 87): Task representation with all metadata (price, variant, commit, etc.)
  - `get_instances()` method (line 454): Loads tasks from CSV and creates task instances
  - `evaluate()` method (line 558): Orchestrates the evaluation flow for a single task
  - `grade()` method (line 174): Runs tests to verify solutions
  - `get_full_summary()` method (line 580): Aggregates and reports final results

#### 3. Task Data & Configuration
Each task has its own directory in `/project/swelancer/issues/[task_id]/`:
- **`issue_data.json`**: Problem statement (title, description, price, etc.)
- **`test.py`**: Automated Playwright test suite
- **`commit_id.txt`**: Git commit SHA with the bug
- **`bug_reintroduce.patch`**: Patch to recreate the bug (for `ic_swe` tasks)
- **`proposals.json`**: Candidate proposals (for `swe_manager` tasks only)

#### 4. Solvers (AI Agents)
- **`/project/swelancer/swelancer/solvers/swelancer_agent/solver.py`**: Real AI solver
  - `SimpleAgentSolver` class: Interacts with LLMs (GPT-4, Claude, etc. via OpenAI API or OpenRouter)
  - `run()` method (line 204): Main solver loop (up to 40 turns)
  - Extracts and executes Python code blocks from model responses
  - Supports `<user-tool>` command to trigger user simulation (runs Playwright tests and provides trace output)
  
- **`/project/swelancer/swelancer/solvers/dummy/solver.py`**: Test solver
  - Used for testing infrastructure without AI
  - Can optionally apply gold solution for validation

#### 5. Prompts
- **`/project/swelancer/swelancer/prompts.py`**: Task prompts for AI systems
  - `construct_ic_task_prompt()`: Prompts for `ic_swe` tasks (includes instructions for using `<user-tool>`)
  - `construct_manager_task_prompt()`: Prompts for `swe_manager` tasks (includes proposal selection instructions)

#### 6. Docker & Runtime Environment
- **`/project/swelancer/Dockerfile_x86_base`**: Base Docker image with all dependencies
  - Ubuntu 22.04 with Python 3.12, Node.js, Ruby, Conda
  - Playwright for browser automation
  - Xvfb for headless display
  - Pusher-fake for websocket mocking
  - NGINX for proxying
  
- **`/project/swelancer/Dockerfile_x86_monolith`**: Single monolith image (required for `swe_manager` tasks)
  
- **`/project/swelancer/Dockerfile_x86_per_task`**: Per-task images (optional, faster setup for `ic_swe` tasks)

- **`/project/swelancer/runtime_scripts/run.sh`**: Container initialization script
  - Sets up display server (Xvfb), window manager (Fluxbox), VNC server
  - Starts Pusher-fake service and NGINX proxy
  - Configures `/etc/hosts` for local DNS overrides
  - Runs Ansible playbooks for Expensify and mitmproxy setup
  - Creates `user-tool` alias for running tests

- **`/project/swelancer/runtime_utils/`**: Utility scripts for runtime
  - `trace_cleaner.py`: Cleans Playwright traces to remove sensitive data
  - `online_guard.py`: Prevents network access during evaluation

## Reusable Components for New Evaluations

The SWE-Lancer codebase is built on top of several highly reusable frameworks in `/project/common/`:

### 1. **nanoeval** (`/project/common/nanoeval/`)
**What it is:** A lightweight, high-performance evaluation framework for running tasks in parallel.

**Key abstractions:**
- `Eval`: Base class for defining evaluations (loads tasks, runs solver, records results)
- `Task`: Represents a single unit of work with `question_id` and `attempt_id`
- `Solver`: Strategy for solving tasks (e.g., prompting a model)
- `EvalSpec`: Configuration for running an eval (concurrency, retries, recording, etc.)
- `RolloutSystemError`: Exception type for retryable system errors (vs model errors)

**Why it's reusable:**
- Minimal indirection: Implement an eval in ~100 lines
- Separation of concerns: Data loading, completions, and execution are separate
- Fast iteration: Imports in <1 second, testable without live LLM backend
- High performance: Uses asyncio + sqlite for task queue, supports multiprocessing
- Framework-agnostic: Can be used for any evaluation (MCQ, coding, multi-turn dialogue, etc.)

**Example usage:** See `nanoeval/examples/gpqa_simple.py` for a minimal GPQA implementation in ~70 lines.

**Reusability verdict:** ⭐⭐⭐⭐⭐ Extremely reusable. This is the core orchestration framework.

### 2. **nanoeval_alcatraz** (`/project/common/nanoeval_alcatraz/`)
**What it is:** Integration layer between nanoeval and alcatraz, providing a `ComputerRuntime` implementation.

**Key components:**
- `AlcatrazComputerRuntime`: Main runtime that creates Docker containers for tasks
- `AlcatrazComputerInterface`: Provides methods for executing code, running shell commands, uploading/downloading files, and disabling internet
- `task_to_alcatraz_config()`: Converts nanoeval `ComputerTask` to alcatraz cluster config

**Why it's reusable:**
- Provides a clean abstraction for running code in Docker containers
- Handles Jupyter kernel management for interactive code execution
- Supports file upload/download, shell commands, and network isolation
- Can be used with any nanoeval-based evaluation that needs code execution

**Reusability verdict:** ⭐⭐⭐⭐⭐ Highly reusable for any evaluation requiring sandboxed code execution.

### 3. **alcatraz** (`/project/common/alcatraz/`)
**What it is:** Low-level Docker container orchestration library for code execution.

**Key components:**
- `BaseAlcatrazCluster`: Abstract interface for container clusters
- `LocalConfig`: Configuration for running containers on local Docker
- Shell command execution with streaming output
- Network isolation via iptables (Linux only)
- Jupyter kernel management for interactive Python execution

**Why it's reusable:**
- Provides low-level primitives for container orchestration
- Can be used independently of nanoeval
- Extensible to support different cluster backends (e.g., Kubernetes, cloud providers)

**Reusability verdict:** ⭐⭐⭐⭐ Reusable for any project needing Docker-based code execution.

### 4. SWE-Lancer-Specific Components (Less Reusable)

The following components are specific to SWE-Lancer but could be adapted:

- **`/project/swelancer/swelancer/eval.py`**: Task loading from CSV, Playwright test execution, manager proposal grading. Could be adapted for similar bug-fixing benchmarks.

- **`/project/swelancer/swelancer/solvers/swelancer_agent/solver.py`**: Multi-turn agent with Python code execution and `<user-tool>` support. Could be adapted for other coding benchmarks.

- **`/project/swelancer/Dockerfile_x86_base`**: Expensify-specific setup (Pusher-fake, NGINX, etc.). Could be used as a template for other web app evaluations.

- **`/project/swelancer/runtime_scripts/`**: Ansible playbooks and shell scripts for Expensify setup. Specific to this benchmark.

## Summary: Building New Evaluations

**To build a new evaluation on top of this infrastructure:**

1. **Use nanoeval** for task orchestration:
   - Define your `Task` class (inherit from `nanoeval.Task`)
   - Define your `Eval` class (inherit from `nanoeval.Eval` or `nanoeval.solvers.computer_tasks.solver.PythonCodingEval`)
   - Implement `get_tasks()`, `evaluate()`, and `get_full_summary()`

2. **Use nanoeval_alcatraz + alcatraz** for code execution:
   - Define Docker images for your tasks
   - Use `AlcatrazComputerRuntime` to run tasks in containers
   - Use `AlcatrazComputerInterface` for executing code, running commands, etc.

3. **Define solvers**:
   - Implement `PythonCodingSolver` or define your own solver interface
   - Use the `ComputerInterface` to interact with the container

4. **Optional: Reuse SWE-Lancer components**:
   - Task prompting patterns from `prompts.py`
   - Multi-turn agent loop from `SimpleAgentSolver`
   - Dockerfile structure from `Dockerfile_x86_base`
   - Test execution patterns from `eval.py` grading methods

**Key advantages of this architecture:**
- **Parallelism**: nanoeval automatically parallelizes task execution
- **Reliability**: Automatic retries for `RolloutSystemError` exceptions
- **Isolation**: Each task runs in its own Docker container
- **Observability**: Structured logging, task-level logs, CSV result exports
- **Configurability**: All parameters configurable via `chz` CLI or Python

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       run_swelancer.py                          │
│                   (Entry point, config parsing)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      nanoeval.run()                             │
│              (Task queue, parallel execution)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SWELancerEval.evaluate()                     │
│           (Load task, call solver, grade result)                │
└────────┬────────────────────────────────────────────────────────┘
         │
         ├──────► get_instances() ──► Load tasks from CSV
         │
         ├──────► Solver.run() ──────► SimpleAgentSolver or DummySolver
         │                              ├─► Multi-turn agent loop
         │                              ├─► Python code execution
         │                              └─► <user-tool> support
         │
         └──────► Task.grade() ───────► Run tests, compute score
                                         ├─► For ic_swe: Run Playwright tests
                                         └─► For swe_manager: Check proposal ID
```

```
Docker Container Lifecycle:
┌────────────────────────────────────────────────────────────────┐
│  1. AlcatrazComputerRuntime.run()                              │
│     └─► Start Docker container with task-specific image       │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  2. Task._setup()                                              │
│     ├─► Apply bug_reintroduce.patch                           │
│     ├─► Zip tests directory (password-protected)              │
│     ├─► Initialize git repo                                   │
│     └─► Disable internet (iptables rules)                     │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  3. Solver.run()                                               │
│     └─► Agent explores codebase and attempts fix              │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  4. Task.grade()                                               │
│     ├─► Unzip tests directory                                 │
│     ├─► Run Playwright tests (ic_swe) or check JSON (manager) │
│     └─► Record results (score, patch, tokens, etc.)           │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  5. Container cleanup                                          │
│     └─► Stop and remove Docker container                      │
└────────────────────────────────────────────────────────────────┘
```
