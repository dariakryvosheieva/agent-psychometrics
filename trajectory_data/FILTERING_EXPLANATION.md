# Trajectory Filtering Explanation

This document explains what the trajectory filter (`llm_judge/trajectory_filter.py`) removes from SWE-bench agent trajectories.

## Goal

Keep only **actual codebase interactions** - the behavioral choices agents make when exploring and modifying code. Filter out everything else.

---

## What Gets FILTERED OUT

### 1. System Prompts & Setup Context

These establish the agent's environment but aren't actions:

```
"Setting: You are an autonomous programmer..."
"You are OpenHands agent, a helpful AI assistant..."
"Instructions: ..."
"Environment: ..."
"Working directly in the command line..."
```

**Why filtered:** This is boilerplate that's identical across all tasks for an agent - not behavioral signal.

---

### 2. Issue Descriptions & Problem Statements

The bug report/task being solved - agents receive this as input:

```
"We're currently solving the following issue..."
"We are currently solving the following issue..."
"Consider the following issue..."
"The reported issue is:"
"<ISSUE>...</ISSUE>"
"<issue_description>...</issue_description>"
"I've uploaded a python code repository..."
"Solve this issue: ..."
"--- BEGIN ISSUE ---"
```

**Why filtered:** This is the task input, not agent behavior. It's the same across all agents for a given task.

---

### 3. Planning & Future-Tense Thinking

Agent reasoning about what it *will* do (not what it *did*):

```
"Let me think about..."
"Let's first..."
"I'll start by..."
"I will help you..."
"I need to..."
"I should..."
"I want to..."
"We need to..."
"We should..."
"I'm going to..."
"My plan is..."
"First, I will..."
"Now, let me..."
"To fix this, I will..."
"To solve this..."
```

**Why filtered:** Planning text reveals reasoning style but not actual codebase interaction. The actual commands/edits that result from planning ARE kept.

---

### 4. Commentary & Analysis (Post-Action)

Agent commentary about observations or results:

```
"Great! Let's analyze the results..."
"Perfect!"
"Excellent!"
"Good!"
"We can see that..."
"We can confirm..."
"It looks like..."
"It seems..."
"It appears..."
"This shows..."
"This indicates..."
"This confirms..."
"The output shows..."
"The result is..."
"The error indicates..."
"The code works..."
"The changes have been..."
"Based on the analysis..."
"Looking at the output..."
"As we can see..."
"I see that..."
"I notice that..."
"I found that..."
```

**Why filtered:** This is interpretation of results, not the action of obtaining them. The actual command outputs ARE kept.

---

### 5. Summary Messages

Wrap-up explanations of what was done:

```
"To summarize..."
"In summary..."
"Summary:"
"The issue has been resolved..."
"The problem has been fixed..."
"This solution addresses..."
"This fix ensures..."
```

**Why filtered:** Summaries describe completed work in prose - the actual edits/patches ARE kept.

---

### 6. Environment Reminders

Mid-conversation system messages:

```
"Environment reminder: You have 15 turns left"
"[System] Remaining budget: ..."
```

**Why filtered:** Infrastructure signals, not agent behavior.

---

## What Gets KEPT

| Category | Examples | Why Kept |
|----------|----------|----------|
| **Shell commands** | `edit file.py`, `python test.py`, `grep -r "error"`, `cat file.py` | Actual codebase exploration |
| **Command outputs** | Tracebacks, test results, file listings, grep matches | Results of agent actions |
| **File contents** | `[File: /path/to/file.py (100 lines)]...` | What agent chose to view |
| **Code modifications** | `diff --git`, edit blocks, `str_replace` | Actual changes made |
| **Tool call JSONs** | `{"action": "read_file", "args": {"path": "..."}}` | Structured action representations |
| **Patches** | `[repair]\nPatch:\ndiff --git a/file.py...` | Final code changes |
| **Past-tense descriptions** | `"I edited the file..."`, `"Successfully modified..."`, `"Opened file..."` | Completed action narration (Amazon-Q style) |

---

## Reduction Rates by Agent Type

| Reduction % | Agent Style | What's Happening |
|-------------|-------------|------------------|
| **0-10%** | Patch-only (patchpilot, moatless) | Almost all content is patches/tool calls - minimal filtering |
| **30-40%** | Action-heavy (bloop, artemis) | More tool outputs than planning |
| **45-55%** | Standard (SWE-agent, OpenHands) | Balanced mix of planning (filtered) and actions (kept) |
| **60-75%** | Verbose (lingma, wandb) | More extensive planning/thinking |
| **85-90%** | Planning-focused (nfactorial, zai_glm) | Mostly "Change plan:" and system setup |
| **95%+** | Log dumps (JoyCode) | Embedded conversation transcripts with lots of setup |

---

## Key Principle

**Kept:** What the agent *did* to the codebase (commands, views, edits, tests)

**Filtered:** What the agent *thought* or *said* about doing it (planning, analysis, commentary)

---

## Usage

```bash
# Filter all unified trajectories
python llm_judge/trajectory_filter.py --all_unified --output_dir trajectory_data/filtered_unified

# Filter single trajectory
python llm_judge/trajectory_filter.py --traj path/to/file.json --output filtered.json

# Preview without saving
python llm_judge/trajectory_filter.py --traj path/to/file.json
```

## Related Files

- `llm_judge/trajectory_filter.py` - Main filtering script
- `llm_judge/trajectory_converter.py` - Converts raw trajectories to unified format
- `trajectory_data/filter_verification.md` - Verification status for each agent
- `trajectory_data/filtered_unified/_filter_summary.json` - Filtering statistics
