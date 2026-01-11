# Experiment B1: Missed Category Analysis

## Executive Summary

Analysis of 10-task grading results reveals systematic patterns in why certain failure mode categories are never correctly detected. The core issue is **semantic granularity mismatch**: judges detect broad failure patterns but miss the specific distinctions that define each category.

## Categories with 0% Detection Rate

| Category | Ground Truth Count | LLM Detection | Lunette Detection |
|----------|-------------------|---------------|-------------------|
| issue_interference | 2 | 0/2 | 0/2 |
| reproduction_output_misreading | 3 | 0/3 | 0/3 |
| control_flow | 2 | 0/2 | 0/2 |
| verification_weakening | 1 | 0/1 | 0/1 |
| issue_misleading | 1 | 1/1* | 1/1* |

*Note: issue_misleading was predicted but in WRONG cases, showing false positive pattern

---

## Detailed Analysis by Category

### 1. `issue_interference` (0/2 detection)

**Definition**: Fix for X broke Y (something that was working before now fails)

**Ground truth cases**:
- astropy__astropy-13236
- django__django-10999

**What judges predicted instead**:
- LLM: algorithmic_implementation, issue_misleading, specific_case_overfitting
- Lunette: issue_misleading, specific_case_overfitting, referred_code_location

**Why it's being missed**:

The judges analyze "what went wrong with the approach" rather than detecting the specific pattern "fix for feature A broke feature B".

For `astropy-13236`, the judge explanations focus on:
- "Agent misunderstood the PR requirements"
- "Added FutureWarning when should have deleted code"
- "Misled by temporal language"

But they never identify: **"The agent's fix introduced a regression in existing functionality"**

**The semantic gap**: `issue_interference` requires understanding that:
1. Feature A had a bug
2. Agent attempted to fix Feature A
3. The fix caused Feature B (previously working) to break

The current prompts ask judges to find "what went wrong" but not specifically "what broke that was working before."

**Potential fix**: Add explicit questions like:
- "Did the agent's changes cause any PREVIOUSLY WORKING functionality to fail?"
- "Are there tests that passed before the agent's changes but fail after?"

---

### 2. `reproduction_output_misreading` (0/3 detection)

**Definition**: Agent RUNS tests but MISINTERPRETS the results

**Ground truth cases**:
- astropy__astropy-13453
- astropy__astropy-14182
- astropy__astropy-8872

**What judges predicted instead**:
- LLM: algorithmic_implementation, component_coordination, verification_abandonment
- Lunette: insufficient_verification_capability, verification_abandonment

**Why it's being missed**:

Judges conflate "verification problems" into broader categories. They see:
- "Agent didn't test properly" → `verification_abandonment`
- "Agent's tests were incomplete" → `insufficient_verification_capability`

But miss the specific case:
- "Agent DID run tests, but MISREAD what the output meant"

**Example from astropy-8872**:
- Ground truth: reproduction_output_misreading (agent thought tests passed when they didn't)
- LLM predicted: recurring_pattern, algorithmic_implementation
- Neither judge noted: "Agent saw test output and drew wrong conclusion"

**The semantic gap**: The distinction requires evidence that:
1. Agent ran a test/script
2. Output was visible to agent
3. Agent's interpretation of output was incorrect

**Potential fix**: Add explicit check:
- "Did the agent run any tests? If so, what did the output actually show vs what did the agent THINK it showed?"

---

### 3. `control_flow` (0/2 detection)

**Definition**: Wrong conditionals, loops, early returns, exception handling

**Ground truth cases**:
- astropy__astropy-8872
- django__django-11087

**What judges predicted instead**:
- LLM: recurring_pattern, algorithmic_implementation
- Lunette: algorithmic_implementation, component_coordination

**Why it's being missed**:

`control_flow` is being absorbed into `algorithmic_implementation`. Both involve "wrong logic", but:
- `algorithmic_implementation`: Wrong algorithm/math/data structure logic
- `control_flow`: Specifically wrong if/else, loops, returns, try/except

**Example from django-11087**:
- Ground truth includes: control_flow, algorithmic_implementation
- LLM predicted: algorithmic_implementation (caught!), but missed control_flow
- The gold patch likely involved fixing conditional logic, but judges don't distinguish

**The semantic gap**: Judges see "wrong implementation" as one category, not distinguishing WHAT KIND of implementation error.

**Potential fix**: When predicting algorithmic_implementation, force a follow-up:
- "Is the error in the algorithm itself (math, data structures) OR in control flow (conditionals, loops, returns, exception handling)?"

---

### 4. `verification_weakening` (0/1 detection)

**Definition**: Agent MODIFIES tests to make them pass instead of fixing code

**Ground truth case**:
- django__django-11087

**What judges predicted instead**:
- LLM: verification_abandonment
- Lunette: verification_abandonment

**Why it's being missed**:

Both judges detected verification problems but chose `verification_abandonment` (stopped testing) instead of `verification_weakening` (modified tests).

**The semantic gap**:
- `verification_abandonment`: Agent stopped running tests
- `verification_weakening`: Agent changed/weakened tests to pass

These require different evidence:
- Abandonment: No test execution after a point
- Weakening: Test file modified to reduce coverage/assertions

**Potential fix**: Explicit check:
- "Did the agent modify any test files? If so, did those modifications REDUCE what was being tested?"

---

## Cross-Cutting Patterns

### Pattern 1: Broad Categories Absorb Specific Ones

| Specific Category | Absorbed Into |
|-------------------|---------------|
| control_flow | algorithmic_implementation |
| verification_weakening | verification_abandonment |
| reproduction_output_misreading | verification_abandonment or insufficient_verification_capability |
| issue_interference | issue_misleading or specific_case_overfitting |

### Pattern 2: "What Went Wrong" vs "How Did It Go Wrong"

Current prompts elicit WHAT failed but not the specific MANNER of failure. Judges say:
- "The fix was wrong" → but not "the fix broke existing functionality"
- "Verification was inadequate" → but not "agent misread test output"

### Pattern 3: False Positive for `issue_misleading`

`issue_misleading` was predicted 4 times across both methods, but only 1 case had it in ground truth. Judges over-attribute "the issue description was confusing" when the real problem was agent error.

---

## Recommendations for Prompt Improvement

### Approach 1: Hierarchical Classification

Instead of asking for all categories at once, use a decision tree:

```
1. VERIFICATION PHASE
   a. Did agent run tests?
      - No → verification_abandonment
      - Yes → continue
   b. Did agent modify test files?
      - Yes, to weaken coverage → verification_weakening
      - No → continue
   c. Did agent correctly interpret test output?
      - No → reproduction_output_misreading
      - Yes → no verification failure

2. REPAIR PHASE
   a. Does agent's fix break previously working functionality?
      - Yes → issue_interference
   b. Is the implementation logic wrong?
      - Wrong conditionals/loops/returns → control_flow
      - Wrong algorithm/math → algorithmic_implementation
```

### Approach 2: Binary Questions per Category

For each hard-to-detect category, ask a specific YES/NO question:

```
issue_interference: "Does the agent's fix cause ANY feature that worked before to fail?"
reproduction_output_misreading: "Does the agent see test output and draw an incorrect conclusion about what it means?"
control_flow: "Are there errors specifically in if/else, loops, returns, or try/except?"
verification_weakening: "Does the agent modify test files to make tests easier to pass?"
```

### Approach 3: Contrastive Examples

Provide examples that distinguish similar categories:

```
algorithmic_implementation vs control_flow:
- algorithmic_implementation: "Used O(n²) algorithm when O(n) was needed"
- control_flow: "Used 'if x > 0' when should have been 'if x >= 0'"

verification_abandonment vs verification_weakening:
- verification_abandonment: "Agent stopped running tests after step 10"
- verification_weakening: "Agent removed assertions from test file to make tests pass"
```

---

## Hierarchical Prompt Experiment Results

Implemented a hierarchical YES/NO question format for each of the 25 categories.

### Detection Rates: Original vs Hierarchical

| Category | Original Prompt | Hierarchical Prompt | Change |
|----------|----------------|---------------------|--------|
| issue_interference | 0% (0/2) | 50% (1/2) | +50% |
| reproduction_output_misreading | 0% (0/3) | 50% (1/2) | +50% |
| control_flow | 0% (0/2) | 50% (1/2) | +50% |
| verification_weakening | 0% (0/1) | 0% (0/1) | no change |
| issue_misleading | 0% (0/1) | 0% (0/1) | no change |

### Overall Metrics

| Metric | Original Prompt | Hierarchical Prompt |
|--------|----------------|---------------------|
| Jaccard | 0.152 | 0.276 |
| Overlap Rate | 50% | 66.7% |

**84% improvement in Jaccard score with hierarchical prompt.**

### Why Some Categories Remain Difficult

**verification_weakening (0% detection)**
- Model response: "NO - No test modifications were made"
- Ground truth says test weakening occurred
- **Root cause**: The model can't reliably detect test file modifications from trajectory alone. Would need explicit file diff analysis.

**issue_misleading (0% detection)**
- Model response: "NO - The PR description was clear about implementing direct ITRS to observed transformations"
- Ground truth says issue was misleading
- **Root cause**: The model interprets "clear structure" as "not misleading" but misses subtle misleading elements. The distinction between "agent misunderstood clear info" vs "info itself was confusing" requires deeper semantic understanding.

### Key Insights

1. **Explicit questions help**: Asking "Does fix for X break Y?" directly detects issue_interference better than asking "what categories apply?"

2. **Some categories require external data**: verification_weakening requires knowing whether test files were modified - this isn't always clear from trajectory text alone.

3. **Subjectivity in annotations**: What makes an issue "misleading" vs "clear" may vary between annotators. The model tends to give benefit of the doubt to issue descriptions.

---

## Recommendations for Full 203-Task Run

1. **Use hierarchical prompt** for improved detection of most categories
2. **Accept limitations** for verification_weakening and issue_misleading - these may require different approaches:
   - verification_weakening: Explicit file diff analysis
   - issue_misleading: More examples of what counts as misleading
3. **Consider ensemble approach**: Run both original and hierarchical prompts, take union of predictions
4. **Report per-category metrics** to understand which categories the judge handles well vs poorly
