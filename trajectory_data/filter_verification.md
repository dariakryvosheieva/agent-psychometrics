# Trajectory Filter Verification Log

This file tracks verification of filtered trajectories for each agent.

**Last Updated:** 2026-01-08
**Total Agents Processed:** 78
**Overall Reduction:** 56.9% (1,419,285 / 3,294,028 messages kept)

## Summary by Category

### Normal Reduction (40-70%) - Expected behavior
These agents show typical filtering where planning/setup is removed but actions are kept.

### High Reduction (70-90%) - More thinking/planning heavy
These agents have more verbose thinking/planning content which is correctly filtered.

### Very High Reduction (>90%) - Special formats
These agents have special formats (log dumps, embedded transcripts) where most content is setup/issue description.

### Low Reduction (<40%) - Action-heavy formats
These agents use action-focused formats (patches, JSON tool calls) where most content IS the action.

## Verification Status

| Agent | Status | Reduction | Notes |
|-------|--------|-----------|-------|
| 20240402_sweagent_claude3opus | OK | 53.8% | Standard SWE-agent format, USER=obs ASSISTANT=planning filtered |
| 20240402_sweagent_gpt4 | OK | 47.9% | Standard SWE-agent format |
| 20240612_MASAI_gpt4o | OK | 63.1% | Multi-agent format, planning filtered |
| 20240620_sweagent_claude3.5sonnet | OK | 50.8% | Standard SWE-agent format, verified manually |
| 20240721_amazon-q-developer-agent-20240719-dev | OK | 72.6% | Narrative format, past-tense action descriptions kept |
| 20240728_sweagent_gpt4o | OK | 50.3% | Standard SWE-agent format |
| 20240820_epam-ai-run-gpt-4o | OK | 41.8% | Action-heavy format |
| 20240820_honeycomb | OK | 49.8% | Standard format |
| 20240918_lingma-agent_lingma-swe-gpt-72b | OK | 75.4% | High planning content filtered |
| 20240918_lingma-agent_lingma-swe-gpt-7b | OK | 76.7% | High planning content filtered |
| 20241002_lingma-agent_lingma-swe-gpt-72b | OK | 75.9% | High planning content filtered |
| 20241002_lingma-agent_lingma-swe-gpt-7b | OK | 74.2% | High planning content filtered |
| 20241007_nfactorial | OK | 61.3% | Standard format |
| 20241016_composio_swekit | OK | 53.0% | Standard format |
| 20241016_epam-ai-run-gpt-4o | OK | 59.9% | Standard format |
| 20241025_composio_swekit | OK | 48.5% | Standard format |
| 20241028_agentless-1.5_gpt4o | OK | 37.4% | Action-heavy, patches and file operations kept |
| 20241029_OpenHands-CodeAct-2.1-sonnet-20241022 | OK | 51.8% | Standard OpenHands format |
| 20241029_epam-ai-run-claude-3-5-sonnet | OK | 54.4% | Standard format |
| 20241030_nfactorial | WARN | 87.3% | Planning-log format, mostly change plans filtered |
| 20241105_nfactorial | WARN | 85.5% | Planning-log format, mostly change plans filtered |
| 20241106_navie-2-gpt4o-sonnet | OK | 73.8% | High planning content |
| 20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022 | OK | 67.9% | Standard format |
| 20241108_devlo | OK | 43.8% | Action-heavy format |
| 20241113_nebius-search-open-weight-models-11-24 | OK | 50.1% | Standard format |
| 20241125_enginelabs | OK | 55.3% | Standard format |
| 20241125_marscode-agent-dev | OK | 56.7% | Standard format |
| 20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128 | OK | 50.2% | Patch-based format (1 trajectory only) |
| 20241202_agentless-1.5_claude-3.5-sonnet-20241022 | OK | 51.3% | Standard format |
| 20241212_epam-ai-run-claude-3-5-sonnet | OK | 58.3% | Standard format |
| 20241213_devlo | OK | 40.9% | Action-heavy format |
| 20241221_codestory_midwit_claude-3-5-sonnet_swe-search | OK | 49.7% | Standard format |
| 20250110_blackboxai_agent_v1.1 | OK | 46.4% | Standard format |
| 20250110_learn_by_interact_claude3.5 | OK | 70.1% | High planning content |
| 20250117_wandb_programmer_o1_crosscheck5 | OK | 72.2% | High planning content |
| 20250118_codeshellagent_gemini_2.0_flash_experimental | OK | 51.1% | Standard format |
| 20250122_autocoderover-v2.1-claude-3-5-sonnet-20241022 | OK | 67.3% | Standard format |
| 20250203_openhands_4x_scaled | OK | 63.3% | Standard OpenHands format |
| 20250226_swerl_llama3_70b | OK | 44.2% | Action-heavy format |
| 20250228_epam-ai-run-claude-3-5-sonnet | OK | 52.3% | Standard format |
| 20250306_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor | OK | 50.1% | Patch-based format (1 trajectory only) |
| 20250410_cortexa | OK | 80.2% | Verbose format with high planning content |
| 20250415_openhands | OK | 63.3% | Standard OpenHands format |
| 20250503_patchpilot-v1.1-o4-mini | OK | 50.7% | Patch + log format, issue descriptions filtered |
| 20250511_sweagent_lm_32b | OK | 53.4% | Standard SWE-agent format |
| 20250515_Refact_Agent | OK | 63.4% | Standard format |
| 20250516_cortexa_o3 | OK | 83.0% | High planning content |
| 20250519_devlo | OK | 40.8% | Action-heavy format |
| 20250519_trae | OK | 73.6% | High planning content |
| 20250520_openhands_devstral_small | OK | 50.8% | Standard OpenHands format |
| 20250522_sweagent_claude-4-sonnet-20250514 | OK | 75.1% | High planning content |
| 20250524_openhands_claude_4_sonnet | OK | 62.6% | Standard OpenHands format |
| 20250527_amazon.nova-premier-v1.0 | OK | 68.4% | High planning content |
| 20250528_patchpilot_Co-PatcheR | OK | 0.1% | All patch content, correctly kept |
| 20250603_Refact_Agent_claude-4-sonnet | OK | 62.7% | Standard format |
| 20250611_moatless_claude-4-sonnet-20250514 | OK | 2.0% | JSON tool calls, issue description filtered |
| 20250612_trae | OK | 72.3% | High planning content |
| 20250616_Skywork-SWE-32B | OK | 51.4% | Standard format |
| 20250616_Skywork-SWE-32B+TTS_Bo8 | OK | 50.8% | Standard format |
| 20250627_agentless_MCTS-Refine-7B | OK | 46.8% | Log stage format, issue logs filtered |
| 20250710_bloop | OK | 31.6% | Action-heavy format, issue content filtered |
| 20250716_openhands_kimi_k2 | OK | 61.2% | Standard OpenHands format |
| 20250728_zai_glm4-5 | WARN | 90.1% | OpenHands-style with lots of system/setup messages |
| 20250804_codesweep_sweagent_kimi_k2_instruct | OK | 53.6% | Standard SWE-agent format |
| 20250804_epam-ai-run-claude-4-sonnet | OK | 55.2% | Standard format |
| 20250807_openhands_gpt5 | OK | 67.8% | Standard OpenHands format |
| 20250901_entroPO_R2E_QwenCoder30BA3B | OK | 50.5% | Standard format |
| 20250901_entroPO_R2E_QwenCoder30BA3B_tts | OK | 50.3% | Standard format |
| 20250915_JoyCode | WARN | 97.9% | Log dump format with embedded transcripts, minimal action content extractable |
| 20250924_artemis_agent_v2 | OK | 38.6% | Action-heavy format, issue descriptions filtered |
| 20250928_trae_doubao_seed_code | OK | 57.4% | Standard format |
| 20250929_Prometheus_v1.2_gpt5 | OK | 54.0% | Standard format |
| 20250930_zai_glm4-6 | WARN | 86.1% | OpenHands-style with lots of system/setup messages |
| 20251015_Prometheus_v1.2.1_gpt5 | OK | 54.0% | Standard format |
| 20251103_SalesforceAIResearch_SAGE_OpenHands | OK | 68.7% | Standard OpenHands format |
| 20251103_sonar-foundation-agent_claude-sonnet-4-5 | OK | 39.7% | Action-heavy format (USER=obs in different role mapping) |
| 20251110_frogboss-32b | OK | 69.7% | Standard format |
| 20251110_frogmini-14b | OK | 70.0% | Standard format |

## Verification Criteria

A filtered trajectory is "reasonable" if:
1. It keeps actual codebase interactions (file views, edits, command execution, test runs)
2. It filters out pure thinking/planning without actions
3. It filters out setup context and environment reminders
4. The remaining content shows behavioral choices the agent made

## Detailed Notes

### Agents with WARN Status

**20241030_nfactorial, 20241105_nfactorial (85-87% reduction)**
- Format: Timestamped log entries with "Change plan:" prefixes
- Content: Mostly planning/analysis messages
- Result: Correctly filtered as planning content, leaves only action descriptions

**20250728_zai_glm4-5, 20250930_zai_glm4-6 (86-90% reduction)**
- Format: OpenHands-style with "system:", "message:", "recall:" prefixes
- Content: Heavy system prompts and issue descriptions
- Result: Keeps "I edited the file" action summaries, correctly filters setup

**20250915_JoyCode (97.9% reduction)**
- Format: Log dump with embedded TURN conversation transcripts
- Content: Mix of issue description + conversation history in single messages
- Result: Hard to parse without agent-specific logic; minimal actionable content extractable

### Low Reduction Agents - Verification

**20250528_patchpilot_Co-PatcheR (0.1% reduction)**
- All 58 ASSISTANT messages are patches starting with `[repair]`
- Verified: These are actual code changes and should all be kept

**20250611_moatless_claude-4-sonnet-20250514 (2.0% reduction)**
- 25 ASSISTANT messages are JSON tool calls (action_args_class)
- 1 USER message filtered (issue description)
- Verified: JSON actions represent tool invocations

**20250710_bloop (31.6% reduction)**
- Mix of USER observations (bash output, file content) and ASSISTANT tool calls
- Issue descriptions with `<ISSUE>` tags now filtered
- Verified: Keeps bash outputs and action content
