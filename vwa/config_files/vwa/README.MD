# README

## Overview
- This folder contains configuration files for each task in **VisualWebArena**, updated with fixes to **intents** and **evaluators**, revised identifiers to facilitate evaluation, and representative subsets to enable faster experimentation. See below for details.

## Key Notes
**1. Data**  
- **`test_vwa.raw.json`** — Flat list containing all tasks configs across all domains.  
- **`test_vwa_{domain}.raw.json`** — Domain-specific sets of tasks, where `{domain}` ∈ {`classifieds`, `reddit`, `shopping`}.
- `intent_fix_note` and `intent_fix_note`, where present, provide a short explanation for the fixes.

**2. Representative Task Subsets**
- The `_lite` JSON files contain **representative subsets** (approximately one-third the size of the full benchmark) that provide similar performance signals and are useful for faster development and testing.

- These subsets are constructed to match the proportion of intent templates and task difficulty of the full benchmark, while minimizing the difference in average scores per domain relative to the averages obtained on the full benchmark for a given agent.  

- The reference scores comes from the `TreeSearch` agent from **@TODO**. New subsets can be generated using baseline scores from other agents. See **@TODO** for the code and instructions to build custom subsets.


**3. Task Identifiers and Evaluation Workflow**
- New identifiers were introduced to simplify task tracking during parallel execution and data processing.
    - Explanation: In the original VisualWebArena release, task IDs were **not unique across domains** (e.g., ID `0` appeared in `shopping`, `classifieds`, and `reddit`).
    - The `task_id` field remains consistent with the original VisualWebArena release for backward compatibility.  
    - For experiments, it is recommended to use one of the **unique identifiers**: `uid`, `domain_task_id`, `domain_task_id_env`  

- The updated codebase allows direct evaluation providing the path to the `.raw.json` files. There is **no need** to generate separate `.json` files for individual tasks.

- See `run.py` for examples iterating through tasks.
---

## Added Fields

| Field | Description |
|:------|:-------------|
| `uid` | Globally unique integer ID for each task, matching the IDs used in the `browsergym` release. |
| `domain_task_id` | Unique task identifier within a given VisualWebArena domain. |
| `domain_task_id_env` | Unique identifier across all domains and environments (VisualWebArena vs. WebArena). |
| `old_intent` | If present, original task intent from the original VisualWebArena release. |
| `intent_fix_note` | Short note describing the modification made to the original intent. |
| `old_eval` | If present, original evaluator configuration from the original VisualWebArena release. |
| `eval_fix_note` | Short note explaining the evaluator fix. |
