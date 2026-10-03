---
name: skill-creator
description: Guide for creating effective skills. Use when users want to create or update a skill that extends DeepWorksEngine with specialized knowledge, workflows, or tool integrations.
---

# Skill Creator

This skill is a template + checklist for creating skills in a workspace.

## What is a skill?

A skill is a folder under `.opencode/skills/<skill-name>/` or `.claude/skills/<skill-name>/` anchored by `SKILL.md`.

## DeepWorks behavior

- In DeepWorks, create skills under `SESSION_WORKING_DIRECTORY/.opencode/skills/<skill-name>/SKILL.md`. `SESSION_WORKING_DIRECTORY` is the effective directory bound to the current chat session, not the shell `pwd` or `Workspace root folder` environment field.
- When the user asks to optimize or update an existing Skill, first search the current session directory and then continue through all visible global Skill sources if the project search has no match. Do not report “not found” after checking only the current workspace.
- Before changing an existing global Skill, always ask which target the user wants: “直接修改全局源文件” or “复制到当前工作区后修改”. Use the structured question tool when available; otherwise ask one concise normal-message question before any file mutation. If the user already chose one of these targets for the same Skill task, keep that choice and do not ask again unless the target changes.
- If the user chooses to copy a global Skill, copy its complete directory, including `SKILL.md` and supporting files, to the current session directory before editing; never edit the global source after that choice.
- If project and global Skills share the same invocation id, ask the same target question instead of silently preferring the project copy.
- Choose `<skill-name>` as the stable invocation id. It must be lowercase ASCII kebab-case matching `^[a-z0-9]+(-[a-z0-9]+)*$`, 1-64 characters, for example `data-security`.
- The `.opencode/skills/<skill-name>/` directory name and the SKILL.md frontmatter `name` value must be exactly the same `<skill-name>` invocation id.
- Do not put Chinese, localized display names, spaces, underscores, or mixed-case titles in frontmatter `name`; put human-readable localized titles in the Markdown H1, body, or description instead.
- When using SkillHub CLI or any third-party skill installer, force the install destination to the same current working directory path: `.opencode/skills/<skill-name>/`. Do not accept installer defaults such as `skills/<skill-name>/` because DeepWorks pages and skill selectors load project skills from `.opencode/skills`.
- For third-party installed skills, preserve bundled files such as `_meta.json`, `references/`, `scripts/`, and `templates/` under `.opencode/skills/<skill-name>/`, and ensure `SKILL.md` frontmatter records the origin with a `metadata.source` field when the source is known.
- Do not create or update global/user-level skills such as `~/.opencode/skills`, `~/.config/opencode/skills`, `~/.claude/skills`, or `~/.agents/skills` before the target question has been answered. A generic request to optimize a named Skill is not permission to modify its global source.
- Do not infer a different skill target directory from `$HOME`, `~`, `Workspace root folder`, shell `pwd`, or a parent directory.
- Use a file mutation tool (`write`, `edit`, or `apply_patch`) on the real skill path instead of pasting the whole skill into chat.
- Writing the skill file lets DeepWorks show the reload banner above the conversation so the user can activate the new skill immediately.

## Design goals

- Portable: safe to copy between machines
- Reconstructable: can recreate any required local state
- Self-building: can bootstrap its own config/state
- Credential-safe: no secrets committed; graceful first-time setup

## Recommended structure

```
.opencode/
  skills/
    my-skill/
      SKILL.md
      README.md
      templates/
      scripts/
```

## Trigger phrases (critical)

The description field is how Claude decides when to use your skill.
Include 2-3 specific phrases that should trigger it.

Bad example:
"Use when working with content"

Good examples:
"Use when user mentions 'content pipeline', 'add to content database', or 'schedule a post'"
"Triggers on: 'rotate PDF', 'flip PDF pages', 'change PDF orientation'"

Quick validation:
- Contains at least one quoted phrase
- Uses "when" or "triggers"
- Longer than ~50 characters

## Frontmatter template

```yaml
---
name: my-skill
description: |
  [What it does in one sentence]

  Triggers when user mentions:
  - "[specific phrase 1]"
  - "[specific phrase 2]"
  - "[specific phrase 3]"
metadata:
  author: "[current DeepWorks skill author from system context]"
---
```

For localized skills, keep `name` as the invocation id and put the localized title below the frontmatter:

```markdown
# 数据安全治理技能
```

When creating or updating a skill through DeepWorks chat, always use the current DeepWorks skill author from the system context as `metadata.author`.

## Authoring checklist

1. Start with a clear purpose statement: when to use it + what it outputs.
2. Specify inputs/outputs and any required permissions.
3. Include "Setup" steps if the skill needs local tooling.
4. Add examples: at least 2 realistic user prompts.
5. Keep it safe: avoid destructive defaults; ask for confirmation.
6. In DeepWorks, finish by writing the final `SKILL.md` file to `.opencode/skills/<skill-name>/SKILL.md` so the reload banner can appear.
