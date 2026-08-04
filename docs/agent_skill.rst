.. _agent_skill:

Using SynPlanner with an AI agent
==================================

SynPlanner ships an **agent skill** — a file that teaches AI coding agents how to
use this package correctly: which API to reach for, how chython differs from
RDKit, and what to do by default.

It follows the `Agent Skills open standard <https://agentskills.io>`_, which is
supported by Claude Code, OpenAI Codex, Cursor, GitHub Copilot, VS Code, Gemini
CLI, OpenCode, OpenHands, Goose, Amp, JetBrains Junie, Roo Code, Kiro and others.

Install it
----------

The skill lives at
`skills/synplanner-usage/ <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/skills/synplanner-usage>`_
in the GitHub repository. Copy that directory into whichever location your agent
reads:

.. list-table::
   :header-rows: 1

   * - Agent
     - Location
   * - Claude Code
     - ``.claude/skills/`` in your project, or ``~/.claude/skills/`` for all projects
   * - Cursor
     - ``.cursor/skills/`` in your project
   * - OpenAI Codex
     - ``~/.codex/skills/``
   * - OpenHands
     - ``.agents/skills/``
   * - Others
     - see your agent's documentation for its skills directory

No further setup is needed. The agent loads the skill when a request matches its
description — planning a route, importing ``synplan``, training a policy, or
hitting a chython-versus-RDKit difference.

If your agent does not support skills, the text below still works as plain
context: paste it, or point the agent at
`the raw file <https://raw.githubusercontent.com/Laboratoire-de-Chemoinformatique/SynPlanner/main/skills/synplanner-usage/SKILL.md>`_.

Scope
-----

The skill is for **using** SynPlanner — planning, curation, training, analysis.
Contributing to SynPlanner itself is out of scope.

It refers to a companion task index that maps each task to the API pieces it
needs; that is reproduced at :doc:`tasks`.

.. _agent_skill_contents:

Skill contents
--------------

The full text of ``SKILL.md``, as the agent reads it:

.. include:: ../skills/synplanner-usage/SKILL.md
   :parser: myst_parser.sphinx_
   :start-line: 20
