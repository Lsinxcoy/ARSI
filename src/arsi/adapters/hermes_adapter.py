"""Hermes adapter — extract behavior traces from Hermes agent.

Data sources:
1. mstar_fitness.db — MSTAR Pro evolution tracking (programs, fitness)
2. failed_trajectories.jsonl — failed execution trajectories
3. Skills directory — procedural memory
4. Plans directory — task plans
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class HermesAdapter:
    """Extracts behavior traces from Hermes agent data sources."""

    def __init__(self, hermes_home: Optional[str] = None):
        self.hermes_home = Path(hermes_home or "C:/Users/41228/.hermes")
        self.hermes_agent = Path("C:/Users/41228/hermes-personal-backup")
        self._extracted_count = 0

    def extract_traces(self) -> list[dict]:
        """Extract behavior traces from all Hermes data sources."""
        traces = []
        traces.extend(self._extract_from_mstar_db())
        traces.extend(self._extract_from_trajectories())
        traces.extend(self._extract_from_skills())
        traces.extend(self._extract_from_plans())
        traces.extend(self._extract_from_config())
        self._extracted_count += len(traces)
        return traces

    def _extract_from_plans(self) -> list[dict]:
        """Extract traces from Hermes plans directory."""
        traces = []
        plans_dir = self.hermes_home / "plans"
        if not plans_dir.exists():
            return traces

        for plan_file in plans_dir.glob("*.md"):
            try:
                content = plan_file.read_text(encoding="utf-8", errors="ignore")
                # Extract task items from markdown
                lines = content.split("\n")
                task_count = sum(1 for l in lines if l.strip().startswith(("- ", "* ", "1.", "2.", "3.")))
                traces.append({
                    "action": f"hermes_plan:{plan_file.stem[:40]}",
                    "outcome": "success" if task_count > 0 else "partial",
                    "effect": min(1.0, task_count / 20),
                    "params": {
                        "source": "plan",
                        "filename": plan_file.name,
                        "content_length": len(content),
                        "task_items": task_count,
                    },
                })
            except Exception:
                continue

        return traces

    def _extract_from_config(self) -> list[dict]:
        """Extract traces from Hermes config (model settings, delegation config)."""
        traces = []
        config_path = self.hermes_home / "config.yaml"
        if not config_path.exists():
            return traces

        try:
            content = config_path.read_text(encoding="utf-8", errors="ignore")
            traces.append({
                "action": "hermes_config",
                "outcome": "recorded",
                "effect": 0.5,
                "params": {
                    "source": "config",
                    "content_length": len(content),
                    "has_model": "model:" in content,
                    "has_delegation": "delegation:" in content,
                },
            })
        except Exception:
            pass

        return traces

    def _extract_from_mstar_db(self) -> list[dict]:
        """Extract traces from MSTAR fitness database."""
        traces = []
        db_path = self.hermes_agent / "mstar_fitness.db"
        if not db_path.exists():
            return traces

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # Extract program fitness data
            cursor = conn.execute(
                "SELECT * FROM programs ORDER BY fitness_score DESC LIMIT 20"
            )
            for row in cursor.fetchall():
                fitness = row["fitness_score"] if "fitness_score" in row.keys() else 0.5
                traces.append({
                    "action": f"mstar_program:{row['program_id'] if 'program_id' in row.keys() else 'unknown'}",
                    "outcome": "success" if fitness > 0.7 else "partial" if fitness > 0.4 else "failure",
                    "effect": min(1.0, max(0.0, fitness)),
                    "params": {
                        "source": "mstar_fitness_db",
                        "program_id": row["program_id"] if "program_id" in row.keys() else "",
                        "fitness": fitness,
                    },
                })

            # Extract fitness snapshots
            cursor = conn.execute(
                "SELECT * FROM fitness_snapshots ORDER BY rowid DESC LIMIT 10"
            )
            for row in cursor.fetchall():
                traces.append({
                    "action": "mstar_snapshot",
                    "outcome": "recorded",
                    "effect": 0.5,
                    "params": {"source": "fitness_snapshot", "data": dict(row)},
                })

            conn.close()
        except Exception as e:
            logger.warning(f"Failed to read MSTAR DB: {e}")

        return traces

    def _extract_from_trajectories(self) -> list[dict]:
        """Extract traces from failed trajectories JSONL."""
        traces = []
        traj_path = self.hermes_agent / "failed_trajectories.jsonl"
        if not traj_path.exists():
            return traces

        try:
            with open(traj_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    if line_num >= 20:  # Limit to 20 trajectories
                        break
                    try:
                        data = json.loads(line.strip())
                        conversations = data.get("conversations", [])
                        # Extract the task from human messages
                        task = ""
                        for msg in conversations:
                            if msg.get("from") == "human":
                                task = msg.get("value", "")[:100]
                                break

                        traces.append({
                            "action": f"hermes_task:{task[:50]}",
                            "outcome": "failure",  # These are failed trajectories
                            "effect": 0.2,
                            "params": {
                                "source": "failed_trajectory",
                                "conversation_count": len(conversations),
                                "task_preview": task,
                            },
                        })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to read trajectories: {e}")

        return traces

    def _extract_from_skills(self) -> list[dict]:
        """Extract traces from Hermes skills directory."""
        traces = []
        skills_dir = self.hermes_home / "skills"
        if not skills_dir.exists():
            return traces

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text(encoding="utf-8", errors="ignore")
                traces.append({
                    "action": f"hermes_skill:{skill_dir.name}",
                    "outcome": "success",
                    "effect": 0.7,
                    "params": {
                        "source": "skill",
                        "skill_name": skill_dir.name,
                        "content_length": len(content),
                    },
                })

        return traces

    def get_stats(self) -> dict:
        """Get Hermes adapter statistics."""
        return {
            "hermes_home": str(self.hermes_home),
            "hermes_agent": str(self.hermes_agent),
            "extracted_count": self._extracted_count,
            "mstar_db_exists": (self.hermes_agent / "mstar_fitness.db").exists(),
            "trajectories_exist": (self.hermes_agent / "failed_trajectories.jsonl").exists(),
            "skills_exist": (self.hermes_home / "skills").exists(),
        }
