"""Feature definitions for environment feature extraction.

All features are deterministic and should produce identical results on repeated runs.
Commands run inside SWE-bench Docker containers where the repo is at /testbed.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class FeatureDefinition:
    """Definition of a single environment feature."""

    name: str
    command: str
    description: str
    parser: Optional[Callable[[str], int | float]] = None

    def parse(self, stdout: str) -> int | float:
        """Parse command output to numeric value."""
        if self.parser:
            return self.parser(stdout)
        # Default: parse as integer
        try:
            return int(stdout.strip())
        except ValueError:
            return -1


def parse_int_or_negative(stdout: str) -> int:
    """Parse stdout as int, return -1 on failure."""
    try:
        return int(stdout.strip())
    except ValueError:
        return -1


def parse_float_or_negative(stdout: str) -> float:
    """Parse stdout as float, return -1.0 on failure."""
    try:
        return float(stdout.strip())
    except ValueError:
        return -1.0


# All feature definitions organized by category
FEATURE_DEFINITIONS: list[FeatureDefinition] = [
    # === File System Structure ===
    FeatureDefinition(
        name="env_total_files",
        command="find /testbed -type f 2>/dev/null | wc -l",
        description="Total files in repo",
    ),
    FeatureDefinition(
        name="env_total_dirs",
        command="find /testbed -type d 2>/dev/null | wc -l",
        description="Total directories",
    ),
    FeatureDefinition(
        name="env_dir_depth_max",
        command="find /testbed -type d 2>/dev/null | sed 's|/testbed||' | awk -F/ '{print NF}' | sort -n | tail -1",
        description="Max directory depth",
    ),
    # === Python Files ===
    FeatureDefinition(
        name="env_python_files",
        command="find /testbed -name '*.py' 2>/dev/null | wc -l",
        description="Python file count",
    ),
    FeatureDefinition(
        name="env_python_loc",
        command="find /testbed -name '*.py' -exec cat {} + 2>/dev/null | wc -l",
        description="Total Python lines of code",
    ),
    FeatureDefinition(
        name="env_init_files",
        command="find /testbed -name '__init__.py' 2>/dev/null | wc -l",
        description="Package __init__.py files",
    ),
    # === Test Infrastructure ===
    FeatureDefinition(
        name="env_test_files",
        command="find /testbed \\( -name 'test_*.py' -o -name '*_test.py' \\) 2>/dev/null | wc -l",
        description="Test file count",
    ),
    FeatureDefinition(
        name="env_test_dirs",
        command="find /testbed -type d -name 'test*' 2>/dev/null | wc -l",
        description="Test directories",
    ),
    FeatureDefinition(
        name="env_conftest_files",
        command="find /testbed -name 'conftest.py' 2>/dev/null | wc -l",
        description="Pytest conftest files",
    ),
    FeatureDefinition(
        name="env_has_pytest_ini",
        command="test -f /testbed/pytest.ini && echo 1 || echo 0",
        description="Has pytest.ini",
    ),
    FeatureDefinition(
        name="env_has_tox",
        command="test -f /testbed/tox.ini && echo 1 || echo 0",
        description="Has tox.ini",
    ),
    # === Build/Package Configuration ===
    FeatureDefinition(
        name="env_has_setup_py",
        command="test -f /testbed/setup.py && echo 1 || echo 0",
        description="Has setup.py",
    ),
    FeatureDefinition(
        name="env_has_setup_cfg",
        command="test -f /testbed/setup.cfg && echo 1 || echo 0",
        description="Has setup.cfg",
    ),
    FeatureDefinition(
        name="env_has_pyproject",
        command="test -f /testbed/pyproject.toml && echo 1 || echo 0",
        description="Has pyproject.toml",
    ),
    FeatureDefinition(
        name="env_has_makefile",
        command="test -f /testbed/Makefile && echo 1 || echo 0",
        description="Has Makefile",
    ),
    FeatureDefinition(
        name="env_has_dockerfile",
        command="test -f /testbed/Dockerfile && echo 1 || echo 0",
        description="Has Dockerfile",
    ),
    # === Dependencies ===
    FeatureDefinition(
        name="env_requirements_count",
        command="cat /testbed/requirements*.txt 2>/dev/null | grep -v '^#' | grep -v '^$' | wc -l",
        description="Lines in requirements files",
    ),
    FeatureDefinition(
        name="env_has_requirements",
        command="ls /testbed/requirements*.txt 2>/dev/null | wc -l",
        description="Number of requirements files",
    ),
    # === Git Repository Stats ===
    FeatureDefinition(
        name="env_git_commits_total",
        command="cd /testbed && git rev-list --count HEAD 2>/dev/null || echo -1",
        description="Total commit count",
    ),
    FeatureDefinition(
        name="env_git_branches",
        command="cd /testbed && git branch -a 2>/dev/null | wc -l",
        description="Branch count",
    ),
    FeatureDefinition(
        name="env_git_tags",
        command="cd /testbed && git tag 2>/dev/null | wc -l",
        description="Tag count",
    ),
    FeatureDefinition(
        name="env_git_contributors",
        command="cd /testbed && git shortlog -sn --all 2>/dev/null | wc -l",
        description="Unique contributors",
    ),
    # === Documentation ===
    FeatureDefinition(
        name="env_doc_files",
        command="find /testbed \\( -name '*.md' -o -name '*.rst' \\) 2>/dev/null | wc -l",
        description="Documentation file count",
    ),
    FeatureDefinition(
        name="env_has_readme",
        command="ls /testbed/README* 2>/dev/null | wc -l",
        description="Has README files",
    ),
    FeatureDefinition(
        name="env_has_docs_dir",
        command="test -d /testbed/docs && echo 1 || echo 0",
        description="Has docs/ directory",
    ),
    FeatureDefinition(
        name="env_sphinx_conf",
        command="find /testbed -name 'conf.py' -path '*/docs/*' 2>/dev/null | wc -l",
        description="Sphinx config files",
    ),
    # === Code Complexity Proxies ===
    FeatureDefinition(
        name="env_class_count",
        command="grep -r '^class ' /testbed --include='*.py' 2>/dev/null | wc -l",
        description="Class definitions",
    ),
    FeatureDefinition(
        name="env_function_count",
        command="grep -r '^def ' /testbed --include='*.py' 2>/dev/null | wc -l",
        description="Function definitions",
    ),
    FeatureDefinition(
        name="env_import_count",
        command="grep -r '^import \\|^from ' /testbed --include='*.py' 2>/dev/null | wc -l",
        description="Import statements",
    ),
    FeatureDefinition(
        name="env_todo_count",
        command="grep -ri 'TODO\\|FIXME\\|XXX' /testbed --include='*.py' 2>/dev/null | wc -l",
        description="TODO/FIXME comments",
    ),
    # === Other File Types ===
    FeatureDefinition(
        name="env_json_files",
        command="find /testbed -name '*.json' 2>/dev/null | wc -l",
        description="JSON files",
    ),
    FeatureDefinition(
        name="env_yaml_files",
        command="find /testbed \\( -name '*.yml' -o -name '*.yaml' \\) 2>/dev/null | wc -l",
        description="YAML files",
    ),
    FeatureDefinition(
        name="env_config_files",
        command="find /testbed \\( -name '*.cfg' -o -name '*.ini' -o -name '*.conf' \\) 2>/dev/null | wc -l",
        description="Config files",
    ),
    FeatureDefinition(
        name="env_shell_scripts",
        command="find /testbed \\( -name '*.sh' -o -name '*.bash' \\) 2>/dev/null | wc -l",
        description="Shell scripts",
    ),
]


# Create a lookup dict for quick access
FEATURE_BY_NAME: dict[str, FeatureDefinition] = {f.name: f for f in FEATURE_DEFINITIONS}


def get_feature_names() -> list[str]:
    """Get list of all feature names in order."""
    return [f.name for f in FEATURE_DEFINITIONS]
