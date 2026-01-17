"""Prompt version tracking and storage for iterative refinement.

Tracks all prompt versions with their performance metrics and lineage.
Supports archival of removed features and human-readable diffs.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib


@dataclass
class FeatureDefinition:
    """Definition of a single feature for extraction.

    Attributes:
        name: Feature identifier (e.g., "fix_complexity")
        description: What the feature measures
        scale_low: Description of lowest value
        scale_high: Description of highest value
        min_value: Minimum numeric value
        max_value: Maximum numeric value
        extraction_prompt: The prompt text for this feature
    """

    name: str
    description: str
    scale_low: str
    scale_high: str
    min_value: int
    max_value: int
    extraction_prompt: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureDefinition":
        return cls(**d)


@dataclass
class PromptVersion:
    """A versioned prompt with its performance metrics.

    Attributes:
        version_id: Unique identifier (e.g., "v001", "v002")
        prompt_text: The full feature extraction instructions
        feature_schema: List of feature definitions
        feature_names: List of feature names (for quick access)
        quick_eval_r: Pearson r with difficulty from quick eval
        quick_eval_entropy: Dict of feature -> entropy from quick eval
        quick_eval_redundant_pairs: List of highly correlated feature pairs
        full_eval_r: Pearson r from full evaluation (if run)
        full_eval_auc: IRT AUC from full evaluation (if run)
        parent_version: ID of parent version (for lineage tracking)
        changes_from_parent: Human-readable description of changes
        created_at: Timestamp
        archived_features: Features that were removed from parent
    """

    version_id: str
    prompt_text: str
    feature_schema: List[FeatureDefinition]
    feature_names: List[str]
    quick_eval_r: Optional[float] = None
    quick_eval_entropy: Dict[str, float] = field(default_factory=dict)
    quick_eval_redundant_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    full_eval_r: Optional[float] = None
    full_eval_auc: Optional[float] = None
    parent_version: Optional[str] = None
    changes_from_parent: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    archived_features: List[FeatureDefinition] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d["feature_schema"] = [f.to_dict() if isinstance(f, FeatureDefinition) else f
                               for f in self.feature_schema]
        d["archived_features"] = [f.to_dict() if isinstance(f, FeatureDefinition) else f
                                  for f in self.archived_features]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "PromptVersion":
        """Create from dict."""
        d = d.copy()
        d["feature_schema"] = [
            FeatureDefinition.from_dict(f) if isinstance(f, dict) else f
            for f in d.get("feature_schema", [])
        ]
        d["archived_features"] = [
            FeatureDefinition.from_dict(f) if isinstance(f, dict) else f
            for f in d.get("archived_features", [])
        ]
        # Handle tuple conversion for redundant pairs
        d["quick_eval_redundant_pairs"] = [
            tuple(p) if isinstance(p, list) else p
            for p in d.get("quick_eval_redundant_pairs", [])
        ]
        return cls(**d)

    def content_hash(self) -> str:
        """Compute hash of prompt content for deduplication."""
        content = self.prompt_text + "".join(self.feature_names)
        return hashlib.md5(content.encode()).hexdigest()[:8]


class PromptStore:
    """Storage and management for prompt versions.

    All versions are persisted to JSON files in the output directory
    for full traceability and resumption.
    """

    def __init__(self, output_dir: Path):
        """Initialize prompt store.

        Args:
            output_dir: Directory for storing prompt versions
        """
        self.output_dir = Path(output_dir)
        self.versions_dir = self.output_dir / "prompt_versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self._versions: Dict[str, PromptVersion] = {}
        self._load_existing()

    def _load_existing(self):
        """Load existing versions from disk."""
        for path in self.versions_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                version = PromptVersion.from_dict(data)
                self._versions[version.version_id] = version
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load {path}: {e}")

    def _next_version_id(self) -> str:
        """Generate next version ID."""
        existing = [v for v in self._versions.keys() if v.startswith("v")]
        if not existing:
            return "v001"

        nums = []
        for v in existing:
            try:
                nums.append(int(v[1:]))
            except ValueError:
                pass

        next_num = max(nums) + 1 if nums else 1
        return f"v{next_num:03d}"

    def add_version(
        self,
        prompt_text: str,
        feature_schema: List[FeatureDefinition],
        parent_version: Optional[str] = None,
        changes_from_parent: str = "",
        archived_features: Optional[List[FeatureDefinition]] = None,
    ) -> PromptVersion:
        """Add a new prompt version.

        Args:
            prompt_text: The feature extraction instructions
            feature_schema: List of feature definitions
            parent_version: ID of parent version
            changes_from_parent: Human-readable diff
            archived_features: Features removed from parent

        Returns:
            The created PromptVersion
        """
        version_id = self._next_version_id()

        version = PromptVersion(
            version_id=version_id,
            prompt_text=prompt_text,
            feature_schema=feature_schema,
            feature_names=[f.name for f in feature_schema],
            parent_version=parent_version,
            changes_from_parent=changes_from_parent,
            archived_features=archived_features or [],
        )

        self._versions[version_id] = version
        self._save_version(version)

        return version

    def _save_version(self, version: PromptVersion):
        """Save version to disk."""
        path = self.versions_dir / f"{version.version_id}.json"
        with open(path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

    def update_version(
        self,
        version_id: str,
        quick_eval_r: Optional[float] = None,
        quick_eval_entropy: Optional[Dict[str, float]] = None,
        quick_eval_redundant_pairs: Optional[List[Tuple[str, str, float]]] = None,
        full_eval_r: Optional[float] = None,
        full_eval_auc: Optional[float] = None,
    ) -> PromptVersion:
        """Update metrics for an existing version.

        Args:
            version_id: Version to update
            quick_eval_r: Pearson r from quick eval
            quick_eval_entropy: Feature entropies from quick eval
            quick_eval_redundant_pairs: Redundant feature pairs
            full_eval_r: Pearson r from full eval
            full_eval_auc: IRT AUC from full eval

        Returns:
            Updated PromptVersion
        """
        version = self._versions[version_id]

        if quick_eval_r is not None:
            version.quick_eval_r = quick_eval_r
        if quick_eval_entropy is not None:
            version.quick_eval_entropy = quick_eval_entropy
        if quick_eval_redundant_pairs is not None:
            version.quick_eval_redundant_pairs = quick_eval_redundant_pairs
        if full_eval_r is not None:
            version.full_eval_r = full_eval_r
        if full_eval_auc is not None:
            version.full_eval_auc = full_eval_auc

        self._save_version(version)
        return version

    def get_version(self, version_id: str) -> Optional[PromptVersion]:
        """Get a specific version."""
        return self._versions.get(version_id)

    def get_latest(self) -> Optional[PromptVersion]:
        """Get the most recently created version."""
        if not self._versions:
            return None
        return max(self._versions.values(), key=lambda v: v.created_at)

    def get_best(self, metric: str = "quick_eval_r") -> Optional[PromptVersion]:
        """Get version with best performance.

        Args:
            metric: Metric to optimize ("quick_eval_r", "full_eval_r", "full_eval_auc")

        Returns:
            Best performing version
        """
        valid = [
            v for v in self._versions.values()
            if getattr(v, metric) is not None
        ]
        if not valid:
            return None

        return max(valid, key=lambda v: abs(getattr(v, metric)))

    def list_versions(self) -> List[PromptVersion]:
        """List all versions sorted by creation time."""
        return sorted(self._versions.values(), key=lambda v: v.created_at)

    def get_lineage(self, version_id: str) -> List[PromptVersion]:
        """Get the lineage (ancestry) of a version.

        Args:
            version_id: Starting version

        Returns:
            List of versions from oldest ancestor to given version
        """
        lineage = []
        current = self._versions.get(version_id)

        while current:
            lineage.append(current)
            if current.parent_version:
                current = self._versions.get(current.parent_version)
            else:
                current = None

        return list(reversed(lineage))

    def format_summary(self) -> str:
        """Format summary of all versions."""
        lines = [
            f"Prompt Store Summary ({len(self._versions)} versions)",
            "=" * 70,
            "",
        ]

        for v in self.list_versions():
            parent = f" (from {v.parent_version})" if v.parent_version else " (initial)"
            quick_r = f"quick_r={v.quick_eval_r:.3f}" if v.quick_eval_r else "not evaluated"
            full_r = f"full_r={v.full_eval_r:.3f}" if v.full_eval_r else ""
            full_auc = f"auc={v.full_eval_auc:.3f}" if v.full_eval_auc else ""

            lines.append(f"{v.version_id}{parent}: {quick_r} {full_r} {full_auc}")
            if v.changes_from_parent:
                lines.append(f"  Changes: {v.changes_from_parent[:80]}")
            lines.append(f"  Features: {', '.join(v.feature_names[:5])}...")

        return "\n".join(lines)


def create_initial_feature_schema() -> List[FeatureDefinition]:
    """Create the initial 9-feature schema from experiment_a.

    Returns:
        List of FeatureDefinition objects for the baseline features
    """
    return [
        FeatureDefinition(
            name="fix_in_description",
            description="Does the problem statement contain or hint at the solution?",
            scale_low="No hint at the solution at all",
            scale_high="Exact code fix or detailed solution provided",
            min_value=0,
            max_value=3,
            extraction_prompt="""### Fix Information in Description (fix_in_description: 0-3)
Does the problem statement contain or hint at the solution?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix or detailed solution provided""",
        ),
        FeatureDefinition(
            name="problem_clarity",
            description="How clear and well-specified is the problem?",
            scale_low="Very vague, unclear what's actually wrong",
            scale_high="Crystal clear with reproduction steps and expected behavior",
            min_value=1,
            max_value=5,
            extraction_prompt="""### Problem Clarity (problem_clarity: 1-5)
How clear and well-specified is the problem?
- 1: Very vague, unclear what's actually wrong
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity
- 4: Clear with good context
- 5: Crystal clear with reproduction steps and expected behavior""",
        ),
        FeatureDefinition(
            name="error_message_provided",
            description="Does the problem include an error message or traceback?",
            scale_low="No error message provided",
            scale_high="Error message, traceback, or exception shown",
            min_value=0,
            max_value=1,
            extraction_prompt="""### Error Message/Traceback (error_message_provided: 0/1)
Does the problem include an error message or traceback?
- 0: No error message provided
- 1: Error message, traceback, or exception shown""",
        ),
        FeatureDefinition(
            name="reproduction_steps",
            description="Are concrete reproduction steps provided?",
            scale_low="No concrete reproduction steps",
            scale_high="Has reproduction steps (code snippet, test case, or commands)",
            min_value=0,
            max_value=1,
            extraction_prompt="""### Reproduction Steps (reproduction_steps: 0/1)
Are concrete reproduction steps provided?
- 0: No concrete reproduction steps
- 1: Has reproduction steps (code snippet, test case, or commands)""",
        ),
        FeatureDefinition(
            name="fix_locality",
            description="How localized is the fix based on the patch?",
            scale_low="Single location, few lines changed (1-5 lines)",
            scale_high="Multiple files or significant changes (>20 lines)",
            min_value=1,
            max_value=3,
            extraction_prompt="""### Fix Locality (fix_locality: 1-3)
How localized is the fix based on the patch?
- 1: Single location, few lines changed (1-5 lines)
- 2: Multiple locations in same file, or moderate changes (6-20 lines)
- 3: Multiple files or significant changes (>20 lines)""",
        ),
        FeatureDefinition(
            name="domain_knowledge_required",
            description="How much specialized knowledge is needed to understand and fix this?",
            scale_low="Basic Python, obvious fix anyone could make",
            scale_high="Obscure APIs, protocols, or highly specialized domain knowledge",
            min_value=1,
            max_value=5,
            extraction_prompt="""### Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed to understand and fix this?
- 1: Basic Python, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge""",
        ),
        FeatureDefinition(
            name="fix_complexity",
            description="How complex is the actual fix?",
            scale_low="Trivial (add parameter, change value, simple one-liner)",
            scale_high="Very complex (architectural changes, subtle edge cases, tricky bugs)",
            min_value=1,
            max_value=5,
            extraction_prompt="""### Fix Complexity (fix_complexity: 1-5)
How complex is the actual fix?
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases, tricky bugs)""",
        ),
        FeatureDefinition(
            name="logical_reasoning_required",
            description="How much logical reasoning is needed to arrive at the fix?",
            scale_low="Mechanical fix, no reasoning needed",
            scale_high="Deep reasoning about edge cases, invariants, or system behavior",
            min_value=1,
            max_value=5,
            extraction_prompt="""### Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed to arrive at the fix?
- 1: Mechanical fix, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants, or system behavior""",
        ),
        FeatureDefinition(
            name="atypicality",
            description="How unusual is this bug pattern?",
            scale_low="Very common bug pattern (typo, off-by-one, missing null check)",
            scale_high="Rare or novel bug pattern",
            min_value=1,
            max_value=5,
            extraction_prompt="""### Atypicality (atypicality: 1-5)
How unusual is this bug pattern?
- 1: Very common bug pattern (typo, off-by-one, missing null check)
- 2: Common pattern (incorrect condition, wrong default)
- 3: Moderately unusual
- 4: Unusual bug pattern
- 5: Rare or novel bug pattern""",
        ),
    ]


def generate_prompt_from_schema(features: List[FeatureDefinition]) -> str:
    """Generate feature extraction instructions from feature schema.

    Args:
        features: List of feature definitions

    Returns:
        Complete feature extraction instructions string
    """
    parts = [
        "Analyze the problem statement and gold patch to evaluate these semantic features.",
        "Be precise and consistent with your ratings.",
        "",
    ]

    for i, f in enumerate(features, 1):
        parts.append(f.extraction_prompt)
        parts.append("")

    # Add output format
    feature_json = ",\n    ".join(
        f'"{f.name}": <{f.min_value}-{f.max_value}>'
        for f in features
    )

    parts.extend([
        "## OUTPUT FORMAT",
        "",
        "Respond with ONLY a JSON object containing all features. No markdown, no extra text.",
        "",
        "{",
        f"    {feature_json},",
        '    "reasoning": "<2-3 sentence summary of the key difficulty factors>"',
        "}",
    ])

    return "\n".join(parts)