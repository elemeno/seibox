"""Pack loader and registry for portable prompt bundles."""

import json
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from seibox.utils.prompt_spec import PromptSpec
from seibox.utils.io import read_jsonl, write_jsonl


@dataclass
class PackCategory:
    """Category definition within a pack."""

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    prompts: str = "prompts.jsonl"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class PackMeta:
    """Metadata for a prompt pack."""

    id: str
    version: str
    path: Path
    name: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    categories: List[PackCategory] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        # Convert dict categories to PackCategory objects
        if self.categories and isinstance(self.categories[0], dict):
            self.categories = [PackCategory(**cat) for cat in self.categories]

    @property
    def prompt_count(self) -> int:
        """Count total prompts across all categories."""
        count = 0
        for category in self.categories:
            prompts_file = self.path / category.prompts
            if prompts_file.exists():
                with open(prompts_file, "r") as f:
                    count += sum(1 for line in f if line.strip())
        return count


def discover_packs(root: str = "packs/") -> List[PackMeta]:
    """Discover all available packs in the root directory.

    Args:
        root: Root directory to search for packs

    Returns:
        List of PackMeta objects for discovered packs
    """
    packs = []
    root_path = Path(root)

    if not root_path.exists():
        return packs

    # Look for pack directories
    for pack_dir in root_path.iterdir():
        if not pack_dir.is_dir():
            continue

        pack_yaml = pack_dir / "pack.yaml"
        if not pack_yaml.exists():
            continue

        try:
            # Load pack metadata
            with open(pack_yaml, "r") as f:
                data = yaml.safe_load(f)

            # Create PackMeta object
            pack = PackMeta(
                id=data.get("id", pack_dir.name),
                version=data.get("version", "0.0.0"),
                path=pack_dir,
                name=data.get("name"),
                author=data.get("author"),
                description=data.get("description"),
                license=data.get("license"),
                categories=data.get("categories", []),
            )

            packs.append(pack)

        except Exception as e:
            print(f"Warning: Failed to load pack from {pack_dir}: {e}")
            continue

    return sorted(packs, key=lambda p: p.id)


def load_pack(path: str, category: Optional[str] = None) -> List[PromptSpec]:
    """Load prompts from a pack.

    Args:
        path: Path to pack directory or pack.yaml
        category: Optional category filter

    Returns:
        List of PromptSpec objects
    """
    pack_path = Path(path)

    # Handle both directory and file paths
    if pack_path.is_file() and pack_path.name == "pack.yaml":
        pack_dir = pack_path.parent
        pack_yaml = pack_path
    else:
        pack_dir = pack_path
        pack_yaml = pack_dir / "pack.yaml"

    if not pack_yaml.exists():
        raise FileNotFoundError(f"No pack.yaml found in {pack_dir}")

    # Load pack metadata
    with open(pack_yaml, "r") as f:
        pack_data = yaml.safe_load(f)

    prompts = []
    categories = pack_data.get("categories", [])

    for cat_data in categories:
        # Skip if filtering by category and doesn't match
        if category and cat_data.get("id") != category:
            continue

        # Load prompts file
        prompts_file = pack_dir / cat_data.get("prompts", "prompts.jsonl")
        if not prompts_file.exists():
            print(f"Warning: Prompts file not found: {prompts_file}")
            continue

        # Parse prompts
        with open(prompts_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # Add pack source metadata
                    if "metadata" not in data:
                        data["metadata"] = {}
                    data["metadata"]["source"] = f"pack:{pack_data.get('id', pack_dir.name)}"
                    data["metadata"]["pack_version"] = pack_data.get("version", "0.0.0")

                    # Create PromptSpec
                    spec = PromptSpec(**data)
                    prompts.append(spec)

                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num} in {prompts_file}: {e}")
                except Exception as e:
                    print(f"Warning: Failed to parse prompt at line {line_num}: {e}")

    return prompts


def import_pack(
    pack_id: str, category: str, dest: str, dedupe: bool = True, preview: bool = False
) -> Dict[str, Any]:
    """Import prompts from a pack into a dataset directory.

    Args:
        pack_id: ID of the pack to import
        category: Category to import
        dest: Destination directory
        dedupe: Whether to deduplicate by prompt ID
        preview: If True, don't write files, just return summary

    Returns:
        Import summary with statistics
    """
    # Find the pack
    packs = discover_packs()
    pack = None
    for p in packs:
        if p.id == pack_id:
            pack = p
            break

    if not pack:
        raise ValueError(f"Pack not found: {pack_id}")

    # Load prompts from pack
    pack_prompts = load_pack(str(pack.path), category=category)

    if not pack_prompts:
        raise ValueError(f"No prompts found for category '{category}' in pack '{pack_id}'")

    dest_path = Path(dest)
    dest_file = dest_path / "prompts.jsonl"

    # Load existing prompts if file exists
    existing_prompts = []
    existing_ids = set()

    if dest_file.exists():
        with open(dest_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    spec = PromptSpec(**data)
                    existing_prompts.append(spec)
                    existing_ids.add(spec.id)
                except Exception:
                    continue

    # Process imports
    new_prompts = []
    duplicates = []

    for prompt in pack_prompts:
        if dedupe and prompt.id in existing_ids:
            duplicates.append(prompt.id)
        else:
            new_prompts.append(prompt)
            existing_ids.add(prompt.id)

    # Prepare summary
    summary = {
        "pack_id": pack_id,
        "pack_version": pack.version,
        "category": category,
        "destination": str(dest_file),
        "existing_count": len(existing_prompts),
        "imported_count": len(new_prompts),
        "duplicate_count": len(duplicates),
        "total_count": len(existing_prompts) + len(new_prompts),
        "duplicates": duplicates[:10] if duplicates else [],  # Show first 10
        "preview_mode": preview,
    }

    # Write merged prompts if not preview
    if not preview:
        dest_path.mkdir(parents=True, exist_ok=True)

        # Combine existing and new prompts
        all_prompts = existing_prompts + new_prompts

        # Write to file
        write_jsonl(str(dest_file), all_prompts)

        # Also create a backup of original if it existed
        if existing_prompts and len(new_prompts) > 0:
            backup_file = dest_file.with_suffix(".jsonl.bak")
            write_jsonl(str(backup_file), existing_prompts)
            summary["backup_created"] = str(backup_file)

    return summary


def validate_pack(pack_path: str) -> Dict[str, Any]:
    """Validate a pack's structure and contents.

    Args:
        pack_path: Path to pack directory

    Returns:
        Validation results
    """
    results = {"valid": True, "errors": [], "warnings": [], "stats": {}}

    pack_dir = Path(pack_path)

    # Check pack.yaml exists
    pack_yaml = pack_dir / "pack.yaml"
    if not pack_yaml.exists():
        results["valid"] = False
        results["errors"].append("Missing pack.yaml")
        return results

    # Load and validate pack.yaml
    try:
        with open(pack_yaml, "r") as f:
            pack_data = yaml.safe_load(f)
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Invalid pack.yaml: {e}")
        return results

    # Check required fields
    required_fields = ["id", "version", "categories"]
    for field in required_fields:
        if field not in pack_data:
            results["valid"] = False
            results["errors"].append(f"Missing required field: {field}")

    # Validate categories and prompts
    categories = pack_data.get("categories", [])
    total_prompts = 0

    for cat in categories:
        if "id" not in cat:
            results["errors"].append("Category missing 'id' field")
            continue

        prompts_file = pack_dir / cat.get("prompts", "prompts.jsonl")
        if not prompts_file.exists():
            results["errors"].append(f"Prompts file not found: {prompts_file}")
            continue

        # Validate prompts
        cat_prompts = 0
        with open(prompts_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    spec = PromptSpec(**data)
                    cat_prompts += 1
                except json.JSONDecodeError:
                    results["warnings"].append(f"Invalid JSON at {prompts_file}:{line_num}")
                except Exception as e:
                    results["warnings"].append(f"Invalid prompt at {prompts_file}:{line_num}: {e}")

        total_prompts += cat_prompts
        results["stats"][f"category_{cat['id']}_prompts"] = cat_prompts

    results["stats"]["total_prompts"] = total_prompts

    # Check for recommended fields
    recommended = ["name", "author", "description", "license"]
    for field in recommended:
        if field not in pack_data:
            results["warnings"].append(f"Recommended field missing: {field}")

    if results["errors"]:
        results["valid"] = False

    return results
