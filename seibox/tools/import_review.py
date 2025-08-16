"""Import human review labels and create normalized golden labels."""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import re

from seibox.utils.io import write_jsonl


def normalize_label(label_value: str) -> bool:
    """Normalize various label formats to boolean blocked status."""
    if isinstance(label_value, bool):
        return label_value
    
    label_str = str(label_value).lower().strip()
    
    # Handle various positive cases
    if label_str in ['true', '1', 'yes', 'block', 'blocked', 'refuse', 'refused']:
        return True
    
    # Handle various negative cases  
    if label_str in ['false', '0', 'no', 'allow', 'allowed', 'ok', 'fine']:
        return False
    
    # Try to extract from longer text
    if 'block' in label_str or 'refuse' in label_str or 'bad' in label_str:
        return True
    if 'allow' in label_str or 'ok' in label_str or 'fine' in label_str or 'good' in label_str:
        return False
    
    # Default to False (allow) for unclear cases
    return False


def extract_reviewer_initials(reviewer_field: str) -> str:
    """Extract reviewer initials from various formats."""
    if not reviewer_field:
        return "unknown"
    
    reviewer = str(reviewer_field).strip()
    
    # If it's already initials (2-4 uppercase letters), return as-is
    if re.match(r'^[A-Z]{2,4}$', reviewer):
        return reviewer.lower()
    
    # Extract initials from name
    words = reviewer.split()
    if len(words) >= 2:
        initials = ''.join(word[0].upper() for word in words[:2] if word)
        return initials.lower()
    elif len(words) == 1 and len(words[0]) >= 2:
        # Single word - take first 2 characters
        return words[0][:2].lower()
    
    # Fallback
    return reviewer[:4].lower() if reviewer else "unk"


def import_review(labels_path: str, out_path: str) -> Dict[str, Any]:
    """Import human review labels and create normalized golden labels.
    
    Args:
        labels_path: Path to CSV or JSONL file with human labels
        out_path: Output path for normalized golden labels
        
    Returns:
        Import summary with statistics
    """
    labels_file = Path(labels_path)
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Determine file format
    if labels_file.suffix.lower() == '.csv':
        return _import_csv_labels(labels_file, out_path)
    elif labels_file.suffix.lower() == '.jsonl':
        return _import_jsonl_labels(labels_file, out_path)
    else:
        raise ValueError(f"Unsupported file format: {labels_file.suffix}. Use .csv or .jsonl")


def _import_csv_labels(labels_file: Path, out_path: str) -> Dict[str, Any]:
    """Import labels from CSV format."""
    
    golden_labels = []
    rows_processed = 0
    rows_skipped = 0
    
    with open(labels_file, 'r', encoding='utf-8') as csvfile:
        # Detect delimiter
        sample = csvfile.read(1024)
        csvfile.seek(0)
        
        delimiter = ','
        if sample.count('\t') > sample.count(','):
            delimiter = '\t'
        
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        
        # Validate required columns
        required_cols = {'id'}
        available_cols = set(reader.fieldnames or [])
        
        if not required_cols.issubset(available_cols):
            missing = required_cols - available_cols
            raise ValueError(f"Missing required columns: {missing}. Found: {available_cols}")
        
        # Find label columns (flexible naming)
        label_col = None
        reviewer_col = None
        notes_col = None
        
        for col in available_cols:
            col_lower = col.lower()
            if not label_col and any(term in col_lower for term in ['label', 'blocked', 'block', 'decision', 'verdict']):
                label_col = col
            elif not reviewer_col and any(term in col_lower for term in ['reviewer', 'labeler', 'annotator', 'human']):
                reviewer_col = col
            elif not notes_col and any(term in col_lower for term in ['note', 'comment', 'rationale', 'reason']):
                notes_col = col
        
        if not label_col:
            # Try common column names
            for fallback in ['human_label', 'label', 'blocked', 'decision']:
                if fallback in available_cols:
                    label_col = fallback
                    break
            
            if not label_col:
                raise ValueError(f"No label column found. Expected one of: human_label, label, blocked, decision. Available: {available_cols}")
        
        for row in reader:
            try:
                record_id = row.get('id', '').strip()
                if not record_id:
                    rows_skipped += 1
                    continue
                
                # Extract label
                label_value = row.get(label_col, '')
                if not label_value or str(label_value).strip() == '':
                    rows_skipped += 1
                    continue
                
                blocked = normalize_label(label_value)
                
                # Extract reviewer
                reviewer = "unknown"
                if reviewer_col and row.get(reviewer_col):
                    reviewer = extract_reviewer_initials(row[reviewer_col])
                
                # Extract notes/rationale
                rationale = ""
                if notes_col and row.get(notes_col):
                    rationale = str(row[notes_col]).strip()
                
                # Create normalized label
                golden_label = {
                    "id": record_id,
                    "human_label": {
                        "blocked": blocked,
                        "rationale": rationale
                    },
                    "reviewer": reviewer,
                    "ts": datetime.now(timezone.utc).isoformat()
                }
                
                # For calibration compatibility
                golden_label["record_id"] = record_id
                golden_label["human_blocked"] = blocked
                
                golden_labels.append(golden_label)
                rows_processed += 1
                
            except Exception as e:
                print(f"Warning: Failed to process row {rows_processed + rows_skipped + 1}: {e}")
                rows_skipped += 1
                continue
    
    # Write output
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out_path_obj), golden_labels)
    
    return {
        'format': 'csv',
        'input_path': str(labels_file),
        'output_path': str(out_path_obj),
        'rows_processed': rows_processed,
        'rows_skipped': rows_skipped,
        'label_column': label_col,
        'reviewer_column': reviewer_col,
        'notes_column': notes_col,
        'total_labels': len(golden_labels)
    }


def _import_jsonl_labels(labels_file: Path, out_path: str) -> Dict[str, Any]:
    """Import labels from JSONL format."""
    
    golden_labels = []
    rows_processed = 0
    rows_skipped = 0
    
    with open(labels_file, 'r', encoding='utf-8') as jsonlfile:
        for line_num, line in enumerate(jsonlfile, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Extract required fields
                record_id = data.get('id', '').strip()
                if not record_id:
                    rows_skipped += 1
                    continue
                
                # Look for label in various possible fields
                label_value = None
                for field in ['human_label', 'label', 'blocked', 'decision', 'verdict']:
                    if field in data and data[field] is not None:
                        label_value = data[field]
                        break
                
                if label_value is None:
                    rows_skipped += 1
                    continue
                
                # Handle nested human_label object
                if isinstance(label_value, dict):
                    blocked = normalize_label(label_value.get('blocked', False))
                    rationale = str(label_value.get('rationale', '')).strip()
                else:
                    blocked = normalize_label(label_value)
                    rationale = str(data.get('rationale', data.get('notes', ''))).strip()
                
                # Extract reviewer
                reviewer = "unknown"
                if 'reviewer' in data:
                    reviewer = extract_reviewer_initials(data['reviewer'])
                elif 'labeler' in data:
                    reviewer = extract_reviewer_initials(data['labeler'])
                
                # Use existing timestamp if available
                timestamp = data.get('ts', data.get('timestamp', datetime.now(timezone.utc).isoformat()))
                
                # Create normalized label for human review system
                golden_label = {
                    "id": record_id,
                    "human_label": {
                        "blocked": blocked,
                        "rationale": rationale
                    },
                    "reviewer": reviewer,
                    "ts": timestamp
                }
                
                # For calibration compatibility, add fields expected by kappa calculation
                golden_label["record_id"] = record_id
                # The human decision as blocked/allowed
                golden_label["human_blocked"] = blocked
                
                golden_labels.append(golden_label)
                rows_processed += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                rows_skipped += 1
                continue
            except Exception as e:
                print(f"Warning: Failed to process line {line_num}: {e}")
                rows_skipped += 1
                continue
    
    # Write output
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out_path_obj), golden_labels)
    
    return {
        'format': 'jsonl',
        'input_path': str(labels_file),
        'output_path': str(out_path_obj),
        'rows_processed': rows_processed,
        'rows_skipped': rows_skipped,
        'total_labels': len(golden_labels)
    }