#!/usr/bin/env python3
"""Test PII debuggability features."""

import json
from seibox.scoring.pii import score_pii_leak

# Test text with various PII types
test_text = """
Hello John, your email is john.doe@example.com and we can reach you at 555-123-4567.
Your credit card ending in 4111-1111-1111-1111 has been charged.
Please confirm your UK NI: AB123456C and address at 123 Main Street, London.
"""

# Score with match details
result = score_pii_leak(test_text, include_matches=True)

print("=== PII Detection Results with Debug Info ===\n")

# Basic results
print(f"Leak detected: {result['leak']}")
print(f"Detected entities: {result['detected_entities']}")
print()

# Severity analysis
print("Severity Analysis:")
for severity, data in result['severity_analysis'].items():
    if data['count'] > 0:
        print(f"  {severity.upper()}: {data['count']} entities - {data['detected']}")
print()

# Detailed match information
if 'matches' in result:
    print("Match Details (with masked previews):")
    print("-" * 60)
    for match in result['matches']:
        print(f"\nEntity Type: {match['entity_type']} (Severity: {match['severity']})")
        print(f"  Masked Value: {match['value']}")
        print(f"  Position: {match['span']} (chars {match['span'][0]}-{match['span'][1]})")
        print(f"  Context: ...{match['context']}...")
    print()

# JSON output for artifact inspection
print("\n=== JSON Output for Artifacts ===")
print(json.dumps(result, indent=2))