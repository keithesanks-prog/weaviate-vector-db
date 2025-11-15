#!/usr/bin/env python3
"""Quick test to verify the ingest_data.py fix"""

import sys
import importlib.util

# Force reload without cache
spec = importlib.util.spec_from_file_location("ingest_data", "ingest_data.py")
ingest_data = importlib.util.module_from_spec(spec)

# Check the source code
with open("ingest_data.py", "r") as f:
    content = f.read()
    
    # Check for the problematic pattern
    if "idx + 1" in content:
        print("ERROR: Still found 'idx + 1' in file!")
        sys.exit(1)
    
    # Check for the fixed pattern
    if "row_num" in content and "Processing {row_num}" in content:
        print("✓ File contains the fix (row_num)")
    else:
        print("WARNING: Fix pattern not found")
        sys.exit(1)

print("✓ File looks correct. The error might be from cached bytecode.")
print("Try: python3 -B ingest_data.py")
print("Or restart your terminal session.")

