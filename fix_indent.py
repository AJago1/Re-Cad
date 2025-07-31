#!/usr/bin/env python3
"""Fix indentation error on line 3926"""

with open('stl_analyzer/gui.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 3926 (index 3925) - add proper indentation
if len(lines) > 3925:
    current_line = lines[3925]
    if 'metadata = json.load(f)' in current_line and not current_line.startswith('                        '):
        # Fix the indentation - should be 24 spaces (6 levels of 4 spaces)
        lines[3925] = '                        metadata = json.load(f)\n'
        print('Fixed indentation on line 3926')
    else:
        print(f'Line 3926 already fixed or content changed: {repr(current_line[:50])}')

# Write the corrected file
with open('stl_analyzer/gui.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('File saved') 