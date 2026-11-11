#!/usr/bin/env python3
"""
Static check that the histograms an analysis script fills match the ones its
config books (output.histograms). Never runs the analysis.

Usage (from alian/alian/):
  python3 ../.claude/skills/add-histogram/check_hists.py analysis/test/embed_perpcone.py config/embed_perpcone.yaml

Reports:
  ERROR    filled in code but not booked in config  -> KeyError at run time
  ERROR    histogram tag differs from its ROOT name  -> plot macros Get() the wrong name
  ERROR    the same ROOT name booked twice           -> one overwrites the other
  ERROR    an axis uses a binning not in output.bins -> crash at init
  WARNING  booked in config but never filled        -> empty histogram in the output
  NOTE     booked names that only an all-variable f-string (f'{a}_{b}_{c}') could fill
A booked name counts as filled if it appears in the script as a string literal
(names are often passed as arguments) or matches an f-string such as
f'{prefix}_rg'. A loose f-string can hide a typo, so read the pattern list too.
Exit code 1 if there is any ERROR.
"""

import re
import sys

import yaml

# self.hists['name'], self.hists["name"] (literal subscripts only)
KEY_RE = re.compile(r"""hists\[\s*(['"])([^'"{}]+?)\1\s*\]""")
# any string literal, f-string or not: names are often passed as arguments,
# e.g. self.do_eec(j, "pp_eec", ...) -> self.hists[hist_name]
STR_RE = re.compile(r"""(f?)(['"])([A-Za-z0-9_{}\[\]\.]+?)\2""")


def code_keys(path):
    """Return (subscripted literals, all string literals, f-string patterns)."""
    subscripts, strings, patterns, loose = set(), set(), set(), set()
    with open(path) as f:
        for line in f:
            code = line.split('#', 1)[0]
            subscripts.update(key for _, key in KEY_RE.findall(code))
            for is_f, _, s in STR_RE.findall(code):
                if is_f and '{' in s:
                    parts = re.split(r'\{[^}]*\}', s)
                    rx = re.compile('^' + '.+'.join(map(re.escape, parts)) + '$')
                    # f'{a}_{b}_{c}' has no fixed text and matches almost anything
                    (patterns if any(len(p) > 1 for p in parts) else loose).add((s, rx))
                else:
                    strings.add(s)
    return subscripts, strings, patterns, loose


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    script, config = sys.argv[1:]
    with open(config) as f:
        cfg = yaml.safe_load(f)
    out = cfg.get('output') or {}
    booked = out.get('histograms') or {}
    bins = out.get('bins') or {}

    errors, warnings = [], []
    subscripts, strings, patterns, loose = code_keys(script)
    notes = []

    for key in sorted(subscripts - set(booked)):
        errors.append(f"filled but not booked in {config}: {key}")

    root_names = {}
    for tag, spec in booked.items():
        _, name, _, *axes = spec
        if name != tag:
            errors.append(f"tag '{tag}' has ROOT name '{name}' (keep them equal)")
        root_names.setdefault(name, []).append(tag)
        for ax in axes:
            if ax not in bins:
                errors.append(f"{tag}: binning '{ax}' not defined in output.bins")
    for name, tags in root_names.items():
        if len(tags) > 1:
            errors.append(f"ROOT name '{name}' booked {len(tags)} times: {tags}")

    for tag in booked:
        if tag in strings:
            continue
        if any(rx.match(tag) for _, rx in patterns):
            continue
        hit = [key for key, rx in loose if rx.match(tag)]
        if hit:
            notes.append(f"{tag}  (only via {', '.join(sorted(hit))})")
        else:
            warnings.append(f"booked but never filled in {script}: {tag}")

    for msg in errors:
        print(f"ERROR    {msg}")
    for msg in warnings:
        print(f"WARNING  {msg}")
    if notes:
        print("\nNOTE  not found as a name; only an f-string built entirely from variables could fill these:")
        for msg in notes:
            print(f"  {msg}")
    print(f"\n{len(booked)} booked. Booked names reached only through these f-strings (check them by eye):")
    for key, rx in sorted(patterns, key=lambda p: p[0]):
        hits = [tag for tag in booked if tag not in strings and rx.match(tag)]
        if hits:
            print(f"  {key}: {len(hits)}")
    if not errors and not warnings:
        print("OK")
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
