import re
import glob
from pathlib import Path

# LOG_DIR = "/rstorage/lbergmann/jewel/z_r/vacuum_5020GeV/jewel_logs"
LOG_DIR = "/rstorage/lbergmann/jewel/z_r/medium_recoilON/jewel_logs"

xsec_pattern = re.compile(r"cross section in .+ channel:\s+([\dE+\-\.]+)\s+mb")
weight_pattern = re.compile(r"sum of event weights in .+ channel:\s+([\dE+\-\.]+)")
good_events_pattern = re.compile(r"number of good events:\s+(\d+)")

total_xsec = 0.0
total_weight = 0.0
total_good_events = 0
files_processed = 0
files_with_errors = []

log_files = sorted(glob.glob(f"{LOG_DIR}/*.log"))

for log_file in log_files:
    text = Path(log_file).read_text()

    xsecs = xsec_pattern.findall(text)
    weights = weight_pattern.findall(text)
    good_events = good_events_pattern.findall(text)

    name = Path(log_file).name
    if len(xsecs) != 4 or len(weights) != 4 or len(good_events) != 1:
        files_with_errors.append(
            f"{name}: xsecs={len(xsecs)}, weights={len(weights)}, good_events={len(good_events)}"
        )
        continue

    total_xsec += sum(float(x) for x in xsecs)
    total_weight += sum(float(w) for w in weights)
    total_good_events += int(good_events[0])
    files_processed += 1

print(f"Files processed:       {files_processed} / {len(log_files)}")
print(f"Total cross section:   {total_xsec:.10E} mb")
print(f"Total event weights:   {total_weight:.10E}")
print(f"Total good events:     {total_good_events}")

if files_with_errors:
    print(f"\nFiles skipped ({len(files_with_errors)}):")
    for err in files_with_errors:
        print(f"  {err}")
