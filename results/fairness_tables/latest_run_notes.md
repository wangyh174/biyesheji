# Latest Run Notes

- Metadata rows: **72**
- Metadata groups: **4**
- Status rule: `current_aligned` means detector score rows == current metadata rows.
- If status is `stale_or_mixed`, do not use it as current run evidence in thesis tables.

## Sample-size caution
- If each group has very few samples (e.g., 1-3 per label), fairness metrics are unstable.
- Prefer at least dozens of samples per group before drawing substantive fairness conclusions.