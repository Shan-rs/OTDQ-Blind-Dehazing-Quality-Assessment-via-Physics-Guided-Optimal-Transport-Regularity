# Anonymous Release Checklist

- [ ] Confirm the repository contains no datasets, generated outputs, local paths, logs, or private notes.
- [ ] Run `python -m compileall OTDQ.py otdq_eval scripts`.
- [ ] Run the smoke-test commands in `README.md` on a tiny local sample.
- [ ] Replace template paths in `configs/reproduction.json` only in a local copy, or keep dataset paths out of git.
- [ ] Decide whether to add a license file before public release.
- [ ] Confirm paper and repository terminology use the same OTDQ fusion weights as `OTDQ.py`.
