# Packages added in 0.2.0 (shipped scope)

Subset of the parent `.licenses/` artifacts: packages in the `v0.2.0` inventory
but not the `v0.1.0` inventory (matched by normalized name; version-only changes
excluded), restricted to packages reachable from the runtime dependencies and
optional extras (`aimnet`, `ase`, `cu12`, `cu13`, `mace`, `pymatgen`,
`tensorboard`, `uma`).

Excluded: packages reachable only through the `dev`, `docs`, `build` and
`distribution` dependency groups (for example Sphinx and its extensions,
pytest-cov, twine, pyarmor).

Same format as the parent files: `summary.md`, `details.json`,
`Third_party_attr.txt`.
