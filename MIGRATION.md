# Structured feedback migration

Opening an existing event database with drone4rf automatically upgrades it in
place. `EventStore` checks `PRAGMA table_info(events)` and adds these nullable
columns when they are absent:

- `verdict`
- `drone_model`
- `fp_class`
- `fp_class_detail`
- `label_notes`
- `labeled_by`
- `labeled_at`

The existing `user_feedback` column and every existing event row are retained.
Legacy `drone` / `true_positive` labels are mapped to `confirmed_drone` with an
`other/unknown` model, while `false_positive` / `not_drone` labels are mapped to
`false_positive` with an `other` device class. The original flat value remains
unchanged.

No manual command is required: the next `drone4rf web`, scan, GUI, dataset
build, label export, or other `EventStore` open upgrades `data/events.db` inside
its normal SQLite transaction. The database remains in WAL mode. As with any
important field dataset, making a backup before upgrading is prudent.

New labels continue to populate `user_feedback` (`drone`, `false_positive`, or
an empty string for `unsure`) so v0.7-era dataset and GUI consumers keep working.
