"""
Common flow/alert schema, shared by both deployment sides:

  - sensor_agent/zeek_adapter.py and sensor_agent/suricata_adapter.py
    build records that conform to this schema before pushing them to
    the host.
  - flow_store.py (host) uses these field lists to create the SQLite
    tables and validate incoming records.
  - rule_engine.py, correlation_engine.py, train_isolation_forest.py and
    train_baseline.py read only columns declared here — never a raw
    Zeek/Suricata field name directly — so the two sides can't drift
    out of sync silently.

This module is deliberately free of thresholds or config values: it
only defines what columns exist, their types, and which subset is fed
to the ML models. Everything network-specific lives in config.yaml.

Design notes (see conversation for the full reasoning):
  - Flow records come from Zeek conn.log ONLY. Suricata is an alert
    source, matched onto a flow after the fact — it never contributes
    its own flow record, to avoid double-counting the same connection.
  - `community_id` is the primary key used to join a Suricata alert
    back to its Zeek flow record (config.join.primary_key). It is
    optional on a record (not every protocol gets one), which is why
    FLOW_METADATA_COLUMNS lists it as nullable.
  - `conn_state` and `history` are carried as raw strings; the derived
    boolean/numeric features below (is_scan_like, is_established,
    history_len, has_fwd_data, has_bwd_data, is_syn_only) are computed
    by zeek_adapter.py at ingestion time and stored alongside, so every
    downstream consumer sees the same precomputed values instead of
    re-parsing conn_state/history itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str  # "str", "int", "float", "bool" — kept simple, maps 1:1 to SQLite affinities


# ============================================================
# FLOW RECORD — one row per Zeek connection (uid).
# ============================================================

# Identity + context. Not fed to the ML models directly, but required
# for correlation (grouping by src/dst), rule evaluation, and joining
# alerts.
FLOW_METADATA_FIELDS: list[Field] = [
    Field("uid", "str"),            # Zeek connection uid — primary key
    Field("community_id", "str"),   # nullable — join key to Suricata alerts
    Field("ts", "str"),             # flow start time, ISO 8601 UTC
    Field("src_ip", "str"),
    Field("src_port", "int"),
    Field("dst_ip", "str"),
    Field("dst_port", "int"),
    Field("proto", "str"),          # "tcp" / "udp" / "icmp" (Zeek's own string, not CICIDS's numeric code)
    Field("service", "str"),        # Zeek-inferred app protocol, "-" if unknown
    Field("conn_state", "str"),     # raw Zeek conn_state code (S0, SF, REJ, ...)
    Field("history", "str"),        # raw Zeek history string
]

# Numeric features fed to Isolation Forest and the per-host baseline.
# Every one of these is either read directly from conn.log or derived
# from fields conn.log actually provides — nothing here assumes
# CICFlowMeter-style packet-level stats that Zeek doesn't produce.
FLOW_NUMERIC_FEATURE_FIELDS: list[Field] = [
    Field("duration", "float"),             # seconds (Zeek's own unit, NOT microseconds like CICIDS)
    Field("orig_bytes", "float"),
    Field("resp_bytes", "float"),
    Field("orig_pkts", "float"),
    Field("resp_pkts", "float"),
    Field("bytes_per_sec", "float"),        # derived: (orig_bytes+resp_bytes) / max(duration, epsilon)
    Field("pkts_per_sec", "float"),         # derived: (orig_pkts+resp_pkts) / max(duration, epsilon)
    Field("down_up_ratio", "float"),        # derived: resp_pkts / max(orig_pkts, epsilon)
    Field("avg_fwd_segment_size", "float"), # derived: orig_bytes / max(orig_pkts, epsilon)
    Field("avg_bwd_segment_size", "float"), # derived: resp_bytes / max(resp_pkts, epsilon)
]

# Derived from conn_state / history at ingestion time (see zeek_adapter.py).
# Kept separate from FLOW_NUMERIC_FEATURE_FIELDS because they are
# booleans/counts used mainly by the rule engine, not necessarily by
# the Isolation Forest feature vector (schema consumers decide which
# subset they train/evaluate on).
FLOW_DERIVED_FIELDS: list[Field] = [
    Field("is_scan_like", "bool"),      # conn_state in config.rules.port_scan.scan_like_conn_states
    Field("is_established", "bool"),    # conn_state indicates a real two-way session (SF/S1/S2/S3)
    Field("history_len", "int"),        # len(history) — rough proxy for exchange complexity
    Field("has_fwd_data", "bool"),      # lowercase 'd' present in history (originator sent payload)
    Field("has_bwd_data", "bool"),      # uppercase 'D' present in history (responder sent payload)
    Field("is_syn_only", "bool"),       # history indicates a bare SYN / SYN+ack-less probe
]

FLOW_FIELDS: list[Field] = FLOW_METADATA_FIELDS + FLOW_NUMERIC_FEATURE_FIELDS + FLOW_DERIVED_FIELDS
FLOW_COLUMN_NAMES: list[str] = [f.name for f in FLOW_FIELDS]

# Convenience subsets other modules import instead of hardcoding lists.
FLOW_NUMERIC_FEATURE_NAMES: list[str] = [f.name for f in FLOW_NUMERIC_FEATURE_FIELDS]
FLOW_METADATA_NAMES: list[str] = [f.name for f in FLOW_METADATA_FIELDS]
FLOW_DERIVED_NAMES: list[str] = [f.name for f in FLOW_DERIVED_FIELDS]


# ============================================================
# ALERT RECORD — one row per Suricata alert (event_type == "alert").
# Never merged into the flow table; matched onto a flow by
# community_id (fallback: 5-tuple + time) at correlation time.
# ============================================================

ALERT_FIELDS: list[Field] = [
    Field("alert_id", "str"),        # synthetic id (e.g. hash of ts+signature_id+community_id)
    Field("community_id", "str"),    # nullable — primary join key to FLOW_FIELDS.community_id
    Field("ts", "str"),              # alert time, ISO 8601 UTC
    Field("src_ip", "str"),
    Field("src_port", "int"),
    Field("dst_ip", "str"),
    Field("dst_port", "int"),
    Field("proto", "str"),
    Field("signature", "str"),       # alert.signature
    Field("signature_id", "int"),    # alert.signature_id
    Field("category", "str"),        # alert.category
    Field("severity", "int"),        # alert.severity
    Field("mitre_technique", "str"), # nullable — pulled from alert.metadata if the ruleset provides it
    Field("raw_metadata", "str"),    # JSON-encoded alert.metadata, kept for investigation context
]

ALERT_COLUMN_NAMES: list[str] = [f.name for f in ALERT_FIELDS]


# ============================================================
# Validation helpers
# ============================================================

_DTYPE_CHECKERS = {
    "str": lambda v: v is None or isinstance(v, str),
    "int": lambda v: v is None or isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: v is None or isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: v is None or isinstance(v, bool),
}


def _validate(record: dict, fields: list[Field], required: list[str]) -> list[str]:
    errors = []
    for field in fields:
        if field.name not in record:
            if field.name in required:
                errors.append(f"missing required field: {field.name}")
            continue
        value = record[field.name]
        checker = _DTYPE_CHECKERS[field.dtype]
        if not checker(value):
            errors.append(
                f"field {field.name!r} expected dtype {field.dtype!r}, got {type(value).__name__}"
            )
    return errors


# A flow record is useless without identity + the core counters needed
# by every downstream consumer (rules, baseline, Isolation Forest).
FLOW_REQUIRED_FIELDS = [
    "uid", "ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto",
    "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
    "conn_state",
]

ALERT_REQUIRED_FIELDS = [
    "alert_id", "ts", "src_ip", "dst_ip", "proto", "signature", "signature_id", "category",
]


def validate_flow_record(record: dict) -> list[str]:
    """Returns a list of validation error strings; empty list = valid."""
    return _validate(record, FLOW_FIELDS, FLOW_REQUIRED_FIELDS)


def validate_alert_record(record: dict) -> list[str]:
    """Returns a list of validation error strings; empty list = valid."""
    return _validate(record, ALERT_FIELDS, ALERT_REQUIRED_FIELDS)
