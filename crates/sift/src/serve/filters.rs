//! Per-hit filter predicates evaluated against the stored JSON payload
//! (arbitrary fields), with numeric rank attributes as a fallback source.

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub(crate) struct FilterClause {
    pub field: String,
    #[serde(default)]
    pub lt: Option<f32>,
    #[serde(default)]
    pub lte: Option<f32>,
    #[serde(default)]
    pub gt: Option<f32>,
    #[serde(default)]
    pub gte: Option<f32>,
    /// Exact match. Number, string, or bool (matched against the payload field,
    /// or against a numeric rank attribute).
    #[serde(default)]
    pub eq: Option<serde_json::Value>,
    #[serde(default)]
    pub neq: Option<serde_json::Value>,
    /// Keyword set membership: the field value must equal one of these.
    #[serde(default, rename = "in")]
    pub in_set: Option<Vec<serde_json::Value>>,
    /// Field presence (non-null). `true` requires it, `false` requires absence.
    #[serde(default)]
    pub exists: Option<bool>,
}

/// Numeric view of a JSON value, for range comparisons.
fn json_num(v: &serde_json::Value) -> Option<f64> {
    v.as_f64()
}

/// Equality across JSON scalars, comparing numbers by value.
fn json_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::Number(_), serde_json::Value::Number(_)) => a.as_f64() == b.as_f64(),
        _ => a == b,
    }
}

/// Evaluate one filter clause against a field value (None = field absent).
fn clause_matches(c: &FilterClause, val: Option<&serde_json::Value>) -> bool {
    if let Some(want) = c.exists {
        let present = matches!(val, Some(v) if !v.is_null());
        if present != want {
            return false;
        }
    }
    let has_range = c.lt.is_some() || c.lte.is_some() || c.gt.is_some() || c.gte.is_some();
    if has_range {
        let n = match val.and_then(json_num) {
            Some(n) => n,
            None => return false,
        };
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        {
            if let Some(t) = c.lt {
                if !(n < t as f64) {
                    return false;
                }
            }
            if let Some(t) = c.lte {
                if !(n <= t as f64) {
                    return false;
                }
            }
            if let Some(t) = c.gt {
                if !(n > t as f64) {
                    return false;
                }
            }
            if let Some(t) = c.gte {
                if !(n >= t as f64) {
                    return false;
                }
            }
        }
    }
    if let Some(eq) = &c.eq {
        match val {
            Some(v) if json_eq(v, eq) => {}
            _ => return false,
        }
    }
    if let Some(neq) = &c.neq {
        if let Some(v) = val {
            if json_eq(v, neq) {
                return false;
            }
        }
    }
    if let Some(set) = &c.in_set {
        match val {
            Some(v) if set.iter().any(|x| json_eq(x, v)) => {}
            _ => return false,
        }
    }
    true
}

/// True if a doc passes every filter clause. `payload` is the doc's stored JSON
/// (if any); `rank` looks up a numeric rank attribute as a fallback field source.
pub(crate) fn passes_filters(
    payload: Option<&str>,
    clauses: &[FilterClause],
    rank: impl Fn(&str) -> Option<f32>,
) -> bool {
    let pv: Option<serde_json::Value> = payload.and_then(|s| serde_json::from_str(s).ok());
    clauses.iter().all(|c| {
        let val: Option<serde_json::Value> = pv
            .as_ref()
            .and_then(|o| o.get(&c.field).cloned())
            .or_else(|| rank(&c.field).map(serde_json::Value::from));
        clause_matches(c, val.as_ref())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clause(field: &str) -> FilterClause {
        FilterClause {
            field: field.to_string(),
            lt: None,
            lte: None,
            gt: None,
            gte: None,
            eq: None,
            neq: None,
            in_set: None,
            exists: None,
        }
    }

    #[test]
    fn combines_scalar_and_range_predicates() {
        let mut brand = clause("brand");
        brand.in_set = Some(vec!["acme".into(), "globex".into()]);
        let mut price = clause("price");
        price.gte = Some(10.0);
        price.lt = Some(20.0);
        let payload = r#"{"brand":"acme","price":12.5}"#;

        assert!(passes_filters(payload.into(), &[brand, price], |_| None));
    }

    #[test]
    fn missing_and_null_have_consistent_exists_semantics() {
        let mut absent = clause("missing");
        absent.exists = Some(false);
        let mut null = clause("nullable");
        null.exists = Some(false);

        assert!(passes_filters(
            Some(r#"{"nullable":null}"#),
            &[absent, null],
            |_| None
        ));
    }

    #[test]
    fn rank_attribute_is_used_when_payload_field_is_absent() {
        let mut rating = clause("rating");
        rating.gte = Some(4.0);

        assert!(passes_filters(Some("{}"), &[rating], |field| {
            (field == "rating").then_some(4.5)
        }));
    }

    #[test]
    fn malformed_payload_fails_value_predicates() {
        let mut brand = clause("brand");
        brand.eq = Some("acme".into());

        assert!(!passes_filters(Some("not-json"), &[brand], |_| None));
    }
}
