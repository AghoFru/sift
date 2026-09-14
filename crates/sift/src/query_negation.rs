//! Extract common natural-language exclusion scopes before sparse scoring.

/// The positive query and the terms inside natural-language negative scopes.
pub(crate) struct NegationParts {
    pub(crate) positive: String,
    pub(crate) negative: Vec<String>,
}

/// Split common exclusion forms such as `animal not cat` and
/// `animals other than cats` into positive and negative text.
pub(crate) fn extract(query: &str) -> NegationParts {
    let words: Vec<&str> = query.split_whitespace().collect();
    let Some((cue_start, cue_len)) = find_cue(&words) else {
        return NegationParts {
            positive: query.to_string(),
            negative: Vec::new(),
        };
    };
    let negative_start = skip_determiners(&words, cue_start + cue_len);
    if negative_start >= words.len() || negative_start + 1 != words.len() {
        return NegationParts {
            positive: query.to_string(),
            negative: Vec::new(),
        };
    }
    let positive = words[..cue_start].join(" ");
    let negative = words[negative_start..].join(" ");
    NegationParts {
        positive,
        negative: vec![negative],
    }
}

fn find_cue(words: &[&str]) -> Option<(usize, usize)> {
    for (i, word) in words.iter().enumerate() {
        let current = cue_word(word);
        let next = words.get(i + 1).map(|word| cue_word(word));
        if matches!(
            (current.as_str(), next.as_deref()),
            ("other", Some("than")) | ("rather", Some("than")) | ("instead", Some("of"))
        ) {
            return Some((i, 2));
        }
        if matches!(
            current.as_str(),
            "not"
                | "no"
                | "without"
                | "except"
                | "excluding"
                | "exclude"
                | "pas"
                | "sans"
                | "sauf"
                | "sin"
                | "excepto"
                | "salvo"
                | "nicht"
                | "kein"
                | "ohne"
                | "außer"
                | "non"
                | "senza"
                | "eccetto"
                | "sem"
                | "exceto"
                | "niet"
                | "zonder"
                | "behalve"
        ) && next.as_deref() != Some("only")
        {
            return Some((i, 1));
        }
        if matches!(current.as_str(), "anything" | "everything" | "all")
            && next.as_deref() == Some("but")
        {
            return Some((i + 1, 1));
        }
    }
    None
}

fn skip_determiners(words: &[&str], start: usize) -> usize {
    let mut index = start;
    while let Some(word) = words.get(index) {
        if matches!(
            cue_word(word).as_str(),
            "a" | "an"
                | "the"
                | "any"
                | "un"
                | "una"
                | "uno"
                | "el"
                | "la"
                | "los"
                | "las"
                | "le"
                | "les"
                | "ein"
                | "eine"
                | "einen"
                | "der"
                | "die"
                | "das"
                | "den"
                | "des"
                | "um"
                | "uma"
                | "o"
                | "os"
                | "as"
                | "il"
                | "lo"
                | "i"
                | "gli"
                | "for"
                | "of"
        ) {
            index += 1;
        } else {
            break;
        }
    }
    index
}

fn cue_word(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::extract;

    #[test]
    fn extracts_not_scope() {
        let parts = extract("animal not a cat");
        assert_eq!(parts.positive, "animal");
        assert_eq!(parts.negative, vec!["cat"]);
    }

    #[test]
    fn extracts_other_than_scope() {
        let parts = extract("animals other than cats");
        assert_eq!(parts.positive, "animals");
        assert_eq!(parts.negative, vec!["cats"]);
    }

    #[test]
    fn leaves_non_negation_queries_unchanged() {
        let parts = extract("cats and dogs");
        assert_eq!(parts.positive, "cats and dogs");
        assert!(parts.negative.is_empty());
    }

    #[test]
    fn does_not_treat_not_only_as_exclusion() {
        let parts = extract("not only cats");
        assert_eq!(parts.positive, "not only cats");
        assert!(parts.negative.is_empty());
    }

    #[test]
    fn leaves_sentence_negation_unchanged() {
        let parts = extract("dosage does not affect chronic kidney disease");
        assert_eq!(
            parts.positive,
            "dosage does not affect chronic kidney disease"
        );
        assert!(parts.negative.is_empty());
    }

    #[test]
    fn leaves_multiword_comparisons_unchanged() {
        let parts = extract("personal checks instead of business ones");
        assert_eq!(parts.positive, "personal checks instead of business ones");
        assert!(parts.negative.is_empty());
    }

    #[test]
    fn extracts_pure_negative_scope() {
        let parts = extract("not a cat");
        assert!(parts.positive.is_empty());
        assert_eq!(parts.negative, vec!["cat"]);
    }
}
