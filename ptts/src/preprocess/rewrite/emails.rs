//! Email addresses: "jane.smith@gmail.com" becomes "jane dot smith at gmail dot com".

use super::{spell_short, suffix};
use crate::preprocess::Lang;

/// An email address with its separators spoken in `lang` and a two-letter top-level domain
/// spelled out: "a.b@gra-dium.fr" becomes "a dot b at gra dash dium dot F-R".
pub(super) fn emails(word: &str, lang: Lang) -> Option<String> {
    let (user, rest) = word.split_once('@')?;
    if user.is_empty() || !user.chars().all(|c| c.is_ascii_alphanumeric() || "-_.+".contains(c)) {
        return None;
    }
    // The domain: labels of letters, digits and dashes, separated by dots, at least two of them.
    let (label, mut rest) = label(rest)?;
    let mut labels = vec![label];
    while let Some((label, after)) = rest.strip_prefix('.').and_then(self::label) {
        labels.push(label);
        rest = after;
    }
    let suffix = suffix(rest)?;
    let (tld, domain) = labels.split_last().filter(|(_, domain)| !domain.is_empty())?;

    let sc = lang.special_chars();
    let mut words = vec![];
    speak(user, &mut words, |c| match c {
        '-' => Some(sc.dash),
        '.' => Some(sc.dot),
        '+' => Some(sc.plus),
        '_' => Some(sc.underscore),
        _ => None,
    });
    words.push(sc.at);
    for (i, label) in domain.iter().enumerate() {
        if i > 0 {
            words.push(sc.dot);
        }
        speak(label, &mut words, |c| (c == '-').then_some(sc.dash));
    }
    words.push(sc.dot);
    Some(format!("{} {}{suffix}", words.join(" "), spell_short(tld)))
}

/// The domain label `s` starts with, and the rest of `s`.
fn label(s: &str) -> Option<(&str, &str)> {
    let n = s.bytes().take_while(|&b| b.is_ascii_alphanumeric() || b == b'-').count();
    (n > 0).then(|| s.split_at(n))
}

/// Push the words of `part` onto `words`, with each separator for which `spoken` has a word
/// replaced by that word.
fn speak<'a>(part: &'a str, words: &mut Vec<&'a str>, spoken: impl Fn(char) -> Option<&'a str>) {
    let mut start = 0;
    for (i, c) in part.char_indices() {
        if let Some(word) = spoken(c) {
            if start < i {
                words.push(&part[start..i]);
            }
            words.push(word);
            start = i + c.len_utf8();
        }
    }
    if start < part.len() {
        words.push(&part[start..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_are_read_out() {
        let cases = [
            (Lang::En, "foo@bar.com", "foo at bar dot com"),
            (Lang::En, "foo@bar.baz.com?", "foo at bar dot baz dot com?"),
            (Lang::En, "jane.smith@gmail.com", "jane dot smith at gmail dot com"),
            (Lang::En, "jane.smith@gra-dium.fr?", "jane dot smith at gra dash dium dot F-R?"),
            (Lang::En, "jane+tag@gmail.com", "jane plus tag at gmail dot com"),
            (Lang::En, "first_last@x.co.uk.", "first underscore last at x dot co dot U-K."),
            (Lang::En, "a..b@x.io", "a dot dot b at x dot I-O"),
            (Lang::Fr, "alex@gmail.com", "alex arobaze gmail point com"),
            (Lang::De, "max-mustermann@web.de", "max Bindestrich mustermann ät web Punkt D-E"),
            (Lang::Es, "a_b@x.es", "a guion bajo b arroba x punto E-S"),
            // The domain is not checked against the top-level domains.
            (Lang::En, "foo@bar.xyzzy", "foo at bar dot xyzzy"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(emails(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        for input in
            ["foo@com...", "user@host", "@bar.com", "foo@", "foo@.com", "fé@bar.com", "a@b.c/d"]
        {
            assert_eq!(emails(input, Lang::En), None, "{input:?}");
        }
    }
}
