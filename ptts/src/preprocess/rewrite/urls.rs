//! Web addresses: `https://www.kyutai.fr` becomes "H-T-T-P-S colon slash slash W-W-W dot kyutai
//! dot F-R".

use super::tlds::is_tld;
use super::{letters, spell_short, suffix};
use crate::preprocess::{Lang, SpecialChars};
use std::borrow::Cow;

/// A web address, with an optional `http://` or `https://` in front and an optional path of
/// letters, digits, dashes and slashes after it, read with its separators spoken in `lang`. Its
/// last label must be a top-level domain, which is what tells "kyutai.fr" from a sentence that
/// lost the space after its period. So must its case: "it.Now" is such a sentence, unless a
/// scheme or `www.` says it is an address.
pub(super) fn urls(word: &str, lang: Lang) -> Option<String> {
    let sc = lang.special_chars();
    let mut words: Vec<Cow<str>> = vec![];
    let rest = if let Some(rest) = word.strip_prefix("https://") {
        words.push("H-T-T-P-S".into());
        rest
    } else if let Some(rest) = word.strip_prefix("http://") {
        words.push("H-T-T-P".into());
        rest
    } else {
        word
    };
    if !words.is_empty() {
        words.extend([sc.colon, sc.slash, sc.slash].map(Cow::from));
    }

    let (label, mut rest) = run(rest, |c| c.is_alphanumeric() || c == '-')?;
    let mut labels = vec![label];
    while let Some((label, after)) =
        rest.strip_prefix('.').and_then(|r| run(r, |c| c.is_alphanumeric() || c == '-'))
    {
        labels.push(label);
        rest = after;
    }
    let (path, rest) = match run(rest, |c| c.is_alphanumeric() || c == '-' || c == '/') {
        Some((path, rest)) => (Some(path), rest),
        None => (None, rest),
    };
    let suffix = suffix(rest)?;
    let tld = labels[labels.len() - 1];
    if labels.len() < 2 || !is_tld(tld) {
        return None;
    }
    let capitalized = tld.chars().any(char::is_uppercase) && tld.chars().any(char::is_lowercase);
    if capitalized && words.is_empty() && !labels[0].eq_ignore_ascii_case("www") {
        return None;
    }
    // The languages with letter names of their own spell with them.
    if letters::names(lang).is_some() {
        let address = &word[..word.len() - suffix.len()];
        return Some(format!("{}{suffix}", letters::read(address, lang)));
    }

    for (i, label) in labels.into_iter().enumerate() {
        if i > 0 {
            words.push(sc.dot.into());
        }
        speak(label, sc, &mut words);
    }
    if let Some(path) = path {
        speak(path, sc, &mut words);
    }
    Some(format!("{}{suffix}", words.join(" ")))
}

/// The run of characters `s` starts with that `keep` accepts, and the rest of `s`; `None` when it
/// is empty.
fn run(s: &str, keep: impl Fn(char) -> bool) -> Option<(&str, &str)> {
    let n = s.find(|c| !keep(c)).unwrap_or(s.len());
    (n > 0).then(|| s.split_at(n))
}

/// Push the words of a label or a path onto `words`: dashes and slashes spoken, each ASCII digit
/// a word of its own, and two-letter words spelled out.
fn speak<'a>(part: &'a str, sc: &SpecialChars, words: &mut Vec<Cow<'a, str>>) {
    let mut start = 0;
    for (i, c) in part.char_indices() {
        let word = match c {
            '-' => sc.dash,
            '/' => sc.slash,
            '0'..='9' => &part[i..=i],
            _ => continue,
        };
        if start < i {
            words.push(spell_short(&part[start..i]));
        }
        words.push(word.into());
        start = i + 1;
    }
    if start < part.len() {
        words.push(spell_short(&part[start..]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_are_read_out() {
        let cases = [
            (Lang::En, "www.example.com", "W-W-W dot example dot com"),
            (Lang::En, "www.example.com/", "W-W-W dot example dot com slash"),
            (
                Lang::En,
                "https://www.example.com/path/to-page/",
                "H-T-T-P-S colon slash slash W-W-W dot example dot com slash path slash T-O dash \
                 page slash",
            ),
            (
                Lang::En,
                "http://sub.domain.co.uk",
                "H-T-T-P colon slash slash sub dot domain dot C-O dot U-K",
            ),
            (
                Lang::Fr,
                "https://www.kyutai.fr",
                "H-T-T-P-S deux-points slash slash W-W-W point kyutai point F-R",
            ),
            (
                Lang::Fr,
                "www.it-management.com/promo",
                "W-W-W point I-T tiret management point com slash promo",
            ),
            (Lang::En, "example.com.", "example dot com."),
            (Lang::En, "abc123.io/v2", "abc 1 2 3 dot I-O slash v 2"),
            (Lang::En, "hello.world", "hello dot world"),
            (Lang::De, "café.de", "café Punkt D-E"),
            // Domains the top-level domain list once had a stray comma in.
            (Lang::En, "abc.net.au", "abc dot net dot A-U"),
            (Lang::En, "gmail.gmail", "gmail dot gmail"),
            // A capitalized domain is an address when a scheme or `www.` says so.
            (Lang::En, "https://kyutai.Fr", "H-T-T-P-S colon slash slash kyutai dot F-R"),
            (Lang::En, "www.example.Com", "W-W-W dot example dot Com"),
            (Lang::En, "EXAMPLE.COM", "EXAMPLE dot COM"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(urls(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        let declined = [
            "test.",
            "3.14",
            "1.234",
            "e.g.",
            "example.html",
            "example.com/index.html",
            "example.com?q=1",
            "ftp://example.com",
            "https://",
            "example..com",
            "foo@bar.com",
            // Sentences that lost the space after their period.
            "it.Now",
            "yes.No",
        ];
        for input in declined {
            assert_eq!(urls(input, Lang::En), None, "{input:?}");
        }
    }
}
