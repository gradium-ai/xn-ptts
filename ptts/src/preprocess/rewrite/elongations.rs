use super::split_suffix;
use crate::preprocess::Lang;

pub(super) fn elongations(word: &str, lang: Lang) -> Option<String> {
    if lang != Lang::En {
        return None;
    }
    let (body, suffix) = split_suffix(word);
    let chars: Vec<char> = body.chars().collect();
    let stretched = |c: char| "aeiouyhw".contains(c.to_ascii_lowercase());
    if !chars.iter().all(char::is_ascii_alphabetic)
        || chars.iter().all(char::is_ascii_uppercase)
        || !chars.windows(3).any(|w| w[0] == w[1] && w[1] == w[2] && stretched(w[0]))
    {
        return None;
    }
    let mut out = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let run = i >= 1 && chars[i - 1] == c && stretched(c);
        let long = chars[i.saturating_sub(2)..(i + 3).min(chars.len())]
            .windows(3)
            .any(|w| w.iter().all(|&x| x == c));
        if !(run && long) {
            out.push(c);
        }
    }
    Some(format!("{out}{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elongated_words_are_shortened() {
        let cases = [
            ("sooo", Some("so")),
            ("Nooooo,", Some("No,")),
            ("Hiii", Some("Hi")),
            ("Whyyy", Some("Why")),
            ("Ewww,", Some("Ew,")),
            ("Ahhh!", Some("Ah!")),
            ("good", None),
            ("Zzz", None),
            ("Brrr", None),
            ("WWW", None),
            ("Mmm", None),
        ];
        for (input, expected) in cases {
            assert_eq!(elongations(input, Lang::En).as_deref(), expected, "{input:?}");
        }
    }
}
