//! Numbers read out with their scale words.

use super::{all_digits, split_suffix, suffix};
use crate::preprocess::Lang;

/// Large numbers with their scale words: "1,234.56" becomes "1 thousand 234 point 56", and in
/// French "1234,56" becomes "mille 234 virgule 56". Values under a thousand and years stay as
/// digits, and a leading zero marks a code, not a quantity.
pub(super) fn numbers(word: &str, lang: Lang) -> Option<String> {
    let (body, end) = split_suffix(word);
    if body.len() >= 2 && body.starts_with('0') && all_digits(body) {
        return Some(format!("{}{end}", spell(body, lang)));
    }
    if let Some(range) = range(body, lang) {
        return Some(format!("{range}{end}"));
    }
    // The Spanish ordinals written "1.er" and "3.er", which the URL rule would otherwise read
    // as a domain.
    if lang == Lang::Es {
        match body.strip_suffix("er").map(|n| n.strip_suffix('.').unwrap_or(n)) {
            Some("1") => return Some(format!("primer{end}")),
            Some("3") => return Some(format!("tercer{end}")),
            _ => {}
        }
    }
    if let Some((words, suffix)) = number_words(word, lang) {
        return Some(format!("{words}{suffix}"));
    }
    let (head, tail) = word.split_at(word.find('-')?);
    if !tail[1..].starts_with(|c: char| c.is_alphabetic()) {
        return None;
    }
    match number_words(head, lang)? {
        (words, "") => Some(format!("{words}{tail}")),
        _ => None,
    }
}

fn range(body: &str, lang: Lang) -> Option<String> {
    let (from, to) = body.split_once('-')?;
    // German ranges start from a single digit too: "8-10 Stunden", "0-2 Jahren".
    let shortest = if lang == Lang::De { 1 } else { 2 };
    let small = |s: &str| {
        (shortest..=3).contains(&s.len()) && all_digits(s) && (s.len() == 1 || !s.starts_with('0'))
    };
    let year =
        |s: &str| s.len() == 4 && all_digits(s) && (s.starts_with("1") || s.starts_with("20"));
    let ok = match (from.len(), to.len()) {
        _ if small(from) && small(to) => from.parse::<u32>().ok()? < to.parse::<u32>().ok()?,
        (4, 4) => year(from) && year(to) && from < to,
        (4, 2) => year(from) && all_digits(to) && &from[2..] < to,
        _ => false,
    };
    if !ok {
        return None;
    }
    let word = match lang {
        Lang::En => "to",
        Lang::Fr => "à",
        Lang::De => "bis",
        Lang::Es | Lang::Pt => "a",
    };
    Some(format!("{from} {word} {to}"))
}

fn spell(s: &str, lang: Lang) -> String {
    let words: [&str; 10] = match lang {
        Lang::En => {
            ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        }
        Lang::Fr => {
            ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]
        }
        Lang::De => {
            ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
        }
        Lang::Es => {
            ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
        }
        Lang::Pt => {
            ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove"]
        }
    };
    let spoken = s.chars().filter(|&c| c != '-').map(|c| match c.to_digit(10) {
        Some(d) => words[d as usize].to_string(),
        None => c.to_string(),
    });
    spoken.collect::<Vec<_>>().join(" ")
}

pub(super) fn digit_run(words: &[&str], lang: Lang) -> Option<(String, usize)> {
    let group = |s: &str| {
        (1..=8).contains(&s.len())
            && s.bytes().all(|b| b.is_ascii_digit() || b.is_ascii_uppercase())
            && s.bytes().any(|b| b.is_ascii_digit())
    };
    let plus = words.first()?.starts_with('+');
    let mut bodies = vec![];
    let mut end = "";
    for word in words {
        let (body, suffix) = split_suffix(word);
        let body = if bodies.is_empty() && plus { &body[1..] } else { body };
        if !group(body) {
            break;
        }
        bodies.push(body);
        if !suffix.is_empty() {
            end = suffix;
            break;
        }
    }
    let years = bodies.iter().all(|b| b.len() == 4 && matches!(&b[..2], "19" | "20"));
    let long = bodies.len() >= 3 && !years;
    if !long && !(bodies.len() == 2 && (plus || bodies.iter().any(|b| b.starts_with('0')))) {
        return None;
    }
    let spelled = bodies.iter().map(|b| spell(b, lang)).collect::<Vec<_>>().join(", ");
    let spelled = if plus { format!("{} {spelled}", lang.special_chars().plus) } else { spelled };
    Some((format!("{spelled}{end}"), bodies.len()))
}

pub(super) fn ending_in(prev: Option<&str>, words: &[&str], lang: Lang) -> Option<(String, usize)> {
    let prev = prev?.to_lowercase();
    if !matches!(prev.as_str(), "ending" | "ends" | "ended") || *words.first()? != "in" {
        return None;
    }
    let (body, end) = split_suffix(words.get(1)?);
    let code = body.strip_prefix('-')?;
    let ok = |b: u8| b.is_ascii_digit() || b.is_ascii_uppercase() || b == b'-';
    if !code.bytes().any(|b| b.is_ascii_digit()) || !code.bytes().all(ok) {
        return None;
    }
    Some((format!("in {}{end}", spell(code, lang)), 2))
}

/// What [`numbers`] reads `word` as, and the trailing punctuation it carries through, kept apart
/// so that an amount of money can put its currency in between.
pub(super) fn number_words(word: &str, lang: Lang) -> Option<(String, &str)> {
    let mut lead = word.chars();
    if lead.next() == Some('0') && lead.next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    // English groups thousands with commas and marks decimals with a dot. French does neither.
    // The other three mark decimals with a comma and may group with dots, but a dot only groups
    // when every group after the first has three digits: "1.234" is a thousand, "1.23" is not a
    // number at all.
    let (group, point) = match lang {
        Lang::En => (Some(','), '.'),
        Lang::Fr => (None, ','),
        _ => {
            let int = word.rsplit_once(',').map_or(word, |(before, _)| before);
            let grouped = int.contains('.')
                && int.split('.').enumerate().all(|(i, g)| i == 0 || g.len() == 3);
            (grouped.then_some('.'), ',')
        }
    };
    let (negative, rest) = match word.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, word),
    };
    let end = rest.find(|c: char| !c.is_ascii_digit() && Some(c) != group).unwrap_or(rest.len());
    let int = rest[..end].trim_end_matches(|c: char| Some(c) == group);
    if int.is_empty() {
        return None;
    }
    let rest = &rest[int.len()..];
    let (decimal, rest) = match rest.strip_prefix(point) {
        Some(after) => match after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len()) {
            0 => (None, rest),
            end => (Some(&after[..end]), &after[end..]),
        },
        None => (None, rest),
    };
    let suffix = suffix(rest)?;
    let digits: String = int.chars().filter(char::is_ascii_digit).collect();
    let value: i64 = digits.parse().ok()?;
    if value >= 1_000_000_000_000 {
        return None;
    }
    let mut words = if value < 1000 || (1900..2100).contains(&value) {
        value.to_string()
    } else {
        // Scale words from a milliard down, singular and plural.
        let scales = match lang {
            Lang::En => [
                (1_000_000_000, "billion", "billion"),
                (1_000_000, "million", "million"),
                (1000, "thousand", "thousand"),
            ],
            Lang::Fr => [
                (1_000_000_000, "milliard", "milliards"),
                (1_000_000, "million", "millions"),
                (1000, "mille", "mille"),
            ],
            Lang::De => [
                (1_000_000_000, "Milliarde", "Milliarden"),
                (1_000_000, "Million", "Millionen"),
                (1000, "Tausend", "Tausend"),
            ],
            Lang::Es => [
                (1_000_000_000, "mil millones", "mil millones"),
                (1_000_000, "millón", "millones"),
                (1000, "mil", "mil"),
            ],
            Lang::Pt => [
                (1_000_000_000, "bilhão", "bilhões"),
                (1_000_000, "milhão", "milhões"),
                (1000, "mil", "mil"),
            ],
        };
        let mut parts: Vec<String> = scales
            .iter()
            .filter_map(|(div, one, many)| match (value / div) % 1000 {
                0 => None,
                1 => Some(match (lang, *div) {
                    // "mille" and "mil" are numeral adjectives: no "1" in front of them.
                    (Lang::Fr | Lang::Es | Lang::Pt, 1000) => (*one).to_string(),
                    // A bare "1" would be read "eins", so German inflects the count like an
                    // article, with the gender of the scale noun.
                    (Lang::De, 1000) => format!("ein {one}"),
                    (Lang::De, _) => format!("eine {one}"),
                    _ => format!("1 {one}"),
                }),
                group => Some(format!("{group} {many}")),
            })
            .collect();
        if value % 1000 != 0 {
            parts.push((value % 1000).to_string());
        }
        parts.join(" ")
    };
    if let Some(decimal) = decimal {
        words = format!("{words} {} {decimal}", lang.decimal_separator());
    }
    if negative {
        let minus = match lang {
            Lang::Fr => "moins",
            Lang::Es | Lang::Pt => "menos",
            _ => "minus",
        };
        words = format!("{minus} {words}");
    }
    Some((words, suffix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_read_out() {
        let cases = [
            ("0", Some("0")),
            ("123", Some("123")),
            ("2024", Some("2024")),
            ("1234", Some("1 thousand 234")),
            ("1,234.56...", Some("1 thousand 234 point 56...")),
            ("3.14", Some("3 point 14")),
            ("12.10", Some("12 point 10")),
            ("-12.00", Some("minus 12 point 00")),
            ("1000000", Some("1 million")),
            ("2500000", Some("2 million 500 thousand")),
            ("2,500,000,", Some("2 million 500 thousand,")),
            ("1002003004", Some("1 billion 2 million 3 thousand 4")),
            ("-4500", Some("minus 4 thousand 500")),
            ("-12.", Some("minus 12.")),
            // Not quantities: a code, a time, a range, a unit, a value past a trillion.
            ("007", Some("zero zero seven")),
            ("0600.", Some("zero six zero zero.")),
            ("7,000-ton", Some("7 thousand-ton")),
            ("20-year-old", Some("20-year-old")),
            ("20-25", Some("20 to 25")),
            ("10-5", None),
            ("22-22", None),
            ("3-5", None),
            ("12:30", None),
            ("10-15", Some("10 to 15")),
            ("1445-1450", Some("1445 to 1450")),
            ("2024-25", Some("2024 to 25")),
            ("1999-1998", None),
            ("3.5%", None),
            ("1234567890123", None),
            ("hello", None),
            ("", None),
            ("12,345", Some("12 thousand 345")),
            // Commas are stripped wherever they fall, and a grouped number can still look like
            // a year. Both are quirks, pinned here because the readings have to stay identical
            // to the ones the serving stack already speaks.
            ("1,2,3", Some("123")),
            ("1,23", Some("123")),
            ("1234,567", Some("1 million 234 thousand 567")),
            ("1,999", Some("1999")),
            ("0,123", Some("123")),
        ];
        for (input, expected) in cases {
            assert_eq!(numbers(input, Lang::En).as_deref(), expected, "{input:?}");
        }

        // Outside English a comma is the decimal mark, not a thousands separator, and the
        // count in front of a scale word follows the language's grammar.
        let rest = [
            (Lang::Fr, "1234", Some("mille 234")),
            (Lang::Fr, "2234", Some("2 mille 234")),
            (Lang::Fr, "2000000", Some("2 millions")),
            (Lang::Fr, "2001000", Some("2 millions mille")),
            (Lang::Fr, "1500000", Some("1 million 500 mille")),
            (Lang::Fr, "2024", Some("2024")),
            (Lang::Fr, "-4500", Some("moins 4 mille 500")),
            (Lang::Fr, "1234,56", Some("mille 234 virgule 56")),
            (Lang::Fr, "1,234", Some("1 virgule 234")),
            (Lang::De, "1000", Some("ein Tausend")),
            (Lang::De, "1000000", Some("eine Million")),
            (Lang::De, "1000000000", Some("eine Milliarde")),
            (Lang::De, "2001000", Some("2 Millionen ein Tausend")),
            (Lang::De, "-1000", Some("minus ein Tausend")),
            (Lang::De, "1,234", Some("1 Komma 234")),
            // A dot groups thousands only when the groups after the first have three digits.
            (Lang::De, "1.042", Some("ein Tausend 42")),
            (Lang::De, "1.234,567", Some("ein Tausend 234 Komma 567")),
            (Lang::De, "1.23", None),
            (Lang::De, "1.2.3", None),
            // A trailing sentence period breaks the grouping, so the number is left as written.
            (Lang::De, "1.500.", None),
            (Lang::Es, "1000000", Some("1 millón")),
            (Lang::Es, "2001000", Some("2 millones mil")),
            (Lang::Es, "-4500", Some("menos 4 mil 500")),
            (Lang::Es, "1.234,5", Some("mil 234 coma 5")),
            (Lang::Pt, "1000000", Some("1 milhão")),
            (Lang::Pt, "2000000", Some("2 milhões")),
            (Lang::Pt, "2001000", Some("2 milhões mil")),
            (Lang::Pt, "1000000000", Some("1 bilhão")),
            (Lang::De, "8-10", Some("8 bis 10")),
            (Lang::De, "0-2", Some("0 bis 2")),
            (Lang::De, "3-1", None),
            (Lang::De, "01-5", None),
            (Lang::Fr, "3-5", None),
            (Lang::Es, "3.er", Some("tercer")),
            (Lang::Es, "1er,", Some("primer,")),
            (Lang::Es, "2.er", None),
        ];
        for (lang, input, expected) in rest {
            assert_eq!(numbers(input, lang).as_deref(), expected, "{lang:?} {input:?}");
        }
    }
}
