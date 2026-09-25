//! Word-level rewrites run by [`super::normalize_text`], one whitespace-separated word at a
//! time. The first rule in [`RULES`] that claims a word wins, and rules are never chained.

use super::Lang;

/// A rule: it rewrites a whole word in a language, or declines it.
type Rewrite = fn(&str, Lang) -> Option<String>;

/// Every rule, in the order they are tried, each with the name it goes by in the frontends'
/// `--rewrites` flag. Adding a rule is a function and a row here.
const RULES: &[(&str, Rewrite)] = &[("numbers", numbers)];

/// Which of the rewrite rules run, on top of the character normalization.
///
/// It travels on [`super::Normalize`] rather than beside it, so that callers which tokenize by
/// hand cannot pick up the language and forget the rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules(u32);

impl Rules {
    /// No rewrites: the character pass alone.
    pub const NONE: Self = Self(0);

    /// Every rule this version implements.
    pub const ALL: Self = Self((1 << RULES.len()) - 1);

    /// Parse the frontends' `--rewrites` flag: `all`, `none`, or a comma-separated list of rule
    /// names.
    pub fn parse(s: &str) -> crate::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "all" => Ok(Self::ALL),
            "none" | "off" => Ok(Self::NONE),
            list => list.split(',').try_fold(Self::NONE, |set, name| {
                match RULES.iter().position(|(rule, _)| *rule == name.trim()) {
                    Some(i) => Ok(Self(set.0 | 1 << i)),
                    None => Err(crate::Error::invalid_argument(format!(
                        "unknown rewrite rule: {name}; expected all, none, or any of: {}",
                        RULES.iter().map(|(rule, _)| *rule).collect::<Vec<_>>().join(", ")
                    ))),
                }
            }),
        }
    }
}

/// Rewrite `word` under `rules`, or `None` when no rule claims it.
pub fn rewrite_word(word: &str, lang: Lang, rules: Rules) -> Option<String> {
    RULES
        .iter()
        .enumerate()
        .filter(|(i, _)| rules.0 & (1 << i) != 0)
        .find_map(|(_, (_, rewrite))| rewrite(word, lang))
}

/// The trailing punctuation a rule carries through, or `None` when `rest` is anything else --
/// which means the rule did not account for the whole word and must not fire.
fn suffix(rest: &str) -> Option<&str> {
    rest.chars().all(|c| "!?.:,;…".contains(c)).then_some(rest)
}

/// Large numbers with their scale words: "1,234.56" becomes "1 thousand 234 point 56", and in
/// French "1234,56" becomes "mille 234 virgule 56". Values under a thousand and years stay as
/// digits, and a leading zero marks a code, not a quantity.
fn numbers(word: &str, lang: Lang) -> Option<String> {
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
    Some(format!("{words}{suffix}"))
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
            ("007", None),
            ("12:30", None),
            ("10-15", None),
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
        ];
        for (lang, input, expected) in rest {
            assert_eq!(numbers(input, lang).as_deref(), expected, "{lang:?} {input:?}");
        }
    }

    #[test]
    fn rules_pick_what_runs() {
        assert_eq!(Rules::parse("all").unwrap(), Rules::ALL);
        assert_eq!(Rules::parse("numbers").unwrap(), Rules::ALL);
        assert_eq!(Rules::parse("none").unwrap(), Rules::NONE);
        assert!(Rules::parse("dates").is_err());
        assert_eq!(rewrite_word("1234", Lang::En, Rules::ALL).as_deref(), Some("1 thousand 234"));
        assert_eq!(rewrite_word("1234", Lang::Fr, Rules::ALL).as_deref(), Some("mille 234"));
        assert_eq!(rewrite_word("1234", Lang::En, Rules::NONE), None);
        assert_eq!(rewrite_word("1234", Lang::De, Rules::ALL).as_deref(), Some("ein Tausend 234"));
    }
}
