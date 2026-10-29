//! Times of day: "3:45PM" becomes "3-45 PM", "08:20" in German "8 Uhr 20".

use super::{digits, suffix};
use crate::preprocess::Lang;

/// A time of day in the way `lang` writes it, read the way it is spoken. Spanish and Portuguese
/// times are left alone.
pub(super) fn times(word: &str, lang: Lang) -> Option<String> {
    match lang {
        Lang::En => en(word),
        Lang::Fr => fr(word),
        Lang::De => de(word),
        _ => None,
    }
}

/// `H:MM` or `H.MM`, with an optional `AM` / `PM`, read "H-MM PM", or "H PM" on the hour. A dot
/// needs the `AM` / `PM`: "12.30" is as likely a decimal number.
fn en(word: &str) -> Option<String> {
    let (hour, minutes, separator, rest) = parse(word, &[':', '.'])?;
    let meridiem =
        ["AM", "PM", "am", "pm"].into_iter().find_map(|m| Some((m, rest.strip_prefix(m)?)));
    let (meridiem, rest) = meridiem.unwrap_or(("", rest));
    let suffix = suffix(rest)?;
    if separator == '.' && meridiem.is_empty() {
        return None;
    }
    let time = if minutes == "00" { hour.to_string() } else { format!("{hour}-{minutes}") };
    match meridiem {
        "" => Some(format!("{time}{suffix}")),
        meridiem => Some(format!("{time} {}{suffix}", meridiem.to_uppercase())),
    }
}

/// `H:MM` or `HhMM`, read "HhMM".
fn fr(word: &str) -> Option<String> {
    let (hour, minutes, _, rest) = parse(word, &[':', 'h', 'H'])?;
    Some(format!("{hour}h{minutes}{}", suffix(rest)?))
}

/// `H:MM` or `H.MM`, read "H Uhr MM", or "H Uhr" on the hour.
fn de(word: &str) -> Option<String> {
    let (hour, minutes, _, rest) = parse(word, &[':', '.'])?;
    let suffix = suffix(rest)?;
    match minutes {
        "00" => Some(format!("{hour} Uhr{suffix}")),
        minutes => Some(format!("{hour} Uhr {minutes}{suffix}")),
    }
}

/// One or two digits of hours, one of `separators`, two to four digits of minutes, and the rest
/// of `word`. A leading zero is dropped from the hours.
fn parse<'a>(word: &'a str, separators: &[char]) -> Option<(&'a str, &'a str, char, &'a str)> {
    let (hour, rest) = digits(word, 1, 2)?;
    let separator = rest.chars().next().filter(|c| separators.contains(c))?;
    let (minutes, rest) = digits(&rest[separator.len_utf8()..], 2, 4)?;
    let hour = if hour.len() == 2 { hour.strip_prefix('0').unwrap_or(hour) } else { hour };
    Some((hour, minutes, separator, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_are_read_out() {
        let cases = [
            (Lang::En, "3:45PM!", "3-45 PM!"),
            (Lang::En, "03:45PM!", "3-45 PM!"),
            (Lang::En, "03.45am!", "3-45 AM!"),
            (Lang::En, "03.00pm!", "3 PM!"),
            (Lang::En, "03:00.", "3."),
            (Lang::En, "1:06", "1-06"),
            (Lang::Fr, "00h15,", "0h15,"),
            (Lang::Fr, "0h15,", "0h15,"),
            (Lang::Fr, "09h15,", "9h15,"),
            (Lang::Fr, "14:00?", "14h00?"),
            (Lang::Fr, "18H30", "18h30"),
            (Lang::De, "8:20.", "8 Uhr 20."),
            (Lang::De, "08:20.", "8 Uhr 20."),
            (Lang::De, "22.45!", "22 Uhr 45!"),
            (Lang::De, "0:00.", "0 Uhr."),
            (Lang::De, "8:00.", "8 Uhr."),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(times(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        let declined = [
            // A dot without AM / PM may be a decimal number.
            (Lang::En, "12.30"),
            (Lang::En, "12.30."),
            (Lang::En, "123:06"),
            (Lang::En, "3:45Pm"),
            (Lang::Fr, "14.00?"),
            (Lang::De, "22h45!"),
            (Lang::Es, "14:00"),
            (Lang::Pt, "14:00"),
            (Lang::En, "3:4"),
            (Lang::En, ":45"),
        ];
        for (lang, input) in declined {
            assert_eq!(times(input, lang), None, "{lang:?} {input:?}");
        }
    }
}
