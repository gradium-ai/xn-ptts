//! Dates and clock times in Swedish and Norwegian: "2026-03-14" becomes "den fjortonde mars
//! 2026", "14.03.2026" in Norwegian "fjortende mars 2026", and "kl. 10.00" "klockan 10".
//!
//! The day is read as an ordinal word and the month by its name, which the models cannot tell
//! from the digits. The year and the hours stay digits, which they read on their own.
//!
//! Only shapes that are dates or times without doubt are read: "1/2" and "3/4" are as often
//! fractions, "3.14" a decimal and "20.11" a version, so a day and a month without a year need
//! a sign that they are a date (see [`span`]).

use super::{all_digits, split_suffix};
use crate::preprocess::Lang;

/// How a language says dates and times.
struct Local {
    /// The ordinal words of the days 1 to 31.
    days: [&'static str; 31],
    months: [&'static str; 12],
    /// What "kl." stands for.
    clock: &'static str,
    /// The word between the two ends of a range.
    to: &'static str,
    /// The word between two times: "kl. 9 och 17".
    and: &'static str,
    /// The words that open a range of times: "från 22.00 till 06.00".
    from: [&'static str; 2],
    /// Whether a date is read with "den" before it, as Swedish does: "den fjortonde mars".
    den: bool,
    /// Whether a day and month are written with points, "14.03.", as Norwegian does.
    dotted: bool,
}

const SV: Local = Local {
    days: [
        "första",
        "andra",
        "tredje",
        "fjärde",
        "femte",
        "sjätte",
        "sjunde",
        "åttonde",
        "nionde",
        "tionde",
        "elfte",
        "tolfte",
        "trettonde",
        "fjortonde",
        "femtonde",
        "sextonde",
        "sjuttonde",
        "artonde",
        "nittonde",
        "tjugonde",
        "tjugoförsta",
        "tjugoandra",
        "tjugotredje",
        "tjugofjärde",
        "tjugofemte",
        "tjugosjätte",
        "tjugosjunde",
        "tjugoåttonde",
        "tjugonionde",
        "trettionde",
        "trettioförsta",
    ],
    months: [
        "januari",
        "februari",
        "mars",
        "april",
        "maj",
        "juni",
        "juli",
        "augusti",
        "september",
        "oktober",
        "november",
        "december",
    ],
    clock: "klockan",
    to: "till",
    and: "och",
    from: ["från", "mellan"],
    den: true,
    dotted: false,
};

const NO: Local = Local {
    days: [
        "første",
        "andre",
        "tredje",
        "fjerde",
        "femte",
        "sjette",
        "sjuende",
        "åttende",
        "niende",
        "tiende",
        "ellevte",
        "tolvte",
        "trettende",
        "fjortende",
        "femtende",
        "sekstende",
        "syttende",
        "attende",
        "nittende",
        "tjuende",
        "tjueførste",
        "tjueandre",
        "tjuetredje",
        "tjuefjerde",
        "tjuefemte",
        "tjuesjette",
        "tjuesjuende",
        "tjueåttende",
        "tjueniende",
        "trettiende",
        "trettiførste",
    ],
    months: [
        "januar",
        "februar",
        "mars",
        "april",
        "mai",
        "juni",
        "juli",
        "august",
        "september",
        "oktober",
        "november",
        "desember",
    ],
    clock: "klokka",
    to: "til",
    and: "og",
    from: ["fra", "mellom"],
    den: false,
    dotted: true,
};

fn local(lang: Lang) -> Option<&'static Local> {
    match lang {
        Lang::Sv => Some(&SV),
        Lang::No => Some(&NO),
        _ => None,
    }
}

/// A day of a month, with the year when it is written.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Date<'a> {
    day: usize,
    month: usize,
    year: Option<&'a str>,
}

impl Date<'_> {
    /// The day and month read out, with the year as digits: "fjortonde mars 2026".
    fn read(self, local: &Local) -> String {
        let date = format!("{} {}", local.days[self.day - 1], local.months[self.month - 1]);
        match self.year {
            Some(year) => format!("{date} {year}"),
            None => date,
        }
    }

    /// Whether a day and a month alone are a date and not a fraction or a rating: the day is
    /// past the month ("23/9", "12/10"), or one of them is written with a leading zero ("02/10").
    fn plain(self, day: &str, month: &str) -> bool {
        self.day > self.month || day.starts_with('0') || month.starts_with('0')
    }
}

/// `s` as a number when it is one to `max` digits.
fn number(s: &str, max: usize) -> Option<usize> {
    (s.len() <= max && all_digits(s)).then(|| s.parse().ok()).flatten()
}

/// A day of `month`, which must exist.
fn date(day: usize, month: usize, year: Option<&str>) -> Option<Date<'_>> {
    const LENGTHS: [usize; 12] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let length = *LENGTHS.get(month.checked_sub(1)?)?;
    (1..=length).contains(&day).then_some(Date { day, month, year })
}

/// A year of four digits from 1000 to 2999.
fn year(s: &str) -> Option<&str> {
    (s.len() == 4 && all_digits(s) && (s.starts_with('1') || s.starts_with('2'))).then_some(s)
}

/// A whole date with its year: ISO "2026-03-14", and "14.03.2026" or "14/3/2026".
fn full(body: &str) -> Option<Date<'_>> {
    let mut parts = body.split('-');
    if let (Some(y), Some(m), Some(d), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    {
        if m.len() == 2 && d.len() == 2 {
            return date(number(d, 2)?, number(m, 2)?, Some(year(y)?));
        }
        return None;
    }
    let separator = if body.contains('/') { '/' } else { '.' };
    let mut parts = body.split(separator);
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(d), Some(m), Some(y), None) => date(number(d, 2)?, number(m, 2)?, Some(year(y)?)),
        _ => None,
    }
}

/// A day and month with a slash, "23/9", and whether it is a date on its own ([`Date::plain`]).
/// "24/7" is round the clock.
fn slashed(body: &str) -> Option<(Date<'_>, bool)> {
    let (d, m) = body.split_once('/')?;
    let date = date(number(d, 2)?, number(m, 2)?, None)?;
    (body != "24/7").then_some((date, date.plain(d, m)))
}

/// A day and month with a point and two digits each, "14.03", as Norwegian writes them.
fn dotted(body: &str) -> Option<Date<'_>> {
    let (d, m) = body.split_once('.')?;
    (d.len() == 2 && m.len() == 2).then(|| date(number(d, 2)?, number(m, 2)?, None)).flatten()
}

/// A day of the month written as a number before the month's name: "14" in Swedish, "14:e" or
/// "14:a" after "den" or before a name, "14." in Norwegian.
fn day(word: &str, local: &Local) -> Option<usize> {
    let (body, end) = split_suffix(word);
    let n = if local.dotted {
        (end == ".").then(|| number(body, 2)).flatten()?
    } else {
        let n = body.strip_suffix(":e").or_else(|| body.strip_suffix(":a")).unwrap_or(body);
        (end.is_empty()).then(|| number(n, 2)).flatten()?
    };
    (1..=31).contains(&n).then_some(n)
}

/// A range of days before the month's name: "3-17" in Swedish, "3.-17." in Norwegian.
fn days(word: &str, local: &Local) -> Option<(usize, usize)> {
    let (from, to) = if local.dotted {
        let (from, to) = word.strip_suffix('.')?.split_once(".-")?;
        (from, to)
    } else {
        word.split_once('-')?
    };
    let (from, to) = (number(from, 2)?, number(to, 2)?);
    (1 <= from && from < to && to <= 31).then_some((from, to))
}

/// Whether `word` is the name of a month.
fn month(word: Option<&str>, local: &Local) -> bool {
    word.is_some_and(|w| local.months.contains(&split_suffix(w).0))
}

/// A time of day as the models read it: the hour without its leading zero and the minutes after
/// a point or a colon, or the hour alone on the hour ("07.05" is "7.05", "10.00" is "10").
/// A bare hour ("15") is taken only when `bare`.
fn time(body: &str, bare: bool) -> Option<String> {
    let (hour, minutes) = match body.find(['.', ':']) {
        Some(i) => (&body[..i], Some((&body[i..i + 1], &body[i + 1..]))),
        None if bare => (body, None),
        None => return None,
    };
    if number(hour, 2)? > 24 {
        return None;
    }
    let hour = if hour.len() == 2 { hour.strip_prefix('0').unwrap_or(hour) } else { hour };
    match minutes {
        None | Some((_, "00")) => Some(hour.to_string()),
        Some((separator, minutes)) => (minutes.len() == 2 && number(minutes, 2)? < 60)
            .then(|| format!("{hour}{separator}{minutes}")),
    }
}

/// A time, or a range of times "14.00-16.00" read with the word for "to" between.
fn times(body: &str, bare: bool, local: &Local) -> Option<String> {
    match body.split_once('-') {
        Some((from, to)) => Some(format!("{} {} {}", time(from, bare)?, local.to, time(to, bare)?)),
        None => time(body, bare),
    }
}

/// Whether `word` stands for the clock: "kl.", "kl", or the word itself.
fn is_clock(word: &str, local: &Local) -> bool {
    let word = word.to_lowercase();
    matches!(word.as_str(), "kl." | "kl") || word == local.clock || word == "klokken"
}

/// Whether `word` starts with a capital letter.
fn capitalized(word: &str) -> bool {
    word.chars().next().is_some_and(char::is_uppercase)
}

/// `word` with its first letter in capitals when `upper`.
fn capitalize(word: &str, upper: bool) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) if upper => first.to_uppercase().chain(chars).collect(),
        _ => word.to_string(),
    }
}

/// The words before "20.11" that make it a version, a time or a number rather than a date.
const NOT_BEFORE_A_DATE: &[&str] = &[
    "versjon", "version", "v", "utgave", "utgåva", "release", "build", "node", "python", "ios",
    "macos", "android", "windows", "kl", "kl.", "klokka", "klokken", "klockan", "nr", "nr.",
];

/// The dates and times that start `words`, with how many words they take, in Swedish and
/// Norwegian; `prev` is the word before, as rewritten.
///
/// - Whole dates: "2026-03-14", "14.03.2026", "14/3/2026".
/// - A day and month with a slash, "23/9", when the day is past the month or written with a
///   leading zero, after "den", or with another date in a range ("1/7-30/9"). The character pass
///   turns an en dash between two such dates into a space, so two in a row are a range too.
/// - A Norwegian day and month with points, "14.03.", when the point after them is written and
///   the word before is not a version or a clock.
/// - A day before the month's name, "14 mars", "14. mars", "3-17 november", and a Swedish
///   "den 17:e" or a Norwegian "den 17." at the end of a sentence.
/// - A time after "kl.": "kl. 10.00" becomes "klockan 10", with the time or range after "och"
///   or "till" that may follow; and a range of times "12.00-14.00" or "från 22.00 till 06.00".
///
/// Swedish reads a date with "den" before it, which is added when the text has none.
pub(super) fn span(prev: Option<&str>, words: &[&str], lang: Lang) -> Option<(String, usize)> {
    let local = local(lang)?;
    let word = *words.first()?;
    let (reading, used) = read(prev, words, local)?;
    Some((capitalize(&reading, prev.is_none() || capitalized(word)), used))
}

/// What [`span`] reads, in lower case.
fn read(prev: Option<&str>, words: &[&str], local: &Local) -> Option<(String, usize)> {
    let word = *words.first()?;
    let next = words.get(1).copied();
    let (body, end) = split_suffix(word);
    let after_den = prev.is_some_and(|p| p.eq_ignore_ascii_case("den"));
    // The "den" a Swedish date takes when the text has none.
    let den = if local.den && !after_den { "den " } else { "" };

    if is_clock(word, local) && matches!(end, "" | ".") {
        return clock(word, &words[1..], local);
    }

    // "från 22.00 till 06.00", "mellom 11.00 og 11.30".
    if local.from.contains(&body.to_lowercase().as_str()) && end.is_empty() && words.len() >= 4 {
        let (first, first_end) = split_suffix(words[1]);
        let (last, last_end) = split_suffix(words[3]);
        if first_end.is_empty()
            && (words[2] == local.to || words[2] == local.and)
            && let (Some(first), Some(last)) = (time(first, false), time(last, false))
        {
            return Some((format!("{word} {first} {} {last}{last_end}", words[2]), 4));
        }
    }

    if let Some(date) = full(body) {
        return Some((format!("{den}{}{end}", date.read(local)), 1));
    }

    if let Some((from, to)) = body.split_once('-') {
        // "12.00-14.00".
        if let (Some(from), Some(to)) = (time(from, false), time(to, false)) {
            return Some((format!("{from} {} {to}{end}", local.to), 1));
        }
        // "1/7-30/9".
        if let (Some((from, a)), Some((to, b))) = (slashed(from), slashed(to))
            && (a || b)
        {
            return Some((format!("{}{end}", range(den, from, to, local)), 1));
        }
    }

    if let Some((date, plain)) = slashed(body) {
        // "1/7 30/9", where the character pass dropped the en dash between them.
        let to = next.and_then(|next| Some((slashed(split_suffix(next).0)?, split_suffix(next).1)));
        if let Some(((to, to_plain), to_end)) = to.filter(|_| end.is_empty())
            && (plain || to_plain)
        {
            return Some((format!("{}{to_end}", range(den, date, to, local)), 2));
        }
        if plain || after_den {
            return Some((format!("{den}{}{end}", date.read(local)), 1));
        }
    }

    // "14.03." in Norwegian. The point after it is the date's own, and the sentence's too when
    // nothing or a capital follows.
    if local.dotted && end.starts_with('.') {
        let excluded = prev.is_some_and(|p| NOT_BEFORE_A_DATE.contains(&p.to_lowercase().as_str()));
        if let Some(date) = dotted(body).filter(|_| !excluded) {
            let end = match next {
                Some(next) if !capitalized(next) => &end[1..],
                _ => end,
            };
            return Some((format!("{}{end}", date.read(local)), 1));
        }
    }

    // "14 mars", "14. mars", "3:e maj", "3-17 november".
    if month(next, local) {
        if let Some(day) = day(word, local) {
            return Some((local.days[day - 1].to_string(), 1));
        }
        if let Some((from, to)) = days(word, local) {
            return Some((
                format!("{} {} {}", local.days[from - 1], local.to, local.days[to - 1]),
                1,
            ));
        }
    }

    // "den 17:e", and a Norwegian "den 17." that ends a sentence.
    if after_den {
        let n = if local.dotted {
            (end == "." && next.is_none_or(capitalized)).then(|| number(body, 2)).flatten()
        } else {
            body.strip_suffix(":e").or_else(|| body.strip_suffix(":a")).and_then(|n| number(n, 2))
        };
        if let Some(n) = n.filter(|n| (1..=31).contains(n)) {
            return Some((format!("{}{end}", local.days[n - 1]), 1));
        }
    }
    None
}

/// Two dates read as a range, the first with `den` before it.
fn range(den: &str, from: Date, to: Date, local: &Local) -> String {
    let to_den = if local.den { "den " } else { "" };
    format!("{den}{} {} {to_den}{}", from.read(local), local.to, to.read(local))
}

/// "kl." and the time after it in `words`: "kl. 10.00" becomes "klockan 10". A time after "och"
/// or "till" that follows is read too ("kl. 08.00 och 20.00"), and so is one right after it,
/// where the character pass turned an en dash into a space ("kl. 14.00 16.00").
fn clock(word: &str, words: &[&str], local: &Local) -> Option<(String, usize)> {
    let (body, end) = split_suffix(words.first()?);
    let reading = times(body, true, local)?;
    let clock = if split_suffix(word).0.eq_ignore_ascii_case("kl") { local.clock } else { word };
    if end.is_empty() {
        let (joined, at) = match words.get(1).copied() {
            Some(w) if w == local.and || w == local.to => (w, 2),
            _ if body.contains(['.', ':']) && !body.contains('-') => (local.to, 1),
            _ => ("", 0),
        };
        let then = words.get(at).filter(|_| at > 0).map(|w| split_suffix(w));
        if let Some((then, then_end)) = then
            && let Some(then) = times(then, false, local)
        {
            return Some((format!("{clock} {reading} {joined} {then}{then_end}"), 2 + at));
        }
    }
    Some((format!("{clock} {reading}{end}"), 2))
}

/// The rule's word-level reading, for a word read on its own: a whole date with its year,
/// "2026-03-14", in Swedish and Norwegian. The rest needs the words around it, in [`span`].
pub(super) fn local_dates(word: &str, lang: Lang) -> Option<String> {
    let local = local(lang)?;
    let (body, end) = split_suffix(word);
    let den = if local.den { "den " } else { "" };
    Some(format!("{den}{}{end}", full(body)?.read(local)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `text` split at its spaces and read from the start, as [`super::super::rewrite_text`]
    /// does, with the words no span claims kept.
    fn read_text(text: &str, lang: Lang) -> String {
        let words: Vec<&str> = text.split(' ').collect();
        let mut out: Vec<String> = vec![];
        let mut i = 0;
        while i < words.len() {
            let prev = out.last().map(String::as_str);
            match span(prev, &words[i..], lang) {
                Some((reading, used)) => {
                    out.push(reading);
                    i += used;
                }
                None => {
                    out.push(words[i].to_string());
                    i += 1;
                }
            }
        }
        out.join(" ")
    }

    #[test]
    fn swedish_dates_are_read_out() {
        let cases = [
            ("Din tid är 2026-03-14 kl. 10.00.", "Din tid är den fjortonde mars 2026 klockan 10."),
            (
                "Du är född 1990-04-07, stämmer det?",
                "Du är född den sjunde april 1990, stämmer det?",
            ),
            ("Mötet har flyttats till 23/9.", "Mötet har flyttats till den tjugotredje september."),
            ("skickas den 14/03 och", "skickas den fjortonde mars och"),
            ("skickades 02/10.", "skickades den andra oktober."),
            ("utfärdades den 9/10.", "utfärdades den nionde oktober."),
            ("perioden 1/7 30/9.", "perioden den första juli till den trettionde september."),
            ("perioden 1/7-30/9.", "perioden den första juli till den trettionde september."),
            ("den 14 mars 2026.", "den fjortonde mars 2026."),
            ("förfaller den 3:e maj.", "förfaller den tredje maj."),
            ("tisdagen den 17:e.", "tisdagen den sjuttonde."),
            ("pågår 3-17 november.", "pågår tredje till sjuttonde november."),
            ("2026-03-14 gäller.", "Den fjortonde mars 2026 gäller."),
            ("född 14/3/1990.", "född den fjortonde mars 1990."),
            ("den 29/2 och 31/12", "den tjugonionde februari och den trettioförsta december"),
        ];
        for (input, expected) in cases {
            assert_eq!(read_text(input, Lang::Sv), expected, "{input:?}");
        }
    }

    #[test]
    fn norwegian_dates_are_read_out() {
        let cases = [
            ("Timen din er 14.03.2026 kl. 10.00.", "Timen din er fjortende mars 2026 klokka 10."),
            ("Garantien utløper 2027-06-30.", "Garantien utløper trettiende juni 2027."),
            ("Møtet er flyttet til 23.09.", "Møtet er flyttet til tjuetredje september."),
            ("sendes 14.03. og kommer", "sendes fjortende mars og kommer"),
            ("dekker 01.07. til 30.09.", "dekker første juli til trettiende september."),
            ("Levert 12.10. Takk.", "Levert tolvte oktober. Takk."),
            ("Timen din er 14. mars 2026.", "Timen din er fjortende mars 2026."),
            ("forfaller den 3. mai.", "forfaller den tredje mai."),
            ("Møtet er tirsdag den 17.", "Møtet er tirsdag den syttende."),
            ("varer 3.-17. november.", "varer tredje til syttende november."),
            ("5. mai er fri.", "Femte mai er fri."),
            ("Født 7.4.1990.", "Født sjuende april 1990."),
        ];
        for (input, expected) in cases {
            assert_eq!(read_text(input, Lang::No), expected, "{input:?}");
        }
    }

    #[test]
    fn times_are_read_out() {
        let cases = [
            (Lang::Sv, "avgår kl. 07.05 från", "avgår klockan 7.05 från"),
            (Lang::Sv, "efter kl. 15.", "efter klockan 15."),
            (Lang::Sv, "Kl. 9 prick.", "Klockan 9 prick."),
            (Lang::Sv, "öppet kl. 8-20.", "öppet klockan 8 till 20."),
            (Lang::Sv, "tid kl. 14.00-16.00 i morgon", "tid klockan 14 till 16 i morgon"),
            (Lang::Sv, "mellan kl. 11.00 och 11.30.", "mellan klockan 11 och 11.30."),
            (Lang::Sv, "mg kl. 08.00 och 20.00.", "mg klockan 8 och 20."),
            (Lang::Sv, "pass från 22.00 till 06.00.", "pass från 22 till 6."),
            (Lang::Sv, "serveras 12.00-14.00.", "serveras 12 till 14."),
            (Lang::Sv, "vid klockan 10:00.", "vid klockan 10."),
            (Lang::No, "tid kl. 14.00 16.00 i morgen", "tid klokka 14 til 16 i morgen"),
            (Lang::No, "fra kl. 8 til kl. 20.", "fra klokka 8 til klokka 20."),
            (Lang::No, "mellom kl. 9 og 17 for", "mellom klokka 9 og 17 for"),
            (Lang::No, "til klokken 10.00.", "til klokken 10."),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(read_text(input, lang), expected, "{lang:?} {input:?}");
        }
    }

    /// Fractions, ratings, decimals, versions, codes, card expiry dates and durations stay.
    #[test]
    fn what_is_not_a_date_stays() {
        let cases = [
            (Lang::Sv, "Bara 1/4 av platserna"),
            (Lang::Sv, "Fyll tanken till 3/4."),
            (Lang::Sv, "Betyg 5/5 och 10/10."),
            (Lang::Sv, "9/10 tandläkare"),
            (Lang::Sv, "Öppet 24/7."),
            (Lang::Sv, "Ditt kort går ut 08/27."),
            (Lang::Sv, "Ärendet CS-2026-0813 och 2026/0917/44."),
            (Lang::Sv, "Version 3.10 och 2.4.1."),
            (Lang::Sv, "Utgåva 24.04 är ute."),
            (Lang::Sv, "Lopptiden var 2:03:45."),
            (Lang::Sv, "Det tar 1:42."),
            (Lang::Sv, "Den 5 är bäst."),
            (Lang::Sv, "Order 2026-118-4409 och 2026-13-01."),
            (Lang::Sv, "Kl. är bra."),
            (Lang::Sv, "kl. 25.00"),
            (Lang::No, "Node 20.11 for denne."),
            (Lang::No, "Versjon 20.11."),
            (Lang::No, "Oppgrader fra 0.9 til 1.0."),
            (Lang::No, "ankomst 23.50."),
            (Lang::No, "Den 2. sesongen"),
            (Lang::No, "Den 1. og 15. er betalingsdager."),
            (Lang::No, "hvert 10.-15. minutt"),
            (Lang::No, "Elever på 7.-9. trinn"),
            (Lang::No, "Møtet 23.09 kl"),
            (Lang::No, "Sak nr. 2026-77 er åpen."),
            (Lang::No, "Bare 1/4 av setene."),
        ];
        for (lang, input) in cases {
            assert_eq!(read_text(input, lang), input, "{lang:?} {input:?}");
        }
    }

    #[test]
    fn only_swedish_and_norwegian() {
        for lang in [Lang::En, Lang::De, Lang::Fr, Lang::Nl, Lang::Fi] {
            for text in ["2026-03-14", "kl. 10.00", "14.03.2026", "den 14 mars"] {
                assert_eq!(read_text(text, lang), text, "{lang:?} {text:?}");
            }
            assert_eq!(local_dates("2026-03-14", lang), None);
        }
        assert_eq!(
            local_dates("2026-03-14.", Lang::Sv).as_deref(),
            Some("den fjortonde mars 2026.")
        );
        assert_eq!(local_dates("14.03.2026", Lang::No).as_deref(), Some("fjortende mars 2026"));
        assert_eq!(local_dates("14.03.", Lang::No), None);
    }
}
