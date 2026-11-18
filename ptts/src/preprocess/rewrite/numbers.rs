//! Numbers read out with their scale words.
//!
//! The digits stay and the scale words go in between them, "1 thousand 234": the models read
//! digits below a thousand (ten thousand in Japanese and Korean) on their own, with the
//! agreement and the sound changes of the language, and lose their place above that. Hindi is
//! the exception: its model reads round numbers in English ("500" as "five hundred"), so every
//! number is spelled out in words there, years included.

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
    if is_identifier(body, lang) {
        return Some(format!("{}{end}", spell(body, lang)));
    }
    // Korean needs no rewriting below ten thousand: the model reads "1,000" as written.
    if lang == Lang::Ko
        && body.chars().all(|c| c.is_ascii_digit() || c == ',')
        && body.replace(',', "").parse::<u32>().is_ok_and(|n| n < 10_000)
    {
        return None;
    }
    if let Some(range) = range(body, lang) {
        return Some(format!("{range}{end}"));
    }
    // Arabic marks a year's calendar with a letter: "2024م" (Gregorian) is read as the year
    // alone, "1445هـ" with the calendar's name.
    if lang == Lang::Ar {
        if let Some(year) = body.strip_suffix('م').filter(|y| y.len() == 4 && all_digits(y)) {
            return Some(format!("{year}{end}"));
        }
        if let Some(year) =
            body.strip_suffix("هـ").filter(|y| (3..=4).contains(&y.len()) && all_digits(y))
        {
            return Some(format!("{year} هجري{end}"));
        }
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

/// The languages whose readings follow the serving stack's, quirks included.
fn serving(lang: Lang) -> bool {
    matches!(lang, Lang::En | Lang::Fr | Lang::De | Lang::Es | Lang::Pt)
}

/// Whether `body`, a plain run of seven or more digits, is an identifier, an order or a case
/// number, read digit by digit. The other languages group a quantity that long ("1 234 567",
/// "1.234.567"), so an ungrouped one is a code; the languages the serving stack reads keep its
/// reading as a quantity. Japanese and Korean decide by the counter after the digits, in
/// [`super::rewrite_word`].
pub(super) fn is_identifier(body: &str, lang: Lang) -> bool {
    !serving(lang) && !matches!(lang, Lang::Ja | Lang::Ko) && body.len() >= 7 && all_digits(body)
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
        // "2024-25"; in the new languages only a few years on, as "nr. 2026-77" is a code.
        (4, 2) => {
            year(from)
                && all_digits(to)
                && &from[2..] < to
                && (serving(lang) || to.parse::<u32>().ok()? <= from[2..].parse::<u32>().ok()? + 10)
        }
        _ => false,
    };
    if !ok {
        return None;
    }
    let word = match lang {
        Lang::En => "to",
        Lang::Fr => "à",
        Lang::De => "bis",
        Lang::Es | Lang::Pt | Lang::It => "a",
        Lang::Ar => "إلى",
        Lang::Hi => "से",
        Lang::Ja => "から",
        Lang::Sv => "till",
        Lang::No => "til",
        Lang::Nl => "tot",
        Lang::Pl => "do",
        // Finnish reads a range in two cases or with a pause: nothing to put between.
        _ => return None,
    };
    Some(format!("{from} {word} {to}"))
}

pub(super) fn spell(s: &str, lang: Lang) -> String {
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
        Lang::Ar => {
            ["صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
        }
        Lang::Hi => {
            return s
                .chars()
                .filter(|&c| c != '-')
                .map(|c| match c.to_digit(10) {
                    Some(d) => HINDI[d as usize].to_string(),
                    None => c.to_string(),
                })
                .collect::<Vec<_>>()
                .join(" ");
        }
        Lang::Ja => ["ゼロ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"],
        // The zero of a digit string, not of a quantity.
        Lang::Ko => ["공", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"],
        Lang::Fi => [
            "nolla",
            "yksi",
            "kaksi",
            "kolme",
            "neljä",
            "viisi",
            "kuusi",
            "seitsemän",
            "kahdeksan",
            "yhdeksän",
        ],
        Lang::Sv => ["noll", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio"],
        Lang::No => ["null", "en", "to", "tre", "fire", "fem", "seks", "sju", "åtte", "ni"],
        Lang::It => {
            ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"]
        }
        Lang::Nl => ["nul", "een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen"],
        Lang::Pl => [
            "zero",
            "jeden",
            "dwa",
            "trzy",
            "cztery",
            "pięć",
            "sześć",
            "siedem",
            "osiem",
            "dziewięć",
        ],
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
    // English, Arabic, Hindi, Japanese and Korean group thousands with commas and mark decimals
    // with a dot. French, Finnish, Swedish, Norwegian and Polish do neither, once the character
    // pass has joined the groups they space. The others mark decimals with a comma and may group
    // with dots, but a dot only groups when every group after the first has three digits:
    // "1.234" is a thousand, "1.23" is not a number at all.
    let (group, point) = match lang {
        Lang::En | Lang::Ar | Lang::Hi | Lang::Ja | Lang::Ko => (Some(','), '.'),
        Lang::Fr | Lang::Fi | Lang::Sv | Lang::No | Lang::Pl => (None, ','),
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
    // Japanese and Korean count in myriads, so their first scale word is at ten thousand.
    let small = if matches!(lang, Lang::Ja | Lang::Ko) { 10_000 } else { 1000 };
    // The Korean model reads a small decimal as written ("4.2%"), better than spelled.
    if lang == Lang::Ko && value < small && decimal.is_some() && !negative {
        return None;
    }
    let mut words = if lang == Lang::Hi {
        hindi_words(value)
    } else if value < small || (1900..2100).contains(&value) {
        value.to_string()
    } else {
        with_scales(value, lang)
    };
    if let Some(decimal) = decimal {
        // Hindi spells the digits after the point one by one; Japanese and Korean read them one
        // by one too, which the models only do when they stand apart ("3 点 1 0", not "3 点 10").
        let decimal = match lang {
            Lang::Hi => spell(decimal, lang),
            Lang::Ja | Lang::Ko => decimal.chars().map(String::from).collect::<Vec<_>>().join(" "),
            _ => decimal.to_string(),
        };
        words = format!("{words} {} {decimal}", lang.decimal_separator());
    }
    if negative {
        words = format!("{} {words}", minus(lang));
    }
    Some((words, suffix))
}

fn minus(lang: Lang) -> &'static str {
    match lang {
        Lang::Fr => "moins",
        Lang::Es | Lang::Pt => "menos",
        Lang::Fi => "miinus",
        Lang::It => "meno",
        Lang::Nl => "min",
        Lang::Ar => "سالب",
        Lang::Hi => "माइनस",
        Lang::Ja => "マイナス",
        Lang::Ko => "마이너스",
        _ => "minus",
    }
}

/// `value`, a thousand or more, with its scale words between its groups of digits: "1 thousand
/// 234", "12 लाख 34 हजार 567", "123万4567".
fn with_scales(value: i64, lang: Lang) -> String {
    match lang {
        Lang::Ja | Lang::Ko => myriads(value, lang),
        Lang::Hi => hindi_words(value),
        _ => thousands(value, lang),
    }
}

/// Groups of a thousand, each with its scale word: a milliard, a million, a thousand.
/// Norwegian says "og" before a last group under a hundred: "tusen og én".
fn thousands(value: i64, lang: Lang) -> String {
    let mut parts: Vec<String> = [1_000_000_000, 1_000_000, 1000]
        .into_iter()
        .enumerate()
        .filter_map(|(scale, div)| match (value / div) % 1000 {
            0 => None,
            n => Some(scale_words(n, scale, lang)),
        })
        .collect();
    if value % 1000 != 0 {
        if lang == Lang::No && value % 1000 < 100 {
            parts.push("og".to_string());
        }
        parts.push((value % 1000).to_string());
    }
    parts.join(" ")
}

/// `n` of the scale `scale`, 0 a milliard, 1 a million and 2 a thousand, with the form the
/// count takes in `lang`: the bare word for one where the language says it so ("mille",
/// "tusen"), an article where it needs one ("eine Million", "en miljon"), the Arabic dual for
/// two and plural from three to ten, and the Polish few form from two to four.
pub(super) fn scale_words(n: i64, scale: usize, lang: Lang) -> String {
    let (one, many): ([&str; 3], [&str; 3]) = match lang {
        Lang::En => (["1 billion", "1 million", "1 thousand"], ["billion", "million", "thousand"]),
        Lang::Fr => (["1 milliard", "1 million", "mille"], ["milliards", "millions", "mille"]),
        Lang::De => (
            ["eine Milliarde", "eine Million", "ein Tausend"],
            ["Milliarden", "Millionen", "Tausend"],
        ),
        Lang::Es => (["1 mil millones", "1 millón", "mil"], ["mil millones", "millones", "mil"]),
        Lang::Pt => (["1 bilhão", "1 milhão", "mil"], ["bilhões", "milhões", "mil"]),
        Lang::Ar => (["مليار", "مليون", "ألف"], ["مليار", "مليون", "ألف"]),
        Lang::Fi => (["miljardi", "miljoona", "tuhat"], ["miljardia", "miljoonaa", "tuhatta"]),
        Lang::Sv => (["en miljard", "en miljon", "tusen"], ["miljarder", "miljoner", "tusen"]),
        Lang::No => (["en milliard", "en million", "tusen"], ["milliarder", "millioner", "tusen"]),
        Lang::It => (["1 miliardo", "1 milione", "mille"], ["miliardi", "milioni", "mila"]),
        Lang::Nl => (["1 miljard", "1 miljoen", "duizend"], ["miljard", "miljoen", "duizend"]),
        Lang::Pl => (["miliard", "milion", "tysiąc"], ["miliardów", "milionów", "tysięcy"]),
        // Hindi, Japanese and Korean group differently: see `indian` and `myriads`.
        Lang::Hi | Lang::Ja | Lang::Ko => {
            return format!("{n} {}", ["billion", "million", "thousand"][scale]);
        }
    };
    let word = match (lang, n) {
        (_, 1) => return one[scale].to_string(),
        (Lang::Ar, 2) => return ["مليارين", "مليونين", "ألفين"][scale].to_string(),
        (Lang::Ar, 3..=10) => ["مليارات", "ملايين", "آلاف"][scale],
        (Lang::Pl, n) if polish_few(n) => ["miliardy", "miliony", "tysiące"][scale],
        _ => many[scale],
    };
    format!("{n} {word}")
}

/// Whether a Polish count takes the few form of its noun, "2 tysiące" against "5 tysięcy": a
/// last digit from 2 to 4, except in the teens.
pub(super) fn polish_few(n: i64) -> bool {
    matches!(n % 10, 2..=4) && !matches!(n % 100, 12..=14)
}

/// The Hindi numbers up to ninety-nine, each a word of its own.
const HINDI: [&str; 100] = [
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पाँच",
    "छह",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "ग्यारह",
    "बारह",
    "तेरह",
    "चौदह",
    "पन्द्रह",
    "सोलह",
    "सत्रह",
    "अठारह",
    "उन्नीस",
    "बीस",
    "इक्कीस",
    "बाईस",
    "तेईस",
    "चौबीस",
    "पच्चीस",
    "छब्बीस",
    "सत्ताईस",
    "अट्ठाईस",
    "उनतीस",
    "तीस",
    "इकतीस",
    "बत्तीस",
    "तैंतीस",
    "चौंतीस",
    "पैंतीस",
    "छत्तीस",
    "सैंतीस",
    "अड़तीस",
    "उनतालीस",
    "चालीस",
    "इकतालीस",
    "बयालीस",
    "तैंतालीस",
    "चौवालीस",
    "पैंतालीस",
    "छियालीस",
    "सैंतालीस",
    "अड़तालीस",
    "उनचास",
    "पचास",
    "इक्यावन",
    "बावन",
    "तिरेपन",
    "चौवन",
    "पचपन",
    "छप्पन",
    "सत्तावन",
    "अट्ठावन",
    "उनसठ",
    "साठ",
    "इकसठ",
    "बासठ",
    "तिरेसठ",
    "चौंसठ",
    "पैंसठ",
    "छियासठ",
    "सड़सठ",
    "अड़सठ",
    "उनहत्तर",
    "सत्तर",
    "इकहत्तर",
    "बहत्तर",
    "तिहत्तर",
    "चौहत्तर",
    "पचहत्तर",
    "छिहत्तर",
    "सतहत्तर",
    "अठहत्तर",
    "उनासी",
    "अस्सी",
    "इक्यासी",
    "बयासी",
    "तिरासी",
    "चौरासी",
    "पचासी",
    "छियासी",
    "सत्तासी",
    "अट्ठासी",
    "नवासी",
    "नब्बे",
    "इक्यानबे",
    "बानबे",
    "तिरानबे",
    "चौरानबे",
    "पंचानबे",
    "छियानबे",
    "सत्तानबे",
    "अट्ठानबे",
    "निन्यानबे",
];

/// `n`, under a thousand, in Hindi words: "नौ सौ निन्यानबे".
pub(super) fn hindi_group(n: i64) -> String {
    match (n / 100, n % 100) {
        (0, rest) => HINDI[rest as usize].to_string(),
        (hundreds, 0) => format!("{} सौ", HINDI[hundreds as usize]),
        (hundreds, rest) => format!("{} सौ {}", HINDI[hundreds as usize], HINDI[rest as usize]),
    }
}

/// `value` in Hindi words, in groups of a hundred above a thousand as Hindi counts: a kharab,
/// an arab, a crore, a lakh, a thousand, "बारह लाख चौंतीस हजार पाँच सौ सड़सठ". The years of the
/// last century are read in hundreds: "उन्नीस सौ पचासी".
pub(super) fn hindi_words(value: i64) -> String {
    if (1900..2000).contains(&value) {
        return match value % 100 {
            0 => "उन्नीस सौ".to_string(),
            rest => format!("उन्नीस सौ {}", HINDI[rest as usize]),
        };
    }
    let scales = [
        (100_000_000_000, 100, "खरब"),
        (1_000_000_000, 100, "अरब"),
        (10_000_000, 100, "करोड़"),
        (100_000, 100, "लाख"),
        (1000, 100, "हजार"),
    ];
    let mut parts: Vec<String> = scales
        .into_iter()
        .filter_map(|(div, modulus, word)| match (value / div) % modulus {
            0 => None,
            n => Some(format!("{} {word}", hindi_group(n))),
        })
        .collect();
    if value % 1000 != 0 || value == 0 {
        parts.push(hindi_group(value % 1000));
    }
    parts.join(" ")
}

/// Groups of ten thousand, as Japanese and Korean count: "123万4567", "1억 2345만 6789". Korean
/// spaces the groups, as its spelling rules have it.
fn myriads(value: i64, lang: Lang) -> String {
    let (words, between) = match lang {
        Lang::Ja => (["億", "万"], ""),
        _ => (["억", "만"], " "),
    };
    let mut parts: Vec<String> = [100_000_000, 10_000]
        .into_iter()
        .zip(words)
        .filter_map(|(div, word)| match (value / div) % 10_000 {
            0 => None,
            n => Some(format!("{n}{word}")),
        })
        .collect();
    if value % 10_000 != 0 {
        parts.push((value % 10_000).to_string());
    }
    parts.join(between)
}

/// Hindi reads a half or a quarter before a scale word by its own name: "1.5 लाख" is "डेढ़ लाख",
/// "2.5 करोड़" "ढाई करोड़", "3.5 हजार" "साढ़े 3 हजार", "1.25 लाख" "सवा लाख" and "1.75 लाख"
/// "पौने 2 लाख".
pub(super) fn hindi_fraction(words: &[&str], lang: Lang) -> Option<(String, usize)> {
    let [word, next, ..] = words else { return None };
    let scale = split_suffix(next).0;
    if lang != Lang::Hi || !matches!(scale, "सौ" | "हजार" | "हज़ार" | "लाख" | "करोड़" | "अरब")
    {
        return None;
    }
    let (body, end) = split_suffix(word);
    let (int, fraction) = body.split_once('.')?;
    if !(1..=3).contains(&int.len()) || !all_digits(int) || int.starts_with('0') {
        return None;
    }
    let n: i64 = int.parse().ok()?;
    let spoken = match (fraction, n) {
        ("5", 1) => "डेढ़".to_string(),
        ("5", 2) => "ढाई".to_string(),
        ("5", n) => format!("साढ़े {}", hindi_group(n)),
        ("25", 1) => "सवा".to_string(),
        ("25", n) => format!("सवा {}", hindi_group(n)),
        ("75", n) => format!("पौने {}", hindi_group(n + 1)),
        _ => return None,
    };
    Some((format!("{spoken}{end}"), 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_new_languages_read_numbers_with_their_scale_words() {
        let cases = [
            (Lang::Ar, "123", Some("123")),
            (Lang::Ar, "1985", Some("1985")),
            (Lang::Ar, "1234", Some("ألف 234")),
            (Lang::Ar, "2000", Some("2000")),
            (Lang::Ar, "2500", Some("ألفين 500")),
            (Lang::Ar, "3000", Some("3 آلاف")),
            (Lang::Ar, "12345", Some("12 ألف 345")),
            (Lang::Ar, "1,234,567", Some("مليون 234 ألف 567")),
            (Lang::Ar, "2,000,000", Some("مليونين")),
            (Lang::Ar, "3.5", Some("3 فاصلة 5")),
            (Lang::Ar, "-5", Some("سالب 5")),
            (Lang::Ar, "2024م.", Some("2024.")),
            (Lang::Ar, "1445هـ", Some("1445 هجري")),
            (Lang::Ar, "20-25", Some("20 إلى 25")),
            (Lang::Hi, "0", Some("शून्य")),
            (Lang::Hi, "7", Some("सात")),
            (Lang::Hi, "123", Some("एक सौ तेईस")),
            (Lang::Hi, "250", Some("दो सौ पचास")),
            (Lang::Hi, "500", Some("पाँच सौ")),
            (Lang::Hi, "999", Some("नौ सौ निन्यानबे")),
            (Lang::Hi, "1500", Some("एक हजार पाँच सौ")),
            (Lang::Hi, "12345", Some("बारह हजार तीन सौ पैंतालीस")),
            (Lang::Hi, "12,34,567", Some("बारह लाख चौंतीस हजार पाँच सौ सड़सठ")),
            (Lang::Hi, "1,234,567", Some("बारह लाख चौंतीस हजार पाँच सौ सड़सठ")),
            (Lang::Hi, "1,00,00,000", Some("एक करोड़")),
            (Lang::Hi, "6,48,78,30,00,000", Some("छह खरब अड़तालीस अरब अठहत्तर करोड़ तीस लाख")),
            (Lang::Hi, "2.5", Some("दो दशमलव पाँच")),
            (Lang::Hi, "-3", Some("माइनस तीन")),
            (Lang::Hi, "1985", Some("उन्नीस सौ पचासी")),
            (Lang::Hi, "1900", Some("उन्नीस सौ")),
            (Lang::Hi, "2026", Some("दो हजार छब्बीस")),
            (Lang::Hi, "007", Some("शून्य शून्य सात")),
            (Lang::Hi, "20-25", Some("20 से 25")),
            (Lang::Ja, "1500", Some("1500")),
            (Lang::Ja, "12345", Some("1万2345")),
            (Lang::Ja, "70449", Some("7万449")),
            (Lang::Ja, "1,234,567", Some("123万4567")),
            (Lang::Ja, "100,000,000", Some("1億")),
            (Lang::Ja, "123,456,789", Some("1億2345万6789")),
            (Lang::Ja, "2.5", Some("2 点 5")),
            (Lang::Ja, "-3", Some("マイナス 3")),
            (Lang::Ja, "007", Some("ゼロ ゼロ なな")),
            (Lang::Ko, "12345", Some("1만 2345")),
            (Lang::Ko, "1,234,567", Some("123만 4567")),
            (Lang::Ko, "100,000,000", Some("1억")),
            (Lang::Ko, "100,010,000", Some("1억 1만")),
            (Lang::Ko, "010", Some("공 일 공")),
            (Lang::Fi, "1000", Some("tuhat")),
            (Lang::Fi, "1234", Some("tuhat 234")),
            (Lang::Fi, "12345", Some("12 tuhatta 345")),
            (Lang::Fi, "1000000,5", Some("miljoona pilkku 5")),
            (Lang::Fi, "123456", Some("123 tuhatta 456")),
            (Lang::Fi, "2,5", Some("2 pilkku 5")),
            (Lang::Fi, "-5", Some("miinus 5")),
            (Lang::Fi, "20-25", None),
            (Lang::Sv, "1234", Some("tusen 234")),
            (Lang::Sv, "12345", Some("12 tusen 345")),
            (Lang::Sv, "1000000,0", Some("en miljon komma 0")),
            (Lang::Sv, "2,5", Some("2 komma 5")),
            (Lang::Sv, "20-25", Some("20 till 25")),
            (Lang::No, "1234", Some("tusen 234")),
            (Lang::No, "1001", Some("tusen og 1")),
            (Lang::No, "81079", Some("81 tusen og 79")),
            (Lang::No, "-5", Some("minus 5")),
            (Lang::It, "1234", Some("mille 234")),
            (Lang::It, "1.500", Some("mille 500")),
            (Lang::It, "12.345", Some("12 mila 345")),
            (Lang::It, "1.234.567", Some("1 milione 234 mila 567")),
            (Lang::It, "2.000.000", Some("2 milioni")),
            (Lang::It, "2,5", Some("2 virgola 5")),
            (Lang::It, "-5", Some("meno 5")),
            (Lang::Nl, "1.500", Some("duizend 500")),
            (Lang::Nl, "70449", Some("70 duizend 449")),
            (Lang::Nl, "2.000.000", Some("2 miljoen")),
            (Lang::Nl, "1.234.567", Some("1 miljoen 234 duizend 567")),
            (Lang::Nl, "-5", Some("min 5")),
            (Lang::Nl, "20-25", Some("20 tot 25")),
            (Lang::Pl, "1000", Some("tysiąc")),
            (Lang::Pl, "2500", Some("2 tysiące 500")),
            (Lang::Pl, "5000", Some("5 tysięcy")),
            (Lang::Pl, "12000", Some("12 tysięcy")),
            (Lang::Pl, "22000", Some("22 tysiące")),
            (Lang::Pl, "1234567,5", Some("milion 234 tysiące 567 przecinek 5")),
            (Lang::Pl, "2000000,5", Some("2 miliony przecinek 5")),
            (Lang::Pl, "5000000,5", Some("5 milionów przecinek 5")),
            (Lang::Pl, "2,5", Some("2 przecinek 5")),
            (Lang::Pl, "20-25", Some("20 do 25")),
            // Identifiers: a plain run of seven or more digits, digit by digit.
            (
                Lang::Fi,
                "48213907",
                Some("neljä kahdeksan kaksi yksi kolme yhdeksän nolla seitsemän"),
            ),
            (Lang::Pl, "1234567.", Some("jeden dwa trzy cztery pięć sześć siedem.")),
            (Lang::Ar, "48213907", Some("أربعة ثمانية اثنان واحد ثلاثة تسعة صفر سبعة")),
            (Lang::Hi, "1234567", Some("एक दो तीन चार पाँच छह सात")),
            (Lang::It, "123456", Some("123 mila 456")),
            (Lang::En, "48213907", Some("48 million 213 thousand 907")),
            // A year and two digits is a range a few years on only.
            (Lang::No, "2026-77", None),
            (Lang::No, "2024-27", Some("2024 til 27")),
            (Lang::Ja, "3.10", Some("3 点 1 0")),
            (Lang::Ko, "2.75", None),
            (Lang::Ko, "12345.75", Some("1만 2345 점 7 5")),
            // Not claimed: a time, a word, a value past a trillion.
            (Lang::Fi, "10.30", None),
            (Lang::It, "10.30", None),
            (Lang::Pl, "10:30", None),
            (Lang::Ja, "hello", None),
            (Lang::Ko, "1234567890123", None),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(numbers(input, lang).as_deref(), expected, "{lang:?} {input:?}");
        }
    }

    #[test]
    fn hindi_reads_fractions_before_their_words() {
        assert_eq!(hindi_fraction(&["1.5", "लाख"], Lang::Hi), Some(("डेढ़".into(), 1)));
        assert_eq!(hindi_fraction(&["2.5", "करोड़"], Lang::Hi), Some(("ढाई".into(), 1)));
        assert_eq!(hindi_fraction(&["3.5", "हजार"], Lang::Hi), Some(("साढ़े तीन".into(), 1)));
        assert_eq!(hindi_fraction(&["1.25", "लाख"], Lang::Hi), Some(("सवा".into(), 1)));
        assert_eq!(hindi_fraction(&["1.75", "लाख."], Lang::Hi), Some(("पौने दो".into(), 1)));
        assert_eq!(hindi_fraction(&["1.5", "किलो"], Lang::Hi), None);
        assert_eq!(hindi_fraction(&["1.4", "लाख"], Lang::Hi), None);
    }

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
