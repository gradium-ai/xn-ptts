//! Hindi readings of what the numbers rule leaves as digits: clock times, dates, ranges,
//! ordinals, percentages, fractions, ratios, phone numbers, units after a number and
//! comparisons. The Hindi model reads digits in English, so every number here is spelled out.
//!
//! Clock times read as Hindi says them: "7:30" is "साढ़े सात", "2:15" "सवा दो" and "6:45" "पौने
//! सात" up to twelve o'clock, any other time "चौदह बजकर तीस मिनट", and a full hour "छह बजे".

use super::numbers::{hindi_words, number_words, spell};
use super::{all_digits, split_suffix};
use crate::preprocess::Lang;

/// What the words at the start of `words` read as, and how many of them that takes. `prev` is
/// the word before, as written.
pub(super) fn span(prev: Option<&str>, words: &[&str]) -> Option<(String, usize)> {
    let first = *words.first()?;
    phone(prev, words)
        .or_else(|| comparison(words))
        .or_else(|| operator(prev, words))
        .or_else(|| time(prev, words))
        .or_else(|| after_number(prev, first).map(|unit| (unit, 1)))
        .or_else(|| single(first, words.get(1).copied()).map(|spoken| (spoken, 1)))
}

/// `body` as a quantity: "1,050", "2.4", "-3".
fn quantity(body: &str) -> Option<String> {
    match number_words(body, Lang::Hi)? {
        (words, "") => Some(words),
        _ => None,
    }
}

/// Digits as a quantity, a leading zero dropped: the "07" of a time or a date.
fn cardinal(digits: &str) -> Option<String> {
    all_digits(digits).then(|| digits.parse().ok().map(hindi_words)).flatten()
}

/// One word on its own.
fn single(word: &str, next: Option<&str>) -> Option<String> {
    let (body, end) = split_suffix(word);
    if body == "नं" {
        return end.strip_prefix('.').map(|end| format!("नंबर{end}"));
    }
    let spoken = fraction(body)
        .or_else(|| score(body, next))
        .or_else(|| date(body))
        .or_else(|| percent(body))
        .or_else(|| ordinal(body))
        .or_else(|| dashed(body, next))?;
    Some(format!("{spoken}{end}"))
}

/// The months, as the dates read them.
const MONTHS: [&str; 12] = [
    "जनवरी",
    "फ़रवरी",
    "मार्च",
    "अप्रैल",
    "मई",
    "जून",
    "जुलाई",
    "अगस्त",
    "सितंबर",
    "अक्टूबर",
    "नवंबर",
    "दिसंबर",
];

/// A date written with digits: "14/03/2026", "2026-10-11", "14/03", and a card's "08/27".
fn date(body: &str) -> Option<String> {
    let month = |m: &str| {
        let n: usize = m.parse().ok()?;
        (m.len() <= 2 && (1..=12).contains(&n)).then(|| MONTHS[n - 1])
    };
    let day = |d: &str| {
        let n: u32 = d.parse().ok()?;
        ((1..=2).contains(&d.len()) && (1..=31).contains(&n)).then(|| cardinal(d)).flatten()
    };
    let year = |y: &str| (y.len() == 4 && all_digits(y)).then(|| quantity(y)).flatten();
    let parts: Vec<&str> = body.split(['/', '-', '.']).collect();
    if parts.len() < 2 || !parts.iter().all(|p| all_digits(p)) {
        return None;
    }
    let slashed = body.contains('/');
    match parts[..] {
        [y, m, d] if y.len() == 4 => Some(format!("{} {} {}", day(d)?, month(m)?, year(y)?)),
        [d, m, y] if y.len() == 4 => Some(format!("{} {} {}", day(d)?, month(m)?, year(y)?)),
        // "24/7" is round the clock, not a date.
        ["24", "7"] => None,
        // A card's expiry, "08/27": a month written with its zero and a year after 2012.
        [m, y] if slashed && m.len() == 2 && m.starts_with('0') && y.len() == 2 && y > "12" => {
            Some(format!("{} {}", month(m)?, hindi_words(2000 + y.parse::<i64>().ok()?)))
        }
        [d, m] if slashed => Some(format!("{} {}", day(d)?, month(m)?)),
        _ => None,
    }
}

/// Marks out of a total before a word for marks: "7/10 अंक" is "दस में से सात अंक", not a date.
fn score(body: &str, next: Option<&str>) -> Option<String> {
    let marks = next.is_some_and(|w| matches!(split_suffix(w).0, "अंक" | "अंकों" | "नंबर" | "मार्क्स"));
    let (got, of) = body.split_once('/').filter(|_| marks)?;
    let short = |s: &str| (1..=3).contains(&s.len()) && all_digits(s);
    (short(got) && short(of)).then(|| Some(format!("{} में से {}", quantity(of)?, quantity(got)?)))?
}

/// A fraction: "1/4" is "एक-चौथाई", "1½" "डेढ़", "¾" "तीन-चौथाई".
fn fraction(body: &str) -> Option<String> {
    if let Some((n, d)) = body.split_once('/') {
        return match (n, d) {
            ("1", "2") => Some("आधा".into()),
            ("1" | "2", "3") => Some(format!("{}-तिहाई", cardinal(n)?)),
            ("1" | "3", "4") => Some(format!("{}-चौथाई", cardinal(n)?)),
            _ => None,
        };
    }
    let last = body.chars().next_back()?;
    let whole = &body[..body.len() - last.len_utf8()];
    let n: i64 = if whole.is_empty() {
        0
    } else if all_digits(whole) && !whole.starts_with('0') {
        whole.parse().ok()?
    } else {
        return None;
    };
    Some(match (last, n) {
        ('½', 0) => "आधा".into(),
        ('½', 1) => "डेढ़".into(),
        ('½', 2) => "ढाई".into(),
        ('½', n) => format!("साढ़े {}", hindi_words(n)),
        ('¼', 0) => "एक-चौथाई".into(),
        ('¼', 1) => "सवा".into(),
        ('¼', n) => format!("सवा {}", hindi_words(n)),
        ('¾', 0) => "तीन-चौथाई".into(),
        ('¾', n) => format!("पौने {}", hindi_words(n + 1)),
        ('⅓', 0) => "एक-तिहाई".into(),
        ('⅔', 0) => "दो-तिहाई".into(),
        _ => return None,
    })
}

/// "15%", "3.5%", "10-20%".
fn percent(body: &str) -> Option<String> {
    let number = body.strip_suffix('%')?;
    let spoken = quantity(number).or_else(|| range(number, false))?;
    Some(format!("{spoken} प्रतिशत"))
}

/// A number with the ending of an ordinal: "1ला", "3रा", "4थी", "6ठा", "12वीं", "40वें".
fn ordinal(body: &str) -> Option<String> {
    let at = body.find(|c: char| !c.is_ascii_digit()).filter(|&at| at > 0)?;
    let (digits, ending) = body.split_at(at);
    if digits.starts_with('0') {
        return None;
    }
    // Which of the three forms: masculine, oblique, feminine.
    let form = match ending {
        "ला" | "रा" | "था" | "ठा" | "वाँ" | "वां" | "वा" => 0,
        "ले" | "रे" | "थे" | "ठे" | "वें" | "वे" => 1,
        "ली" | "री" | "थी" | "ठी" | "वीं" | "वी" => 2,
        _ => return None,
    };
    let n: i64 = digits.parse().ok()?;
    let own = match n {
        1 => ["पहला", "पहले", "पहली"],
        2 => ["दूसरा", "दूसरे", "दूसरी"],
        3 => ["तीसरा", "तीसरे", "तीसरी"],
        4 => ["चौथा", "चौथे", "चौथी"],
        6 => ["छठा", "छठे", "छठी"],
        _ => {
            // "सौवाँ", "हजारवाँ": a round hundred or thousand without its "एक".
            let words = hindi_words(n);
            let words = match n {
                100 | 1000 | 100_000 | 10_000_000 => words.trim_start_matches("एक ").to_string(),
                _ => words,
            };
            return Some(format!("{words}{}", ["वाँ", "वें", "वीं"][form]));
        }
    };
    Some(own[form].to_string())
}

/// Numbers joined by a dash: a range, "3-5" "तीन से पाँच", a score, "3-2" "तीन-दो", or a code
/// read digit by digit in its groups, "3391-8820" "तीन तीन नौ एक, आठ आठ दो शून्य". A range may
/// run down to a smaller number before बजे: "9-5 बजे".
fn dashed(body: &str, next: Option<&str>) -> Option<String> {
    let hours = next.is_some_and(|w| is_baje(split_suffix(w).0));
    if let Some(range) = range(body, hours) {
        return Some(range);
    }
    let parts: Vec<&str> = body.split('-').collect();
    if parts.len() < 2 || !parts.iter().all(|p| all_digits(p)) {
        return None;
    }
    let small = |p: &&str| p.len() <= 3 && (p.len() == 1 || !p.starts_with('0'));
    if parts.len() == 2 && parts.iter().all(small) {
        return Some(format!("{}-{}", cardinal(parts[0])?, cardinal(parts[1])?));
    }
    Some(parts.iter().map(|p| spell(p, Lang::Hi)).collect::<Vec<_>>().join(", "))
}

/// "3-5", "0-2", "201-210", "2019-2023", "2024-25"; `any_order` takes "9-5" too.
fn range(body: &str, any_order: bool) -> Option<String> {
    let (from, to) = body.split_once('-')?;
    let small = |s: &str| {
        (1..=3).contains(&s.len()) && all_digits(s) && (s.len() == 1 || !s.starts_with('0'))
    };
    let year =
        |s: &str| s.len() == 4 && all_digits(s) && (s.starts_with('1') || s.starts_with("20"));
    let ok = match (from.len(), to.len()) {
        _ if small(from) && small(to) => {
            any_order || from.parse::<u32>().ok()? < to.parse::<u32>().ok()?
        }
        (4, 4) => year(from) && year(to) && from < to,
        // "2024-25", a few years on only: "2026-77" is a code.
        (4, 2) => {
            year(from)
                && all_digits(to)
                && &from[2..] < to
                && to.parse::<u32>().ok()? <= from[2..].parse::<u32>().ok()? + 10
        }
        _ => false,
    };
    ok.then(|| Some(format!("{} से {}", quantity(from)?, quantity(to)?))).flatten()
}

/// How Hindi says a clock time.
enum Clock {
    /// A full hour, with its बजे: "छह बजे".
    Hour(String),
    /// A quarter past, half past or quarter to, which take बजे after them as they stand:
    /// "साढ़े सात".
    Quarter(String),
    /// Any other time, which takes no बजे: "चौदह बजकर तीस मिनट".
    Minutes(String),
}

impl Clock {
    /// The time in `body`, hours and minutes split by `sep`.
    fn parse(body: &str, sep: char) -> Option<Self> {
        let (h, m) = body.split_once(sep)?;
        if !(1..=2).contains(&h.len()) || m.len() != 2 || !all_digits(h) || !all_digits(m) {
            return None;
        }
        let (h, m): (i64, i64) = (h.parse().ok()?, m.parse().ok()?);
        if h > 24 || m > 59 || (h == 24 && m > 0) {
            return None;
        }
        // Midnight is twelve.
        let hour = hindi_words(if h == 0 { 12 } else { h });
        let twelve = (1..=12).contains(&h);
        Some(match m {
            0 => Self::Hour(format!("{hour} बजे")),
            30 if h == 1 => Self::Quarter("डेढ़".into()),
            30 if h == 2 => Self::Quarter("ढाई".into()),
            30 if twelve => Self::Quarter(format!("साढ़े {hour}")),
            15 if twelve => Self::Quarter(format!("सवा {hour}")),
            45 if twelve => Self::Quarter(format!("पौने {}", hindi_words(h % 12 + 1))),
            _ => Self::Minutes(format!("{hour} बजकर {} मिनट", hindi_words(m))),
        })
    }

    fn alone(self) -> String {
        match self {
            Self::Hour(s) | Self::Quarter(s) | Self::Minutes(s) => s,
        }
    }
}

/// The words a time may be followed by when its बजे goes: "14:30 बजे के लिए" is "चौदह बजकर तीस
/// मिनट के लिए", where "14:30 बजे है" is "चौदह बजकर तीस मिनट पर है".
const POSTPOSITIONS: [&str; 11] =
    ["के", "का", "की", "को", "तक", "से", "पर", "में", "वाला", "वाले", "वाली"];

/// The ratio words a pair of numbers with a colon follows.
const RATIOS: [&str; 4] = ["अनुपात", "रेशियो", "रेश्यो", "ratio"];

/// Whether `word` is a form of बजना that follows a time ("बजे", "बजकर"), not "बजट".
fn is_baje(word: &str) -> bool {
    matches!(word, "बज" | "बजे" | "बजकर" | "बजने")
}

/// A clock time, a range of them, a duration or a ratio, and the बजे after a time.
fn time(prev: Option<&str>, words: &[&str]) -> Option<(String, usize)> {
    let (body, end) = split_suffix(words.first()?);
    let next = words.get(1).map(|w| split_suffix(w));
    let ratio = prev.is_some_and(|p| RATIOS.contains(&split_suffix(p).0));
    // Hindi writes a time with a period before बजे: "11.30 बजे".
    let before_baje = next.is_some_and(|(n, _)| is_baje(n));
    let mut clock = Clock::parse(body, ':').filter(|_| !ratio);
    if clock.is_none() && before_baje {
        clock = Clock::parse(body, '.');
    }
    let mut lead = String::new();
    if clock.is_none()
        && let Some((from, to)) = body.split_once('-')
        && let (Some(from), Some(to)) = (Clock::parse(from, ':'), Clock::parse(to, ':'))
    {
        lead = format!("{} से ", from.alone());
        clock = Some(to);
    }
    let Some(clock) = clock else {
        let spoken = duration(body).or_else(|| colon_ratio(body, ratio))?;
        return Some((format!("{spoken}{end}"), 1));
    };
    if !end.is_empty() {
        return Some((format!("{lead}{}{end}", clock.alone()), 1));
    }
    let baje = next.filter(|(n, _)| *n == "बजे");
    Some(match (clock, baje) {
        (Clock::Hour(hour), Some((_, after))) => (format!("{lead}{hour}{after}"), 2),
        (Clock::Minutes(time), Some((_, after))) => {
            let post = words.get(2).is_some_and(|w| POSTPOSITIONS.contains(&split_suffix(w).0));
            let at = if after.is_empty() && !post { " पर" } else { "" };
            (format!("{lead}{time}{at}{after}"), 2)
        }
        (clock, _) => (format!("{lead}{}", clock.alone()), 1),
    })
}

/// "2:03:45" in hours, minutes and seconds, and "48:12" in minutes and seconds when it is too
/// many for a time.
fn duration(body: &str) -> Option<String> {
    let parts: Vec<&str> = body.split(':').collect();
    let two = |p: &&str| p.len() == 2 && all_digits(p) && p < &"60";
    let (hours, minutes, seconds) = match parts[..] {
        [h, m, s] if (1..=2).contains(&h.len()) && all_digits(h) && two(&m) && two(&s) => {
            (Some(h), m, s)
        }
        [m, s] if m.len() == 2 && all_digits(m) && m >= "24" && two(&s) => (None, m, s),
        _ => return None,
    };
    let mut out = vec![];
    if let Some(h) = hours.filter(|h| !h.trim_start_matches('0').is_empty()) {
        let unit = if h.trim_start_matches('0') == "1" { "घंटा" } else { "घंटे" };
        out.push(format!("{} {unit}", cardinal(h)?));
    }
    for (n, unit) in [(minutes, "मिनट"), (seconds, "सेकंड")] {
        if n != "00" {
            out.push(format!("{} {unit}", cardinal(n)?));
        }
    }
    Some(out.join(" "))
}

/// "16:9", "1:25,000": a colon before anything but two digits, which make a time.
fn colon_ratio(body: &str, ratio: bool) -> Option<String> {
    let (a, b) = body.split_once(':')?;
    if !ratio && b.len() == 2 && all_digits(b) {
        return None;
    }
    if !all_digits(a) || !b.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    Some(format!("{} अनुपात {}", quantity(a)?, quantity(b)?))
}

/// The words before a number that make it a code read digit by digit, with the groups after
/// it: "पिन 4471", "कोड 302 118".
const CODES: [&str; 6] = ["पिन", "pin", "कोड", "एक्सटेंशन", "otp", "ओटीपी"];

/// A phone number or a code, read digit by digit in its groups: "+1 (415) 555-0132", "011-4946
/// 0018", "98765 43210", "पिन 4471".
fn phone(prev: Option<&str>, words: &[&str]) -> Option<(String, usize)> {
    let (first, mut end) = split_suffix(words.first()?);
    let digits = |s: &str| spell(s, Lang::Hi);
    let mut groups = vec![];
    let plus = first.strip_prefix('+');
    if let Some(number) = plus {
        let mut parts = number.split('-');
        let code = parts.next()?;
        if !(1..=3).contains(&code.len()) || !all_digits(code) {
            return None;
        }
        groups.push(format!("प्लस {}", digits(code)));
        for part in parts {
            groups.push(all_digits(part).then(|| digits(part))?);
        }
    } else if let Some((area, rest)) = first.split_once('-')
        && area.starts_with('0')
        && (3..=5).contains(&area.len())
        && all_digits(area)
        && (3..=4).contains(&rest.len())
        && all_digits(rest)
    {
        groups.extend([digits(area), digits(rest)]);
    } else if prev.is_some_and(|p| CODES.contains(&split_suffix(p).0.to_lowercase().as_str()))
        && (3..=8).contains(&first.len())
        && all_digits(first)
    {
        groups.push(digits(first));
    } else if first.len() == 5
        && all_digits(first)
        && end.is_empty()
        && words.get(1).is_some_and(|w| {
            let next = split_suffix(w).0;
            next.len() == 5 && all_digits(next)
        })
    {
        // A mobile number in two groups of five.
        groups.push(digits(first));
    } else {
        return None;
    }
    let mut used = 1;
    while end.is_empty() || end == "," {
        let Some((body, after)) = words.get(used).map(|w| split_suffix(w)) else { break };
        let parts: Vec<&str> = body.split('-').collect();
        if !parts.iter().all(|p| (2..=5).contains(&p.len()) && all_digits(p)) {
            break;
        }
        groups.extend(parts.into_iter().map(digits));
        end = after;
        used += 1;
    }
    // A plus and a number is a phone number only with enough digits after it: "+5 अंक" is not.
    if plus.is_some() && groups.iter().map(|g| g.split(' ').count()).sum::<usize>() < 8 {
        return None;
    }
    Some((format!("{}{end}", groups.join(", ")), used))
}

/// A comparison before a number, which Hindi says after it: "< 100" is "एक सौ से कम", and "≥ 18",
/// which the character pass has spelled, "अठारह या उससे अधिक".
fn comparison(words: &[&str]) -> Option<(String, usize)> {
    let (than, skip) = match words {
        ["<", ..] => ("से कम", 1),
        [">", ..] => ("से अधिक", 1),
        ["से", "कम", "या", "बराबर", ..] => ("या उससे कम", 4),
        ["से", "अधिक", "या", "बराबर", ..] => ("या उससे अधिक", 4),
        _ => return None,
    };
    let (body, mut end) = split_suffix(words.get(skip)?);
    let mut spoken = quantity(body)?;
    let mut used = skip + 1;
    if end.is_empty()
        && let Some((unit, after)) = words.get(used).and_then(|w| unit(w, body))
    {
        spoken = format!("{spoken} {unit}");
        end = after;
        used += 1;
    }
    Some((format!("{spoken} {than}{end}"), used))
}

/// A dash between two numbers is a minus before a result ("7 - 3 = 4") and a range otherwise,
/// and the × the character pass spelled "गुणा" is "बाई" between two lengths ("3 × 4 मीटर") and
/// "गुना" after a number alone ("2× स्पीड"). A product with its result stays "गुणा".
fn operator(prev: Option<&str>, words: &[&str]) -> Option<(String, usize)> {
    let [word, next, ..] = words else { return None };
    let prev_word = prev?;
    let (prev, prev_end) = split_suffix(prev_word);
    let after_number = prev_end.is_empty() && prev.ends_with(|c: char| c.is_ascii_digit());
    let before_number = next.starts_with(|c: char| c.is_ascii_digit());
    let equals = words.iter().skip(2).take(3).any(|w| split_suffix(w).0 == "बराबर");
    match *word {
        "-" if after_number && before_number && equals => Some(("माइनस".into(), 1)),
        "-" if after_number && before_number => {
            let (from, to) = (prev.parse::<f64>().ok()?, split_suffix(next).0.parse::<f64>().ok()?);
            (from < to).then(|| ("से".into(), 1))
        }
        "गुणा" if before_number && (after_number || unit(prev_word, "").is_some()) => {
            (!equals).then(|| ("बाई".into(), 1))
        }
        "गुणा" if after_number => Some(("गुना".into(), 1)),
        _ => None,
    }
}

/// A unit written short after a number: "2.4 कि.ग्रा.", "250 ml", "30 कि.मी./घंटा".
fn after_number(prev: Option<&str>, word: &str) -> Option<String> {
    let (prev, prev_end) = split_suffix(prev?);
    let count = prev.rsplit('-').next().unwrap_or(prev);
    let number = prev_end.is_empty() && prev.ends_with(|c: char| c.is_ascii_digit());
    if !number && !matches!(prev, "वर्ग" | "घन") {
        return None;
    }
    let (unit, end) = unit(word, count)?;
    Some(format!("{unit}{end}"))
}

/// The unit `word` names, and the punctuation after it; `count` is the number before it.
fn unit<'a>(word: &'a str, count: &str) -> Option<(String, &'a str)> {
    let (body, end) = split_suffix(word);
    if let Some((unit, per)) = body.split_once('/') {
        let unit = short_unit(unit.trim_end_matches('.'), true, count)?;
        let per = match per.trim_end_matches('.') {
            "घंटा" | "घं" | "h" | "hr" => "घंटा",
            "मिनट" | "मि" | "min" => "मिनट",
            "सेकंड" | "से" | "s" => "सेकंड",
            "दिन" | "day" => "दिन",
            _ => return None,
        };
        return Some((format!("{unit} प्रति {per}"), end));
    }
    // A Devanagari abbreviation ends with its period, which is not the sentence's.
    let (dotted, end) = match end.strip_prefix('.') {
        Some(rest) => (true, rest),
        None => (body.contains('.'), end),
    };
    if let Some(unit) = short_unit(body, dotted, count) {
        return Some((unit.to_string(), end));
    }
    let latin = match body {
        "ml" | "mL" => "मिलीलीटर",
        "mg" => "मिलीग्राम",
        "kg" => "किलोग्राम",
        "km" => "किलोमीटर",
        "cm" => "सेंटीमीटर",
        "mm" => "मिलीमीटर",
        "km/h" | "kmph" => "किलोमीटर प्रति घंटा",
        "mph" => "मील प्रति घंटा",
        "kWh" => "किलोवाट घंटा",
        "kW" => "किलोवाट",
        "mAh" => "एमएएच",
        "hp" => "हॉर्सपावर",
        "KB" => "केबी",
        "MB" => "एमबी",
        "GB" => "जीबी",
        "TB" => "टीबी",
        "Hz" => "हर्ट्ज़",
        "MHz" => "मेगाहर्ट्ज़",
        "GHz" => "गीगाहर्ट्ज़",
        "tsp" => "टी-स्पून",
        "tbsp" => "टेबल-स्पून",
        "bar" => "बार",
        _ => return None,
    };
    Some((latin.to_string(), end))
}

/// The Devanagari abbreviation of a unit, without its last period, when written `dotted`.
fn short_unit(body: &str, dotted: bool, count: &str) -> Option<&'static str> {
    if !dotted {
        return None;
    }
    Some(match body {
        "कि.ग्रा" => "किलोग्राम",
        "ग्रा" => "ग्राम",
        "मि.ग्रा" => "मिलीग्राम",
        "कि.मी" => "किलोमीटर",
        "मी" => "मीटर",
        "से.मी" | "सें.मी" => "सेंटीमीटर",
        "मि.मी" => "मिलीमीटर",
        "ली" => "लीटर",
        "मि.ली" => "मिलीलीटर",
        "हे" => "हेक्टेयर",
        "मि" => "मिनट",
        "सेकं" => "सेकंड",
        "घं" if count == "1" => "घंटा",
        "घं" => "घंटे",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(words: &str) -> Option<(String, usize)> {
        let words: Vec<&str> = words.split(' ').collect();
        span(None, &words)
    }

    #[test]
    fn times_read_as_hindi_says_them() {
        let cases = [
            ("7:30 बजे", "साढ़े सात", 1),
            ("2:15 बजे", "सवा दो", 1),
            ("6:45", "पौने सात", 1),
            ("12:45", "पौने एक", 1),
            ("1:30", "डेढ़", 1),
            ("2:30", "ढाई", 1),
            ("11.30 बजे", "साढ़े ग्यारह", 1),
            ("06:00 बजे का", "छह बजे", 2),
            ("17:00।", "सत्रह बजे।", 1),
            ("11:00 से", "ग्यारह बजे", 1),
            ("14:30 बजे के लिए", "चौदह बजकर तीस मिनट", 2),
            ("07:05 बजे छूटेगी", "सात बजकर पाँच मिनट पर", 2),
            ("23:50 बजे।", "तेईस बजकर पचास मिनट।", 2),
            ("00:15 बजे", "बारह बजकर पन्द्रह मिनट पर", 2),
            ("10:40 की", "दस बजकर चालीस मिनट", 1),
            ("12:00-14:00 बजे तक", "बारह बजे से चौदह बजे", 2),
            ("2:03:45", "दो घंटे तीन मिनट पैंतालीस सेकंड", 1),
            ("48:12", "अड़तालीस मिनट बारह सेकंड", 1),
            ("16:9", "सोलह अनुपात नौ", 1),
            ("1:25,000", "एक अनुपात पच्चीस हजार", 1),
        ];
        for (input, expected, used) in cases {
            assert_eq!(read(input), Some((expected.into(), used)), "{input:?}");
        }
        assert_eq!(span(Some("अनुपात"), &["16:10"]), Some(("सोलह अनुपात दस".into(), 1)));
        // A period makes a time only before बजे, and minutes run to 59.
        assert_eq!(read("11.30 मिनट"), None);
        assert_eq!(read("11.3 बजे"), None);
        assert_eq!(read("25:61"), None);
    }

    #[test]
    fn dates_ranges_and_codes() {
        let cases = [
            ("14/03/2026", "चौदह मार्च दो हजार छब्बीस"),
            ("07/04/1990", "सात अप्रैल उन्नीस सौ नब्बे"),
            ("2026-10-11", "ग्यारह अक्टूबर दो हजार छब्बीस"),
            ("12/10", "बारह अक्टूबर"),
            ("08/27", "अगस्त दो हजार सत्ताईस"),
            ("3-5", "तीन से पाँच"),
            ("0-2", "शून्य से दो"),
            ("201-210", "दो सौ एक से दो सौ दस"),
            ("2019-2023", "दो हजार उन्नीस से दो हजार तेईस"),
            ("2024-25", "दो हजार चौबीस से पच्चीस"),
            ("3-2", "तीन-दो"),
            ("2026-77", "दो शून्य दो छह, सात सात"),
            ("91-220-0017", "नौ एक, दो दो शून्य, शून्य शून्य एक सात"),
        ];
        for (input, expected) in cases {
            assert_eq!(read(input), Some((expected.into(), 1)), "{input:?}");
        }
        assert_eq!(read("9-5 बजे"), Some(("नौ से पाँच".into(), 1)));
        assert_eq!(read("32/13"), None);
        assert_eq!(read("24/7"), None);
    }

    #[test]
    fn ordinals_percents_and_fractions() {
        let cases = [
            ("1ला", "पहला"),
            ("2रे", "दूसरे"),
            ("3रा", "तीसरा"),
            ("4थी", "चौथी"),
            ("6ठा", "छठा"),
            ("8वीं", "आठवीं"),
            ("21वीं", "इक्कीसवीं"),
            ("40वें", "चालीसवें"),
            ("100वें", "सौवें"),
            ("12वाँ", "बारहवाँ"),
            ("15%", "पन्द्रह प्रतिशत"),
            ("3.5%", "तीन दशमलव पाँच प्रतिशत"),
            ("0.5%", "शून्य दशमलव पाँच प्रतिशत"),
            ("100%", "एक सौ प्रतिशत"),
            ("10-20%", "दस से बीस प्रतिशत"),
            ("1/4", "एक-चौथाई"),
            ("3/4", "तीन-चौथाई"),
            ("1/2", "आधा"),
            ("1½", "डेढ़"),
            ("3½", "साढ़े तीन"),
            ("¾", "तीन-चौथाई"),
            ("नं.", "नंबर"),
        ];
        for (input, expected) in cases {
            assert_eq!(read(input), Some((expected.into(), 1)), "{input:?}");
        }
        assert_eq!(read("नं"), None);
        assert_eq!(read("3बजे"), None);
    }

    #[test]
    fn phones_and_codes_read_digit_by_digit() {
        assert_eq!(
            read("011-4946 0018 पर"),
            Some(("शून्य एक एक, चार नौ चार छह, शून्य शून्य एक आठ".into(), 2))
        );
        assert_eq!(
            read("+1, 415, 555-0132 पर"),
            Some(("प्लस एक, चार एक पाँच, पाँच पाँच पाँच, शून्य एक तीन दो".into(), 3))
        );
        assert_eq!(span(Some("पिन"), &["4471", "है।"]), Some(("चार चार सात एक".into(), 1)));
        assert_eq!(
            span(Some("कोड"), &["302", "118", "डालें।"]),
            Some(("तीन शून्य दो, एक एक आठ".into(), 2))
        );
        assert_eq!(
            read("+91-98765-43210"),
            Some(("प्लस नौ एक, नौ आठ सात छह पाँच, चार तीन दो एक शून्य".into(), 1))
        );
        assert_eq!(read("98765 43210 पर"), Some(("नौ आठ सात छह पाँच, चार तीन दो एक शून्य".into(), 2)));
        // A plus before a small number is not a phone number.
        assert_eq!(read("+5 अंक"), None);
        // Marks out of a total, not a date.
        assert_eq!(read("7/10 अंक").map(|(s, _)| s).as_deref(), Some("दस में से सात"));
        assert_eq!(read("12/10").map(|(s, _)| s).as_deref(), Some("बारह अक्टूबर"));
        // A five-digit number before a five-letter word is a count, not a mobile number.
        assert_eq!(read("10000 steps"), None);
        assert_eq!(span(Some("कुल"), &["4471"]), None);
    }

    #[test]
    fn units_operators_and_comparisons() {
        let after = |prev, word| span(Some(prev), &[word]).map(|(s, _)| s);
        assert_eq!(after("2.4", "कि.ग्रा.").as_deref(), Some("किलोग्राम"));
        assert_eq!(after("2", "कि.ग्रा.।").as_deref(), Some("किलोग्राम।"));
        assert_eq!(after("25", "मी.").as_deref(), Some("मीटर"));
        assert_eq!(after("वर्ग", "मी.").as_deref(), Some("मीटर"));
        assert_eq!(after("1", "घं.").as_deref(), Some("घंटा"));
        assert_eq!(after("80-100", "कि.मी./घंटा").as_deref(), Some("किलोमीटर प्रति घंटा"));
        assert_eq!(after("250", "ml").as_deref(), Some("मिलीलीटर"));
        // Without its period or a number before it, the word is left alone.
        assert_eq!(after("25", "मी"), None);
        assert_eq!(after("वह", "मी."), None);
        assert_eq!(read("< 100 होना"), Some(("एक सौ से कम".into(), 2)));
        assert_eq!(read("से अधिक या बराबर 18 हो।"), Some(("अठारह या उससे अधिक".into(), 5)));
        assert_eq!(read("से कम या बराबर 5 कि.ग्रा. रखें।"), Some(("पाँच किलोग्राम या उससे कम".into(), 6)));
        let op = |prev, words: &[&str]| span(Some(prev), words).map(|(s, _)| s);
        assert_eq!(op("3", &["गुणा", "4", "मीटर"]).as_deref(), Some("बाई"));
        assert_eq!(op("मी.", &["गुणा", "1", "मी."]).as_deref(), Some("बाई"));
        assert_eq!(op("3", &["गुणा", "4", "बराबर", "12"]), None);
        assert_eq!(op("2", &["गुणा", "स्पीड"]).as_deref(), Some("गुना"));
        assert_eq!(op("7", &["-", "3", "बराबर", "4"]).as_deref(), Some("माइनस"));
        assert_eq!(op("10", &["-", "15", "मिनट"]).as_deref(), Some("से"));
        assert_eq!(op("7", &["-", "3"]), None);
        assert_eq!(op("दो", &["गुणा", "तीन"]), None);
    }

    /// Whole sentences, through the character pass and the other rules.
    #[test]
    fn sentences() {
        let cases = [
            ("हाँ, 3-5 दिन।", "हाँ, तीन से पाँच दिन।"),
            ("आपकी कॉल 14:30 बजे के लिए बुक है।", "आपकी कॉल चौदह बजकर तीस मिनट के लिए बुक है।"),
            ("अगला स्लॉट 11:20 बजे है।", "अगला स्लॉट ग्यारह बजकर बीस मिनट पर है।"),
            ("दरवाज़े शाम 7:30 बजे खुलेंगे।", "दरवाज़े शाम साढ़े सात बजे खुलेंगे।"),
            ("आपकी शिफ़्ट 22:00 से 06:00 तक है।", "आपकी शिफ़्ट बाईस बजे से छह बजे तक है।"),
            ("यह आपका 3रा रिमाइंडर है।", "यह आपका तीसरा रिमाइंडर है।"),
            ("हमारी 10वीं सालगिरह जून में है।", "हमारी दसवीं सालगिरह जून में है।"),
            ("बैटरी 23% है।", "बैटरी तेईस प्रतिशत है।"),
            ("आपका जन्म 07/04/1990 को हुआ था।", "आपका जन्म सात अप्रैल उन्नीस सौ नब्बे को हुआ था।"),
            ("ठीक है, 2 कि.ग्रा.।", "ठीक है, दो किलोग्राम।"),
            ("कमरा 3 × 4 मीटर का है।", "कमरा तीन बाई चार मीटर का है।"),
            ("7 − 3 = 4", "सात माइनस तीन बराबर चार"),
            ("ऐसा मान दर्ज करें जो ≥ 18 हो।", "ऐसा मान दर्ज करें जो अठारह या उससे अधिक हो।"),
            ("हो गया, ऑर्डर #4471।", "हो गया, ऑर्डर नंबर चार हजार चार सौ इकहत्तर।"),
            (
                "उनसे +1 (415) 555-0132 पर संपर्क करें।",
                "उनसे प्लस एक, चार एक पाँच, पाँच पाँच पाँच, शून्य एक तीन दो पर संपर्क करें।",
            ),
            ("कुल योग ₹3.5 मिलियन है।", "कुल योग तीन दशमलव पाँच मिलियन रुपये है।"),
            ("बजट ₹2 लाख है।", "बजट दो लाख रुपये है।"),
            ("बजट ₹1.5 लाख है।", "बजट डेढ़ लाख रुपये है।"),
        ];
        for (input, expected) in cases {
            let rules = crate::preprocess::Rules::DEFAULT;
            assert_eq!(crate::preprocess::normalize_text(input, Lang::Hi, rules), expected);
        }
    }
}
