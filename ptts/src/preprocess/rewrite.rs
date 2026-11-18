//! Word-level rewrites run by [`super::normalize_text`], one whitespace-separated word at a
//! time. The first rule in [`RULES`] that claims a word wins, and rules are never chained.

use super::Lang;

mod abbreviations;
mod codes;
mod currency;
mod dashed_digits;
mod dates;
mod elongations;
mod emails;
mod hindi;
mod japanese;
mod letters;
mod local_dates;
mod numbers;
mod phones;
mod times;
mod tlds;
mod urls;

pub(super) use currency::is_symbol as is_currency_symbol;

/// A rule: it rewrites a whole word in a language, or declines it.
type Rewrite = fn(&str, Lang) -> Option<String>;

/// A row of [`RULES`].
struct Rule {
    /// What the rule goes by in the frontends' `--rewrites` flag.
    name: &'static str,
    rewrite: Rewrite,
    /// Whether [`Rules::DEFAULT`] runs it.
    default: bool,
}

/// Every rule, in the order they are tried. Adding a rule is a module and a row here.
///
/// Phone numbers, times and dates are left out of the defaults, as the serving stack leaves them
/// out: "1/2" is as often a fraction as a date, and a run of digits as often a quantity as a phone
/// number. Numbers and times are the only rules that claim the same words, German ones like
/// "1.234", and numbers come first so that those read as a thousand rather than a time.
/// Swedish and Norwegian dates and times have a rule of their own, on by default, which reads
/// only the shapes that cannot be anything else.
const RULES: &[Rule] = &[
    Rule { name: "numbers", rewrite: numbers::numbers, default: true },
    Rule { name: "currency", rewrite: currency::currency, default: true },
    Rule { name: "dashed-digits", rewrite: dashed_digits::dashed_digits, default: true },
    Rule { name: "codes", rewrite: codes::codes, default: true },
    Rule { name: "emails", rewrite: emails::emails, default: true },
    Rule { name: "urls", rewrite: urls::urls, default: true },
    Rule { name: "abbreviations", rewrite: abbreviations::abbreviations, default: true },
    Rule { name: "elongations", rewrite: elongations::elongations, default: true },
    Rule { name: "local-dates", rewrite: local_dates::local_dates, default: true },
    Rule { name: "phones", rewrite: phones::phones, default: false },
    Rule { name: "times", rewrite: times::times, default: false },
    Rule { name: "dates", rewrite: dates::dates, default: false },
];

/// Which of the rewrite rules run, on top of the character normalization.
///
/// It travels on [`super::Normalize`] rather than beside it, so that callers which tokenize by
/// hand cannot pick up the language and forget the rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules(u32);

impl Rules {
    /// No rewrites: the character pass alone.
    pub const NONE: Self = Self(0);

    /// The rules that run unless a caller picks others: every rule but phone numbers, times and
    /// dates.
    pub const DEFAULT: Self = {
        let mut bits = 0;
        let mut i = 0;
        while i < RULES.len() {
            if RULES[i].default {
                bits |= 1 << i;
            }
            i += 1;
        }
        Self(bits)
    };

    /// Every rule this version implements.
    pub const ALL: Self = Self((1 << RULES.len()) - 1);
}

/// Parse the frontends' `--rewrites` flag: `default`, `all`, `none`, or a comma-separated list
/// of rule names.
impl std::str::FromStr for Rules {
    type Err = crate::Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "default" => Ok(Self::DEFAULT),
            "all" => Ok(Self::ALL),
            "none" | "off" => Ok(Self::NONE),
            list => list.split(',').try_fold(Self::NONE, |set, name| {
                match RULES.iter().position(|rule| rule.name == name.trim()) {
                    Some(i) => Ok(Self(set.0 | 1 << i)),
                    None => Err(crate::Error::invalid_argument(format!(
                        "unknown rewrite rule: {name}; expected default, all, none, or any of: {}",
                        RULES.iter().map(|rule| rule.name).collect::<Vec<_>>().join(", ")
                    ))),
                }
            }),
        }
    }
}

/// Rewrite `word` under `rules`, or `None` when no rule claims it.
pub fn rewrite_word(word: &str, lang: Lang, rules: Rules) -> Option<String> {
    // Japanese and Korean join a number to the word around it, "12345人", "5,000원입니다": the
    // rules read each run of digits inside such a word on its own.
    if matches!(lang, Lang::Ja | Lang::Ko) && !word.is_ascii() {
        let marked = (lang == Lang::Ja && rules.has("numbers")).then(|| japanese::marks(word));
        let marked = marked.flatten();
        let word = marked.as_deref().unwrap_or(word);
        return codes::within(word, lang, rules)
            .or_else(|| rewrite_runs(word, lang, rules))
            .or(marked);
    }
    if lang == Lang::Ja {
        let ascii = japanese::ascii_word(word, rules.has("numbers"), rules.has("dashed-digits"));
        if ascii.is_some() {
            return ascii;
        }
    }
    // A plain run of seven digits or more on its own is a code there too.
    if matches!(lang, Lang::Ja | Lang::Ko) && rules.has("numbers") {
        let (body, end) = split_suffix(word);
        if body.len() >= 7 && all_digits(body) {
            return Some(format!("{}{end}", numbers::spell(body, lang)));
        }
    }
    claim(word, lang, rules)
}

/// Whether `prev`, the word before, is a number sign: "nr.", "No.", "nummer". Hindi spells
/// every number out, so it keeps its rule there too, and the languages the serving stack reads
/// keep its readings.
fn numbered(prev: Option<&str>, lang: Lang) -> bool {
    let Some(prev) = prev else { return false };
    let prev = prev.to_lowercase();
    !matches!(lang, Lang::Hi | Lang::En | Lang::Fr | Lang::De | Lang::Es | Lang::Pt)
        && (matches!(prev.as_str(), "nr" | "nr." | "nro" | "nro." | "no." | "n." | "nº" | "#")
            || prev.ends_with(".nr.")
            || prev == lang.number_sign())
}

/// The counters a Japanese or Korean count is written before.
fn counters(lang: Lang) -> &'static str {
    match lang {
        Lang::Ja => "人名円個件本枚台冊回年歳点匹杯分秒時日月倍万億",
        Lang::Ko => "명원개건권대장회년살세분초시월배점만억",
        _ => "",
    }
}

/// Whether `body`, a word without its closing point, is a short address rather than a dotted
/// abbreviation: a first part of two letters or more and a common top-level domain, "rai.it",
/// "nu.nl". Abbreviations start with a single letter ("t.ex", "m.in") or end on one that is no
/// such domain ("ref.nr").
fn ends_a_sentence_as_address(body: &str) -> bool {
    const COMMON: &[&str] = &[
        "com", "org", "net", "info", "eu", "io", "it", "nl", "se", "no", "fi", "pl", "dk", "de",
        "fr", "es", "pt", "uk", "in", "jp", "kr", "ae", "sa", "eg", "ma",
    ];
    let labels: Vec<&str> = body.split('.').collect();
    labels.len() >= 2
        && labels[0].len() >= 2
        && labels.last().is_some_and(|tld| COMMON.contains(&tld.to_lowercase().as_str()))
}

/// What the first rule of `rules` that claims `word` makes of it.
fn claim(word: &str, lang: Lang, rules: Rules) -> Option<String> {
    // A dotted abbreviation, "ref.nr.", "t.ex.", is not an address: every part short letters and
    // a closing point. An address that ends a sentence, "rai.it.", still is one.
    let body = word.strip_suffix('.');
    if !matches!(lang, Lang::En | Lang::Fr | Lang::De | Lang::Es | Lang::Pt)
        && word.contains('.')
        && body.is_some_and(|w| {
            w.split('.').all(|p| (1..=4).contains(&p.len()) && p.chars().all(char::is_alphabetic))
        })
        && !body.is_some_and(ends_a_sentence_as_address)
    {
        return None;
    }
    RULES
        .iter()
        .enumerate()
        .filter(|(i, _)| rules.0 & (1 << i) != 0)
        .find_map(|(_, rule)| (rule.rewrite)(word, lang))
}

/// `word` with every run of digits inside it rewritten as a word of its own, or `None` when no
/// run changes. A run holds digits and the points, commas, colons, slashes and dashes between
/// them, and takes a minus or a currency symbol before it ("-3.5", "¥1500"), a symbol after
/// it, and the currency unit written right after it ("1,500円"). A run after a Latin letter is
/// a code ("ABC123") and stays.
fn rewrite_runs(word: &str, lang: Lang, rules: Rules) -> Option<String> {
    let units: &[&str] = match lang {
        Lang::Ja => &["円", "ドル", "ユーロ"],
        Lang::Ko => &["원", "달러", "유로", "엔"],
        _ => &[],
    };
    let chars: Vec<(usize, char)> = word.char_indices().collect();
    let mut out = String::new();
    let mut changed = false;
    let mut i = 0;
    while i < chars.len() {
        let (start, c) = chars[i];
        let before_digit = chars.get(i + 1).is_some_and(|(_, d)| d.is_ascii_digit());
        let after_mark =
            !chars.get(i.wrapping_sub(1)).is_some_and(|(_, p)| p.is_ascii_alphanumeric());
        let leads = before_digit && (is_currency_symbol(c) || (c == '-' && after_mark));
        if !c.is_ascii_digit() && !leads {
            out.push(c);
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < chars.len() {
            let c = chars[j].1;
            let joins = matches!(c, '.' | ',' | ':' | '-' | '/')
                && chars.get(j + 1).is_some_and(|(_, d)| d.is_ascii_digit());
            if !c.is_ascii_digit() && !joins {
                break;
            }
            j += 1;
        }
        let mut end = chars.get(j).map_or(word.len(), |(k, _)| *k);
        if let Some(unit) = units.iter().find(|unit| word[end..].starts_with(*unit)) {
            end += unit.len();
        } else if let Some(symbol) =
            chars.get(j).map(|(_, c)| *c).filter(|&c| is_currency_symbol(c))
        {
            end += symbol.len_utf8();
        }
        let segment = &word[start..end];
        let after_letter = i > 0 && matches!(chars[i - 1].1, 'a'..='z' | 'A'..='Z' | '#');
        // Japanese dates, times, ranges and codes; after a "v", a version alone.
        let next = chars.iter().position(|(k, _)| *k >= end).unwrap_or(chars.len());
        if lang == Lang::Ja {
            let (numbers, codes) = (rules.has("numbers"), rules.has("dashed-digits"));
            let version = matches!(chars.get(i.wrapping_sub(1)), Some((_, 'v' | 'V')));
            let read = match (after_letter, version) {
                (false, _) => japanese::run(segment, &word[..start], &word[end..], numbers, codes),
                (true, true) => japanese::version_after_v(segment, numbers),
                (true, false) => None,
            };
            if let Some(read) = read {
                out.push_str(&read);
                changed = true;
                i = next;
                continue;
            }
        }
        // Seven digits or more are a count before a counter ("1234567人") and a code otherwise
        // ("番号は48213907です").
        if rules.has("numbers") && segment.len() >= 7 && all_digits(segment) && !after_letter {
            let counter = word[end..].chars().next().is_some_and(|c| counters(lang).contains(c));
            if !counter {
                out.push_str(&format!(" {} ", numbers::spell(segment, lang)));
                changed = true;
                i = chars.iter().position(|(k, _)| *k >= end).unwrap_or(chars.len());
                continue;
            }
        }
        match claim(segment, lang, rules).filter(|rewritten| !after_letter && rewritten != segment)
        {
            Some(rewritten) => {
                out.push_str(&rewritten);
                changed = true;
            }
            None => out.push_str(segment),
        }
        i = chars.iter().position(|(k, _)| *k >= end).unwrap_or(chars.len());
    }
    // A code spelled at either end of the word would leave a second space beside the one that
    // joins the words.
    changed.then(|| out.trim().to_string())
}

impl Rules {
    pub(super) fn has(self, name: &str) -> bool {
        RULES.iter().position(|rule| rule.name == name).is_some_and(|i| self.0 & (1 << i) != 0)
    }
}

pub(super) fn rewrite_text(
    text: &str,
    lang: Lang,
    rules: Rules,
    unclaimed: impl Fn(&str) -> String,
) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let prev = out.last().map(String::as_str);
        // A number after a number sign is a code, read as written: "nr. 4471", "No. 12".
        if numbered(prev, lang) && split_suffix(words[i]).0.bytes().all(|b| b.is_ascii_digit()) {
            out.push(unclaimed(words[i]));
            i += 1;
            continue;
        }
        let span = None
            .or_else(|| {
                rules.has("currency").then(|| currency::scaled(&words[i..], lang)).flatten()
            })
            .or_else(|| {
                rules.has("currency").then(|| currency::unit_after(&words[i..], lang)).flatten()
            })
            .or_else(|| {
                let before = i.checked_sub(1).map(|j| words[j]);
                (lang == Lang::Hi && rules.has("numbers"))
                    .then(|| hindi::span(before, &words[i..]))
                    .flatten()
            })
            .or_else(|| {
                rules.has("numbers").then(|| numbers::hindi_fraction(&words[i..], lang)).flatten()
            })
            .or_else(|| {
                rules.has("numbers").then(|| numbers::digit_run(&words[i..], lang)).flatten()
            })
            .or_else(|| {
                rules.has("numbers").then(|| numbers::ending_in(prev, &words[i..], lang)).flatten()
            })
            .or_else(|| {
                rules
                    .has("abbreviations")
                    .then(|| abbreviations::ranks(&words[i..], lang))
                    .flatten()
            })
            .or_else(|| {
                rules
                    .has("local-dates")
                    .then(|| local_dates::span(prev, &words[i..], lang))
                    .flatten()
            })
            .or_else(|| {
                (lang == Lang::Ja && rules.has("numbers"))
                    .then(|| {
                        japanese::date_before_weekday(&words[i..])
                            .or_else(|| japanese::weekday(prev, &words[i..]))
                            .or_else(|| japanese::country_code(&words[i..]))
                    })
                    .flatten()
            });
        if let Some((rewritten, used)) = span {
            out.push(rewritten);
            i += used;
            continue;
        }
        // A Japanese or Korean word a rule rewrote inside may still carry symbols to spell.
        let spelled =
            |w: String| if matches!(lang, Lang::Ja | Lang::Ko) { unclaimed(&w) } else { w };
        out.push(rewrite_word(words[i], lang, rules).map_or_else(|| unclaimed(words[i]), spelled));
        i += 1;
    }
    out.join(" ")
}

/// The sentence punctuation a rule carries through at the end of a word, the other scripts'
/// marks included: the Hindi danda, the Arabic comma and question mark, the Japanese stops.
const PUNCTUATION: &str = "!?.:,;…।॥،؟؛。、！？，";

/// The trailing punctuation a rule carries through, or `None` when `rest` is anything else --
/// which means the rule did not account for the whole word and must not fire.
fn suffix(rest: &str) -> Option<&str> {
    rest.chars().all(|c| PUNCTUATION.contains(c)).then_some(rest)
}

/// `word` without its trailing punctuation, and that punctuation.
fn split_suffix(word: &str) -> (&str, &str) {
    let body = word.trim_end_matches(|c| PUNCTUATION.contains(c));
    (body, &word[body.len()..])
}

/// The ASCII digits `s` starts with, as many as there are up to `max`, and the rest of `s`; or
/// `None` when there are fewer than `min`.
fn digits(s: &str, min: usize, max: usize) -> Option<(&str, &str)> {
    let n = s.bytes().take_while(u8::is_ascii_digit).count().min(max);
    (n >= min).then(|| s.split_at(n))
}

/// Whether `s` is a run of ASCII digits.
fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// A two-letter part of an address spelled out, as an abbreviation reads: "fr" becomes "F-R",
/// and "www" "W-W-W". Anything else is read as a word.
fn spell_short(word: &str) -> std::borrow::Cow<'_, str> {
    match word.as_bytes() {
        b"www" | b"WWW" => "W-W-W".into(),
        [a, b] if a.is_ascii() && b.is_ascii() => {
            format!("{}-{}", a.to_ascii_uppercase() as char, b.to_ascii_uppercase() as char).into()
        }
        _ => word.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_pick_what_runs() {
        assert_eq!("all".parse::<Rules>().unwrap(), Rules::ALL);
        assert_eq!("default".parse::<Rules>().unwrap(), Rules::DEFAULT);
        let default = "numbers,currency,dashed-digits,codes,emails,urls,abbreviations,elongations,\
                       local-dates"
            .parse::<Rules>()
            .unwrap();
        assert_eq!(default, Rules::DEFAULT);
        let opt_in = "phones,times,dates".parse::<Rules>().unwrap();
        assert_eq!(Rules(default.0 | opt_in.0), Rules::ALL);
        assert_eq!("none".parse::<Rules>().unwrap(), Rules::NONE);
        assert!("colours".parse::<Rules>().is_err());
        assert_eq!(rewrite_word("1234", Lang::En, Rules::ALL).as_deref(), Some("1 thousand 234"));
        assert_eq!(rewrite_word("1234", Lang::Fr, Rules::ALL).as_deref(), Some("mille 234"));
        assert_eq!(rewrite_word("1234", Lang::En, Rules::NONE), None);
        assert_eq!(rewrite_word("1234", Lang::De, Rules::ALL).as_deref(), Some("ein Tausend 234"));
        assert_eq!(rewrite_word("$5", Lang::En, Rules::DEFAULT).as_deref(), Some("5 dollars"));
        // The opt-in rules stay off by default.
        assert_eq!(rewrite_word("3:45PM", Lang::En, Rules::DEFAULT), None);
        assert_eq!(rewrite_word("3:45PM", Lang::En, Rules::ALL).as_deref(), Some("3-45 PM"));
    }

    #[test]
    fn japanese_and_korean_read_the_numbers_inside_their_words() {
        let rules = Rules::DEFAULT;
        let ja = |w| rewrite_word(w, Lang::Ja, rules);
        assert_eq!(ja("人口は1234567人です。").as_deref(), Some("人口は123万4567人です。"));
        assert_eq!(
            ja("ご注文番号は48213907です。").as_deref(),
            Some("ご注文番号は よん はち に いち さん きゅう ゼロ なな です。")
        );
        assert_eq!(ja("これは1,500円です。").as_deref(), Some("これは1500 円です。"));
        assert_eq!(ja("重さは2.5キロです。").as_deref(), Some("重さは2 点 5キロです。"));
        assert_eq!(ja("会議は10:30です。").as_deref(), Some("会議は10時30分です。"));
        assert_eq!(ja("2024年3月15日"), None);
        assert_eq!(ja("ABC123です"), None);
        assert_eq!(ja("3万円"), None);
        assert_eq!(ja("価格は¥1500です。").as_deref(), Some("価格は1500 円です。"));
        assert_eq!(ja("5€と$3").as_deref(), Some("5 ユーロと3 ドル"));
        assert_eq!(ja("気温は-3.5度です").as_deref(), Some("気温はマイナス 3 点 5度です"));
        assert_eq!(ja("2026/03/14の10:00").as_deref(), Some("2026年3月14日の10時"));
        assert_eq!(ja("A-3"), None);
        let ko = |w| rewrite_word(w, Lang::Ko, rules);
        assert_eq!(ko("12345명입니다.").as_deref(), Some("1만 2345명입니다."));
        assert_eq!(ko("5,000원입니다.").as_deref(), Some("5000 원입니다."));
        assert_eq!(ko("61%에서"), None);
        assert_eq!(ko("₩5,000입니다.").as_deref(), Some("5000 원입니다."));
        assert_eq!(ko("12345").as_deref(), Some("1만 2345"));
        assert_eq!(ko("48213907입니다.").as_deref(), Some("사 팔 이 일 삼 구 공 칠 입니다."));
        assert_eq!(ko("48213907").as_deref(), Some("사 팔 이 일 삼 구 공 칠"));
        assert_eq!(ko("1234567명").as_deref(), Some("123만 4567명"));
        assert_eq!(ko("#773102가"), None);
    }

    /// Numbers are tried before times, so a German thousand is not read as a time, while a
    /// German time, which is not a number, still is.
    #[test]
    fn numbers_win_over_times() {
        let all = Rules::ALL;
        assert_eq!(rewrite_word("1.234", Lang::De, all).as_deref(), Some("ein Tausend 234"));
        assert_eq!(rewrite_word("22.45", Lang::De, all).as_deref(), Some("22 Uhr 45"));
    }

    #[test]
    fn rules_parse_lists_spaces_and_case() {
        // What a CLI flag or a Python string is likely to hold.
        let numbers = "numbers".parse::<Rules>().unwrap();
        assert_eq!(" Numbers ".parse::<Rules>().unwrap(), numbers);
        assert_eq!("numbers, numbers".parse::<Rules>().unwrap(), numbers);
        assert_eq!(" Emails , URLS".parse::<Rules>().unwrap(), "emails,urls".parse().unwrap());
        assert_eq!("ALL".parse::<Rules>().unwrap(), Rules::ALL);
        assert_eq!(" Default ".parse::<Rules>().unwrap(), Rules::DEFAULT);
        assert_eq!("Off".parse::<Rules>().unwrap(), Rules::NONE);
        let err = "numbers,colours".parse::<Rules>().unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("colours") && msg.contains("dashed-digits"), "{msg}");
        assert!("".parse::<Rules>().is_err());
        // The whole-string values are not rule names.
        assert!("numbers,all".parse::<Rules>().is_err());
    }

    #[test]
    fn helpers() {
        assert_eq!(split_suffix("12,..."), ("12", ",..."));
        assert_eq!(split_suffix("12"), ("12", ""));
        assert_eq!(split_suffix("42।"), ("42", "।"));
        assert_eq!(split_suffix("12،"), ("12", "،"));
        assert_eq!(split_suffix("..."), ("", "..."));
        assert_eq!(digits("12345x", 2, 4), Some(("1234", "5x")));
        assert_eq!(digits("1x", 2, 4), None);
        assert_eq!(spell_short("fr"), "F-R");
        assert_eq!(spell_short("www"), "W-W-W");
        assert_eq!(spell_short("com"), "com");
        // One two-byte letter is not two letters.
        assert_eq!(spell_short("é"), "é");
    }
}
