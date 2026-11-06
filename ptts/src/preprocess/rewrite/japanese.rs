//! The shapes digits take in Japanese text, written the way Japanese writes them out: a date
//! "2026/03/14" becomes "2026年3月14日", a time "07:05" "7時5分", a fraction "1/4" "4分の1", a
//! range "3-5" or "8〜10" "3から5", a version "2.0.3" "2 点 0 点 3", and a phone or order number
//! "090-7946-0018" its digits in kana, each group joined by の.
//!
//! The digits stay where the model reads them well, before a counter; the separators go, since
//! the model reads a slash, a colon, a dash or a wave dash between digits as noise.

use super::{all_digits, numbers, split_suffix};
use crate::preprocess::Lang;

/// A run of digits and separators inside a Japanese word, as [`super::rewrite_runs`] cuts it,
/// rewritten; or `None` to leave it to the other rules. `before` and `after` are the text of
/// the word around it. Codes go by `codes`, the rest by `numbers`.
pub(super) fn run(
    run: &str,
    before: &str,
    after: &str,
    numbers: bool,
    codes: bool,
) -> Option<String> {
    let read = if numbers {
        None.or_else(|| date(run))
            .or_else(|| expiry(run, before))
            .or_else(|| slashed(run))
            .or_else(|| time_range(run))
            .or_else(|| time(run))
            .or_else(|| version(run))
            .or_else(|| range(run, before, after))
    } else {
        None
    };
    read.or_else(|| codes.then(|| code(run)).flatten())
}

/// A version after its "v", "v2.4.1".
pub(super) fn version_after_v(run: &str, numbers: bool) -> Option<String> {
    numbers.then(|| version(run)).flatten()
}

/// `word`, a Japanese word that is all ASCII, rewritten as [`run`] would: "090-7946-0018" on its
/// own, between spaces.
pub(super) fn ascii_word(word: &str, numbers: bool, codes: bool) -> Option<String> {
    let (body, end) = split_suffix(word);
    let rewritten = run(body, "", end, numbers, codes)?;
    Some(format!("{}{end}", rewritten.trim()))
}

/// The wave dash of a range between two numbers read から, "8〜10時間", "11月3日〜17日", a
/// weekday in brackets after a date read in full, "5日（月）", and a country code at the end of
/// the word, "海外からは+81", read digit by digit; or `None` when `word` has none of them.
pub(super) fn marks(word: &str) -> Option<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut out = String::with_capacity(word.len());
    let mut changed = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let prev = i.checked_sub(1).map(|j| chars[j]);
        if matches!(c, '〜' | '～')
            && prev.is_some_and(char::is_alphanumeric)
            && chars.get(i + 1).is_some_and(char::is_ascii_digit)
        {
            out.push_str("から");
            changed = true;
        } else if c == '+'
            && !prev.is_some_and(|p| p.is_ascii_digit())
            && (2..=4).contains(&(chars.len() - i))
            && chars[i + 1..].iter().all(char::is_ascii_digit)
        {
            // A country code at the end of a word, before the rest of the phone number.
            let code: String = chars[i + 1..].iter().collect();
            out.push_str(&format!(
                "{} {}、",
                Lang::Ja.special_chars().plus,
                numbers::spell(&code, Lang::Ja)
            ));
            changed = true;
            break;
        } else if c == '（'
            && prev == Some('日')
            && chars.get(i + 2) == Some(&'）')
            && is_weekday(chars[i + 1])
        {
            out.push_str(&format!("（{}曜日）", chars[i + 1]));
            changed = true;
            i += 3;
            continue;
        } else {
            out.push(c);
        }
        i += 1;
    }
    changed.then_some(out)
}

/// A weekday the character pass cut out of its brackets after a date, "5日, 月,", read in full:
/// the word after `prev` when `prev` ends on 日 and the word is a weekday alone.
pub(super) fn weekday(prev: Option<&str>, words: &[&str]) -> Option<(String, usize)> {
    let (day, _) = split_suffix(prev?);
    let (body, end) = split_suffix(words.first()?);
    let mut chars = body.chars();
    let c = chars.next().filter(|&c| is_weekday(c))?;
    (day.ends_with('日') && chars.next().is_none()).then(|| (format!("{c}曜日{end}"), 1))
}

/// A country code, "+81", before the rest of a phone number, read digit by digit in kana.
pub(super) fn country_code(words: &[&str]) -> Option<(String, usize)> {
    let code = words.first()?.strip_prefix('+')?;
    let next = words.get(1)?;
    if !(1..=3).contains(&code.len())
        || !all_digits(code)
        || !next.starts_with(|c: char| c.is_ascii_digit())
    {
        return None;
    }
    Some((format!("{} {}、", Lang::Ja.special_chars().plus, numbers::spell(code, Lang::Ja)), 1))
}

fn is_weekday(c: char) -> bool {
    "月火水木金土日".contains(c)
}

/// The groups of `run` cut at `separator`, when every group is digits.
fn groups(run: &str, separator: char) -> Option<Vec<&str>> {
    let groups: Vec<&str> = run.split(separator).collect();
    (groups.len() >= 2 && groups.iter().all(|g| all_digits(g))).then_some(groups)
}

/// `s` without its leading zeros, "0" for zero.
fn value(s: &str) -> &str {
    let trimmed = s.trim_start_matches('0');
    if trimmed.is_empty() { "0" } else { trimmed }
}

/// A month and a day, each one or two digits, read "3月14日".
fn month_day(month: &str, day: &str) -> Option<String> {
    let ok =
        |s: &str, max: u32| s.len() <= 2 && s.parse::<u32>().is_ok_and(|n| (1..=max).contains(&n));
    (ok(month, 12) && ok(day, 31)).then(|| format!("{}月{}日", value(month), value(day)))
}

/// A year, a month and a day, "2026/03/14", "2026-03-14" or "2026.03.14".
fn date(run: &str) -> Option<String> {
    let separator = run.chars().find(|c| matches!(c, '/' | '-' | '.'))?;
    match groups(run, separator)?.as_slice() {
        [year, month, day] if year.len() == 4 => {
            Some(format!("{year}年{}", month_day(month, day)?))
        }
        _ => None,
    }
}

/// A card's expiry, "08/27" after 有効期限, read as a month of a year.
fn expiry(run: &str, before: &str) -> Option<String> {
    if !before.trim_end_matches('は').ends_with("有効期限") {
        return None;
    }
    match groups(run, '/')?.as_slice() {
        [month, year] if month.len() == 2 && year.len() == 2 => {
            month_day(month, "1")?;
            Some(format!("20{year}年{}月", value(month)))
        }
        _ => None,
    }
}

/// Two numbers with a slash: a fraction when it is one in lowest terms with a small
/// denominator, "1/4" read "4分の1", and a month and a day otherwise, "3/15", "10/12" or "07/01".
fn slashed(run: &str) -> Option<String> {
    let [a, b] = groups(run, '/')?[..] else { return None };
    let padded = (a.len() > 1 && a.starts_with('0')) || (b.len() > 1 && b.starts_with('0'));
    if !padded && a.len() <= 2 && b.len() <= 2 {
        let (x, y) = (a.parse::<u32>().ok()?, b.parse::<u32>().ok()?);
        if 0 < x && x < y && y <= 10 && gcd(x, y) == 1 {
            return Some(format!("{b}分の{a}"));
        }
    }
    month_day(a, b)
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A time of day, `H:MM`, read "7時5分", or "7時" on the hour. More than 24 hours is a duration
/// in minutes and seconds, "48:12" read "48分12秒".
fn time(run: &str) -> Option<String> {
    let [h, m] = groups(run, ':')?[..] else { return None };
    if h.len() > 2 || m.len() != 2 || m.parse::<u32>().ok()? >= 60 {
        return None;
    }
    let (h, m) = (value(h), value(m));
    if h.parse::<u32>().ok()? > 24 {
        return Some(format!("{h}分{m}秒"));
    }
    Some(if m == "0" { format!("{h}時") } else { format!("{h}時{m}分") })
}

/// Two times with a dash, "14:00-16:00".
fn time_range(run: &str) -> Option<String> {
    let (from, to) = run.split_once('-')?;
    Some(format!("{}から{}", time(from)?, time(to)?))
}

/// A version or an address with two points or more, "2.0.3" read "2 点 0 点 3". A part with a
/// leading zero, or as long as a code, is read digit by digit.
fn version(run: &str) -> Option<String> {
    let parts = groups(run, '.')?;
    if parts.len() < 3 {
        return None;
    }
    let read = |p: &&str| {
        if (p.len() > 1 && p.starts_with('0')) || p.len() > 4 {
            p.chars().map(String::from).collect::<Vec<_>>().join(" ")
        } else {
            p.to_string()
        }
    };
    Some(format!(" {} ", parts.iter().map(read).collect::<Vec<_>>().join(" 点 ")))
}

/// Two numbers with a dash: a range when the first is the smaller and both are short or years,
/// "3-5" read "3から5"; a score, "3-2で" or "スコア3-2", read "3対2"; and a section, "2-1", read
/// "2の1". Longer groups are a code: see [`code`].
fn range(run: &str, before: &str, after: &str) -> Option<String> {
    let [a, b] = groups(run, '-')?[..] else { return None };
    let short = |s: &str| s.len() <= 3 && (s.len() == 1 || !s.starts_with('0'));
    let year = |s: &str| s.len() == 4 && (s.starts_with("19") || s.starts_with("20"));
    let (x, y) = (a.parse::<u32>().ok()?, b.parse::<u32>().ok()?);
    // "2024-25" a few years on, as the numbers rule reads it; "2026-77" is a code.
    let years_on = year(a) && b.len() == 2 && (x % 100 < y && y <= x % 100 + 10);
    if ((short(a) && short(b)) || (year(a) && year(b))) && x < y || years_on {
        return Some(format!("{a}から{b}"));
    }
    if a.len() <= 2 && b.len() <= 2 && short(a) && short(b) {
        let score = after.starts_with('で') || before.ends_with("スコア");
        let joint = if score { "対" } else { "の" };
        return Some(format!("{a}{joint}{b}"));
    }
    None
}

/// A phone, order or case number, groups of digits with dashes or slashes, "090-7946-0018":
/// every digit in kana, the groups joined by の, as it is read out. A date is not one.
fn code(run: &str) -> Option<String> {
    let separator = run.chars().find(|c| matches!(c, '-' | '/'))?;
    let groups = groups(run, separator)?;
    if groups.iter().map(|g| g.len()).sum::<usize>() < 4 || date(run).is_some() {
        return None;
    }
    let spelled: Vec<String> = groups.iter().map(|g| numbers::spell(g, Lang::Ja)).collect();
    Some(format!(" {} ", spelled.join(" の ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_read_as_japanese_writes_them() {
        let cases: &[(&str, &str, &str, Option<&str>)] = &[
            ("2026/03/14", "", "の", Some("2026年3月14日")),
            ("2027-06-30", "", "まで", Some("2027年6月30日")),
            ("2026.10.02", "", "に", Some("2026年10月2日")),
            // Not a date: a code.
            ("2026/13/14", "", "", Some(" に ゼロ に ろく の いち さん の いち よん ")),
            ("09/23", "", "に", Some("9月23日")),
            ("10/12", "", "に", Some("10月12日")),
            ("3/15", "", "", Some("3月15日")),
            ("1/4", "", "しか", Some("4分の1")),
            ("3/4", "", "まで", Some("4分の3")),
            ("1/3", "約", "が", Some("3分の1")),
            ("08/27", "カードの有効期限は", "です", Some("2027年8月")),
            ("07:05", "", "に", Some("7時5分")),
            ("06:00", "", "に", Some("6時")),
            ("00:15", "", "に", Some("0時15分")),
            ("23:50", "", "", Some("23時50分")),
            ("48:12", "", "", Some("48分12秒")),
            ("12:75", "", "", None),
            ("2:03:45", "", "", None),
            ("14:00-16:00", "", "", Some("14時から16時")),
            ("2.0.3", "", "", Some(" 2 点 0 点 3 ")),
            ("535.129.03", "", "", Some(" 535 点 129 点 0 3 ")),
            ("10.0.19045", "", "", Some(" 10 点 0 点 1 9 0 4 5 ")),
            ("3.10", "", "", None),
            ("3-5", "", "営業日", Some("3から5")),
            ("201-210", "", "号室", Some("201から210")),
            ("2019-2023", "", "年", Some("2019から2023")),
            ("2024-25", "", "年度", Some("2024から25")),
            ("2026-77", "ケース番号", "は", Some(" に ゼロ に ろく の なな なな ")),
            ("3-2", "", "で勝ちました", Some("3対2")),
            ("3-2", "スコア", "。", Some("3対2")),
            ("2-1", "第", "章", Some("2の1")),
            (
                "090-7946-0018",
                "",
                "まで",
                Some(" ゼロ きゅう ゼロ の なな きゅう よん ろく の ゼロ ゼロ いち はち "),
            ),
            ("3391-8820", "", "", Some(" さん さん きゅう いち の はち はち に ゼロ ")),
            (
                "2026/0917/44",
                "",
                "",
                Some(" に ゼロ に ろく の ゼロ きゅう いち なな の よん よん "),
            ),
            ("1-2", "", "", Some("1から2")),
            ("1:25,000", "", "", None),
        ];
        for &(input, before, after, expected) in cases {
            assert_eq!(run(input, before, after, true, true).as_deref(), expected, "{input}");
        }
        // Codes go by their own rule.
        assert_eq!(run("3391-8820", "", "", true, false), None);
        assert_eq!(run("2026/03/14", "", "", false, true), None);
    }

    /// The whole normalization, character spacing included.
    #[test]
    fn japanese_sentences() {
        use crate::preprocess::{Rules, normalize_text};
        let ja = |text| normalize_text(text, Lang::Ja, Rules::DEFAULT);
        let cases = [
            ("ご予約は2026/03/14の10:00です。", "ご 予 約 は 2026 年 3 月 14 日 の 10 時 で す 。"),
            ("会議は09/23に変更", "会 議 は 9 月 23 日 に 変 更 。"),
            ("席は1/4しか", "席 は 4 分 の 1 し か 。"),
            ("8〜10時間", "8 か ら 10 時 間 。"),
            ("8~10時間と~5", "8 か ら 10 時 間 と 約 5 。"),
            (
                "講座は1月5日(月)に始まります。",
                "講 座 は 1 月 5 日 , 月 曜 日 , に 始 ま り ま す 。",
            ),
            ("まずv2.4.1に", "ま ず v 2 点 4 点 1 に 。"),
            (
                "機種名はHP-V2-114です。",
                "機 種 名 は エ イ チ ピ ー ブ イ に い ち い ち よ ん で す 。",
            ),
            ("AB-4471-Zの", "エ ー ビ ー よ ん よ ん な な い ち ゼ ッ ト の 。"),
            (
                "+81 3-5555-0132で",
                "プ ラ ス は ち い ち 、 さ ん の ご ご ご ご の ゼ ロ い ち さ ん に で 。",
            ),
            (
                "海外からは+81 50-7946-0018に",
                "海 外 か ら は プ ラ ス は ち い ち 、 ご ゼ ロ の な な き ゅ う よ ん ろ く の ゼ ロ ゼ ロ い ち は ち に 。",
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(ja(text), expected, "{text}");
        }
        // The rules off leave it all as written.
        assert_eq!(
            normalize_text("2026/03/14の10:00", Lang::Ja, Rules::NONE),
            "2026 / 03 / 14 の 10 : 00 。"
        );
    }

    #[test]
    fn marks_and_words_around_runs() {
        assert_eq!(marks("8〜10時間").as_deref(), Some("8から10時間"));
        assert_eq!(marks("11月3日〜17日").as_deref(), Some("11月3日から17日"));
        assert_eq!(marks("5日（月）").as_deref(), Some("5日（月曜日）"));
        assert_eq!(marks("〜10"), None);
        assert_eq!(marks("（月）"), None);
        assert_eq!(marks("海外からは+81").as_deref(), Some("海外からはプラス はち いち、"));
        assert_eq!(marks("1+81"), None);
        assert_eq!(marks("+81です"), None);
        assert_eq!(weekday(Some("1月5日,"), &["月,", "に"]), Some(("月曜日,".into(), 1)));
        assert_eq!(weekday(Some("1月,"), &["月,"]), None);
        assert_eq!(weekday(Some("5日,"), &["月曜,"]), None);
        assert_eq!(
            country_code(&["+81", "50-7946-0018に"]),
            Some(("プラス はち いち、".into(), 1))
        );
        assert_eq!(country_code(&["+81", "です"]), None);
        assert_eq!(
            ascii_word("090-7946-0018", true, true).as_deref(),
            Some("ゼロ きゅう ゼロ の なな きゅう よん ろく の ゼロ ゼロ いち はち")
        );
        assert_eq!(ascii_word("1234", true, true), None);
    }
}
