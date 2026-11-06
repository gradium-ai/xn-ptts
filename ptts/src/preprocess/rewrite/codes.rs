//! File names, paths and letter-and-digit codes, in the languages with letter names of their
//! own ([`letters::names`]): "report_final_v3.pdf" becomes in Finnish "report alaviiva final
//! alaviiva vee 3 piste pee dee äf", and "AB-4471-Z" "aa bee neljä neljä seitsemän yksi tset".
//! The five languages the serving stack reads are left to its rules.

use super::{emails::emails, letters, numbers, split_suffix, tlds::is_tld, urls::urls};
use crate::preprocess::{Lang, Rules};

/// A file name, a path or a code, read out in `lang`.
pub(super) fn codes(word: &str, lang: Lang) -> Option<String> {
    letters::names(lang)?;
    let (body, suffix) = split_suffix(word);
    // Finnish inflects a name or a code after a colon ("AB-4471-Z:n", "B12:ssa"). The ending
    // goes, as the character pass drops it after a digit.
    let body = match body.rsplit_once(':') {
        Some((name, ending))
            if lang == Lang::Fi
                && !name.is_empty()
                && !ending.is_empty()
                && ending.chars().all(char::is_lowercase) =>
        {
            name
        }
        _ => body,
    };
    let read = file(body, lang).or_else(|| code(body, lang))?;
    Some(format!("{read}{suffix}"))
}

/// The characters an address, a path or a file name is made of.
fn address_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "._-/@:+".contains(c)
}

/// A file name ("report_final_v3.pdf"), with a path or an address in front or not, or a path
/// from the root ("/var/log"), or an address with a scheme or `www.` that the URL rule leaves
/// alone because of what its path holds. A name whose extension is also a top-level domain
/// ("deploy.sh") is the URL or email rule's when either takes it, which reads it the same way.
fn file(body: &str, lang: Lang) -> Option<String> {
    if !body.chars().all(address_char)
        || !body.starts_with(|c: char| c.is_alphanumeric() || c == '/')
    {
        return None;
    }
    let letter = body.chars().any(|c| c.is_ascii_alphabetic());
    let named = body.rsplit_once('.').is_some_and(|(name, ext)| {
        !name.is_empty()
            && name.ends_with(|c: char| c.is_ascii_alphanumeric())
            && letters::is_extension(ext)
    });
    let rooted = body.starts_with('/')
        && body.len() > 1
        && body[1..].split('/').all(|p| !p.is_empty() && p.chars().all(|c| c != '@' && c != ':'));
    let schemed = ["http://", "https://", "www."].iter().any(|s| body.starts_with(s));
    if !letter || !(named || rooted || schemed) {
        return None;
    }
    let tld = body.rsplit_once('.').is_some_and(|(_, ext)| is_tld(ext));
    if named && tld && (emails(body, lang).is_some() || urls(body, lang).is_some()) {
        return None;
    }
    if !named && schemed && urls(body, lang).is_some() {
        return None;
    }
    Some(letters::read(body, lang))
}

/// A group of a code that is a word rather than letters to spell: four letters or more, two of
/// them vowels at least, and no three consonants in a row ("SAVE", "WELCOME", "COVID").
fn wordlike(group: &str) -> bool {
    let vowel = |c: char| "AEIOUY".contains(c);
    let consonants = group.split(vowel).map(str::len).max().unwrap_or(0);
    group.len() >= 4
        && group.chars().all(|c| c.is_ascii_uppercase())
        && group.chars().filter(|&c| vowel(c)).count() >= 2
        && consonants < 3
}

/// The runs of letters and of digits `s` is made of, in order.
fn runs(s: &str) -> Vec<&str> {
    let mut out = vec![];
    let mut start = 0;
    let bytes = s.as_bytes();
    for i in 1..=bytes.len() {
        if i == bytes.len() || bytes[i].is_ascii_digit() != bytes[i - 1].is_ascii_digit() {
            out.push(&s[start..i]);
            start = i;
        }
    }
    out
}

/// A code of capital letters and digits: one group ("B12", "X7K", "14C"), or groups joined by
/// dashes ("AB-4471-Z", "SAVE-20", "QZ-7731-mt"). The letters are spelled and the digits of a
/// dashed code read one by one, as a code is read out, apart from the number after a word
/// ("SAVE-20"). A group alone is shorter and its digits are read as a number ("B12"). The dashes
/// are not spoken.
///
/// Words with lowercase letters are left alone, so that a compound such as "5G-verkko" is not
/// spelled, and so are a number followed by a unit letter ("100W") and groups of more than three
/// capitals that are not words.
fn code(body: &str, lang: Lang) -> Option<String> {
    let groups: Vec<&str> = body.split('-').collect();
    let ok = |g: &&str| {
        !g.is_empty()
            && g.chars().all(|c| c.is_ascii_alphanumeric())
            && (!g.chars().any(|c| c.is_ascii_lowercase())
                || (g.len() <= 2 && g.chars().all(|c| c.is_ascii_lowercase())))
    };
    if !groups.iter().all(ok)
        || !body.chars().any(|c| c.is_ascii_digit())
        || !body.chars().any(|c| c.is_ascii_uppercase())
    {
        return None;
    }
    // Letters to spell come three at most to a run in a dashed code, two in a code alone.
    let most = if groups.len() == 1 { 2 } else { 3 };
    let letters_ok = |g: &str| {
        runs(g).iter().all(|r| r.as_bytes()[0].is_ascii_digit() || r.len() <= most || wordlike(r))
    };
    if !groups.iter().all(|g| letters_ok(g)) {
        return None;
    }
    let mut words = vec![];
    if let [group] = groups[..] {
        let runs = runs(group);
        if group.len() > 8 || runs.len() < 2 {
            return None;
        }
        // One digit and one capital is a term read as written: "3D", "4K", "5G".
        if group.len() == 2
            && group.as_bytes()[0].is_ascii_digit()
            && group.as_bytes()[1].is_ascii_uppercase()
        {
            return None;
        }
        // A number and a unit: "100W", "12V", "5A".
        if let [n, unit] = runs[..]
            && n.as_bytes()[0].is_ascii_digit()
            && (unit.len() > 1 || "WVAJN".contains(unit))
        {
            return None;
        }
        for run in runs {
            if run.as_bytes()[0].is_ascii_digit() {
                words.push(numbers::numbers(run, lang).unwrap_or_else(|| run.to_string()));
            } else if wordlike(run) {
                words.push(run.to_lowercase());
            } else {
                words.push(letters::spell(run, lang));
            }
        }
        return Some(words.join(" "));
    }
    // "A-3" is as often a range or a size as a code.
    if body.bytes().filter(u8::is_ascii_digit).count() < 2 {
        return None;
    }
    let mut after_word = false;
    for group in groups {
        if group.chars().all(|c| c.is_ascii_digit()) && after_word {
            words.push(numbers::numbers(group, lang).unwrap_or_else(|| group.to_string()));
            after_word = false;
            continue;
        }
        after_word = wordlike(group);
        if after_word {
            words.push(group.to_lowercase());
            continue;
        }
        for run in runs(group) {
            if run.as_bytes()[0].is_ascii_digit() {
                words.push(numbers::spell(run, lang));
            } else {
                words.push(letters::spell(run, lang));
            }
        }
    }
    Some(words.join(" "))
}

/// A Japanese or Korean word with an address, a file name or a code written into it ("ファイル名は
/// report_final_v3.pdfです", "AB-4471-Z입니다"): each run of ASCII letters, digits and
/// separators that holds a letter and that a rule claims is read on its own, and the rest of
/// the word goes to [`super::rewrite_runs`] as before. `None` when no such run is claimed.
pub(super) fn within(word: &str, lang: Lang, rules: Rules) -> Option<String> {
    letters::names(lang)?;
    let mut out = String::new();
    let mut changed = false;
    let mut rest = word;
    let flush = |out: &mut String, text: &str| {
        out.push_str(&super::rewrite_runs(text, lang, rules).unwrap_or_else(|| text.to_string()));
    };
    let mut plain = String::new();
    while let Some(start) = rest.find(|c: char| c.is_ascii_alphanumeric() || c == '/') {
        let len = rest[start..].find(|c: char| !address_char(c)).unwrap_or(rest.len() - start);
        let run = rest[start..start + len].trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
        let claimed = run
            .chars()
            .any(|c| c.is_ascii_alphabetic())
            .then(|| super::claim(run, lang, rules))
            .flatten()
            .filter(|read| read != run);
        match claimed {
            Some(read) => {
                plain.push_str(&rest[..start]);
                flush(&mut out, &plain);
                plain.clear();
                if lang == Lang::Ja {
                    out.push_str(&format!(" {read} "));
                } else {
                    out.push_str(&read);
                }
                changed = true;
                rest = &rest[start + run.len()..];
            }
            None => {
                let end = start + run.len().max(1);
                let end =
                    (end..=rest.len()).find(|&i| rest.is_char_boundary(i)).unwrap_or(rest.len());
                plain.push_str(&rest[..end]);
                rest = &rest[end..];
            }
        }
    }
    if !changed {
        return None;
    }
    plain.push_str(rest);
    flush(&mut out, &plain);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_and_paths_are_read() {
        let cases = [
            (
                Lang::Fi,
                "report_final_v3.pdf.",
                "report alaviiva final alaviiva vee 3 piste pee dee äf.",
            ),
            (Lang::Fi, "sales_2026-Q3.xlsx", "sales alaviiva 2026 viiva kuu 3 piste äks äl äs äks"),
            (
                Lang::Fi,
                "/var/log/app/error.log",
                "kauttaviiva var kauttaviiva log kauttaviiva app kauttaviiva error piste log",
            ),
            (
                Lang::Sv,
                "downloads.example.com/setup.exe",
                "downloads punkt example punkt com snedstreck setup punkt e eks e",
            ),
            (
                Lang::No,
                "Documents/backup_2026.zip",
                "Documents skråstrek backup understrek 2026 punktum zip",
            ),
            (
                Lang::It,
                "Q4_review_final.pptx",
                "cu 4 trattino basso review trattino basso final punto pi pi ti ics",
            ),
            (
                Lang::Nl,
                "logo_dark@2x.png",
                "logo liggend streepje dark apenstaartje 2 iks punt pee en gee",
            ),
            (Lang::Pl, "customers_export.csv,", "customers podkreślnik export kropka ce es fał,"),
            (Lang::Fi, "report.pdf:ssä", "report piste pee dee äf"),
            (Lang::Ar, "notes.txt", "notes نقطة تي إكس تي"),
            (Lang::Ko, "photo_01.jpg", "photo 언더바 공 일 닷 제이피지"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(codes(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        // Names whose extension is a top-level domain are the URL rule's.
        assert_eq!(codes("deploy.sh", Lang::Fi), None);
        // Not file names.
        for input in ["e.g.", "3.14", "example.com", "it.Now", "and/or", "a/b.Pdf"] {
            assert_eq!(codes(input, Lang::Fi), None, "{input:?}");
        }
    }

    #[test]
    fn codes_are_spelled() {
        let cases = [
            (Lang::Fi, "AB-4471-Z.", "aa bee neljä neljä seitsemän yksi tset."),
            (Lang::Fi, "X7K-29Q", "äks seitsemän koo kaksi yhdeksän kuu"),
            (Lang::Fi, "SAVE-20", "save 20"),
            (Lang::Fi, "QZ-7731-mt", "kuu tset seitsemän seitsemän kolme yksi äm tee"),
            (Lang::Fi, "B12", "bee 12"),
            (Lang::Fi, "14C", "14 see"),
            (Lang::Fi, "AB-4471-Z:n", "aa bee neljä neljä seitsemän yksi tset"),
            (Lang::Fi, "B12:ssa.", "bee 12."),
            (Lang::Sv, "INV-00928", "i enn ve noll noll nio två åtta"),
            (Lang::No, "4F-8H-2Q", "fire eff åtte hå to ku"),
            (Lang::It, "WELCOME-15", "welcome 15"),
            (Lang::Nl, "HP-V2-114", "haa pee vee twee een een vier"),
            (Lang::Pl, "L-2208-K7", "el dwa dwa zero osiem ka siedem"),
            (Lang::Hi, "XK-2290-B", "एक्स के दो दो नौ शून्य बी"),
            (Lang::Hi, "SAVE-20", "save बीस"),
            (Lang::Ar, "TK-30977", "تي كيه ثلاثة صفر تسعة سبعة سبعة"),
            (Lang::Ko, "AB-4471-Z", "에이비 사 사 칠 일 제트"),
            (Lang::Ja, "E-404-B", "イー よん ゼロ よん ビー"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(codes(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        let declined = [
            "Wi-Fi",
            "5G-verkko",
            "mp3-soitin",
            "100W",
            "12V",
            "10GB",
            "2024-2025",
            "ABCDE1",
            "COVID-19-pandemia",
            "T-paita",
            "1080p",
            "abc123",
        ];
        for input in declined {
            assert_eq!(codes(input, Lang::Fi), None, "{input:?}");
        }
        // The serving stack's languages keep its rules.
        assert_eq!(codes("AB-4471-Z", Lang::En), None);
        assert_eq!(codes("report.pdf", Lang::De), None);
    }

    #[test]
    fn japanese_and_korean_words_hold_codes() {
        let rules = Rules::DEFAULT;
        assert_eq!(
            within("ファイル名はreport_final_v3.pdfです。", Lang::Ja, rules).as_deref(),
            Some(
                "ファイル名は report アンダーバー final アンダーバー ブイ 3 ドット ピーディーエフ \
                 です。"
            )
        );
        assert_eq!(
            within("AB-4471-Z입니다.", Lang::Ko, rules).as_deref(),
            Some("에이비 사 사 칠 일 제트입니다.")
        );
        assert_eq!(
            within("support@example.com으로", Lang::Ko, rules).as_deref(),
            Some("support 골뱅이 example 닷 com으로")
        );
        // Letters a rule does not claim stay, and the digits around go on as before.
        assert_eq!(within("Wi-Fiのパスワード", Lang::Ja, rules), None);
        assert_eq!(within("IDは1,500円です", Lang::Ja, rules), None);
        assert_eq!(within("ABC123です", Lang::Ja, rules), None);
    }
}
