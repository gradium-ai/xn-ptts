//! The names of the Latin letters in the languages that spell them their own way, and the
//! reading of an address, a path or a file name in them: "gradium.sh/docs" becomes in Finnish
//! "gradium piste äs hoo kauttaviiva docs".
//!
//! The five languages the serving stack reads keep its readings, so they have no table here.
//! The names are the ones dictionaries and school alphabets give: Kielitoimisto for Finnish,
//! Svenska Akademiens ordlista for Swedish, Bokmålsordboka for Norwegian, Treccani for Italian,
//! Van Dale for Dutch, PWN for Polish, and the usual transcriptions of the English names in
//! Arabic, Hindi, Korean and Japanese.

use super::numbers;
use crate::preprocess::Lang;

/// The names of the letters a to z in `lang`, or `None` in a language that keeps the serving
/// stack's readings or has no rules.
pub(super) fn names(lang: Lang) -> Option<&'static [&'static str; 26]> {
    Some(match lang {
        Lang::Fi => &[
            "aa",
            "bee",
            "see",
            "dee",
            "ee",
            "äf",
            "gee",
            "hoo",
            "ii",
            "jii",
            "koo",
            "äl",
            "äm",
            "än",
            "oo",
            "pee",
            "kuu",
            "är",
            "äs",
            "tee",
            "uu",
            "vee",
            "kaksoisvee",
            "äks",
            "yy",
            "tset",
        ],
        Lang::Sv => &[
            "a",
            "be",
            "se",
            "de",
            "e",
            "eff",
            "ge",
            "hå",
            "i",
            "ji",
            "kå",
            "ell",
            "emm",
            "enn",
            "o",
            "pe",
            "ku",
            "ärr",
            "ess",
            "te",
            "u",
            "ve",
            "dubbel-ve",
            "eks",
            "y",
            "säta",
        ],
        Lang::No => &[
            "a",
            "be",
            "se",
            "de",
            "e",
            "eff",
            "ge",
            "hå",
            "i",
            "jådd",
            "kå",
            "ell",
            "emm",
            "enn",
            "o",
            "pe",
            "ku",
            "ærr",
            "ess",
            "te",
            "u",
            "ve",
            "dobbelt-ve",
            "eks",
            "y",
            "sett",
        ],
        Lang::It => &[
            "a",
            "bi",
            "ci",
            "di",
            "e",
            "effe",
            "gi",
            "acca",
            "i",
            "i lunga",
            "cappa",
            "elle",
            "emme",
            "enne",
            "o",
            "pi",
            "cu",
            "erre",
            "esse",
            "ti",
            "u",
            "vu",
            "doppia vu",
            "ics",
            "ipsilon",
            "zeta",
        ],
        Lang::Nl => &[
            "aa",
            "bee",
            "cee",
            "dee",
            "ee",
            "ef",
            "gee",
            "haa",
            "ie",
            "jee",
            "kaa",
            "el",
            "em",
            "en",
            "oo",
            "pee",
            "kuu",
            "er",
            "es",
            "tee",
            "uu",
            "vee",
            "wee",
            "iks",
            "griekse ij",
            "zet",
        ],
        Lang::Pl => &[
            "a", "be", "ce", "de", "e", "ef", "gie", "ha", "i", "jot", "ka", "el", "em", "en", "o",
            "pe", "ku", "er", "es", "te", "u", "fał", "wu", "iks", "igrek", "zet",
        ],
        Lang::Ar => &[
            "إيه",
            "بي",
            "سي",
            "دي",
            "إي",
            "إف",
            "جي",
            "إتش",
            "آي",
            "جيه",
            "كيه",
            "إل",
            "إم",
            "إن",
            "أو",
            "بي",
            "كيو",
            "آر",
            "إس",
            "تي",
            "يو",
            "في",
            "دبليو",
            "إكس",
            "واي",
            "زد",
        ],
        Lang::Hi => &[
            "ए",
            "बी",
            "सी",
            "डी",
            "ई",
            "एफ़",
            "जी",
            "एच",
            "आई",
            "जे",
            "के",
            "एल",
            "एम",
            "एन",
            "ओ",
            "पी",
            "क्यू",
            "आर",
            "एस",
            "टी",
            "यू",
            "वी",
            "डब्ल्यू",
            "एक्स",
            "वाई",
            "ज़ेड",
        ],
        Lang::Ko => &[
            "에이",
            "비",
            "씨",
            "디",
            "이",
            "에프",
            "지",
            "에이치",
            "아이",
            "제이",
            "케이",
            "엘",
            "엠",
            "엔",
            "오",
            "피",
            "큐",
            "알",
            "에스",
            "티",
            "유",
            "브이",
            "더블유",
            "엑스",
            "와이",
            "제트",
        ],
        Lang::Ja => &[
            "エー",
            "ビー",
            "シー",
            "ディー",
            "イー",
            "エフ",
            "ジー",
            "エイチ",
            "アイ",
            "ジェー",
            "ケー",
            "エル",
            "エム",
            "エヌ",
            "オー",
            "ピー",
            "キュー",
            "アール",
            "エス",
            "ティー",
            "ユー",
            "ブイ",
            "ダブリュー",
            "エックス",
            "ワイ",
            "ゼット",
        ],
        _ => return None,
    })
}

/// `letters` spelled out by their names in `lang`, which must have them: Korean and Japanese
/// join the names as they write a spelled abbreviation ("에이비"), the others space them.
pub(super) fn spell(letters: &str, lang: Lang) -> String {
    let Some(names) = names(lang) else { return letters.to_string() };
    let spoken = letters.chars().map(|c| match c.to_ascii_lowercase() {
        c @ 'a'..='z' => names[(c as u8 - b'a') as usize].to_string(),
        c => c.to_string(),
    });
    let joint = if matches!(lang, Lang::Ko | Lang::Ja) { "" } else { " " };
    spoken.collect::<Vec<_>>().join(joint)
}

/// The file extensions that are read as letters although they could be read as words. Those
/// without a vowel ("pdf", "xlsx") are spelled anyway, and the others ("zip", "json", "log")
/// are read as words.
const SPELLED_EXTENSIONS: &[&str] = &[
    "exe", "doc", "docx", "odt", "ods", "odp", "apk", "avi", "gif", "ico", "iso", "heic", "aac",
    "ogg", "ini", "eps", "bat", "deb", "dat", "obj", "msi", "epub", "xlsm", "jpeg",
];

/// The extensions that make a word a file name: every one above, those without a vowel, and
/// those read as words.
pub(super) fn is_extension(ext: &str) -> bool {
    let lower = ext.to_ascii_lowercase();
    const OTHERS: &[&str] = &[
        "pdf", "xlsx", "xls", "csv", "tsv", "pptx", "ppt", "txt", "rtf", "png", "jpg", "svg",
        "bmp", "tif", "tiff", "webp", "mp3", "mp4", "mkv", "wav", "flac", "mov", "html", "htm",
        "xml", "css", "js", "ts", "py", "rs", "sh", "md", "sql", "php", "dll", "dmg", "zip", "rar",
        "tar", "gz", "json", "yaml", "yml", "toml", "conf", "cfg", "log", "csv", "key",
    ];
    (ext.chars().all(|c| c.is_ascii_lowercase()) || ext.chars().all(|c| c.is_ascii_uppercase()))
        && (SPELLED_EXTENSIONS.contains(&lower.as_str()) || OTHERS.contains(&lower.as_str()))
}

/// Whether a run of letters is read out letter by letter: one or two letters, as an address
/// spells "io" and "uk", no vowel at all ("pdf", "kTx", "https"), or a spelled extension.
fn spelled(run: &str, extension: bool, short: bool) -> bool {
    (short && run.len() <= 2)
        || !run.chars().any(|c| "aeiouyAEIOUY".contains(c))
        || (extension && SPELLED_EXTENSIONS.contains(&run.to_ascii_lowercase().as_str()))
}

/// What "www" is called: three of the language's names for w, or for v where that is how it
/// is said.
fn www(lang: Lang) -> String {
    let letter = if matches!(lang, Lang::Fi | Lang::Sv | Lang::It) { "v" } else { "w" };
    vec![spell(letter, lang); 3].join(" ")
}

/// A digit run inside an address: a number, unless a leading zero makes it a code read digit
/// by digit ("0412").
fn digits(run: &str, lang: Lang) -> String {
    numbers::numbers(run, lang).unwrap_or_else(|| run.to_string())
}

/// Read an address, a path or a file name, made of ASCII letters, digits and the separators
/// `. _ - / @ : +`, in `lang`, which must have letter names. The separators are spoken, digits
/// are read as numbers, and runs of letters are spelled when [`spelled`] says so and kept as
/// words otherwise. With an `@`, short words stay words but for the last ("no-reply@x.co.uk").
pub(super) fn read(address: &str, lang: Lang) -> String {
    let sc = lang.special_chars();
    let last_dot = address.rfind('.');
    let at = address.find('@');
    let mut words: Vec<String> = vec![];
    let mut start = 0;
    let bytes = address.as_bytes();
    while start < bytes.len() {
        let c = bytes[start];
        let separator = match c {
            b'.' => Some(sc.dot),
            b'_' => Some(sc.underscore),
            b'-' => Some(sc.dash),
            b'/' => Some(sc.slash),
            b'@' => Some(sc.at),
            b':' => Some(sc.colon),
            b'+' => Some(sc.plus),
            _ => None,
        };
        if let Some(separator) = separator {
            words.push(separator.to_string());
            start += 1;
            continue;
        }
        // A run of letters or of digits, or any other character on its own.
        let digit = c.is_ascii_digit();
        let len = if c.is_ascii_alphanumeric() {
            bytes[start..]
                .iter()
                .take_while(|b| if digit { b.is_ascii_digit() } else { b.is_ascii_alphabetic() })
                .count()
        } else {
            address[start..].chars().next().map_or(1, char::len_utf8)
        };
        let run = &address[start..start + len];
        if digit {
            words.push(digits(run, lang));
        } else if run.eq_ignore_ascii_case("www") {
            words.push(www(lang));
        } else if c.is_ascii_alphabetic() {
            let extension = last_dot.is_some_and(|dot| dot < start) && start + len == address.len();
            // An email address spells its top-level domain alone, as the email rule does.
            let short = at.is_none() || extension;
            if spelled(run, extension, short) {
                words.push(spell(run, lang));
            } else {
                words.push(run.to_string());
            }
        } else {
            words.push(run.to_string());
        }
        start += len;
    }
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letters_have_names() {
        assert_eq!(spell("pdf", Lang::Fi), "pee dee äf");
        assert_eq!(spell("XLSX", Lang::Sv), "eks ell ess eks");
        assert_eq!(spell("jpg", Lang::No), "jådd pe ge");
        assert_eq!(spell("WJ", Lang::It), "doppia vu i lunga");
        assert_eq!(spell("y", Lang::Nl), "griekse ij");
        assert_eq!(spell("vw", Lang::Pl), "fał wu");
        assert_eq!(spell("AB", Lang::Ar), "إيه بي");
        assert_eq!(spell("AZ", Lang::Hi), "ए ज़ेड");
        assert_eq!(spell("pdf", Lang::Ko), "피디에프");
        assert_eq!(spell("sh", Lang::Ja), "エスエイチ");
        // The five languages the serving stack reads have none.
        assert_eq!(names(Lang::En), None);
        assert_eq!(spell("pdf", Lang::De), "pdf");
    }

    #[test]
    fn addresses_are_read() {
        let cases = [
            (
                Lang::Fi,
                "report_final_v3.pdf",
                "report alaviiva final alaviiva vee 3 piste pee dee äf",
            ),
            (Lang::Fi, "gradium.sh/docs", "gradium piste äs hoo kauttaviiva docs"),
            (
                Lang::Fi,
                "invoice_0412.png",
                "invoice alaviiva nolla neljä yksi kaksi piste pee än gee",
            ),
            (Lang::Fi, "setup.exe", "setup piste ee äks ee"),
            (Lang::Fi, "settings.json", "settings piste json"),
            (Lang::Fi, "www.example.org", "vee vee vee piste example piste org"),
            (
                Lang::Fi,
                "no-reply@shop.example.com",
                "no viiva reply ät shop piste example piste com",
            ),
            // An email address spells its top-level domain alone.
            (
                Lang::It,
                "billing@company.co.uk",
                "billing chiocciola company punto co punto u cappa",
            ),
            (Lang::Fi, "bit.ly/3kTx9Q", "bit piste äl yy kauttaviiva 3 koo tee äks 9 kuu"),
            (
                Lang::Sv,
                "https://status.example.com",
                "hå te te pe ess kolon snedstreck snedstreck status punkt example punkt com",
            ),
            (Lang::Nl, "www.example.org", "wee wee wee punt example punt org"),
            (Lang::Pl, "logo_dark@2x.png", "logo podkreślnik dark małpa 2 iks kropka pe en gie"),
            (Lang::Ko, "notes.txt", "notes 닷 티엑스티"),
            (Lang::Hi, "photo_01.jpg", "photo अंडरस्कोर शून्य एक डॉट जे पी जी"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(read(input, lang), expected, "{lang:?} {input:?}");
        }
    }

    #[test]
    fn extensions_are_known() {
        assert!(is_extension("pdf") && is_extension("PDF") && is_extension("json"));
        assert!(!is_extension("Pdf") && !is_extension("com") && !is_extension("docs"));
    }
}
