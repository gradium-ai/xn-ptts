//! Text normalization applied before tokenization.
//!
//! The model is trained on ordinary written prose, so characters it never saw -- typographic
//! quotes, bullets, arrows, emoji -- are best removed rather than tokenized, and symbols that
//! are read aloud (`@`, `+`, `=`) are best spelled out in the target language. [`normalize_text`]
//! does both, driven by a [`Lang`].
//!
//! Every caller says which language, or says not to normalize: [`Normalize`] is a required
//! argument to [`crate::synth::SynthBuilder::new`], and every frontend takes it as a required
//! flag. There is no default, deliberately. Output is noticeably better with normalization than
//! without, but normalizing German as English speaks `@` as "at" rather than "ät", so guessing
//! the language is worse than doing nothing.

mod rewrite;

pub use rewrite::{Rules, rewrite_word};

/// Spoken forms of the punctuation characters that are read aloud rather than dropped: `@`, `+`
/// and `=` anywhere in the text, and the separators inside emails, URLs and codes.
#[derive(Debug, Clone)]
pub struct SpecialChars {
    pub at: &'static str,
    pub plus: &'static str,
    pub equals: &'static str,
    pub colon: &'static str,
    pub slash: &'static str,
    pub dash: &'static str,
    pub dot: &'static str,
    pub underscore: &'static str,
}

pub const SPECIAL_CHARS_EN: SpecialChars = SpecialChars {
    at: "at",
    plus: "plus",
    equals: "equals",
    colon: "colon",
    slash: "slash",
    dash: "dash",
    dot: "dot",
    underscore: "underscore",
};

pub const SPECIAL_CHARS_FR: SpecialChars = SpecialChars {
    at: "arobaze",
    plus: "plus",
    equals: "égal",
    colon: "deux-points",
    slash: "slash",
    dash: "tiret",
    dot: "point",
    underscore: "underscore",
};

pub const SPECIAL_CHARS_DE: SpecialChars = SpecialChars {
    at: "ät",
    plus: "Plus",
    equals: "Gleich",
    colon: "Doppelpunkt",
    slash: "Slash",
    dash: "Bindestrich",
    dot: "Punkt",
    underscore: "Unterstrich",
};

pub const SPECIAL_CHARS_ES: SpecialChars = SpecialChars {
    at: "arroba",
    plus: "mas",
    equals: "igual",
    colon: "dos-puntos",
    slash: "slash",
    dash: "guion",
    dot: "punto",
    underscore: "guion bajo",
};

pub const SPECIAL_CHARS_PT: SpecialChars = SpecialChars {
    at: "arroba",
    plus: "mais",
    equals: "igual",
    colon: "dois-pontos",
    slash: "slash",
    dash: "hifen",
    dot: "ponto",
    underscore: "underscore",
};

pub const SPECIAL_CHARS_AR: SpecialChars = SpecialChars {
    at: "آت",
    plus: "زائد",
    equals: "يساوي",
    colon: "نقطتان",
    slash: "شرطة مائلة",
    dash: "شرطة",
    dot: "نقطة",
    underscore: "شرطة سفلية",
};

pub const SPECIAL_CHARS_HI: SpecialChars = SpecialChars {
    at: "ऐट",
    plus: "प्लस",
    equals: "बराबर",
    colon: "कोलन",
    slash: "स्लैश",
    dash: "डैश",
    dot: "डॉट",
    underscore: "अंडरस्कोर",
};

pub const SPECIAL_CHARS_KO: SpecialChars = SpecialChars {
    at: "골뱅이",
    plus: "플러스",
    equals: "이퀄",
    colon: "콜론",
    slash: "슬래시",
    dash: "하이픈",
    dot: "닷",
    underscore: "언더바",
};

pub const SPECIAL_CHARS_PL: SpecialChars = SpecialChars {
    at: "małpa",
    plus: "plus",
    equals: "równa się",
    colon: "dwukropek",
    slash: "ukośnik",
    dash: "myślnik",
    dot: "kropka",
    underscore: "podkreślnik",
};

pub const SPECIAL_CHARS_SV: SpecialChars = SpecialChars {
    at: "snabel-a",
    plus: "plus",
    equals: "lika med",
    colon: "kolon",
    slash: "snedstreck",
    dash: "bindestreck",
    dot: "punkt",
    underscore: "understreck",
};

pub const SPECIAL_CHARS_FI: SpecialChars = SpecialChars {
    at: "ät",
    plus: "plus",
    equals: "on yhtä kuin",
    colon: "kaksoispiste",
    slash: "kauttaviiva",
    dash: "viiva",
    dot: "piste",
    underscore: "alaviiva",
};

pub const SPECIAL_CHARS_NO: SpecialChars = SpecialChars {
    at: "alfakrøll",
    plus: "pluss",
    equals: "er lik",
    colon: "kolon",
    slash: "skråstrek",
    dash: "bindestrek",
    dot: "punktum",
    underscore: "understrek",
};

pub const SPECIAL_CHARS_IT: SpecialChars = SpecialChars {
    at: "chiocciola",
    plus: "più",
    equals: "uguale",
    colon: "due punti",
    slash: "barra",
    dash: "trattino",
    dot: "punto",
    underscore: "trattino basso",
};

pub const SPECIAL_CHARS_NL: SpecialChars = SpecialChars {
    at: "apenstaartje",
    plus: "plus",
    equals: "is gelijk aan",
    colon: "dubbele punt",
    slash: "schuine streep",
    dash: "streepje",
    dot: "punt",
    underscore: "liggend streepje",
};

pub const SPECIAL_CHARS_JA: SpecialChars = SpecialChars {
    at: "アットマーク",
    plus: "プラス",
    equals: "イコール",
    colon: "コロン",
    slash: "スラッシュ",
    dash: "ハイフン",
    dot: "ドット",
    underscore: "アンダーバー",
};

/// Language driving the spoken forms used by [`normalize_text`].
///
/// Deliberately has no `Default`: the spoken forms differ per language, so a caller that has not
/// said which language it has is better off not normalizing at all. See [`Normalize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Lang {
    En,
    Fr,
    De,
    Es,
    Pt,
    Ar,
    Hi,
    Ko,
    Pl,
    Sv,
    Fi,
    No,
    It,
    Nl,
    Ja,
}

impl std::str::FromStr for Lang {
    type Err = crate::Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        let lower = s.trim().to_lowercase();
        // Bokmål's own code too, for YAML files, where a bare `no` reads as false.
        let code = if lower == "nb" { "no" } else { lower.as_str() };
        Lang::ALL.iter().copied().find(|lang| lang.as_str() == code).ok_or_else(|| {
            crate::Error::invalid_argument(format!(
                "unsupported language code: {s}; expected one of {}, or none to skip \
                 normalization",
                Lang::ALL.iter().map(|lang| lang.as_str()).collect::<Vec<_>>().join(", ")
            ))
        })
    }
}

impl Lang {
    /// Every language, in the order the frontends list them.
    pub const ALL: &'static [Lang] = &[
        Lang::En,
        Lang::Fr,
        Lang::De,
        Lang::Es,
        Lang::Pt,
        Lang::Ar,
        Lang::Hi,
        Lang::Ko,
        Lang::Pl,
        Lang::Sv,
        Lang::Fi,
        Lang::No,
        Lang::It,
        Lang::Nl,
        Lang::Ja,
    ];

    /// The language code this variant parses from.
    pub fn as_str(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Fr => "fr",
            Lang::De => "de",
            Lang::Es => "es",
            Lang::Pt => "pt",
            Lang::Ar => "ar",
            Lang::Hi => "hi",
            Lang::Ko => "ko",
            Lang::Pl => "pl",
            Lang::Sv => "sv",
            Lang::Fi => "fi",
            Lang::No => "no",
            Lang::It => "it",
            Lang::Nl => "nl",
            Lang::Ja => "ja",
        }
    }

    pub fn special_chars(self) -> &'static SpecialChars {
        match self {
            Lang::En => &SPECIAL_CHARS_EN,
            Lang::Fr => &SPECIAL_CHARS_FR,
            Lang::De => &SPECIAL_CHARS_DE,
            Lang::Es => &SPECIAL_CHARS_ES,
            Lang::Pt => &SPECIAL_CHARS_PT,
            Lang::Ar => &SPECIAL_CHARS_AR,
            Lang::Hi => &SPECIAL_CHARS_HI,
            Lang::Ko => &SPECIAL_CHARS_KO,
            Lang::Pl => &SPECIAL_CHARS_PL,
            Lang::Sv => &SPECIAL_CHARS_SV,
            Lang::Fi => &SPECIAL_CHARS_FI,
            Lang::No => &SPECIAL_CHARS_NO,
            Lang::It => &SPECIAL_CHARS_IT,
            Lang::Nl => &SPECIAL_CHARS_NL,
            Lang::Ja => &SPECIAL_CHARS_JA,
        }
    }

    fn about(self) -> &'static str {
        match self {
            Lang::En => "about",
            Lang::Fr => "environ",
            Lang::De => "etwa",
            Lang::Es | Lang::Pt => "aproximadamente",
            Lang::Ar => "حوالي",
            Lang::Hi => "लगभग",
            Lang::Ko => "약",
            Lang::Pl => "około",
            Lang::Sv | Lang::No => "cirka",
            Lang::Fi => "noin",
            Lang::It => "circa",
            Lang::Nl => "ongeveer",
            Lang::Ja => "約",
        }
    }

    fn plus_minus(self) -> &'static str {
        match self {
            Lang::En => "plus or minus",
            Lang::Fr => "plus ou moins",
            Lang::De => "plus minus",
            Lang::Es => "más o menos",
            Lang::Pt => "mais ou menos",
            Lang::Ar => "زائد أو ناقص",
            Lang::Hi => "प्लस माइनस",
            Lang::Ko => "플러스 마이너스",
            Lang::Pl | Lang::Sv => "plus minus",
            Lang::Fi => "plus miinus",
            Lang::No => "pluss minus",
            Lang::It => "più o meno",
            Lang::Nl => "plus of min",
            Lang::Ja => "プラスマイナス",
        }
    }

    fn love(self) -> &'static str {
        match self {
            Lang::En => "love",
            Lang::Fr => "aime",
            Lang::De => "liebe",
            Lang::Es | Lang::Pt => "amo",
            Lang::Ar => "أحب",
            Lang::Hi => "प्यार",
            Lang::Ko => "사랑해",
            Lang::Pl => "kocham",
            Lang::Sv => "älskar",
            Lang::Fi => "rakastan",
            Lang::No => "elsker",
            Lang::It => "amo",
            Lang::Nl => "hou van",
            Lang::Ja => "大好き",
        }
    }

    fn star(self) -> &'static str {
        match self {
            Lang::En => "star",
            Lang::Fr => "étoile",
            Lang::De => "Stern",
            Lang::Es | Lang::Pt | Lang::It => "asterisco",
            Lang::Ar => "نجمة",
            Lang::Hi => "तारा",
            Lang::Ko => "별표",
            Lang::Pl => "gwiazdka",
            Lang::Sv => "stjärna",
            Lang::Fi => "tähti",
            Lang::No => "stjerne",
            Lang::Nl => "sterretje",
            Lang::Ja => "アスタリスク",
        }
    }

    /// The word for `&`, in the languages whose tokenizers have no piece for the sign. The
    /// others keep it for the model to read.
    fn ampersand(self) -> Option<&'static str> {
        Some(match self {
            Lang::Ar => "و",
            Lang::Hi => "और",
            Lang::Fi => "ja",
            Lang::Sv => "och",
            Lang::No => "og",
            Lang::It => "e",
            Lang::Nl => "en",
            Lang::Pl => "i",
            _ => return None,
        })
    }

    /// Whether a `#` before a number is spelled as the number sign, where the tokenizer has no
    /// piece for it.
    fn spells_number_sign(self) -> bool {
        matches!(
            self,
            Lang::Ar | Lang::Hi | Lang::Fi | Lang::Sv | Lang::No | Lang::It | Lang::Nl | Lang::Pl
        )
    }

    /// The word after the degrees for `°C`, or `None` where it goes unsaid, as Finnish,
    /// Italian, Swedish, Norwegian, Japanese and Korean read a temperature.
    fn celsius(self) -> Option<&'static str> {
        Some(match self {
            Lang::Fi | Lang::It | Lang::Sv | Lang::No | Lang::Ja | Lang::Ko => return None,
            Lang::Pl => "Celsjusza",
            Lang::Hi => "सेल्सियस",
            Lang::Ar => "مئوية",
            _ => "Celsius",
        })
    }

    /// The word after the degrees for `°F`, as [`Self::celsius`].
    fn fahrenheit(self) -> Option<&'static str> {
        Some(match self {
            Lang::Ja | Lang::Ko => return None,
            Lang::Pl => "Fahrenheita",
            Lang::Hi => "फ़ारेनहाइट",
            Lang::Ar => "فهرنهايت",
            _ => "Fahrenheit",
        })
    }

    /// The word "n°" abbreviates, as in "le n°5" or "el n.º 5".
    fn number_sign(self) -> &'static str {
        match self {
            Lang::En => "number",
            Lang::Fr => "numéro",
            Lang::De => "Nummer",
            Lang::Es | Lang::Pt => "número",
            Lang::Ar => "رقم",
            Lang::Hi => "नंबर",
            Lang::Ko => "번호",
            Lang::Pl => "numer",
            Lang::Sv | Lang::No | Lang::Nl => "nummer",
            Lang::Fi | Lang::It => "numero",
            Lang::Ja => "番号",
        }
    }

    fn degrees(self, one: bool) -> &'static str {
        match (self, one) {
            (Lang::En, true) => "degree",
            (Lang::En, false) => "degrees",
            (Lang::Fr, true) => "degré",
            (Lang::Fr, false) => "degrés",
            (Lang::De, _) => "Grad",
            (Lang::Es, true) => "grado",
            (Lang::Es, false) => "grados",
            (Lang::Pt, true) => "grau",
            (Lang::Pt, false) => "graus",
            (Lang::Ar, true) => "درجة",
            (Lang::Ar, false) => "درجات",
            (Lang::Hi, _) => "डिग्री",
            (Lang::Ko, _) => "도",
            (Lang::Pl, true) => "stopień",
            (Lang::Pl, false) => "stopni",
            (Lang::Sv | Lang::No, true) => "grad",
            (Lang::Sv | Lang::No, false) => "grader",
            (Lang::Fi, true) => "aste",
            (Lang::Fi, false) => "astetta",
            (Lang::It, true) => "grado",
            (Lang::It, false) => "gradi",
            (Lang::Nl, true) => "graad",
            (Lang::Nl, false) => "graden",
            (Lang::Ja, _) => "度",
        }
    }

    /// The spoken form of `c`, one of `×`, `÷`, `≤`, `≥`, `≠`, `≈`, `√`, `∞`, `∑` and `‰`: each
    /// language's row below reads them in that order.
    fn symbol(self, c: char) -> Option<&'static str> {
        let i = "×÷≤≥≠≈√∞∑‰".chars().position(|s| s == c)?;
        let row = match self {
            Lang::En => {
                "times|divided by|less than or equal to|greater than or equal to|is not equal to|approximately|square root of|infinity|sigma|per mille"
            }
            Lang::Fr => {
                "fois|divisé par|inférieur ou égal à|supérieur ou égal à|différent de|environ|racine carrée de|infini|sigma|pour mille"
            }
            Lang::De => {
                "mal|geteilt durch|kleiner oder gleich|größer oder gleich|ungleich|ungefähr|Wurzel aus|unendlich|Sigma|Promille"
            }
            Lang::Es => {
                "por|dividido entre|menor o igual que|mayor o igual que|distinto de|aproximadamente|raíz cuadrada de|infinito|sigma|por mil"
            }
            Lang::Pt => {
                "vezes|dividido por|menor ou igual a|maior ou igual a|diferente de|aproximadamente|raiz quadrada de|infinito|sigma|por mil"
            }
            Lang::Ar => {
                "ضرب|تقسيم|أصغر من أو يساوي|أكبر من أو يساوي|لا يساوي|تقريبا|الجذر التربيعي لـ|ما لا نهاية|سيغما|في الألف"
            }
            Lang::Hi => {
                "गुणा|भाग|से कम या बराबर|से अधिक या बराबर|बराबर नहीं|लगभग|वर्गमूल|अनंत|सिग्मा|प्रति हज़ार"
            }
            Lang::Ko => {
                "곱하기|나누기|작거나 같다|크거나 같다|같지 않다|약|루트|무한대|시그마|퍼밀"
            }
            Lang::Pl => {
                "razy|podzielić przez|mniejsze lub równe|większe lub równe|różne od|około|pierwiastek z|nieskończoność|sigma|promil"
            }
            Lang::Sv => {
                "gånger|delat med|mindre än eller lika med|större än eller lika med|inte lika med|ungefär|roten ur|oändligheten|sigma|promille"
            }
            Lang::Fi => {
                "kertaa|jaettuna|pienempi tai yhtä suuri kuin|suurempi tai yhtä suuri kuin|eri suuri kuin|noin|neliöjuuri|ääretön|sigma|promille"
            }
            Lang::No => {
                "ganger|delt på|mindre enn eller lik|større enn eller lik|ikke lik|omtrent|kvadratroten av|uendelig|sigma|promille"
            }
            Lang::It => {
                "per|diviso|minore o uguale a|maggiore o uguale a|diverso da|circa|radice quadrata di|infinito|sigma|per mille"
            }
            Lang::Nl => {
                "keer|gedeeld door|kleiner dan of gelijk aan|groter dan of gelijk aan|is niet gelijk aan|ongeveer|wortel van|oneindig|sigma|promille"
            }
            Lang::Ja => "かける|わる|以下|以上|ノットイコール|約|ルート|無限大|シグマ|パーミル",
        };
        row.split('|').nth(i)
    }

    fn stars(self, n: &str, of: Option<usize>) -> String {
        match (self, of) {
            (Lang::En, Some(of)) => format!("{n} out of {of} stars"),
            (Lang::En, None) => format!("{n} stars"),
            (Lang::Fr, Some(of)) => format!("{n} étoiles sur {of}"),
            (Lang::Fr, None) => format!("{n} étoiles"),
            (Lang::De, Some(of)) => format!("{n} von {of} Sternen"),
            (Lang::De, None) => format!("{n} Sterne"),
            (Lang::Es, Some(of)) => format!("{n} de {of} estrellas"),
            (Lang::Es, None) => format!("{n} estrellas"),
            (Lang::Pt, Some(of)) => format!("{n} de {of} estrelas"),
            (Lang::Pt, None) => format!("{n} estrelas"),
            (Lang::Ar, Some(of)) => format!("{n} من {of} نجوم"),
            (Lang::Ar, None) => format!("{n} نجوم"),
            (Lang::Hi, Some(of)) => format!("{of} में से {n} सितारे"),
            (Lang::Hi, None) => format!("{n} सितारे"),
            (Lang::Ko, Some(of)) => format!("{of}개 중 별 {n}개"),
            (Lang::Ko, None) => format!("별 {n}개"),
            (Lang::Pl, Some(of)) => format!("{n} z {of} gwiazdek"),
            (Lang::Pl, None) => format!("{n} gwiazdek"),
            (Lang::Sv, Some(of)) => format!("{n} av {of} stjärnor"),
            (Lang::Sv, None) => format!("{n} stjärnor"),
            (Lang::Fi, Some(of)) => format!("{n} tähteä {of}:stä"),
            (Lang::Fi, None) => format!("{n} tähteä"),
            (Lang::No, Some(of)) => format!("{n} av {of} stjerner"),
            (Lang::No, None) => format!("{n} stjerner"),
            (Lang::It, Some(of)) => format!("{n} stelle su {of}"),
            (Lang::It, None) => format!("{n} stelle"),
            (Lang::Nl, Some(of)) => format!("{n} van de {of} sterren"),
            (Lang::Nl, None) => format!("{n} sterren"),
            (Lang::Ja, Some(of)) => format!("{of}つ星のうち{n}つ"),
            (Lang::Ja, None) => format!("星{n}つ"),
        }
    }

    fn power(self, n: &str) -> String {
        match (self, n) {
            (Lang::En, "2") => "squared".into(),
            (Lang::En, "3") => "cubed".into(),
            (Lang::En, n) => format!("to the power of {n}"),
            (Lang::Fr, "2") => "au carré".into(),
            (Lang::Fr, "3") => "au cube".into(),
            (Lang::Fr, n) => format!("puissance {n}"),
            (Lang::De, n) => format!("hoch {n}"),
            (Lang::Es, "2") => "al cuadrado".into(),
            (Lang::Es, "3") => "al cubo".into(),
            (Lang::Es, n) => format!("elevado a {n}"),
            (Lang::Pt, "2") => "ao quadrado".into(),
            (Lang::Pt, "3") => "ao cubo".into(),
            (Lang::Pt, n) => format!("elevado a {n}"),
            (Lang::Ar, n) => format!("أس {n}"),
            (Lang::Hi, "2") => "का वर्ग".into(),
            (Lang::Hi, "3") => "का घन".into(),
            (Lang::Hi, n) => format!("की घात {n}"),
            (Lang::Ko, "2") => "제곱".into(),
            (Lang::Ko, "3") => "세제곱".into(),
            (Lang::Ko, n) => format!("{n}제곱"),
            (Lang::Pl, "2") => "do kwadratu".into(),
            (Lang::Pl, "3") => "do sześcianu".into(),
            (Lang::Pl, n) => format!("do potęgi {n}"),
            (Lang::Sv, "2") => "i kvadrat".into(),
            (Lang::Sv, "3") => "i kubik".into(),
            (Lang::Sv, n) => format!("upphöjt till {n}"),
            (Lang::Fi, "2") => "toiseen".into(),
            (Lang::Fi, "3") => "kolmanteen".into(),
            (Lang::Fi, n) => format!("potenssiin {n}"),
            (Lang::No, "2") => "i andre".into(),
            (Lang::No, "3") => "i tredje".into(),
            (Lang::No, n) => format!("opphøyd i {n}"),
            (Lang::It, "2") => "al quadrato".into(),
            (Lang::It, "3") => "al cubo".into(),
            (Lang::It, n) => format!("alla {n}"),
            (Lang::Nl, "2") => "kwadraat".into(),
            (Lang::Nl, "3") => "tot de derde".into(),
            (Lang::Nl, n) => format!("tot de macht {n}"),
            (Lang::Ja, n) => format!("の{n}乗"),
        }
    }

    fn greek(self, c: char) -> Option<String> {
        let i = match c {
            'α'..='ρ' => c as usize - 'α' as usize,
            'ς' => 17,
            'σ'..='ω' => c as usize - 'α' as usize - 1,
            'Α'..='Ρ' => c as usize - 'Α' as usize,
            'Σ'..='Ω' => c as usize - 'Α' as usize - 1,
            _ => return None,
        };
        let names = match self {
            Lang::En | Lang::Fr | Lang::De | Lang::Es | Lang::Pt => {
                "alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega"
            }
            Lang::Ar => {
                "ألفا|بيتا|غاما|دلتا|إبسيلون|زيتا|إيتا|ثيتا|يوتا|كابا|لامدا|مو|نو|كسي|أوميكرون|باي|رو|سيغما|تاو|أبسيلون|فاي|خي|بساي|أوميغا"
            }
            Lang::Hi => {
                "अल्फा|बीटा|गामा|डेल्टा|एप्सिलॉन|जीटा|ईटा|थीटा|आयोटा|कप्पा|लैम्ब्डा|म्यू|न्यू|क्साई|ओमिक्रॉन|पाई|रो|सिग्मा|टाऊ|अप्सिलॉन|फाई|काई|साई|ओमेगा"
            }
            Lang::Ko => {
                "알파|베타|감마|델타|엡실론|제타|에타|세타|요타|카파|람다|뮤|뉴|크시|오미크론|파이|로|시그마|타우|입실론|피|카이|프사이|오메가"
            }
            Lang::Pl => {
                "alfa|beta|gamma|delta|epsilon|dzeta|eta|theta|jota|kappa|lambda|mi|ni|ksi|omikron|pi|ro|sigma|tau|ypsilon|fi|chi|psi|omega"
            }
            Lang::Sv => {
                "alfa|beta|gamma|delta|epsilon|zeta|eta|theta|jota|kappa|lambda|my|ny|xi|omikron|pi|rho|sigma|tau|ypsilon|fi|chi|psi|omega"
            }
            Lang::Fi => {
                "alfa|beeta|gamma|delta|epsilon|zeeta|eeta|theeta|ioota|kappa|lambda|myy|nyy|ksii|omikron|pii|rhoo|sigma|tau|ypsilon|fii|khii|psii|oomega"
            }
            Lang::No => {
                "alfa|beta|gamma|delta|epsilon|zeta|eta|theta|jota|kappa|lambda|my|ny|ksi|omikron|pi|rho|sigma|tau|ypsilon|fi|khi|psi|omega"
            }
            Lang::It => {
                "alfa|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mi|ni|xi|omicron|pi greco|rho|sigma|tau|ipsilon|fi|chi|psi|omega"
            }
            Lang::Nl => {
                "alfa|bèta|gamma|delta|epsilon|zèta|èta|thèta|jota|kappa|lambda|mu|nu|xi|omikron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega"
            }
            Lang::Ja => {
                "アルファ|ベータ|ガンマ|デルタ|イプシロン|ゼータ|イータ|シータ|イオタ|カッパ|ラムダ|ミュー|ニュー|クシー|オミクロン|パイ|ロー|シグマ|タウ|ウプシロン|ファイ|カイ|プサイ|オメガ"
            }
        };
        let name = names.split('|').nth(i)?;
        Some(match self {
            Lang::De => name[..1].to_uppercase() + &name[1..],
            _ => name.to_string(),
        })
    }

    /// The letter `c` folds to, when the tokenizer of the checkpoints for this language has no
    /// piece for it. A new language must check its own letters against this table.
    fn fold(self, c: char) -> Option<char> {
        match self {
            Lang::En | Lang::Fr | Lang::De | Lang::Es | Lang::Pt => unaccented(c),
            // The table was made against the tokenizer of the five above, which has no piece
            // for these letters; a checkpoint that speaks the language must.
            Lang::It if "ìòÌÒ".contains(c) => None,
            Lang::Pl if "ąćęńśźżĄĆĘŃŚŹŻ".contains(c) => None,
            Lang::Ar
            | Lang::Hi
            | Lang::Ko
            | Lang::Pl
            | Lang::Sv
            | Lang::Fi
            | Lang::No
            | Lang::It
            | Lang::Nl
            | Lang::Ja => unaccented(c),
        }
    }

    pub fn decimal_separator(self) -> &'static str {
        match self {
            Lang::En => "point",
            Lang::Fr => "virgule",
            Lang::De => "Komma",
            Lang::Es => "coma",
            Lang::Pt => "vírgula",
            Lang::Ar => "فاصلة",
            Lang::Hi => "दशमलव",
            Lang::Ko => "점",
            Lang::Pl => "przecinek",
            Lang::Sv | Lang::No | Lang::Nl => "komma",
            Lang::Fi => "pilkku",
            Lang::It => "virgola",
            Lang::Ja => "点",
        }
    }
}

/// Whether to normalize, in which language, and with which [`Rules`].
///
/// There is no default and no "unset": [`crate::synth::SynthBuilder::new`] takes one of these,
/// so choosing is not something a caller can forget. [`Normalize::OFF`] is the way to say "hand
/// the text to the tokenizer as written", which is for callers that normalize it themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Normalize {
    lang: Option<Lang>,
    rules: Rules,
}

impl Normalize {
    /// Hand text to the tokenizer as written.
    pub const OFF: Self = Self { lang: None, rules: Rules::NONE };

    /// Normalize as `lang`, with the default rewrite rules, [`Rules::DEFAULT`].
    pub const fn for_lang(lang: Lang) -> Self {
        Self { lang: Some(lang), rules: Rules::DEFAULT }
    }

    /// This policy with `rules` instead. A no-op on [`Self::OFF`], which rewrites nothing.
    pub const fn with_rules(self, rules: Rules) -> Self {
        match self.lang {
            Some(lang) => Self { lang: Some(lang), rules },
            None => Self::OFF,
        }
    }

    /// A warning when this policy normalizes text as a language the model was not trained for,
    /// given the `languages` its config lists. A config that lists none is not checked.
    pub fn language_warning(self, languages: &[String]) -> Option<String> {
        let lang = self.lang?.as_str();
        if languages.is_empty() || languages.iter().any(|l| l == lang) {
            return None;
        }
        Some(format!(
            "text is normalized as {lang}, but the model was trained for {}",
            languages.join(", ")
        ))
    }

    /// Which rewrites this policy applies.
    pub const fn rules(self) -> Rules {
        self.rules
    }

    /// The language half of this policy, round-tripping through its `FromStr`. The rules are
    /// not part of it, since they parse from a flag of their own: see [`Rules`].
    pub fn as_str(self) -> &'static str {
        match self.lang {
            Some(lang) => lang.as_str(),
            None => "none",
        }
    }

    /// Normalize `text`, or hand it back untouched when this is [`Self::OFF`].
    ///
    /// Borrows when off, so opting out costs no allocation per request.
    pub fn apply<'a>(self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match self.lang {
            None => std::borrow::Cow::Borrowed(text),
            Some(lang) => std::borrow::Cow::Owned(normalize_text(text, lang, self.rules)),
        }
    }
}

/// Normalizes as `lang`, with the default rewrite rules: what [`Normalize::for_lang`] makes.
impl From<Lang> for Normalize {
    fn from(lang: Lang) -> Self {
        Self::for_lang(lang)
    }
}

/// Parse what the frontends' `--lang` flag accepts: a language code, or `none` / `off`. The
/// rules are a flag of their own, see [`Rules`].
impl std::str::FromStr for Normalize {
    type Err = crate::Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "none" | "off" => Ok(Self::OFF),
            other => other.parse().map(Self::for_lang),
        }
    }
}

/// Character sink that keeps the output free of the punctuation pile-ups the substitutions
/// below would otherwise produce: a `.` or `,` swallows any whitespace and punctuation
/// immediately before it, and runs of whitespace collapse to a single space.
struct StringAppender {
    buffer: Vec<char>,
}

impl StringAppender {
    fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    fn push(&mut self, c: char) {
        if c == '.' || c == ',' {
            while self
                .buffer
                .last()
                .is_some_and(|&l| l.is_whitespace() || (l.is_ascii_punctuation() && !self.keeps(l)))
            {
                self.buffer.pop();
            }
        }
        self.buffer.push(c);
    }

    fn push_str(&mut self, s: &str) {
        for c in s.chars() {
            self.push(c);
        }
    }

    /// Whether `last`, the last character pushed, stays before a `.` or `,`. Quotes do: dropping
    /// a closing one would leave the opening one unbalanced. So do the symbols the rewrite rules
    /// read: `@` and `+`, which are spelled out after the rules, and the `$` of an amount, as in
    /// "it costs 5$.". So do `%`, `#` and `&`, which the model reads aloud: dropping one changes
    /// what is said.
    fn keeps(&self, last: char) -> bool {
        match last {
            '"' | '\'' | '@' | '+' | '%' | '#' | '&' => true,
            '$' => self.buffer.len() >= 2 && self.buffer[self.buffer.len() - 2].is_ascii_digit(),
            _ => false,
        }
    }

    fn ends_with_one(&self) -> bool {
        let b = &self.buffer
            [..self.buffer.iter().rposition(|c| !c.is_whitespace()).map_or(0, |i| i + 1)];
        b.last() == Some(&'1')
            && !b
                .len()
                .checked_sub(2)
                .is_some_and(|i| b[i].is_ascii_digit() || b[i] == '.' || b[i] == ',')
    }

    /// Whether the text ends with an "n" or "N" that is a word of its own, with or without a
    /// period: before a `°` or the ordinal sign `º`, it is the number sign, written "n°" in
    /// French and "n.º" or "nº" in Spanish and Portuguese, rather than "n degrees".
    fn ends_with_n(&self) -> bool {
        let b = self.buffer.strip_suffix(&['.']).unwrap_or(&self.buffer);
        matches!(b.last(), Some('n' | 'N'))
            && !b.len().checked_sub(2).is_some_and(|i| b[i].is_alphanumeric())
    }

    /// Whether the text ends with a digit and then whitespace, as before the `€` of "250 €".
    fn ends_with_spaced_digit(&self) -> bool {
        let b = &self.buffer;
        self.last_is_whitespace()
            && b.iter().rev().find(|c| !c.is_whitespace()).is_some_and(char::is_ascii_digit)
    }

    /// How many digits the number the text ends with has, when French can group thousands after
    /// it, "40" before "40 000": one to three digits, not starting with zero, and a word of their own
    /// that does not follow another number.
    fn thousands_lead(&self) -> Option<usize> {
        let b = &self.buffer;
        let n = b.iter().rev().take_while(|c| c.is_ascii_digit()).count();
        let before = &b[..b.len() - n];
        let word_before = before.iter().rev().find(|c| !c.is_whitespace());
        ((1..=3).contains(&n)
            && b[b.len() - n] != '0'
            && !before.last().is_some_and(|&c| c.is_alphanumeric() || ",.+".contains(c))
            && !word_before.is_some_and(char::is_ascii_digit))
        .then_some(n)
    }

    /// Whether the text ends with a word of plain digits, as "9" before the dash of "9–17" and
    /// not "1/7" before the dash between two dates.
    fn ends_with_plain_number(&self) -> bool {
        let word = self.buffer.iter().rev().take_while(|c| !c.is_whitespace());
        let mut any = false;
        for c in word {
            if !c.is_ascii_digit() {
                return false;
            }
            any = true;
        }
        any
    }

    /// Turn the minus of the number the text ends with into Korean "영하", below zero.
    fn minus_to_below_zero(&mut self) {
        let b = &self.buffer;
        let n = b.iter().rev().take_while(|c| c.is_ascii_digit() || matches!(c, '.' | ',')).count();
        let Some(minus) = b.len().checked_sub(n + 1) else { return };
        if n == 0 || b[minus] != '-' || (minus > 0 && !b[minus - 1].is_whitespace()) {
            return;
        }
        let number: Vec<char> = self.buffer.drain(minus..).skip(1).collect();
        self.push_str("영하 ");
        self.buffer.extend(number);
    }

    /// `word` as a word of its own, for a symbol read aloud.
    fn push_spoken(&mut self, word: &str) {
        self.push_whitespace();
        self.push_str(word);
        self.push_whitespace();
    }

    fn into_string(mut self) -> String {
        self.pop_whitespace();
        self.buffer.into_iter().collect()
    }

    fn last_is_whitespace(&self) -> bool {
        self.buffer.last().is_some_and(|c| c.is_whitespace())
    }

    fn push_whitespace(&mut self) {
        if !self.last_is_whitespace() && !self.buffer.is_empty() {
            self.push(' ');
        }
    }

    fn pop_whitespace(&mut self) {
        while self.last_is_whitespace() {
            self.buffer.pop();
        }
    }
}

/// Letters with a diacritic the tokenizer has no piece for, which it would otherwise hand the
/// model as raw bytes, paired with the letter they fold to.
const FOLDS: &str = "ÌIÒOÝYìiòoýyÿyĀAāaĂAăaĄAąaĆCćcĈCĉcĊCċcČCčcĎDďdĒEēeĔEĕeĖEėeĘEęeĚEěeĜGĝgĞGğgĠGġgĢGģgĤHĥhĨIĩiĪIīiĬIĭiĮIįiİIĴJĵjĶKķkĹLĺlĻLļlĽLľlŃNńnŅNņnŇNňnŌOōoŎOŏoŐOőoŔRŕrŖRŗrŘRřrŚSśsŜSŝsŞSşsŠSšsŢTţtŤTťtŨUũuŪUūuŬUŭuŮUůuŰUűuŲUųuŴWŵwŶYŷyŸYŹZźzŻZżzŽZžzƠOơoƯUưuǍAǎaǏIǐiǑOǒoǓUǔuǕUǖuǗUǘuǙUǚuǛUǜuǞAǟaǠAǡaǦGǧgǨKǩkǪOǫoǬOǭoǰjǴGǵgǸNǹnǺAǻaȀAȁaȂAȃaȄEȅeȆEȇeȈIȉiȊIȋiȌOȍoȎOȏoȐRȑrȒRȓrȔUȕuȖUȗuȘSșsȚTțtȞHȟhȦAȧaȨEȩeȪOȫoȬOȭoȮOȯoȰOȱoȲYȳyḀAḁaḂBḃbḄBḅbḆBḇbḈCḉcḊDḋdḌDḍdḎDḏdḐDḑdḒDḓdḔEḕeḖEḗeḘEḙeḚEḛeḜEḝeḞFḟfḠGḡgḢHḣhḤHḥhḦHḧhḨHḩhḪHḫhḬIḭiḮIḯiḰKḱkḲKḳkḴKḵkḶLḷlḸLḹlḺLḻlḼLḽlḾMḿmṀMṁmṂMṃmṄNṅnṆNṇnṈNṉnṊNṋnṌOṍoṎOṏoṐOṑoṒOṓoṔPṕpṖPṗpṘRṙrṚRṛrṜRṝrṞRṟrṠSṡsṢSṣsṤSṥsṦSṧsṨSṩsṪTṫtṬTṭtṮTṯtṰTṱtṲUṳuṴUṵuṶUṷuṸUṹuṺUṻuṼVṽvṾVṿvẀWẁwẂWẃwẄWẅwẆWẇwẈWẉwẊXẋxẌXẍxẎYẏyẐZẑzẒZẓzẔZẕzẖhẗtẘwẙyẠAạaẢAảaẤAấaẦAầaẨAẩaẪAẫaẬAậaẮAắaẰAằaẲAẳaẴAẵaẶAặaẸEẹeẺEẻeẼEẽeẾEếeỀEềeỂEểeỄEễeỆEệeỈIỉiỊIịiỌOọoỎOỏoỐOốoỒOồoỔOổoỖOỗoỘOộoỚOớoỜOờoỞOởoỠOỡoỢOợoỤUụuỦUủuỨUứuỪUừuỬUửuỮUữuỰUựuỲYỳyỴYỵyỶYỷyỸYỹy";

/// The letter `c` folds to in [`FOLDS`], if it has a diacritic listed there.
pub(crate) fn unaccented(c: char) -> Option<char> {
    if (c as u32) < 0xCC {
        return None;
    }
    let mut it = FOLDS.chars();
    while let (Some(a), Some(b)) = (it.next(), it.next()) {
        if a == c {
            return Some(b);
        }
    }
    None
}

/// The ASCII digit `c` stands for, when it is an Arabic-Indic, Persian, Devanagari or
/// full-width digit.
fn native_digit(c: char) -> Option<char> {
    let zero = match c {
        '٠'..='٩' => '٠',
        '۰'..='۹' => '۰',
        '०'..='९' => '०',
        '０'..='９' => '０',
        _ => return None,
    };
    char::from_u32('0' as u32 + (c as u32 - zero as u32))
}

/// The ASCII form of a full-width Latin letter or symbol, and the narrow yen and won signs.
/// Full-width punctuation stays: the Japanese and Korean tokenizers have pieces for it.
fn fullwidth(c: char) -> Option<char> {
    match c {
        'Ａ'..='Ｚ'
        | 'ａ'..='ｚ'
        | '＄'
        | '％'
        | '＋'
        | '＝'
        | '＠'
        | '／'
        | '＃'
        | '＆'
        | '＊'
        | '－' => char::from_u32(c as u32 - 0xFEE0),
        '￥' => Some('¥'),
        '￦' => Some('₩'),
        _ => None,
    }
}

/// Whether one space and then an amount follow, as after the symbol of "€ 5,50": digits that
/// do not run into a dash, which would make them a range.
fn spaced_amount_follows(chars: &std::iter::Peekable<std::str::Chars<'_>>) -> bool {
    let mut ahead = chars.clone();
    ahead.next().is_some_and(char::is_whitespace)
        && ahead.clone().next().is_some_and(|c| c.is_ascii_digit())
        && ahead.find(|c| !c.is_ascii_digit()) != Some('-')
}

fn subscript(c: char) -> Option<char> {
    Some(match c {
        '₀'..='₉' => char::from(b'0' + (c as u32 - '₀' as u32) as u8),
        '₊' => '+',
        '₋' => '-',
        'ₐ' => 'a',
        'ₑ' => 'e',
        'ₒ' => 'o',
        'ₓ' => 'x',
        'ₕ' => 'h',
        'ₖ' => 'k',
        'ₗ' => 'l',
        'ₘ' => 'm',
        'ₙ' => 'n',
        'ₚ' => 'p',
        'ₛ' => 's',
        'ₜ' => 't',
        'ᵢ' => 'i',
        'ⱼ' => 'j',
        _ => return None,
    })
}

fn superscript(c: char) -> Option<char> {
    Some(match c {
        '⁰' => '0',
        '¹' => '1',
        '²' => '2',
        '³' => '3',
        '⁴'..='⁹' => char::from(b'4' + (c as u32 - '⁴' as u32) as u8),
        '⁻' => '-',
        '⁺' => '+',
        'ⁿ' => 'n',
        _ => return None,
    })
}

fn is_emoji(c: char) -> bool {
    let c = c as u32;
    matches!(c,
        // Every pictograph block from mahjong tiles to the symbols and pictographs extension:
        // "🎉", "🏆" and "🌍" sit below the emoticons, and flags (regional indicators) in between.
        0x1F000..=0x1FAFF |
        0x2600..=0x27BF |
        0x2B00..=0x2BFF |
        // The variation selector and joiner that emoji sequences leave behind: "☀️", "👩‍💻".
        0xFE0F | 0x200D
    )
}

/// Unicode characters that read as a double quotation mark: guillemets,
/// including the single-angle `‹ ›` pair, which quotes speech in the same way
/// the double one does, the curly and reversed variants, the low-9 ones
/// sitting on the baseline (German/Czech opening quotes), double primes,
/// dingbat and CJK corner quotes, and the fullwidth form.
fn is_double_quote(c: char) -> bool {
    matches!(
        c,
        '»' | '«'
            | '‹'
            | '›'
            | '“'
            | '”'
            | '„'
            | '‟'
            | '″'
            | '‶'
            | '⹂'
            | '❝'
            | '❞'
            | '❠'
            | '〝'
            | '〞'
            | '〟'
            | '＂'
    )
}

/// Unicode characters that read as a single quotation mark or apostrophe:
/// the curly and reversed variants, the low-9 one sitting on the baseline,
/// primes, dingbat quotes, the fullwidth form and the modifier letter and
/// accent characters commonly typed in place of an apostrophe.
fn is_single_quote(c: char) -> bool {
    matches!(
        c,
        '‘' | '’'
            | '‚'
            | '‛'
            | '′'
            | '‵'
            | '❛'
            | '❜'
            | '❟'
            | '＇'
            | 'ʼ'
            | 'ʻ'
            | 'ʹ'
            | '´'
            | '`'
    )
}

/// Rewrite `input` into the character set the model was trained on.
///
/// Typographic quotes, dashes, bullets, arrows and emoji are dropped or folded to their ASCII
/// equivalents; `@`, `+` and `=` are spelled out in `lang`; `;`, parentheses and a `:` with
/// whitespace on either side become commas, which is how the model is asked to pause. A `:`
/// between two non-space characters, as in `10:30`, is kept.
///
/// Then each word goes to the `rules`, and `@` and `+` are spelled out only in the words no rule
/// claimed, since the email and phone rules read them.
pub fn normalize_text(input: &str, lang: Lang, rules: Rules) -> String {
    let mut res = StringAppender::new();
    let mut chars = input.chars().peekable();
    let mut prev = None;
    // Where the digits of a French number joined across a thousands space end, so that a further
    // group joins too: "1 000 000".
    let mut grouped = 0;
    while let Some(c) = chars.next() {
        match c {
            c if is_double_quote(c) => res.push('"'),
            c if is_single_quote(c) => res.push('\''),
            '‐' | '‑' | '‒' | '―' => res.push('-'),
            '★' | '☆' => {
                let mut full = usize::from(c == '★');
                let mut empty = usize::from(c == '☆');
                while let Some(&next) = chars.peek().filter(|&&c| c == '★' || c == '☆') {
                    full += usize::from(next == '★');
                    empty += usize::from(next == '☆');
                    chars.next();
                }
                let half = chars.next_if_eq(&'½').is_some();
                let n = match (half, lang) {
                    (false, _) => full.to_string(),
                    (true, Lang::En | Lang::Ar | Lang::Hi | Lang::Ko | Lang::Ja) => {
                        format!("{full}.5")
                    }
                    (true, _) => format!("{full},5"),
                };
                let of = (empty > 0).then_some(full + empty + usize::from(half));
                res.push_spoken(&lang.stars(&n, of));
            }
            '°' | 'º' if res.ends_with_n() => {
                let word = lang.number_sign();
                res.buffer.pop_if(|c| *c == '.');
                if res.buffer.pop() == Some('N') {
                    let mut chars = word.chars();
                    let first = chars.next().map(char::to_uppercase).into_iter().flatten();
                    res.push_spoken(&first.chain(chars).collect::<String>());
                } else {
                    res.push_spoken(word);
                }
            }
            '°' | '℃' | '℉' => {
                // Korean reads a temperature below zero as "영하 12 도", not "minus 12".
                if lang == Lang::Ko {
                    res.minus_to_below_zero();
                }
                res.push_spoken(lang.degrees(res.ends_with_one()));
                // The scale letter, or Arabic's own markers; the languages that leave the
                // scale unsaid still take the letter.
                let scale = match (c, chars.peek()) {
                    ('℃', _) | ('°', Some('C' | 'م')) => Some(lang.celsius()),
                    ('℉', _) | ('°', Some('F' | 'ف')) => Some(lang.fahrenheit()),
                    _ => None,
                };
                let mut after = chars.clone();
                if c == '°' {
                    after.next();
                }
                if let Some(scale) =
                    scale.filter(|_| !after.peek().is_some_and(|c| c.is_ascii_alphanumeric()))
                {
                    if c == '°' {
                        chars.next();
                    }
                    if let Some(scale) = scale {
                        res.push_str(scale);
                    }
                }
            }
            // Digits of the other scripts, and full-width ones, as ASCII: the tokenizers have
            // pieces for those alone.
            c if native_digit(c).is_some() => res.push(native_digit(c).unwrap_or(c)),
            '٫' => res.push('.'),
            '٬' => res.push(','),
            '٪' => res.push('%'),
            // Bidirectional and byte-order marks say nothing aloud.
            '\u{200E}'
            | '\u{200F}'
            | '\u{061C}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2066}'..='\u{2069}'
            | '\u{FEFF}' => {}
            // The minus sign, in the languages the serving stack does not read.
            '−' if !matches!(lang, Lang::En | Lang::Fr | Lang::De | Lang::Es | Lang::Pt) => {
                res.push('-')
            }
            c if fullwidth(c).is_some() => res.push(fullwidth(c).unwrap_or(c)),
            '|' if lang == Lang::Hi => res.push('।'),
            c if lang.fold(c).is_some() => res.push(lang.fold(c).unwrap_or(c)),
            c if subscript(c).is_some() => res.push(subscript(c).unwrap_or(c)),
            c if superscript(c).is_some() => {
                let mut n: String = superscript(c).into_iter().collect();
                while let Some(d) = chars.peek().and_then(|&c| superscript(c)) {
                    n.push(d);
                    chars.next();
                }
                if prev.is_some_and(|p: char| p.is_alphanumeric() || p == ')') {
                    res.push_spoken(&lang.power(&n));
                } else {
                    res.push_str(&n);
                }
            }
            c if lang.greek(c).is_some() => res.push_spoken(&lang.greek(c).unwrap_or_default()),
            c if lang.symbol(c).is_some() => res.push_spoken(lang.symbol(c).unwrap_or_default()),
            '±' => res.push_spoken(lang.plus_minus()),
            '&' if lang.ampersand().is_some()
                && !(prev.is_some_and(|p| p.is_ascii_uppercase())
                    && chars.peek().is_some_and(char::is_ascii_uppercase)) =>
            {
                res.push_spoken(lang.ampersand().unwrap_or_default())
            }
            '#' if lang.spells_number_sign() && chars.peek().is_some_and(char::is_ascii_digit) => {
                res.push_spoken(lang.number_sign())
            }
            '❤' | '♥' => {
                chars.next_if_eq(&'\u{FE0F}');
                res.push_spoken(lang.love());
            }
            // Between two numbers in Japanese, a range: the rules read the wave dash.
            '~' if lang == Lang::Ja
                && rules.has("numbers")
                && prev.is_some_and(|p| p.is_ascii_digit())
                && chars.peek().is_some_and(char::is_ascii_digit) =>
            {
                res.push('〜')
            }
            '~' if chars.peek().is_some_and(char::is_ascii_digit) => res.push_spoken(lang.about()),
            '*' if prev.is_none_or(char::is_whitespace)
                && chars.peek().is_some_and(char::is_ascii_digit) =>
            {
                res.push_spoken(lang.star())
            }
            // An en dash between two numbers is a range, which the numbers rule reads.
            '–' if res.ends_with_plain_number()
                && chars.peek().is_some_and(char::is_ascii_digit) =>
            {
                res.push('-')
            }
            // The two dashes below are not - (ascii 45) but similar unicode chars.
            '–' | '*' | '—' | '[' | ']' | '{' | '}' => res.push_whitespace(),
            '•' | '‣' | '◦' | '·' | '→' | '←' | '↑' | '↓' | '➡' | '➜' => {
                res.push_whitespace();
            }
            // "250 €", "42,00 €": the symbol after the number and a space, as French, German,
            // Spanish and Portuguese write it, joins the number so the currency rule reads it.
            // Not before a digit, where it leads the next amount instead.
            c if rewrite::is_currency_symbol(c)
                && res.ends_with_spaced_digit()
                && !chars.peek().is_some_and(char::is_ascii_digit) =>
            {
                res.pop_whitespace();
                res.push(c);
            }
            // "€ 5,50", "₹ 500": the symbol before the number and a space, as Dutch and Hindi
            // write it, joins the number the same way. Not after a letter ("R$ 40", which has
            // its own rule below), and not before a range, which the numbers rule reads.
            c if rewrite::is_currency_symbol(c)
                && !prev.is_some_and(|p| p.is_ascii_alphabetic())
                && spaced_amount_follows(&chars) =>
            {
                res.push(c);
                chars.next();
            }
            // "40 000", "1 234,56": French, Finnish, Swedish, Norwegian and Polish group thousands
            // with a space, plain or non-breaking, which goes so that the numbers rule reads one
            // number rather than two. Three groups of three digits are a code, as the digit runs
            // of the numbers rule have it.
            c if c.is_whitespace()
                && matches!(lang, Lang::Fr | Lang::Fi | Lang::Sv | Lang::No | Lang::Pl)
                && (res.buffer.len() == grouped
                    || res
                        .thousands_lead()
                        .is_some_and(|n| n < 3 || thousands_groups(chars.clone()) == 1))
                && thousands_groups(chars.clone()) > 0 =>
            {
                grouped = res.buffer.len() + 3;
            }
            c if c.is_whitespace()
                && lang == Lang::Pt
                && res.buffer.ends_with(&['R', '$'])
                && chars.peek().is_some_and(char::is_ascii_digit)
                && chars.clone().find(|c| !c.is_ascii_digit()) != Some('-') => {}
            '…' => res.push('.'),
            '=' if lang == Lang::Ko => {
                let particle =
                    korean_topic_particle(res.buffer.iter().rev().find(|c| !c.is_whitespace()));
                res.pop_whitespace();
                res.push_str(particle);
                res.push_whitespace();
            }
            '=' => res.push_spoken(lang.special_chars().equals),
            ':' if lang == Lang::Fi
                && !prev.is_none_or(|p| p.is_alphabetic() || p.is_whitespace())
                && chars.peek().is_some_and(|c| c.is_lowercase()) =>
            {
                while chars.next_if(|c| c.is_lowercase()).is_some() {}
            }
            ':' if !prev.is_none_or(char::is_whitespace)
                && !chars.peek().is_none_or(|c| c.is_whitespace()) =>
            {
                res.push(':')
            }
            ';' | ':' | '(' | ')' => {
                res.pop_whitespace();
                res.push(',');
                res.push_whitespace();
            }
            c => {
                if is_emoji(c) || c.is_control() || c.is_whitespace() {
                    res.push_whitespace();
                } else {
                    res.push(c)
                }
            }
        }
        prev = Some(c);
    }
    let text = res.into_string();
    let text = rewrite::rewrite_text(&text, lang, rules, |w| spell_symbols(w, lang, rules));
    let text = strip_closing_quote(&text);
    if lang == Lang::Ja { space_chars(text) } else { text.to_string() }
}

/// The topic particle Korean reads `=` as, 은 after a syllable ending on a consonant and 는 after
/// a vowel, as the digit or Latin letter before it is read ("1은 2", "a는 b").
fn korean_topic_particle(before: Option<&char>) -> &'static str {
    let consonant = match before {
        Some(&c @ '가'..='힣') => !(c as u32 - '가' as u32).is_multiple_of(28),
        Some(c) if c.is_ascii_digit() => "013678".contains(*c),
        Some(c) if c.is_ascii_alphabetic() => "lmnrLMNR".contains(*c),
        _ => false,
    };
    if consonant { "은" } else { "는" }
}

/// Japanese as the checkpoints were trained on it: one space between characters, with runs of
/// ASCII letters and digits kept whole, and a `。` after a text that ends on a character, spaced
/// like the rest.
fn space_chars(text: &str) -> String {
    let mut out = String::new();
    let mut run = false;
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            run = false;
            continue;
        }
        // A point or comma between two digits stays in the run: "2.5", "1,500".
        let joins = run
            && (c.is_ascii_alphanumeric()
                || (matches!(c, '.' | ',')
                    && chars.get(i.wrapping_sub(1)).is_some_and(char::is_ascii_digit)
                    && chars.get(i + 1).is_some_and(char::is_ascii_digit)));
        if !joins && !out.is_empty() {
            out.push(' ');
        }
        out.push(c);
        run = joins || c.is_ascii_alphanumeric();
    }
    if out.chars().last().is_some_and(char::is_alphanumeric) {
        out.push_str(" 。");
    }
    out
}

/// How many groups of exactly three digits, each but the first after a space, `chars` starts
/// with: the rest of a French number. None when more digits follow them, as in a code ("71 002
/// 4489").
fn thousands_groups(chars: impl Iterator<Item = char>) -> usize {
    let ahead: Vec<char> = chars.take(64).collect();
    let (mut n, mut i) = (0, 0);
    loop {
        match ahead[i..].iter().take_while(|c| c.is_ascii_digit()).count() {
            3 => n += 1,
            0 => return n,
            _ => return 0,
        }
        i += 3;
        if !ahead.get(i).is_some_and(|c| c.is_whitespace()) {
            return n;
        }
        i += 1;
    }
}

fn strip_closing_quote(text: &str) -> &str {
    let trimmed = text.trim_end_matches('"');
    match trimmed.strip_suffix('\'') {
        Some(rest) if rest.ends_with(['.', '!', '?', ',', ';']) => rest,
        _ => trimmed,
    }
}

/// Spell out the `@` and `+` of a word no rule claimed, as the character pass spells `=`, and
/// give the pieces they leave their own turn at the rules: "1500+20" reads "1 thousand 500 plus
/// 20", as it would had the character pass spelled the `+`. A slash between words ("and/or") is
/// read as a pause rather than as "slash".
fn spell_symbols(word: &str, lang: Lang, rules: Rules) -> String {
    if let Some(text) = split_slash(word) {
        let words =
            text.split(' ').map(|w| rewrite_word(w, lang, rules).unwrap_or_else(|| w.into()));
        return words.collect::<Vec<_>>().join(" ");
    }
    if !word.contains(['@', '+']) {
        return word.to_string();
    }
    let mut res = StringAppender::new();
    for c in word.chars() {
        match c {
            '@' => res.push_spoken(lang.special_chars().at),
            '+' => res.push_spoken(lang.special_chars().plus),
            c => res.push(c),
        }
    }
    let text = res.into_string();
    let words = text.split(' ').map(|w| rewrite_word(w, lang, rules).unwrap_or_else(|| w.into()));
    words.collect::<Vec<_>>().join(" ")
}

/// Units that are written with a slash between letters ("mg/kg", "kJ/mol"), which keep it.
const SLASH_UNITS: &[&str] = &[
    "km", "kg", "mg", "ml", "mL", "dL", "mol", "kJ", "kW", "kWh", "Hz", "min", "hr", "hrs", "sec",
    "ft", "lb", "lbs", "oz", "mi", "mph", "kph", "yd", "cm", "mm", "yr", "wk", "mo", "kcal", "rpm",
];

/// `word` with each slash replaced by a space, when every part it separates is a word of two
/// letters or more and not a unit; `None` otherwise, which leaves "c/o", "km/h", "mg/kg", dates
/// and fractions alone.
fn split_slash(word: &str) -> Option<String> {
    let body = word.trim_end_matches(|c: char| c.is_ascii_punctuation() && c != '/');
    let parts: Vec<&str> = body.split('/').collect();
    let wordlike = |p: &&str| {
        p.chars().count() >= 2 && p.chars().all(char::is_alphabetic) && !SLASH_UNITS.contains(p)
    };
    (parts.len() > 1 && parts.iter().all(wordlike))
        .then(|| format!("{}{}", parts.join(" "), &word[body.len()..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_language_the_model_lacks_is_warned_about() {
        let langs = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let fr = Normalize::for_lang(Lang::Fr);
        assert_eq!(
            fr.language_warning(&langs(&["en"])).as_deref(),
            Some("text is normalized as fr, but the model was trained for en")
        );
        assert_eq!(fr.language_warning(&langs(&["en", "fr"])), None);
        assert_eq!(fr.language_warning(&[]), None);
        assert_eq!(Normalize::OFF.language_warning(&langs(&["en"])), None);
    }

    #[test]
    fn normalize_text_cases() {
        let cases: &[(&str, &str)] = &[
            ("Hello, world!", "Hello, world!"),
            ("", ""),
            ("“hello” world it's", "\"hello\" world it's"),
            ("«hello» ‹world›", "\"hello\" \"world"),
            ("a‐b‑c‒d―e", "a-b-c-d-e"),
            ("a–b—c", "a b c"),
            ("foo (bar) [baz] {qux} *quux*", "foo, bar, baz qux quux"),
            ("• ‣ ◦ · a→b←c↑d↓e ➡ ➜", "a b c d e"),
            ("wait… a…b", "wait. a.b"),
            ("user@host @home", "user at host at home"),
            ("café résumé 日本語", "café résumé 日本語"),
            ("hello 😀 flag 🇫🇷 sun ☀", "hello flag sun"),
            ("party 🎉 cup 🏆 globe 🌍 pizza 🍕 star ⭐", "party cup globe pizza star"),
            ("sun ☀️ dev 👩‍💻 done", "sun dev done"),
            ("Fish &, chips &.", "Fish &, chips &."),
            // ';', '(' / ')' and a ':' next to whitespace collapse to ", " (comma + single
            // space); a ':' between two non-space characters is kept.
            ("a;b:c", "a, b:c"),
            ("time: 10:30", "time, 10:30"),
            ("a :b", "a, b"),
            ("note:", "note,"),
            (":start", ", start"),
            ("; leading", ", leading"),
            ("hello (world)", "hello, world,"),
            // Surrounding whitespace is absorbed into the comma replacement.
            ("foo ; bar  :  baz", "foo, bar, baz"),
            // Runs of ASCII and non-ASCII whitespace collapse to a single space,
            // and trailing whitespace is stripped.
            ("a   b\t\tc\n\nd", "a b c d"),
            ("hello   ", "hello"),
            ("a • b • c", "a b c"),
            (
                "“Hello”; please email user@host (now)… 🚀",
                "\"Hello\", please email user at host, now.",
            ),
            (
                "Numbers: one, two, three, four, five. Special items: at sign, hash, dollar, percent.",
                "Numbers, one, two, three, four, five. Special items, at sign, hash, dollar, percent.",
            ),
            (
                "The conference will be held on Tuesday, March 15th at 3:30 PM.",
                "The conference will be held on Tuesday, March 15th at 3:30 PM.",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(
                &normalize_text(input, Lang::En, Rules::DEFAULT),
                expected,
                "input: {input:?}"
            );
        }
    }

    #[test]
    fn spoken_symbols_follow_the_language() {
        assert_eq!(normalize_text("a@b", Lang::En, Rules::DEFAULT), "a at b");
        assert_eq!(normalize_text("a@b", Lang::Fr, Rules::DEFAULT), "a arobaze b");
        assert_eq!(normalize_text("a@b", Lang::De, Rules::DEFAULT), "a ät b");
        assert_eq!(normalize_text("1+1=2", Lang::Es, Rules::DEFAULT), "1 mas 1 igual 2");
        assert_eq!(normalize_text("1+1=2", Lang::Pt, Rules::DEFAULT), "1 mais 1 igual 2");
        assert_eq!(normalize_text("a@b", Lang::It, Rules::DEFAULT), "a chiocciola b");
        assert_eq!(normalize_text("1+1=2", Lang::Ko, Rules::DEFAULT), "1 플러스 1은 2");
        assert_eq!(normalize_text("2 + 2 = 4", Lang::Ko, Rules::DEFAULT), "2 플러스 2는 4");
        assert_eq!(normalize_text("x=y, n=값", Lang::Ko, Rules::DEFAULT), "x는 y, n은 값");
        assert_eq!(normalize_text("값=3, 나=4", Lang::Ko, Rules::DEFAULT), "값은 3, 나는 4");
        assert_eq!(normalize_text("Δ, ρ, π", Lang::Fi, Rules::DEFAULT), "delta, rhoo, pii");
        assert_eq!(normalize_text("π, Ω", Lang::It, Rules::DEFAULT), "pi greco, omega");
        assert_eq!(normalize_text("π≈3", Lang::De, Rules::DEFAULT), "Pi ungefähr 3");
        for &lang in Lang::ALL {
            for c in 'α'..='ω' {
                assert!(lang.greek(c).is_some_and(|s| !s.is_empty()), "{lang:?} {c}");
            }
        }
        assert_eq!(
            normalize_text("1+1=2", Lang::Ja, Rules::DEFAULT),
            "1 プ ラ ス 1 イ コ ー ル 2 。"
        );
        assert_eq!(
            normalize_text("★★★★☆ α", Lang::Ja, Rules::DEFAULT),
            "5 つ 星 の う ち 4 つ ア ル フ ァ 。"
        );
        for &lang in Lang::ALL {
            for c in "×÷≤≥≠≈√∞∑‰".chars() {
                assert!(lang.symbol(c).is_some_and(|s| !s.is_empty()), "{lang:?} {c}");
            }
        }
    }

    /// The rules read every language, and without them the character pass runs alone.
    #[test]
    fn rules_read_every_language_and_none_leaves_the_character_pass() {
        assert_eq!(rewrite_word("1234", Lang::Nl, Rules::ALL).as_deref(), Some("duizend 234"));
        let it = |text| normalize_text(text, Lang::It, Rules::NONE);
        assert_eq!(
            it("Costa $1,234.56, ~5 ★★★☆☆ e 3²."),
            "Costa $1,234.56, circa 5 3 stelle su 5 e 3 al quadrato."
        );
        assert_eq!(it("jane@x.it, N°5 a 20°C"), "jane chiocciola x.it, Numero 5 a 20 gradi");
        assert_eq!(it("Così è più bello, ťuk"), "Così è più bello, tuk");
        assert_eq!(normalize_text("Źle, więc", Lang::Pl, Rules::DEFAULT), "Źle, więc");
        assert_eq!(normalize_text("Źle, więc", Lang::En, Rules::DEFAULT), "Zle, wiec");
        assert_eq!(
            normalize_text("Işık, ağaç, İzmir", Lang::En, Rules::DEFAULT),
            "Isık, agaç, Izmir"
        );
        assert_eq!(normalize_text("n°5", Lang::Ar, Rules::DEFAULT), "رقم 5");
        assert_eq!(normalize_text("N°5", Lang::Ar, Rules::DEFAULT), "رقم 5");
    }

    /// The digits of the other scripts, full-width forms, the bidirectional marks and the
    /// separators that come with them, as the tokenizers have pieces for ASCII alone.
    #[test]
    fn the_other_scripts_digits_and_symbols_fold_to_ascii() {
        let none = |text, lang| normalize_text(text, lang, Rules::NONE);
        assert_eq!(none("الرقم ٣٤ و ١٢٫٥٪ و ۱۲", Lang::Ar), "الرقم 34 و 12.5% و 12");
        assert_eq!(none("\u{200F}مرحبا\u{200E} بك\u{061C}", Lang::Ar), "مرحبا بك");
        assert_eq!(none("१२ लड़के आए।", Lang::Hi), "12 लड़के आए।");
        assert_eq!(none("वह गया| फिर सो गया|", Lang::Hi), "वह गया। फिर सो गया।");
        assert_eq!(none("１２人が来ました。", Lang::Ja), "12 人 が 来 ま し た 。");
        assert_eq!(none("ＡＢＣ１２３と５０％", Lang::Ja), "ABC123 と 50 %");
        assert_eq!(none("￥１５００", Lang::Ja), "¥ 1500 。");
        assert_eq!(none("−5 °C", Lang::Fi), "-5 astetta");
        assert_eq!(none("−12 °C:een", Lang::Fi), "-12 astetta");
        assert_eq!(none("15 %:n alennus, 22:n", Lang::Fi), "15 % alennus, 22");
        assert_eq!(none("-3,5 °C", Lang::Pl), "-3,5 stopni Celsjusza");
        assert_eq!(none("-12°C까지", Lang::Ko), "영하 12 도 까지");
        assert_eq!(none("기온은 -3.5°C입니다", Lang::Ko), "기온은 영하 3.5 도 입니다");
        assert_eq!(none("2-3°C", Lang::Ko), "2-3 도");
        assert_eq!(none("36.8℃でした", Lang::Ja), "36.8 度 で し た 。");
        assert_eq!(none("-12°م", Lang::Ar), "-12 درجات مئوية");
        assert_eq!(none("21°C", Lang::Hi), "21 डिग्री सेल्सियस");
        assert_eq!(none("21°C", Lang::En), "21 degrees Celsius");
        assert_eq!(none("R&D, #1", Lang::Sv), "R&D, nummer 1");
        assert_eq!(none("Tom & Jerry", Lang::Sv), "Tom och Jerry");
        assert_eq!(none("Tom & Jerry #1", Lang::En), "Tom & Jerry #1");
        // A decimal stays one run in Japanese.
        assert_eq!(none("重さは2.5キロです。", Lang::Ja), "重 さ は 2.5 キ ロ で す 。");
        assert_eq!(none("1,500円", Lang::Ja), "1,500 円 。");
        assert_eq!(none("町でNo.1の店", Lang::Ja), "町 で No . 1 の 店 。");
        assert_eq!(none("1/7–30/9 ja 9–17", Lang::Sv), "1/7 30/9 ja 9-17");
    }

    /// The new languages' rules at work on whole texts: spaced thousands joined, a currency
    /// symbol before a spaced amount joined, en-dash ranges, and the spans around a number.
    #[test]
    fn the_new_languages_read_sentences() {
        let cases = [
            (
                Lang::Fi,
                "Budjetti on 1 234 567 euroa.",
                "Budjetti on miljoona 234 tuhatta 567 euroa.",
            ),
            (Lang::Fi, "Hinta on 5,99 €.", "Hinta on 5 euroa 99 senttiä."),
            (Lang::Fi, "Kokous on klo 10.30.", "Kokous on klo 10.30."),
            (Lang::Sv, "Det bor 12 345 personer där.", "Det bor 12 tusen 345 personer där."),
            (Lang::Sv, "Priset är 5,99 kr.", "Priset är 5 kronor och 99 öre."),
            (Lang::Sv, "Öppet 20–25 maj.", "Öppet tjugonde till tjugofemte maj."),
            (
                Lang::No,
                "Budsjettet er 1 234 567 kroner.",
                "Budsjettet er en million 234 tusen 567 kroner.",
            ),
            (Lang::Pl, "Mieszka tam 12 345 osób.", "Mieszka tam 12 tysięcy 345 osób."),
            (Lang::Pl, "To kosztuje 5,99 zł.", "To kosztuje 5 złotych 99 groszy."),
            (
                Lang::Pl,
                "Zadzwoń pod +48 123 456 789.",
                "Zadzwoń pod plus cztery osiem, jeden dwa trzy, cztery pięć sześć, siedem osiem dziewięć.",
            ),
            (
                Lang::It,
                "Il bilancio è di 1.234.567 euro.",
                "Il bilancio è di 1 milione 234 mila 567 euro.",
            ),
            (Lang::It, "Costa €5.", "Costa 5 euro."),
            (Lang::Nl, "Het kost € 5,50.", "Het kost 5 euro 50."),
            (Lang::Nl, "Er wonen 70449 mensen.", "Er wonen 70 duizend 449 mensen."),
            (Lang::Hi, "इसकी कीमत ₹ 500 है।", "इसकी कीमत पाँच सौ रुपये है।"),
            (Lang::Hi, "बजट 1.5 लाख रुपये है।", "बजट डेढ़ लाख रुपये है।"),
            (Lang::Hi, "बैठक सुबह 11.30 बजे है।", "बैठक सुबह साढ़े ग्यारह बजे है।"),
            (Lang::Hi, "कुल 12,34,567 लोग आए।", "कुल बारह लाख चौंतीस हजार पाँच सौ सड़सठ लोग आए।"),
            (Lang::Hi, "ऑर्डर नंबर 48213907 है।", "ऑर्डर नंबर चार आठ दो एक तीन नौ शून्य सात है।"),
            (Lang::No, "Ordre 48213907 sendes.", "Ordre fire åtte to en tre ni null sju sendes."),
            (Lang::No, "Sak nr. 4471 er lukket.", "Sak nr. 4471 er lukket."),
            (Lang::Nl, "Het ref.nr. is 4471.", "Het ref.nr. is 4 duizend 471."),
            (Lang::Pl, "Są m.in. psy.", "Są m.in. psy."),
            (Lang::It, "Visita rai.it.", "Visita rai punto i ti."),
            (Lang::Ko, "번호는 48213907입니다.", "번호는 사 팔 이 일 삼 구 공 칠 입니다."),
            (Lang::Ja, "バグ#2093", "バ グ # 2093 。"),
            (Lang::Hi, "5G तकनीक और 3D प्रिंटिंग।", "5G तकनीक और 3D प्रिंटिंग।"),
            (Lang::Ko, "나나크(1469–1539)가 1,000파운드", "나나크, 1469-1539, 가 1,000파운드"),
            (Lang::It, "I B&B più alti", "I B&B più alti"),
            (Lang::Sv, "Ärende nummer 773102.", "Ärende nummer 773102."),
            (Lang::Hi, "मेरे पास 123 किताबें हैं।", "मेरे पास एक सौ तेईस किताबें हैं।"),
            (Lang::Hi, "कुल: €42।", "कुल, बयालीस यूरो।"),
            (Lang::Ar, "السعر $50؟", "السعر 50 دولار؟"),
            (Lang::Hi, "वह 1985 में पैदा हुआ।", "वह उन्नीस सौ पचासी में पैदा हुआ।"),
            (Lang::Ar, "السعر 50 ر.س فقط.", "السعر 50 ريال فقط."),
            (Lang::Ar, "تأسست عام 2024م.", "تأسست عام 2024."),
            (Lang::Ar, "عدد السكان 1,234,567 نسمة.", "عدد السكان مليون 234 ألف 567 نسمة."),
            (Lang::Ja, "予算は1,234,567円です。", "予 算 は 123 万 4567 円 で す 。"),
            (Lang::Ja, "人口は12345人です。", "人 口 は 1 万 2345 人 で す 。"),
            (Lang::Ko, "예산은 1234567원입니다.", "예산은 123만 4567 원입니다."),
            (Lang::Ko, "기온은 -12°C입니다.", "기온은 영하 12 도 입니다."),
            (Lang::Ko, "인구는 70449명입니다.", "인구는 7만 449명입니다."),
            (Lang::Sv, "Säsongen 2024–25.", "Säsongen 2024 till 25."),
            (Lang::En, "The 2024–25 season.", "The 2024 to 25 season."),
            (Lang::En, "It costs $ 5 and ２０ ٣.", "It costs 5 dollars and 20 3."),
            (Lang::En, "It is 20℃ and 68℉.", "It is 20 degrees Celsius and 68 degrees Fahrenheit."),
            (Lang::Fr, "Ça coûte € 5,50.", "Ça coûte 5 euros et 50 centimes."),
            (Lang::De, "Es kostet € 5.", "Es kostet 5 Euro."),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(normalize_text(input, lang, Rules::DEFAULT), expected, "{lang:?} {input:?}");
        }
    }

    /// Japanese is spaced one character at a time, as the checkpoints were trained, with runs of
    /// ASCII letters and digits whole, and ends on `。`.
    #[test]
    fn japanese_is_spaced_by_character() {
        let ja = |text| normalize_text(text, Lang::Ja, Rules::DEFAULT);
        assert_eq!(ja("公文中学校2年のABC123です"), "公 文 中 学 校 2 年 の ABC123 で す 。");
        assert_eq!(ja("「はい」と言った。次は?"), "「 は い 」 と 言 っ た 。 次 は ?");
        assert_eq!(ja("公 文 中 学 校 2 年 。"), "公 文 中 学 校 2 年 。");
        assert_eq!(ja("Hello world"), "Hello world 。");
        assert_eq!(ja(""), "");
    }

    /// `@` and `+` reach the rules, which read them in emails and phone numbers, and are
    /// spelled out as before everywhere else, even with no rules at all.
    #[test]
    fn symbols_are_spelled_after_the_rules() {
        let cases = [
            ("Write to jane.smith@gmail.com!", "Write to jane dot smith at gmail dot com!"),
            ("Mail (jane+tag@gmail.com)", "Mail, jane plus tag at gmail dot com,"),
            ("user@host @home", "user at host at home"),
            ("a@.", "a at."),
            ("x @, y", "x at, y"),
            ("C++.", "C plus plus."),
            ("1500+20", "1 thousand 500 plus 20"),
            ("+33612345678", "plus 33 billion 612 million 345 thousand 678"),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize_text(input, Lang::En, Rules::DEFAULT), expected, "{input:?}");
        }
        assert_eq!(normalize_text("foo@bar.com", Lang::En, Rules::NONE), "foo at bar.com");
        assert_eq!(normalize_text("C++.", Lang::En, Rules::NONE), "C plus plus.");
        let phones = "phones".parse().unwrap();
        assert_eq!(normalize_text("+33612345678.", Lang::Fr, phones), "plus 33 6 12 34 56 78.");
    }

    #[test]
    fn sentences_are_rewritten() {
        let cases = [
            (Lang::En, "It costs $1500 today.", "It costs 1 thousand 500 dollars today."),
            (Lang::En, "It costs 5$.", "It costs 5 dollars."),
            (Lang::En, "It costs -$5.", "It costs minus 5 dollars."),
            // A `$` that is not part of an amount goes before a period, as other symbols do.
            (Lang::En, "In $.", "In."),
            (Lang::En, "Only £50 left.", "Only 50 pounds left."),
            (Lang::En, "The fee is 2.9% plus 30¢.", "The fee is 2.9% plus 30 cents."),
            (Lang::Fr, "Ça coûte 500€ aujourd'hui.", "Ça coûte 500 euros aujourd'hui."),
            (Lang::De, "Es kostet $2500 heute.", "Es kostet 2 Tausend 500 Dollar heute."),
            (
                Lang::En,
                "Call 555-123-4567 before 2024-05-12, it costs 12345678.",
                "Call 555 123 4 5 6 7 before 2024-05-12, it costs 12 million 345 thousand 678.",
            ),
            (Lang::En, "Visit www.kyutai.fr today.", "Visit W-W-W dot kyutai dot F-R today."),
            (
                Lang::Fr,
                "Voir https://www.kyutai.fr/.",
                "Voir H-T-T-P-S deux-points slash slash W-W-W point kyutai point F-R.",
            ),
            // Times and dates are opt-in.
            (Lang::En, "It's 1:06pm on 20/12/2015.", "It's 1:06pm on 20/12/2015."),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(normalize_text(input, lang, Rules::DEFAULT), expected, "{input:?}");
        }
        let en = [
            ("It sold for £3.2m.", "It sold for 3 point 2 million pounds."),
            (
                "We raised $2 million, then €3bn.",
                "We raised 2 million dollars, then 3 billion euros.",
            ),
            (
                "Ring +44 20 7946 0958.",
                "Ring plus four four, two zero, seven nine four six, zero nine five eight.",
            ),
            (
                "Text 07700 900123 now.",
                "Text zero seven seven zero zero, nine zero zero one two three now.",
            ),
            ("In 2019 2020 2021", "In 2019 2020 2021"),
            ("The card ending in -4098.", "The card ending in four zero nine eight."),
            ("It fell to -12.", "It fell to minus 12."),
            ("Your order number is 48213.", "Your order number is 48 thousand 213."),
            (
                "Lt Col Vann and Lt. Col. Harris.",
                "Lieutenant Colonel Vann and Lieutenant Colonel Harris.",
            ),
            ("Rated ★★★★☆, or ★★★½.", "Rated 4 out of 5 stars, or 3 point 5 stars."),
            ("Wait ~10 minutes, ±2.", "Wait about 10 minutes, plus or minus 2."),
            ("I ❤️ NY, dial *67, it's 4.5%.", "I love NY, dial star 67, it's 4.5%."),
            ("Press 1, then #.", "Press 1, then #."),
            ("That was sooo good.", "That was so good."),
            ("He said, \"go home!\"", "He said, \"go home!"),
            ("It's the boys'", "It's the boys'"),
            ("'Stop it.'", "'Stop it."),
            ("It's -40° outside.", "It's minus 40 degrees outside."),
            (
                "Water boils at 100°C, or 212 °F.",
                "Water boils at 100 degrees Celsius, or 212 degrees Fahrenheit.",
            ),
            (
                "It was -3.5°C, then 1°C.",
                "It was minus 3 point 5 degrees Celsius, then 1 degree Celsius.",
            ),
            ("Turn 90°, then 1°.", "Turn 90 degrees, then 1 degree."),
            ("A 45°Ceiling", "A 45 degrees Ceiling"),
            (
                "1920×1080, 6÷2, x≤3, y≥4, a≠b, π≈3.14",
                "1920 times 1 thousand 80, 6 divided by 2, x less than or equal to 3, y greater than or equal to 4, a is not equal to b, pi approximately 3 point 14",
            ),
            ("√2, ∞, 5‰ and ∑", "square root of 2, infinity, 5 per mille and sigma"),
            ("x² + y³ = 10⁻⁶", "x squared plus y cubed equals 10 to the power of minus 6"),
            ("C₈H₁₈ and ¹³C", "C8H18 and 13C"),
            ("ΔH, α and ω", "delta H, alpha and omega"),
            ("and/or, to/from, reality/the deity.", "and or, to from, reality the deity."),
            ("40 km/h, 5 mg/kg, c/o, 3/4, 05/04/2025", "40 km/h, 5 mg/kg, c/o, 3/4, 05/04/2025"),
            ("bǎohù zìjǐ in İstanbul, città", "baohù ziji in Istanbul, città"),
        ];
        for (input, expected) in en {
            assert_eq!(normalize_text(input, Lang::En, Rules::DEFAULT), expected, "{input:?}");
        }
        assert_eq!(
            normalize_text("Il fait -5°C.", Lang::Fr, Rules::DEFAULT),
            "Il fait moins 5 degrés Celsius."
        );
        // "n°" is the number sign, not "n degrees"; an "n" ending a longer word is not one.
        assert_eq!(
            normalize_text("Appelez le n°5, ou le N° 12.", Lang::Fr, Rules::DEFAULT),
            "Appelez le numéro 5, ou le Numéro 12."
        );
        assert_eq!(
            normalize_text("Turn 90°, then 10n°.", Lang::En, Rules::DEFAULT),
            "Turn 90 degrees, then 10n degrees."
        );
        assert_eq!(
            normalize_text("El n.º 5, el nº 6, el n.° 7 y la N.º 8.", Lang::Es, Rules::DEFAULT),
            "El número 5, el número 6, el número 7 y la Número 8."
        );
        assert_eq!(normalize_text("Casa nº3.", Lang::Pt, Rules::DEFAULT), "Casa número 3.");
        assert_eq!(
            normalize_text("Es ist Haus n°3.", Lang::De, Rules::DEFAULT),
            "Es ist Haus Nummer 3."
        );
        assert_eq!(
            normalize_text("Es sind 21°C.", Lang::De, Rules::DEFAULT),
            "Es sind 21 Grad Celsius."
        );
        assert_eq!(
            normalize_text("Hace 1° y 3×2.", Lang::Es, Rules::DEFAULT),
            "Hace 1 grado y 3 por 2."
        );
        assert_eq!(
            normalize_text("Il a levé 2 € millions, ★★★☆☆.", Lang::Fr, Rules::DEFAULT),
            "Il a levé 2 millions d'euros, 3 étoiles sur 5."
        );
        // A symbol after the number and a space, plain or non-breaking, reads as one without.
        let spaced = [
            (Lang::Fr, "Ça coûte 250 €.", "Ça coûte 250 euros."),
            (Lang::Fr, "Ça coûte 42,00\u{a0}€.", "Ça coûte 42 euros."),
            (Lang::Fr, "Ça coûte 0,07\u{202f}$.", "Ça coûte 7 cents."),
            (Lang::Fr, "Ça coûte 4,59 €.", "Ça coûte 4 euros et 59 centimes."),
            (Lang::De, "Es kostet 250 €.", "Es kostet 250 Euro."),
            (Lang::Es, "Cuesta 250 €.", "Cuesta 250 euros."),
            (Lang::Pt, "Custa 30 ¢.", "Custa 30 centavos."),
            (Lang::En, "From 10 $20 bills.", "From 10 20 dollars bills."),
            (Lang::Fr, "Le prix en €.", "Le prix en €."),
        ];
        let reais = [
            ("O total é de R$ 1.234,56.", "O total é de mil 234 vírgula 56 reais."),
            ("A taxa é de R$ 250.", "A taxa é de 250 reais."),
            ("Saldo: -R$ 12,50.", "Saldo, menos 12 vírgula 50 reais."),
            ("Captamos R$ 3,5 bi no ano.", "Captamos 3 vírgula 5 bilhões de reais no ano."),
            ("O negócio vale R$1,5 mi.", "O negócio vale 1 vírgula 5 milhões de reais."),
            ("Custou R$ 1 mi.", "Custou 1 milhão de reais."),
            ("Vale R$ 1 bi.", "Vale 1 bilhão de reais."),
            ("Fica entre R$ 40-60.", "Fica entre R$ 40 a 60."),
        ];
        for (input, expected) in reais {
            assert_eq!(normalize_text(input, Lang::Pt, Rules::DEFAULT), expected, "{input:?}");
        }
        for (lang, input, expected) in spaced {
            assert_eq!(normalize_text(input, lang, Rules::DEFAULT), expected, "{input:?}");
        }
        // French groups thousands with a space, which joins them into one number.
        let grouped = [
            ("Elle possède 40 000 livres.", "Elle possède 40 mille livres."),
            ("Lot de 1\u{202f}200 unités.", "Lot de mille 200 unités."),
            ("La prime est de 1 000 €.", "La prime est de mille euros."),
            ("Il y a 1 200 000 habitants.", "Il y a 1 million 200 mille habitants."),
            ("Soit 1 234,56 kg.", "Soit mille 234 virgule 56 kg."),
            (
                "Le code est RR 123 456 789 GB.",
                "Le code est RR un deux trois, quatre cinq six, sept huit neuf GB.",
            ),
            // Not thousands: a group of other than three digits, or one after a leading zero.
            (
                "Appelez le 01 23 45 67 89.",
                "Appelez le zéro un, deux trois, quatre cinq, six sept, huit neuf.",
            ),
            (
                "Les numéros 12 1234 et 07 123.",
                "Les numéros 12 mille 234 et zéro sept, un deux trois.",
            ),
        ];
        for (input, expected) in grouped {
            assert_eq!(normalize_text(input, Lang::Fr, Rules::DEFAULT), expected, "{input:?}");
        }
        assert_eq!(
            normalize_text("Sie zahlte $2 Millionen.", Lang::De, Rules::DEFAULT),
            "Sie zahlte 2 Millionen Dollar."
        );
        assert_eq!(
            normalize_text("It's 1:06pm on 20/12/2015.", Lang::En, Rules::ALL),
            "It's 1-06 PM on 20-12 2015."
        );
        assert_eq!(normalize_text("Um 08:20 Uhr.", Lang::De, Rules::ALL), "Um 8 Uhr 20 Uhr.");
    }

    /// The frontends take one flag for the language and for turning
    /// normalization off, so both spellings of "off" have to parse rather than
    /// error, and every policy has to spell itself back.
    #[test]
    fn normalize_parses_and_round_trips() {
        for lang in [Lang::En, Lang::Fr, Lang::De, Lang::Es, Lang::Pt] {
            let norm = Normalize::for_lang(lang);
            assert_eq!(Normalize::from(lang), norm);
            assert_eq!(lang.as_str().parse::<Normalize>().unwrap(), norm);
            assert_eq!(norm.as_str().parse::<Normalize>().unwrap(), norm);
        }
        assert_eq!("EN".parse::<Normalize>().unwrap(), Normalize::for_lang(Lang::En));
        assert_eq!(" en ".parse::<Normalize>().unwrap(), Normalize::for_lang(Lang::En));
        assert_eq!(" Off ".parse::<Normalize>().unwrap(), Normalize::OFF);
        assert_eq!("none".parse::<Normalize>().unwrap(), Normalize::OFF);
        assert_eq!("off".parse::<Normalize>().unwrap(), Normalize::OFF);
        assert_eq!(Normalize::OFF.as_str().parse::<Normalize>().unwrap(), Normalize::OFF);
        let err = "klingon".parse::<Normalize>().unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("klingon"), "{msg}");
        assert!(msg.contains("none"), "the error must name the opt-out: {msg}");
    }

    /// Opting out has to hand the text through untouched, and without
    /// allocating: `apply` runs on every request.
    #[test]
    fn apply_follows_the_policy() {
        use std::borrow::Cow;
        assert_eq!(Normalize::OFF.apply("a@b (c)"), "a@b (c)");
        assert_eq!(Normalize::for_lang(Lang::En).apply("a@b (c)"), "a at b, c,");
        assert_eq!(Normalize::for_lang(Lang::Fr).apply("a@b"), "a arobaze b");
        assert!(matches!(Normalize::OFF.apply("text"), Cow::Borrowed(_)));
        // The rules travel with the policy, so `apply` is all a caller needs.
        let en = Normalize::for_lang(Lang::En);
        assert_eq!(en.apply("I paid 1500."), "I paid 1 thousand 500.");
        assert_eq!(en.with_rules(Rules::NONE).apply("I paid 1500."), "I paid 1500.");
    }

    #[test]
    fn lang_round_trips_through_str() {
        use std::str::FromStr;
        for (s, lang) in [
            ("en", Lang::En),
            ("FR", Lang::Fr),
            ("de", Lang::De),
            ("es", Lang::Es),
            ("pt", Lang::Pt),
            ("ja", Lang::Ja),
        ] {
            assert_eq!(Lang::from_str(s).unwrap(), lang);
        }
        for &lang in Lang::ALL {
            assert_eq!(Lang::from_str(lang.as_str()).unwrap(), lang);
        }
        assert_eq!(Normalize::from_str("no").unwrap(), Normalize::for_lang(Lang::No));
        assert_eq!(Lang::from_str("nb").unwrap(), Lang::No);
        assert_eq!(Normalize::from_str("none").unwrap(), Normalize::OFF);
        assert!(Lang::from_str("klingon").is_err());
    }
}
