//! Text normalization applied before tokenization.
//!
//! The model is trained on ordinary written prose, so characters it never saw -- typographic
//! quotes, bullets, arrows, emoji -- are best removed rather than tokenized, and symbols that
//! are read aloud (`@`, `+`, `=`) are best spelled out in the target language. [`normalize_text`]
//! does both, driven by a [`Lang`].
//!
//! This is deliberately conservative: it does not expand numbers, dates or abbreviations, which
//! the model handles natively.
//!
//! Every caller says which language, or says not to normalize: [`Normalize`] is a required
//! argument to [`crate::synth::SynthBuilder::new`], and every frontend takes it as a required
//! flag. There is no default, deliberately. Output is noticeably better with normalization than
//! without, but normalizing German as English speaks `@` as "at" rather than "ät", so guessing
//! the language is worse than doing nothing.

/// Spoken forms of the punctuation characters that are read aloud rather than dropped.
#[derive(Debug, Clone)]
pub struct SpecialChars {
    pub colon: &'static str,
    pub slash: &'static str,
    pub dash: &'static str,
    pub dot: &'static str,
    pub at: &'static str,
    pub plus: &'static str,
    pub equals: &'static str,
}

pub const SPECIAL_CHARS_EN: SpecialChars = SpecialChars {
    colon: "colon",
    slash: "slash",
    dash: "dash",
    dot: "dot",
    at: "at",
    plus: "plus",
    equals: "equals",
};

pub const SPECIAL_CHARS_FR: SpecialChars = SpecialChars {
    colon: "deux-points",
    slash: "slash",
    dash: "tiret",
    dot: "point",
    at: "arobaze",
    plus: "plus",
    equals: "égal",
};

pub const SPECIAL_CHARS_DE: SpecialChars = SpecialChars {
    colon: "Doppelpunkt",
    slash: "Slash",
    dash: "Bindestrich",
    dot: "Punkt",
    at: "ät",
    plus: "Plus",
    equals: "Gleich",
};

pub const SPECIAL_CHARS_ES: SpecialChars = SpecialChars {
    colon: "dos-puntos",
    slash: "slash",
    dash: "guion",
    dot: "punto",
    at: "arroba",
    plus: "mas",
    equals: "igual",
};

pub const SPECIAL_CHARS_PT: SpecialChars = SpecialChars {
    colon: "dois-pontos",
    slash: "slash",
    dash: "hifen",
    dot: "ponto",
    at: "arroba",
    plus: "mais",
    equals: "igual",
};

/// Language driving the spoken forms used by [`normalize_text`].
///
/// Deliberately has no `Default`: the spoken forms differ per language, so a caller that has not
/// said which language it has is better off not normalizing at all. See [`Normalize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    En,
    Fr,
    De,
    Es,
    Pt,
}

impl std::str::FromStr for Lang {
    type Err = crate::Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.to_lowercase().as_str() {
            "en" => Ok(Lang::En),
            "fr" => Ok(Lang::Fr),
            "de" => Ok(Lang::De),
            "es" => Ok(Lang::Es),
            "pt" => Ok(Lang::Pt),
            _ => Err(crate::Error::invalid_argument(format!(
                "unsupported language code: {s}; expected en, fr, de, es, pt, or none to skip \
                 normalization"
            ))),
        }
    }
}

impl Lang {
    /// The language code this variant parses from.
    pub fn as_str(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Fr => "fr",
            Lang::De => "de",
            Lang::Es => "es",
            Lang::Pt => "pt",
        }
    }

    pub fn special_chars(self) -> &'static SpecialChars {
        match self {
            Lang::En => &SPECIAL_CHARS_EN,
            Lang::Fr => &SPECIAL_CHARS_FR,
            Lang::De => &SPECIAL_CHARS_DE,
            Lang::Es => &SPECIAL_CHARS_ES,
            Lang::Pt => &SPECIAL_CHARS_PT,
        }
    }

    pub fn decimal_separator(self) -> &'static str {
        match self {
            Lang::En => "point",
            Lang::Fr => "virgule",
            Lang::De => "Komma",
            Lang::Es => "coma",
            Lang::Pt => "vírgula",
        }
    }

    pub fn underscore(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr | Lang::Pt => "underscore",
            Lang::De => "Unterstrich",
            Lang::Es => "guion bajo",
        }
    }

    pub fn dollars(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr => "dollars",
            Lang::De => "Dollar",
            Lang::Es | Lang::Pt => "dólares",
        }
    }

    pub fn dollars_singular(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr => "dollar",
            Lang::De => "Dollar",
            Lang::Es | Lang::Pt => "dólar",
        }
    }

    pub fn euros(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr | Lang::Es | Lang::Pt => "euros",
            Lang::De => "Euro",
        }
    }

    pub fn euros_singular(self) -> &'static str {
        match self {
            Lang::En | Lang::Fr | Lang::Es | Lang::Pt => "euro",
            Lang::De => "Euro",
        }
    }

    pub fn pounds(self) -> &'static str {
        match self {
            Lang::En => "pounds",
            Lang::Fr => "livres",
            Lang::De => "Pfund",
            Lang::Es | Lang::Pt => "libras",
        }
    }

    pub fn pounds_singular(self) -> &'static str {
        match self {
            Lang::En => "pound",
            Lang::Fr => "livre",
            Lang::De => "Pfund",
            Lang::Es | Lang::Pt => "libra",
        }
    }

    pub fn currency(&self, symbol: char) -> Option<&'static str> {
        match symbol {
            '$' => Some(self.dollars()),
            '€' => Some(self.euros()),
            '£' => Some(self.pounds()),
            _ => None,
        }
    }

    pub fn currency_singular(&self, symbol: char) -> Option<&'static str> {
        match symbol {
            '$' => Some(self.dollars_singular()),
            '€' => Some(self.euros_singular()),
            '£' => Some(self.pounds_singular()),
            _ => None,
        }
    }
}

/// Whether to normalize, and in which language.
///
/// There is no default and no "unset": [`crate::synth::SynthBuilder::new`] takes one of these,
/// so choosing is not something a caller can forget. [`Normalize::Off`] is the way to say "hand
/// the text to the tokenizer as written", which is for callers that normalize it themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalize {
    For(Lang),
    Off,
}

impl Normalize {
    /// Parse what the frontends' `--lang` flag accepts: a language code, or `none` / `off`.
    pub fn parse(s: &str) -> crate::Result<Self> {
        match s.to_lowercase().as_str() {
            "none" | "off" => Ok(Self::Off),
            other => other.parse().map(Self::For),
        }
    }

    /// How this policy spells itself back, round-tripping through [`Self::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::For(lang) => lang.as_str(),
            Self::Off => "none",
        }
    }

    /// Normalize `text`, or hand it back untouched when this is [`Self::Off`].
    ///
    /// Borrows when off, so opting out costs no allocation per request.
    ///
    /// This has to run before `prepare_text_prompt`, which pads short text with leading spaces
    /// that normalization would collapse away.
    pub fn apply<'a>(self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match self {
            Self::Off => std::borrow::Cow::Borrowed(text),
            Self::For(lang) => std::borrow::Cow::Owned(normalize_text(text, lang)),
        }
    }
}

fn is_emoji(c: char) -> bool {
    let c = c as u32;
    matches!(c,
        0x1F600..=0x1FAFF |
        0x2600..=0x27BF |
        // Flags (regional indicator symbols)
        0x1F1E6..=0x1F1FF
    )
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
            while self.buffer.last().is_some_and(|l| l.is_whitespace() || l.is_ascii_punctuation())
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

/// Rewrite `input` into the character set the model was trained on.
///
/// Typographic quotes, dashes, bullets, arrows and emoji are dropped or folded to their ASCII
/// equivalents; `@`, `+` and `=` are spelled out in `lang`; `;`, `:` and parentheses become
/// commas, which is how the model is asked to pause. Numbers, dates and abbreviations are left
/// alone -- the model reads those natively.
pub fn normalize_text(input: &str, lang: Lang) -> String {
    let mut res = StringAppender::new();
    for c in input.chars() {
        match c {
            '“' | '”' | '"' => res.push_whitespace(),
            '’' | '‘' => res.push('\''),
            '‐' | '‑' | '‒' | '―' => res.push('-'),
            // The two dashes below are not - (ascii 45) but similar unicode chars.
            '–' | '*' | '—' | '[' | ']' | '{' | '}' => res.push_whitespace(),
            '•' | '‣' | '◦' | '·' | '→' | '←' | '↑' | '↓' | '➡' | '➜' => {
                res.push_whitespace();
            }
            '…' => res.push('.'),
            '@' => {
                res.push_whitespace();
                res.push_str(lang.special_chars().at);
                res.push_whitespace();
            }
            '+' => {
                res.push_whitespace();
                res.push_str(lang.special_chars().plus);
                res.push_whitespace();
            }
            '=' => {
                res.push_whitespace();
                res.push_str(lang.special_chars().equals);
                res.push_whitespace();
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
    }
    res.into_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_text_cases() {
        let cases: &[(&str, &str)] = &[
            ("Hello, world!", "Hello, world!"),
            ("", ""),
            ("“hello” world it's", "hello world it's"),
            ("a‐b‑c‒d―e", "a-b-c-d-e"),
            ("a–b—c", "a b c"),
            ("foo (bar) [baz] {qux} *quux*", "foo, bar, baz qux quux"),
            ("• ‣ ◦ · a→b←c↑d↓e ➡ ➜", "a b c d e"),
            ("wait… a…b", "wait. a.b"),
            ("user@host @home", "user at host at home"),
            ("café résumé 日本語", "café résumé 日本語"),
            ("hello 😀 flag 🇫🇷 sun ☀", "hello flag sun"),
            // ';', ':' and '(' / ')' all collapse to ", " (comma + single space).
            ("a;b:c", "a, b, c"),
            ("time: 10:30", "time, 10, 30"),
            ("; leading", ", leading"),
            ("hello (world)", "hello, world,"),
            // Surrounding whitespace is absorbed into the comma replacement.
            ("foo ; bar  :  baz", "foo, bar, baz"),
            // Runs of ASCII and non-ASCII whitespace collapse to a single space,
            // and trailing whitespace is stripped.
            ("a   b\t\tc\n\nd", "a b c d"),
            ("hello   ", "hello"),
            ("a • b • c", "a b c"),
            ("“Hello”; please email user@host (now)… 🚀", "Hello, please email user at host, now."),
            (
                "Numbers: one, two, three, four, five. Special items: at sign, hash, dollar, percent.",
                "Numbers, one, two, three, four, five. Special items, at sign, hash, dollar, percent.",
            ),
            (
                "The conference will be held on Tuesday, March 15th at 3:30 PM.",
                "The conference will be held on Tuesday, March 15th at 3, 30 PM.",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(&normalize_text(input, Lang::En), expected, "input: {input:?}");
        }
    }

    #[test]
    fn spoken_symbols_follow_the_language() {
        assert_eq!(normalize_text("a@b", Lang::En), "a at b");
        assert_eq!(normalize_text("a@b", Lang::Fr), "a arobaze b");
        assert_eq!(normalize_text("a@b", Lang::De), "a ät b");
        assert_eq!(normalize_text("1+1=2", Lang::Es), "1 mas 1 igual 2");
        assert_eq!(normalize_text("1+1=2", Lang::Pt), "1 mais 1 igual 2");
    }

    /// The frontends take one flag for the language and for turning
    /// normalization off, so both spellings of "off" have to parse rather than
    /// error, and every policy has to spell itself back.
    #[test]
    fn normalize_parses_and_round_trips() {
        for lang in [Lang::En, Lang::Fr, Lang::De, Lang::Es, Lang::Pt] {
            let norm = Normalize::For(lang);
            assert_eq!(Normalize::parse(lang.as_str()).unwrap(), norm);
            assert_eq!(Normalize::parse(norm.as_str()).unwrap(), norm);
        }
        assert_eq!(Normalize::parse("EN").unwrap(), Normalize::For(Lang::En));
        assert_eq!(Normalize::parse("none").unwrap(), Normalize::Off);
        assert_eq!(Normalize::parse("off").unwrap(), Normalize::Off);
        assert_eq!(Normalize::parse(Normalize::Off.as_str()).unwrap(), Normalize::Off);
        let err = Normalize::parse("klingon").unwrap_err();
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
        assert_eq!(Normalize::Off.apply("a@b (c)"), "a@b (c)");
        assert_eq!(Normalize::For(Lang::En).apply("a@b (c)"), "a at b, c,");
        assert_eq!(Normalize::For(Lang::Fr).apply("a@b"), "a arobaze b");
        assert!(matches!(Normalize::Off.apply("text"), Cow::Borrowed(_)));
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
        ] {
            assert_eq!(Lang::from_str(s).unwrap(), lang);
        }
        assert!(Lang::from_str("klingon").is_err());
    }
}
