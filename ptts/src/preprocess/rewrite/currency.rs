//! Amounts of money: "$1,234.56" becomes "1 thousand 234 dollars and 56 cents", "30¢" becomes
//! "30 cents".

use super::numbers::{number_words, polish_few, scale_words};
use super::{all_digits, split_suffix};
use crate::preprocess::Lang;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Currency {
    Dollar,
    Euro,
    Pound,
    Yen,
    Cent,
    Real,
    Rupee,
    Won,
    Krona,
    Krone,
    Zloty,
    Riyal,
    Dirham,
    Dinar,
}

impl Currency {
    const SYMBOLS: [char; 7] = ['$', '€', '£', '¥', '¢', '₹', '₩'];

    fn from_symbol(symbol: char) -> Option<Self> {
        match symbol {
            '$' => Some(Self::Dollar),
            '€' => Some(Self::Euro),
            '£' => Some(Self::Pound),
            '¥' => Some(Self::Yen),
            '¢' => Some(Self::Cent),
            '₹' => Some(Self::Rupee),
            '₩' => Some(Self::Won),
            _ => None,
        }
    }

    /// The currency a word or an abbreviation next to an amount names in `lang`: "kr", "zł",
    /// "ر.س", "Rs.", "円". The languages the serving stack reads have none, so that their
    /// readings stay as they are.
    fn from_unit(unit: &str, lang: Lang) -> Option<Self> {
        Some(match (lang, unit) {
            (Lang::Sv, "kr" | "kronor" | "krona" | "SEK") => Self::Krona,
            (Lang::No, "kr" | "kroner" | "krone" | "NOK") => Self::Krone,
            (Lang::Pl, "zł" | "PLN") => Self::Zloty,
            (Lang::Fi, "euroa" | "euro" | "EUR")
            | (Lang::Sv | Lang::No | Lang::It | Lang::Nl | Lang::Pl, "euro" | "EUR") => Self::Euro,
            (Lang::Hi, "रुपये" | "रुपए" | "रुपया" | "Rs" | "Rs." | "INR") => {
                Self::Rupee
            }
            (Lang::Ar, "ريال" | "ر.س" | "ر.ق" | "ر.ع") => Self::Riyal,
            (Lang::Ar, "درهم" | "د.إ" | "د.م") => Self::Dirham,
            (Lang::Ar, "جنيه" | "ج.م") => Self::Pound,
            (Lang::Ar, "دينار" | "د.ك" | "د.أ" | "د.ع" | "د.ب") => Self::Dinar,
            (Lang::Ar, "دولار") => Self::Dollar,
            (Lang::Ar, "يورو") => Self::Euro,
            (Lang::Ja, "円" | "JPY") => Self::Yen,
            (Lang::Ja, "ドル") => Self::Dollar,
            (Lang::Ja, "ユーロ") => Self::Euro,
            (Lang::Ko, "원" | "KRW") => Self::Won,
            (Lang::Ko, "달러") => Self::Dollar,
            (Lang::Ko, "유로") => Self::Euro,
            (Lang::Ko, "엔") => Self::Yen,
            _ => return None,
        })
    }

    /// The name an amount of this currency takes: the plural everywhere the serving stack
    /// reads, even for 1 (see [`currency`]); elsewhere the singular for 1, the Polish few form
    /// from 2 to 4, and the plural otherwise.
    fn name_for(self, lang: Lang, amount: &str) -> Option<&'static str> {
        let (one, many) = self.name(lang)?;
        if matches!(lang, Lang::En | Lang::Fr | Lang::De | Lang::Es | Lang::Pt) {
            return Some(if self == Self::Cent && amount == "1" { one } else { many });
        }
        let n = amount.parse::<i64>().ok();
        Some(match n {
            Some(1) => one,
            Some(n) if lang == Lang::Pl && polish_few(n) => match self {
                Self::Dollar => "dolary",
                Self::Pound => "funty",
                Self::Yen => "jeny",
                Self::Cent => "centy",
                Self::Rupee => "rupie",
                Self::Zloty => "złote",
                _ => many,
            },
            _ => many,
        })
    }

    /// The name of this currency's hundredth in `lang`, for `n` of them, or `None` where the
    /// hundredth is not said and the amount is read as a decimal instead. Dutch says none:
    /// "5 euro 50".
    fn subunit(self, lang: Lang, n: i64) -> Option<&'static str> {
        let few = lang == Lang::Pl && polish_few(n);
        Some(match (self, lang) {
            (Self::Dollar | Self::Euro, Lang::Fi) => {
                if n == 1 {
                    "sentti"
                } else {
                    "senttiä"
                }
            }
            (Self::Dollar | Self::Euro, Lang::Sv | Lang::No) => "cent",
            (Self::Dollar | Self::Euro, Lang::It) => {
                if n == 1 {
                    "centesimo"
                } else {
                    "centesimi"
                }
            }
            (Self::Dollar | Self::Euro, Lang::Nl) => "",
            (Self::Dollar | Self::Euro, Lang::Pl) => {
                if n == 1 {
                    "cent"
                } else if few {
                    "centy"
                } else {
                    "centów"
                }
            }
            (Self::Dollar | Self::Euro, Lang::Ar) => "سنت",
            (Self::Dollar | Self::Euro, Lang::Hi) => "सेंट",
            (Self::Dollar | Self::Euro, Lang::Ja) => "セント",
            (Self::Dollar | Self::Euro, Lang::Ko) => "센트",
            (Self::Riyal, Lang::Ar) => "هللة",
            (Self::Dirham | Self::Dinar, Lang::Ar) => "فلس",
            (Self::Pound, Lang::Ar) => "قرش",
            (Self::Rupee, Lang::Hi) => {
                if n == 1 {
                    "पैसा"
                } else {
                    "पैसे"
                }
            }
            (Self::Krona, Lang::Sv) => "öre",
            (Self::Krone, Lang::No) => "øre",
            (Self::Zloty, Lang::Pl) => {
                if n == 1 {
                    "grosz"
                } else if few {
                    "grosze"
                } else {
                    "groszy"
                }
            }
            _ => return None,
        })
    }

    /// The currency's name in `lang`, singular and plural.
    fn name(self, lang: Lang) -> Option<(&'static str, &'static str)> {
        Some(match (self, lang) {
            (Self::Dollar, Lang::En | Lang::Fr) => ("dollar", "dollars"),
            (Self::Dollar, Lang::De) => ("Dollar", "Dollar"),
            (Self::Dollar, Lang::Es | Lang::Pt) => ("dólar", "dólares"),
            (Self::Dollar, Lang::Fi) => ("dollari", "dollaria"),
            (Self::Dollar, Lang::Sv | Lang::No | Lang::Nl) => ("dollar", "dollar"),
            (Self::Dollar, Lang::It) => ("dollaro", "dollari"),
            (Self::Dollar, Lang::Pl) => ("dolar", "dolarów"),
            (Self::Dollar, Lang::Ar) => ("دولار", "دولار"),
            (Self::Dollar, Lang::Hi) => ("डॉलर", "डॉलर"),
            (Self::Dollar, Lang::Ja) => ("ドル", "ドル"),
            (Self::Dollar, Lang::Ko) => ("달러", "달러"),
            (Self::Euro, Lang::De) => ("Euro", "Euro"),
            (Self::Euro, Lang::Fi) => ("euro", "euroa"),
            (Self::Euro, Lang::Sv | Lang::No | Lang::It | Lang::Nl | Lang::Pl) => ("euro", "euro"),
            (Self::Euro, Lang::Ar) => ("يورو", "يورو"),
            (Self::Euro, Lang::Hi) => ("यूरो", "यूरो"),
            (Self::Euro, Lang::Ja) => ("ユーロ", "ユーロ"),
            (Self::Euro, Lang::Ko) => ("유로", "유로"),
            (Self::Euro, _) => ("euro", "euros"),
            (Self::Pound, Lang::En) => ("pound", "pounds"),
            (Self::Pound, Lang::Fr) => ("livre", "livres"),
            (Self::Pound, Lang::De) => ("Pfund", "Pfund"),
            (Self::Pound, Lang::Es | Lang::Pt) => ("libra", "libras"),
            (Self::Pound, Lang::Fi) => ("punta", "puntaa"),
            (Self::Pound, Lang::Sv | Lang::No) => ("pund", "pund"),
            (Self::Pound, Lang::It) => ("sterlina", "sterline"),
            (Self::Pound, Lang::Nl) => ("pond", "pond"),
            (Self::Pound, Lang::Pl) => ("funt", "funtów"),
            (Self::Pound, Lang::Ar) => ("جنيه", "جنيه"),
            (Self::Pound, Lang::Hi) => ("पाउंड", "पाउंड"),
            (Self::Pound, Lang::Ja) => ("ポンド", "ポンド"),
            (Self::Pound, Lang::Ko) => ("파운드", "파운드"),
            (Self::Yen, Lang::De) => ("Yen", "Yen"),
            (Self::Yen, Lang::En) => ("yen", "yen"),
            (Self::Yen, Lang::Fr) => ("yen", "yens"),
            (Self::Yen, Lang::Es) => ("yen", "yenes"),
            (Self::Yen, Lang::Pt) => ("iene", "ienes"),
            (Self::Yen, Lang::Fi) => ("jeni", "jeniä"),
            (Self::Yen, Lang::Sv | Lang::No | Lang::It | Lang::Nl) => ("yen", "yen"),
            (Self::Yen, Lang::Pl) => ("jen", "jenów"),
            (Self::Yen, Lang::Ar) => ("ين", "ين"),
            (Self::Yen, Lang::Hi) => ("येन", "येन"),
            (Self::Yen, Lang::Ja) => ("円", "円"),
            (Self::Yen, Lang::Ko) => ("엔", "엔"),
            (Self::Cent, Lang::En | Lang::Fr) => ("cent", "cents"),
            (Self::Cent, Lang::De) => ("Cent", "Cent"),
            (Self::Cent, Lang::Es | Lang::Pt) => ("centavo", "centavos"),
            (Self::Cent, Lang::Fi) => ("sentti", "senttiä"),
            (Self::Cent, Lang::Sv | Lang::No | Lang::Nl) => ("cent", "cent"),
            (Self::Cent, Lang::It) => ("centesimo", "centesimi"),
            (Self::Cent, Lang::Pl) => ("cent", "centów"),
            (Self::Cent, Lang::Ar) => ("سنت", "سنت"),
            (Self::Cent, Lang::Hi) => ("सेंट", "सेंट"),
            (Self::Cent, Lang::Ja) => ("セント", "セント"),
            (Self::Cent, Lang::Ko) => ("센트", "센트"),
            (Self::Real, _) => ("real", "reais"),
            (Self::Rupee, Lang::En) => ("rupee", "rupees"),
            (Self::Rupee, Lang::Fr) => ("roupie", "roupies"),
            (Self::Rupee, Lang::De) => ("Rupie", "Rupien"),
            (Self::Rupee, Lang::Es | Lang::Pt) => ("rupia", "rupias"),
            (Self::Rupee, Lang::It) => ("rupia", "rupie"),
            (Self::Rupee, Lang::Fi) => ("rupia", "rupiaa"),
            (Self::Rupee, Lang::Sv) => ("rupie", "rupier"),
            (Self::Rupee, Lang::No) => ("rupi", "rupier"),
            (Self::Rupee, Lang::Nl) => ("roepie", "roepies"),
            (Self::Rupee, Lang::Pl) => ("rupia", "rupii"),
            (Self::Rupee, Lang::Ar) => ("روبية", "روبية"),
            (Self::Rupee, Lang::Hi) => ("रुपया", "रुपये"),
            (Self::Rupee, Lang::Ja) => ("ルピー", "ルピー"),
            (Self::Rupee, Lang::Ko) => ("루피", "루피"),
            (Self::Won, Lang::Ko) => ("원", "원"),
            (Self::Won, Lang::Ja) => ("ウォン", "ウォン"),
            (Self::Won, Lang::Ar) => ("وون", "وون"),
            (Self::Won, Lang::Hi) => ("वॉन", "वॉन"),
            (Self::Won, _) => ("won", "won"),
            (Self::Krona, Lang::Sv) => ("krona", "kronor"),
            (Self::Krone, Lang::No) => ("krone", "kroner"),
            (Self::Zloty, Lang::Pl) => ("złoty", "złotych"),
            (Self::Riyal, Lang::Ar) => ("ريال", "ريال"),
            (Self::Dirham, Lang::Ar) => ("درهم", "درهم"),
            (Self::Dinar, Lang::Ar) => ("دينار", "دينار"),
            _ => return None,
        })
    }
}

/// Whether `c` is a currency symbol the rule reads. The character pass joins one to the number
/// before it, so that "250 €", as French, German, Spanish and Portuguese write it, reads as
/// "250€".
pub(in crate::preprocess) fn is_symbol(c: char) -> bool {
    Currency::from_symbol(c).is_some()
}

/// A number with a currency symbol before or after it, read as [`super::numbers`] reads the
/// number, followed by the currency's name. In English and French an amount with exactly two
/// decimals is read in cents: "$4.59" becomes "4 dollars and 59 cents", "4,59€" becomes "4 euros
/// et 59 centimes". A cent sign is never read in cents of itself: "1.5¢" is "1 point 5 cents".
pub(super) fn currency(word: &str, lang: Lang) -> Option<String> {
    let (body, suffix) = split_suffix(word);
    // "-$500" reads as "$-500" does.
    if let Some(body) = body.strip_prefix('-')
        && let Some(amount) = body
            .strip_prefix(Currency::SYMBOLS)
            .or_else(|| body.strip_prefix("R$").filter(|_| lang == Lang::Pt))
    {
        let symbol = &body[..body.len() - amount.len()];
        return currency(&format!("{symbol}-{amount}{suffix}"), lang);
    }
    // The symbol leads or trails, never both: "$5€" is not an amount.
    let (currency, amount) = split_symbol(body, lang)?;
    if let Some((amount, scale)) = split_scale(amount, lang) {
        return with_scale(amount, scale, currency, lang).map(|spoken| format!("{spoken}{suffix}"));
    }
    if currency != Currency::Cent
        && let Some(spoken) = with_cents(amount, currency, lang)
    {
        return Some(format!("{spoken}{suffix}"));
    }
    let (spoken, _) = number_words(amount, lang)?;
    // Always the plural in the languages the serving stack reads, even for 1: "$1" reads "1
    // dollars". A quirk, kept so that the readings stay identical to the ones it already speaks.
    // The cent sign is newer than that, so "1¢" reads "1 cent".
    let name = currency.name_for(lang, amount)?;
    Some(format!("{}{suffix}", with_name(&spoken, name, lang)))
}

/// An amount and its currency as two words, the currency a word or an abbreviation, after the
/// amount ("5,99 kr", "50 ر.س", "5 euro") or before it ("Rs. 500"), or with a scale word between
/// ("5 mln zł"). The amount is read as it is after a symbol, and the currency by its full name.
pub(super) fn unit_after(words: &[&str], lang: Lang) -> Option<(String, usize)> {
    let [first, second, ..] = words else { return None };
    let (unit, suffix) = split_suffix(second);
    let (amount, currency, scale, used, suffix) = if let Some(currency) =
        Currency::from_unit(first, lang)
            .filter(|_| second.starts_with(|c: char| c.is_ascii_digit()))
    {
        let (amount, suffix) = split_suffix(second);
        (amount, currency, None, 2, suffix)
    } else if let Some(currency) = Currency::from_unit(unit, lang) {
        (*first, currency, None, 2, suffix)
    } else if let Some(third) = words.get(2)
        && let Some(scale) = scale_index(unit, lang)
        && let Some(currency) = Currency::from_unit(split_suffix(third).0, lang)
    {
        (*first, currency, Some(scale), 3, split_suffix(third).1)
    } else {
        return None;
    };
    if !amount.starts_with(|c: char| c.is_ascii_digit() || c == '-') {
        return None;
    }
    let spoken = match scale {
        Some(scale) => {
            let n: i64 = amount.parse().ok().filter(|n| (1..1000).contains(n))?;
            with_name(&scale_words(n, scale, lang), currency.name(lang)?.1, lang)
        }
        None => match with_cents(amount, currency, lang) {
            Some(spoken) => spoken,
            None => {
                let (number, rest) = number_words(amount, lang)?;
                if !rest.is_empty() {
                    return None;
                }
                with_name(&number, currency.name_for(lang, amount)?, lang)
            }
        },
    };
    Some((format!("{spoken}{suffix}"), used))
}

/// The scale a word between an amount and its currency names, as [`scale_words`] counts them:
/// 0 a milliard, 1 a million, 2 a thousand.
fn scale_index(word: &str, lang: Lang) -> Option<usize> {
    let scales: [&[&str]; 3] = match lang {
        Lang::Fi => [
            &["miljardi", "miljardia", "mrd."],
            &["miljoona", "miljoonaa", "milj."],
            &["tuhat", "tuhatta"],
        ],
        Lang::Sv => [&["miljard", "miljarder"], &["miljon", "miljoner"], &["tusen"]],
        Lang::No => {
            [&["milliard", "milliarder", "mrd."], &["million", "millioner", "mill."], &["tusen"]]
        }
        Lang::It => {
            [&["miliardo", "miliardi", "mld"], &["milione", "milioni", "mln"], &["mille", "mila"]]
        }
        Lang::Nl => [&["miljard", "mld"], &["miljoen", "mln"], &["duizend"]],
        Lang::Pl => [
            &["miliard", "miliardy", "miliardów", "mld"],
            &["milion", "miliony", "milionów", "mln"],
            &["tysiąc", "tysiące", "tysięcy", "tys.", "tys"],
        ],
        Lang::Ar => [&["مليار", "مليارات"], &["مليون", "ملايين"], &["ألف", "آلاف"]],
        _ => return None,
    };
    scales.iter().position(|words| words.contains(&word))
}

/// `amount` read as whole units and cents, when it has exactly two decimals: English writes it
/// "1,234.56" and French "1234,56". `None` for any other amount, and in the other languages.
fn split_scale(amount: &str, lang: Lang) -> Option<(&str, &'static str)> {
    let suffixes: &[(&str, &str)] = match lang {
        Lang::En => &[
            ("thousand", "thousand"),
            ("million", "million"),
            ("billion", "billion"),
            ("trillion", "trillion"),
            ("bn", "billion"),
            ("BN", "billion"),
            ("mn", "million"),
            ("tn", "trillion"),
            ("k", "thousand"),
            ("K", "thousand"),
            ("m", "million"),
            ("M", "million"),
            ("b", "billion"),
            ("B", "billion"),
        ],
        _ => &[],
    };
    suffixes.iter().find_map(|&(suffix, scale)| {
        let number = amount.strip_suffix(suffix)?;
        number.ends_with(|c: char| c.is_ascii_digit()).then_some((number, scale))
    })
}

fn scale_word(word: &str, lang: Lang) -> bool {
    let words: &[&str] = match lang {
        Lang::En => &["thousand", "million", "billion", "trillion"],
        Lang::Fr => &["mille", "million", "millions", "milliard", "milliards"],
        Lang::De => &["Tausend", "Million", "Millionen", "Milliarde", "Milliarden"],
        Lang::Es => &["mil", "millón", "millones"],
        Lang::Pt => &["mil", "milhão", "milhões", "bilhão", "bilhões"],
        _ => &[],
    };
    words.contains(&word)
}

fn with_scale(amount: &str, scale: &str, currency: Currency, lang: Lang) -> Option<String> {
    let (amount, rest) = number_words(amount, lang)?;
    if !rest.is_empty() {
        return None;
    }
    let (_, name) = currency.name(lang)?;
    Some(with_name(&format!("{amount} {scale}"), name, lang))
}

pub(super) fn scaled(words: &[&str], lang: Lang) -> Option<(String, usize)> {
    let [word, next, ..] = words else { return None };
    let (scale, suffix) = split_suffix(next);
    // "$2 million", or "2 € millions" once the character pass has joined it to "2€ millions".
    let (currency, amount) = split_symbol(word, lang)?;
    let scale = match (lang, scale, amount) {
        (Lang::Pt, "bi", "1") => "bilhão",
        (Lang::Pt, "bi", _) => "bilhões",
        (Lang::Pt, "mi", "1") => "milhão",
        (Lang::Pt, "mi", _) => "milhões",
        _ => scale,
    };
    if !scale_word(scale, lang) {
        return None;
    }
    let spoken = with_scale(amount, scale, currency, lang)?;
    Some((format!("{spoken}{suffix}"), 2))
}

fn split_symbol(word: &str, lang: Lang) -> Option<(Currency, &str)> {
    if lang == Lang::Pt
        && let Some(amount) = word.strip_prefix("R$")
    {
        return Some((Currency::Real, amount));
    }
    // "Rs.500", "Rs500": the rupee's abbreviation joined to the amount.
    if lang == Lang::Hi
        && let Some(amount) = word.strip_prefix("Rs.").or_else(|| word.strip_prefix("Rs"))
        && amount.starts_with(|c: char| c.is_ascii_digit())
    {
        return Some((Currency::Rupee, amount));
    }
    // "1,500円", "5000원": Japanese and Korean write the unit right after the amount.
    if matches!(lang, Lang::Ja | Lang::Ko)
        && let Some(at) = word.find(|c: char| !(c.is_ascii_digit() || matches!(c, ',' | '.' | '-')))
        && at > 0
        && let Some(currency) = Currency::from_unit(&word[at..], lang)
    {
        return Some((currency, &word[..at]));
    }
    match word.strip_prefix(Currency::SYMBOLS) {
        Some(amount) => Some((Currency::from_symbol(word.chars().next()?)?, amount)),
        None => {
            let amount = word.strip_suffix(Currency::SYMBOLS)?;
            Some((Currency::from_symbol(word.chars().next_back()?)?, amount))
        }
    }
}

fn with_cents(amount: &str, currency: Currency, lang: Lang) -> Option<String> {
    // Whole units grouped by `group`, when `whole` is not plain digits.
    let grouped = |whole: &str, group: char| {
        whole.split(group).enumerate().all(|(i, part)| {
            all_digits(part) && if i == 0 { part.len() <= 3 } else { part.len() == 3 }
        })
    };
    let (whole, cents) = match lang {
        Lang::En => amount.rsplit_once('.').filter(|(whole, _)| grouped(whole, ','))?,
        Lang::Fr | Lang::Fi | Lang::Sv | Lang::No | Lang::Pl => {
            amount.rsplit_once(',').filter(|(whole, _)| all_digits(whole))?
        }
        Lang::Ar | Lang::Hi | Lang::Ja | Lang::Ko => {
            amount.rsplit_once('.').filter(|(whole, _)| grouped(whole, ','))?
        }
        Lang::It | Lang::Nl => amount.rsplit_once(',').filter(|(whole, _)| grouped(whole, '.'))?,
        _ => return None,
    };
    if cents.len() != 2 || !all_digits(cents) {
        return None;
    }
    let digits: String = whole.chars().filter(char::is_ascii_digit).collect();
    let (whole, _) = number_words(whole, lang)?;
    // One unit is singular here even where the whole amount is always plural: "$1.00" reads
    // "1 dollar" where "$1" reads "1 dollars".
    let name =
        if digits == "1" { currency.name(lang)?.0 } else { currency.name_for(lang, &digits)? };
    let units = with_name(&whole, name, lang);
    if cents == "00" {
        return Some(units);
    }
    // "05" is read "5". In English and French the subdivision is always plural, and English
    // says cents even for pounds: quirks again, kept for the same reason as the plural above.
    let cents = cents.strip_prefix('0').unwrap_or(cents);
    let n: i64 = cents.parse().ok()?;
    // Hindi spells its numbers out, the hundredths too.
    let cents = if lang == Lang::Hi { super::numbers::hindi_group(n) } else { cents.to_string() };
    let cents = cents.as_str();
    let (and, cent) = match (lang, currency) {
        (Lang::Fr, Currency::Euro) => ("et", "centimes"),
        (Lang::Fr, _) => ("et", "cents"),
        (Lang::En, _) => ("and", "cents"),
        _ => {
            let and = match lang {
                Lang::Sv => "och",
                Lang::No => "og",
                Lang::It => "e",
                Lang::Ar => "و",
                _ => "",
            };
            (and, currency.subunit(lang, n)?)
        }
    };
    let cents = [cents, cent].into_iter().filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ");
    if whole == "0" {
        return Some(cents);
    }
    Some(
        [units.as_str(), and, &cents]
            .into_iter()
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// `amount` followed by the currency's `name`. Scale words are nouns, so in French, Spanish,
/// Portuguese and Italian a currency right after one takes a linking "de" or "di" ("2 millions
/// de dollars", "2 milioni di euro"), elided before a vowel in French ("2 millions d'euros"). A
/// numeral adjective takes none ("500 mille euros").
fn with_name(amount: &str, name: &str, lang: Lang) -> String {
    let (nouns, link): (&[&str], &str) = match lang {
        Lang::Fr => (&["million", "millions", "milliard", "milliards"], "de"),
        Lang::Es => (&["millón", "millones"], "de"),
        Lang::Pt => (&["milhão", "milhões", "bilhão", "bilhões"], "de"),
        Lang::It => (&["milione", "milioni", "miliardo", "miliardi"], "di"),
        _ => (&[], ""),
    };
    if !nouns.iter().any(|noun| amount.ends_with(noun)) {
        format!("{amount} {name}")
    } else if lang == Lang::Fr && name.starts_with(['a', 'e', 'i', 'o', 'u', 'é']) {
        format!("{amount} d'{name}")
    } else {
        format!("{amount} {link} {name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_new_languages_read_amounts_with_their_currency() {
        let cases = [
            (Lang::Ar, "$50", "50 دولار"),
            (Lang::Ar, "$1", "1 دولار"),
            (Lang::Ar, "€2,500", "ألفين 500 يورو"),
            (Lang::Ar, "€3,500", "3 آلاف 500 يورو"),
            (Lang::Ar, "$4.50", "4 دولار و 50 سنت"),
            (Lang::Ja, "€1,234.56", "1234 ユーロ 56 セント"),
            (Lang::Ko, "$4.50", "4 달러 50 센트"),
            (Lang::Hi, "₹500", "पाँच सौ रुपये"),
            (Lang::Hi, "₹1", "एक रुपया"),
            (Lang::Hi, "Rs.500", "पाँच सौ रुपये"),
            (Lang::Hi, "₹1,23,456", "एक लाख तेईस हजार चार सौ छप्पन रुपये"),
            (Lang::Hi, "₹50.50", "पचास रुपये पचास पैसे"),
            (Lang::Hi, "$5", "पाँच डॉलर"),
            (Lang::Ja, "¥1500", "1500 円"),
            (Lang::Ja, "1,500円", "1500 円"),
            (Lang::Ja, "¥1,234,567", "123万4567 円"),
            (Lang::Ja, "500ドル", "500 ドル"),
            (Lang::Ja, "$5.50", "5 ドル 50 セント"),
            (Lang::Ko, "₩5,000", "5000 원"),
            (Lang::Ko, "5,000원", "5000 원"),
            (Lang::Ko, "12345원", "1만 2345 원"),
            (Lang::Ko, "$5", "5 달러"),
            (Lang::Fi, "5€", "5 euroa"),
            (Lang::Fi, "1€", "1 euro"),
            (Lang::Fi, "5,99€", "5 euroa 99 senttiä"),
            (Lang::Fi, "5,01€", "5 euroa 1 sentti"),
            (Lang::Fi, "$5", "5 dollaria"),
            (Lang::Sv, "5€", "5 euro"),
            (Lang::Sv, "$5", "5 dollar"),
            (Lang::Sv, "5,99$", "5 dollar och 99 cent"),
            (Lang::No, "$5", "5 dollar"),
            (Lang::It, "€5", "5 euro"),
            (Lang::It, "5,99€", "5 euro e 99 centesimi"),
            (Lang::It, "€1.500", "mille 500 euro"),
            (Lang::It, "$5", "5 dollari"),
            (Lang::It, "$1", "1 dollaro"),
            (Lang::Nl, "€5", "5 euro"),
            (Lang::Nl, "€5,50", "5 euro 50"),
            (Lang::Nl, "€5,00", "5 euro"),
            (Lang::Nl, "€1.500,50", "duizend 500 euro 50"),
            (Lang::Pl, "5€", "5 euro"),
            (Lang::Pl, "$5", "5 dolarów"),
            (Lang::Pl, "$2", "2 dolary"),
            (Lang::Pl, "$1", "1 dolar"),
            (Lang::Pl, "5,99$", "5 dolarów 99 centów"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(currency(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        // Amounts that are not numbers.
        assert_eq!(currency("€", Lang::Fi), None);
        assert_eq!(currency("5€x", Lang::Fi), None);
    }

    #[test]
    fn units_next_to_the_amount_are_read_in_full() {
        let unit = |words: &[&str], lang| unit_after(words, lang);
        assert_eq!(unit(&["50", "kr."], Lang::Sv), Some(("50 kronor.".into(), 2)));
        assert_eq!(unit(&["1", "kr"], Lang::Sv), Some(("1 krona".into(), 2)));
        assert_eq!(unit(&["5,99", "kr"], Lang::Sv), Some(("5 kronor och 99 öre".into(), 2)));
        assert_eq!(unit(&["5,99", "kr"], Lang::No), Some(("5 kroner og 99 øre".into(), 2)));
        assert_eq!(unit(&["5,99", "zł"], Lang::Pl), Some(("5 złotych 99 groszy".into(), 2)));
        assert_eq!(unit(&["2", "zł"], Lang::Pl), Some(("2 złote".into(), 2)));
        assert_eq!(unit(&["1", "zł"], Lang::Pl), Some(("1 złoty".into(), 2)));
        assert_eq!(unit(&["5", "mln", "zł"], Lang::Pl), Some(("5 milionów złotych".into(), 3)));
        assert_eq!(unit(&["2", "mld", "zł."], Lang::Pl), Some(("2 miliardy złotych.".into(), 3)));
        assert_eq!(unit(&["2", "mln", "euro"], Lang::It), Some(("2 milioni di euro".into(), 3)));
        assert_eq!(unit(&["5,99", "euro"], Lang::Fi), Some(("5 euroa 99 senttiä".into(), 2)));
        assert_eq!(unit(&["5,99", "euro."], Lang::Nl), Some(("5 euro 99.".into(), 2)));
        assert_eq!(unit(&["50", "ر.س"], Lang::Ar), Some(("50 ريال".into(), 2)));
        assert_eq!(
            unit(&["1,234.56", "ر.س."], Lang::Ar),
            Some(("ألف 234 ريال و 56 هللة.".into(), 2))
        );
        assert_eq!(unit(&["50", "د.إ"], Lang::Ar), Some(("50 درهم".into(), 2)));
        assert_eq!(unit(&["3", "ملايين", "دولار"], Lang::Ar), Some(("3 ملايين دولار".into(), 3)));
        assert_eq!(unit(&["Rs.", "500"], Lang::Hi), Some(("पाँच सौ रुपये".into(), 2)));
        assert_eq!(unit(&["500", "रुपये"], Lang::Hi), Some(("पाँच सौ रुपये".into(), 2)));
        assert_eq!(
            unit(&["12345", "रुपये"], Lang::Hi),
            Some(("बारह हजार तीन सौ पैंतालीस रुपये".into(), 2))
        );
        // Not an amount, and the languages the serving stack reads keep their readings.
        assert_eq!(unit(&["about", "kr"], Lang::Sv), None);
        assert_eq!(unit(&["50", "kr"], Lang::En), None);
        assert_eq!(unit(&["50", "euro"], Lang::De), None);
    }

    #[test]
    fn amounts_are_read_with_their_currency() {
        let cases = [
            // A symbol before the amount.
            (Lang::En, "$1234", "1 thousand 234 dollars"),
            (Lang::En, "$1,234.56", "1 thousand 234 dollars and 56 cents"),
            (Lang::En, "$1,234.56...", "1 thousand 234 dollars and 56 cents..."),
            (Lang::En, "$4.59", "4 dollars and 59 cents"),
            (Lang::En, "$0.01", "1 cents"),
            (Lang::En, "$0.10", "10 cents"),
            (Lang::En, "$1.00", "1 dollar"),
            (Lang::En, "$1.10", "1 dollar and 10 cents"),
            (Lang::En, "$3.14", "3 dollars and 14 cents"),
            (Lang::En, "$0.00", "0 dollars"),
            (Lang::En, "$1", "1 dollars"),
            (Lang::En, "£1.50", "1 pound and 50 cents"),
            (Lang::Fr, "$4,59", "4 dollars et 59 cents"),
            (Lang::Fr, "$0,01", "1 cents"),
            (Lang::Fr, "$1,00", "1 dollar"),
            (Lang::Fr, "$1,10", "1 dollar et 10 cents"),
            (Lang::Fr, "$0,00", "0 dollars"),
            (Lang::Fr, "$100", "100 dollars"),
            (Lang::De, "$2500", "2 Tausend 500 Dollar"),
            (Lang::Es, "$50", "50 dólares"),
            (Lang::Pt, "$50!", "50 dólares!"),
            (Lang::En, "€500...", "500 euros..."),
            (Lang::En, "€0", "0 euros"),
            (Lang::En, "€0.5", "0 point 5 euros"),
            (Lang::Fr, "€1000", "mille euros"),
            (Lang::De, "€1000", "ein Tausend Euro"),
            (Lang::De, "$1000000", "eine Million Dollar"),
            (Lang::En, "£250", "250 pounds"),
            (Lang::Fr, "£250", "250 livres"),
            // A cent sign, plural apart from 1, and never split into cents of cents.
            (Lang::En, "30¢", "30 cents"),
            (Lang::En, "30¢.", "30 cents."),
            (Lang::En, "1¢", "1 cent"),
            (Lang::En, "99¢!", "99 cents!"),
            (Lang::En, "1.50¢", "1 point 50 cents"),
            (Lang::Fr, "30¢", "30 cents"),
            (Lang::De, "30¢", "30 Cent"),
            (Lang::Es, "30¢", "30 centavos"),
            (Lang::Pt, "1¢", "1 centavo"),
            (Lang::De, "£250", "250 Pfund"),
            (Lang::Es, "£250", "250 libras"),
            // A symbol after it.
            (Lang::En, "1234$", "1 thousand 234 dollars"),
            (Lang::Fr, "4,59€", "4 euros et 59 centimes"),
            (Lang::Fr, "0,01€", "1 centimes"),
            (Lang::Fr, "1,00€", "1 euro"),
            (Lang::Fr, "0,00€", "0 euros"),
            (Lang::Fr, "500€!", "500 euros!"),
            (Lang::De, "100£", "100 Pfund"),
            (Lang::De, "1000000000€", "eine Milliarde Euro"),
            // Only English and French read cents: elsewhere two decimals are a plain number.
            (Lang::De, "1,23$", "1 Komma 23 Dollar"),
            (Lang::De, "1,23$!!!", "1 Komma 23 Dollar!!!"),
            // Signs, including a zero integer part.
            (Lang::En, "$-500", "minus 500 dollars"),
            (Lang::En, "-$500.", "minus 500 dollars."),
            (Lang::De, "-€1.000", "minus ein Tausend Euro"),
            (Lang::Fr, "-0,50€", "moins 0 virgule 50 euros"),
            // A linking "de" after a scale noun, elided before a vowel in French, and none after
            // the numeral adjective "mille".
            (Lang::Fr, "€2000000", "2 millions d'euros"),
            (Lang::Fr, "$2000000", "2 millions de dollars"),
            (Lang::Fr, "£3000000000.", "3 milliards de livres."),
            (Lang::Fr, "€2000000,00", "2 millions d'euros"),
            (Lang::Fr, "2000000,50€", "2 millions d'euros et 50 centimes"),
            (Lang::Fr, "€2500000", "2 millions 500 mille euros"),
            (Lang::Es, "€2000000", "2 millones de euros"),
            (Lang::Pt, "$2000000", "2 milhões de dólares"),
            (Lang::En, "$2000000", "2 million dollars"),
            (Lang::De, "$2000000", "2 Millionen Dollar"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(currency(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        for input in ["1234", "$abc", "$", "$5€", "5$5", "$007", "$1.2.3", "$ 5", "-$", "--$5"] {
            assert_eq!(currency(input, Lang::En), None, "{input:?}");
        }
    }
}
