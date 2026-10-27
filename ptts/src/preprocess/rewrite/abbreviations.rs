use super::split_suffix;
use crate::preprocess::Lang;

const EN: &[(&str, &str)] = &[
    ("Lt", "Lieutenant"),
    ("Lt.", "Lieutenant"),
    ("Col.", "Colonel"),
    ("Sgt.", "Sergeant"),
    ("Sgt", "Sergeant"),
    ("Capt.", "Captain"),
    ("Cpl.", "Corporal"),
    ("Gen.", "General"),
    ("Adm.", "Admiral"),
    ("Sen.", "Senator"),
    ("Sens.", "Senators"),
    ("Rep.", "Representative"),
    ("Reps.", "Representatives"),
    ("Gov.", "Governor"),
    ("Pres.", "President"),
    ("Det.", "Detective"),
    ("Insp.", "Inspector"),
    ("Supt.", "Superintendent"),
    ("Ft.", "Fort"),
    ("Mt.", "Mount"),
    ("Jan.", "January"),
    ("Feb.", "February"),
    ("Mar.", "March"),
    ("Apr.", "April"),
    ("Jun.", "June"),
    ("Jul.", "July"),
    ("Aug.", "August"),
    ("Sep.", "September"),
    ("Sept.", "September"),
    ("Oct.", "October"),
    ("Nov.", "November"),
    ("Dec.", "December"),
    ("Mon.", "Monday"),
    ("Tue.", "Tuesday"),
    ("Tues.", "Tuesday"),
    ("Wed.", "Wednesday"),
    ("Thu.", "Thursday"),
    ("Thur.", "Thursday"),
    ("Thurs.", "Thursday"),
    ("Fri.", "Friday"),
    ("Sat.", "Saturday"),
    ("Sun.", "Sunday"),
    ("vol.", "volume"),
    ("Vol.", "Volume"),
    ("ed.", "edition"),
    ("approx.", "approximately"),
    ("Approx.", "Approximately"),
    ("dept.", "department"),
    ("Dept.", "Department"),
    ("vs.", "versus"),
    ("w/", "with"),
    ("W/", "With"),
    ("w/o", "without"),
    ("W/o", "Without"),
];

pub(super) fn abbreviations(word: &str, lang: Lang) -> Option<String> {
    if lang != Lang::En {
        return None;
    }
    let (body, suffix) = split_suffix(word);
    let (abbreviation, suffix) = match suffix.strip_prefix('.') {
        Some(rest) => (&word[..body.len() + 1], rest),
        None => (body, suffix),
    };
    let (_, expansion) = EN.iter().find(|(short, _)| *short == abbreviation)?;
    Some(format!("{expansion}{suffix}"))
}

pub(super) fn ranks(words: &[&str], lang: Lang) -> Option<(String, usize)> {
    let [first, second, ..] = words else { return None };
    let (rank, suffix) = split_suffix(second);
    match (lang, *first, rank) {
        (Lang::En, "Lt", "Col") => Some((format!("Lieutenant Colonel{suffix}"), 2)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abbreviations_are_expanded() {
        let cases = [
            ("Lt.", Some("Lieutenant")),
            ("Sgt.", Some("Sergeant")),
            ("Dec.", Some("December")),
            ("Dec.,", Some("December,")),
            ("Thu.", Some("Thursday")),
            ("ed.", Some("edition")),
            ("w/o", Some("without")),
            ("Col", None),
            ("dec.", None),
            ("Sun", None),
            ("Dr.", None),
        ];
        for (input, expected) in cases {
            assert_eq!(abbreviations(input, Lang::En).as_deref(), expected, "{input:?}");
        }
        assert_eq!(abbreviations("Lt.", Lang::Fr), None);
        assert_eq!(ranks(&["Lt", "Col", "Vann"], Lang::En), Some(("Lieutenant Colonel".into(), 2)));
        assert_eq!(ranks(&["Lt", "Vann"], Lang::En), None);
    }
}
