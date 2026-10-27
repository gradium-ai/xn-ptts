//! Opt-in settings for clients built for OpenAI's own service: voice aliases, for clients that
//! only offer OpenAI's voice names, and accepting a `speed` the model cannot honor. Both are off
//! unless set, so an unknown voice or a speed other than 1 stays an error by default.

use anyhow::{Context as _, Result};

/// The built-in voices OpenAI's speech API names, which `--openai-voices` aliases all at once.
/// Taken from the `voice` type of OpenAI's Python SDK in October 2026; update it when they add one.
pub const OPENAI_VOICES: &[&str] = &[
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage",
    "shimmer", "verse",
];

/// The settings as given on the command line, before they are checked against the voices.
#[derive(Debug, Default)]
pub struct Compat {
    /// `ALIAS=VOICE` entries. Spaces around an entry or its `=` are ignored, and empty entries
    /// are skipped.
    pub voice_aliases: Vec<String>,
    /// The voice every one of [`OPENAI_VOICES`] stands for.
    pub openai_voices: Option<String>,
    /// Accept a `speed` in OpenAI's range and speak at the normal rate.
    pub ignore_speed: bool,
}

/// Extra names a request may give for a voice, each matched in any case like a voice name.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Aliases(Vec<(String, Target)>);

/// What an alias stands for: a voice by its registered name, or `None` for the default voice.
pub type Target = Option<String>;

impl Aliases {
    /// Check `compat`'s aliases against the registered `voices`. A target must be one of them or
    /// `default`, and an alias must not hide a voice or another alias. An OpenAI name that is
    /// already a voice is left to that voice.
    pub fn new(compat: &Compat, voices: &[String]) -> Result<Self> {
        let mut aliases = Self::default();
        for entry in compat.voice_aliases.iter().map(|e| e.trim()).filter(|e| !e.is_empty()) {
            let (alias, target) = entry
                .split_once('=')
                .map(|(a, t)| (a.trim(), t.trim()))
                .filter(|(a, t)| !a.is_empty() && !t.is_empty())
                .with_context(|| format!("--voice-alias takes ALIAS=VOICE, not '{entry}'"))?;
            let target = resolve(target, voices)
                .map_err(|e| anyhow::anyhow!("--voice-alias {entry}: {e}"))?;
            anyhow::ensure!(
                !is_voice(alias, voices),
                "--voice-alias {entry}: '{alias}' is already a voice of this server"
            );
            anyhow::ensure!(
                aliases.get(alias).is_none(),
                "--voice-alias {entry}: '{alias}' is given more than once"
            );
            aliases.0.push((alias.to_string(), target));
        }
        if let Some(voice) = &compat.openai_voices {
            let target = resolve(voice, voices)
                .map_err(|e| anyhow::anyhow!("--openai-voices {voice}: {e}"))?;
            for &alias in OPENAI_VOICES {
                // An explicit --voice-alias for one of the names wins.
                if !is_voice(alias, voices) && aliases.get(alias).is_none() {
                    aliases.0.push((alias.to_string(), target.clone()));
                }
            }
        }
        Ok(aliases)
    }

    /// The voice `name` stands for, if it is an alias.
    pub fn get(&self, name: &str) -> Option<&Target> {
        self.0.iter().find(|(alias, _)| alias.eq_ignore_ascii_case(name)).map(|(_, target)| target)
    }

    /// Each alias and the voice it stands for, `default` for the default voice.
    pub fn to_json(&self) -> serde_json::Value {
        let map = self
            .0
            .iter()
            .map(|(alias, target)| (alias.clone(), target.as_deref().unwrap_or("default").into()));
        serde_json::Value::Object(map.collect())
    }
}

/// Whether a request naming `name` already gets a voice: a registered one, or `default`.
fn is_voice(name: &str, voices: &[String]) -> bool {
    name.eq_ignore_ascii_case("default") || voices.iter().any(|v| v.eq_ignore_ascii_case(name))
}

/// The voice `name` names, matched as a request's `voice` is.
fn resolve(name: &str, voices: &[String]) -> Result<Target, String> {
    if let Some(voice) = voices.iter().find(|v| v.eq_ignore_ascii_case(name)) {
        Ok(Some(voice.clone()))
    } else if name.eq_ignore_ascii_case("default") {
        Ok(None)
    } else {
        Err(format!("no voice '{name}'; this server has {}", voices.join(", ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voices() -> Vec<String> {
        vec!["Freya".into(), "Toby".into()]
    }

    fn aliases(entries: &[&str], openai: Option<&str>) -> Result<Aliases> {
        let compat = Compat {
            voice_aliases: entries.iter().map(|e| e.to_string()).collect(),
            openai_voices: openai.map(str::to_string),
            ignore_speed: false,
        };
        Aliases::new(&compat, &voices())
    }

    #[test]
    fn an_alias_stands_for_a_voice_in_any_case() {
        let got = aliases(&["alloy=freya", "Narrator=default", ""], None).unwrap();
        assert_eq!(got.get("ALLOY"), Some(&Some("Freya".to_string())));
        assert_eq!(got.get("narrator"), Some(&None));
        assert_eq!(got.get("nova"), None);
        assert_eq!(got.to_json(), serde_json::json!({ "alloy": "Freya", "Narrator": "default" }));
        assert_eq!(aliases(&[], None).unwrap(), Aliases::default());
    }

    #[test]
    fn spaces_around_an_alias_are_ignored() {
        // As in `PTTS_VOICE_ALIAS="alloy=Freya, nova = Toby"`.
        let got = aliases(&["alloy=Freya", " nova = Toby ", " "], None).unwrap();
        assert_eq!(got.get("nova"), Some(&Some("Toby".to_string())));
        assert_eq!(got.to_json(), serde_json::json!({ "alloy": "Freya", "nova": "Toby" }));
    }

    #[test]
    fn openai_voices_alias_every_openai_name_and_explicit_aliases_win() {
        let got = aliases(&["nova=Toby"], Some("default")).unwrap();
        assert_eq!(got.get("nova"), Some(&Some("Toby".to_string())));
        for name in OPENAI_VOICES.iter().filter(|&&n| n != "nova") {
            assert_eq!(got.get(name), Some(&None), "{name}");
        }
        // A checkpoint voice with an OpenAI name stays that voice.
        let compat = Compat { openai_voices: Some("Freya".into()), ..Compat::default() };
        let got = Aliases::new(&compat, &["Freya".into(), "Echo".into()]).unwrap();
        assert_eq!(got.get("echo"), None);
        assert_eq!(got.get("alloy"), Some(&Some("Freya".to_string())));
    }

    #[test]
    fn bad_aliases_fail_at_startup() {
        for (entries, openai) in [
            (&["alloy"][..], None),
            (&["alloy="][..], None),
            (&["=Freya"][..], None),
            (&["alloy=nobody"][..], None),
            (&["freya=Toby"][..], None),
            (&["DEFAULT=Toby"][..], None),
            (&["alloy=Toby", "Alloy=Freya"][..], None),
            (&[][..], Some("nobody")),
        ] {
            assert!(aliases(entries, openai).is_err(), "{entries:?} {openai:?}");
        }
    }
}
