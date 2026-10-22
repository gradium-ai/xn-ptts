//! The C interface must hand back exactly what ptts plans for `lang=en`.
//!
//! Each input in `tests/inputs.txt` (one per line, `\n`, `\r` and `\t` unescaped) goes through
//! three paths, which must agree field for field:
//! - `ptts_text_split`, through the C ABI;
//! - `ptts::plan::chunks`, which `Synth::say` / `Synth::stream` call, with `Synth`'s defaults;
//! - the same steps spelled out: `Normalize::apply`, `split_into_best_sentences`,
//!   `prepare_text_prompt`, then the raw `tokenizers` crate's `encode(.., false)`.
//!
//! It also writes the ptts reference in the format `examples/dump.c` prints, to
//! `target/expected.txt`, for `test.sh` to diff the C program's output against on each platform.
//!
//! The tokenizer is the `tokenizer.json` that `PTTS_TOKENIZER` names; any checkpoint's will do.

use ptts::Tokenizer as _;
use ptts::preprocess::{Lang, Normalize};
use ptts::tts_model::{MAX_TOKENS_PER_CHUNK, prepare_text_prompt, split_into_best_sentences};
use ptts_text::*;
use std::ffi::{CStr, CString, c_char};
use std::fmt::Write as _;

#[derive(Debug, PartialEq)]
struct Row {
    text: String,
    prepared: String,
    tokens: Vec<u32>,
    frames_after_eos: u32,
    frame_budget: u32,
    seq_budget: u32,
}

fn tokenizer_path() -> String {
    std::env::var("PTTS_TOKENIZER").expect("set PTTS_TOKENIZER to a checkpoint's tokenizer.json")
}

fn inputs() -> Vec<String> {
    let file = include_str!("inputs.txt");
    file.lines().map(|l| l.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")).collect()
}

fn via_c(h: *const ptts_text::PttsText, text: &str) -> Result<Vec<Row>, String> {
    let c = CString::new(text).unwrap();
    let mut err: *mut c_char = std::ptr::null_mut();
    let out = unsafe { ptts_text_split(h, c.as_ptr(), 0, 0.0, &mut err) };
    if out.is_null() {
        let msg = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
        unsafe { ptts_text_string_free(err) };
        return Err(msg);
    }
    let s = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string();
    let chunks = unsafe { std::slice::from_raw_parts((*out).chunks, (*out).n_chunks) };
    let rows = chunks
        .iter()
        .map(|c| Row {
            text: s(c.text),
            prepared: s(c.prepared),
            tokens: unsafe { std::slice::from_raw_parts(c.tokens, c.n_tokens) }.to_vec(),
            frames_after_eos: c.frames_after_eos,
            frame_budget: c.frame_budget,
            seq_budget: c.seq_budget,
        })
        .collect();
    unsafe { ptts_text_chunks_free(out) };
    Ok(rows)
}

fn via_plan(tok: &ptts::tok::Tok, text: &str) -> Result<Vec<Row>, String> {
    let en = Normalize::for_lang(Lang::En);
    let chunks =
        ptts::plan::chunks(tok, text, en, MAX_TOKENS_PER_CHUNK, 12.5).map_err(|e| e.to_string())?;
    Ok(chunks
        .into_iter()
        .map(|c| Row {
            prepared: prepare_text_prompt(&c.text).0,
            frames_after_eos: c.frames_after_eos as u32,
            frame_budget: c.frame_budget as u32,
            seq_budget: c.seq_budget() as u32,
            text: c.text,
            tokens: c.tokens,
        })
        .collect())
}

fn via_steps(tok: &ptts::tok::Tok, raw: &tokenizers::Tokenizer, text: &str) -> Vec<Row> {
    let normalized = Normalize::for_lang(Lang::En).apply(text);
    let pieces = split_into_best_sentences(tok, &normalized, Some(50)).unwrap();
    pieces
        .into_iter()
        .map(|piece| {
            let (prepared, frames_after_eos) = prepare_text_prompt(&piece);
            let tokens = raw.encode(prepared.as_str(), false).unwrap().get_ids().to_vec();
            let frame_budget = ((tokens.len() as f64 / 3.0 + 2.0) * 12.5).ceil() as u32;
            Row {
                text: piece,
                prepared,
                frames_after_eos: frames_after_eos as u32,
                frame_budget,
                seq_budget: tokens.len() as u32 + frame_budget + 512,
                tokens,
            }
        })
        .collect()
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t")
}

/// The format `examples/dump.c` prints.
fn dump(i: usize, r: &Result<Vec<Row>, String>, out: &mut String) {
    match r {
        Err(e) => writeln!(out, "#{i} error: {e}").unwrap(),
        Ok(rows) => {
            writeln!(out, "#{i} chunks={}", rows.len()).unwrap();
            for row in rows {
                writeln!(out, "  text={}", escape(&row.text)).unwrap();
                writeln!(out, "  prepared={}", escape(&row.prepared)).unwrap();
                writeln!(
                    out,
                    "  frames_after_eos={} frame_budget={} seq_budget={} n_tokens={}",
                    row.frames_after_eos,
                    row.frame_budget,
                    row.seq_budget,
                    row.tokens.len()
                )
                .unwrap();
                let ids: Vec<String> = row.tokens.iter().map(u32::to_string).collect();
                writeln!(out, "  ids={}", ids.join(" ")).unwrap();
            }
        }
    }
}

#[test]
fn c_interface_matches_ptts_for_english() {
    let path = tokenizer_path();
    let tok = ptts::tok::Tok::open(std::path::Path::new(&path)).unwrap();
    let raw = tokenizers::Tokenizer::from_file(&path).unwrap();
    let (cpath, lang) = (CString::new(path.clone()).unwrap(), CString::new("en").unwrap());
    let mut err: *mut c_char = std::ptr::null_mut();
    let h = unsafe { ptts_text_new(cpath.as_ptr(), lang.as_ptr(), &mut err) };
    assert!(!h.is_null());

    let inputs = inputs();
    assert!(inputs.len() >= 30);
    let (mut expected, mut multi, mut ok) = (String::new(), 0, 0);
    for (i, text) in inputs.iter().enumerate() {
        let c = via_c(h, text);
        let plan = via_plan(&tok, text);
        assert_eq!(c, plan, "input #{i} {text:?}");
        if let Ok(rows) = &plan {
            assert_eq!(rows, &via_steps(&tok, &raw, text), "input #{i} {text:?}");
            ok += 1;
            multi += (rows.len() > 1) as usize;
            for row in rows {
                // What the generation loop is handed is a function of the prepared text alone.
                assert_eq!(tok.encode(&row.prepared).unwrap(), row.tokens);
            }
        }
        dump(i, &plan, &mut expected);
    }
    assert!(ok >= 30, "only {ok} inputs produced chunks");
    assert!(multi >= 2, "only {multi} inputs needed more than one chunk");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("expected.txt"), expected).unwrap();
    unsafe { ptts_text_free(h) };
}

#[test]
fn errors_are_reported() {
    let open = |path: &str, lang: &str| {
        let (p, l) = (CString::new(path).unwrap(), CString::new(lang).unwrap());
        let mut err: *mut c_char = std::ptr::null_mut();
        let h = unsafe { ptts_text_new(p.as_ptr(), l.as_ptr(), &mut err) };
        if h.is_null() {
            let msg = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
            unsafe { ptts_text_string_free(err) };
            Err(msg)
        } else {
            Ok(h)
        }
    };
    assert!(open(&tokenizer_path(), "xx").is_err());
    assert!(open("/nonexistent/tokenizer.json", "en").is_err());
    let h = open(&tokenizer_path(), "en").unwrap();
    assert!(via_c(h, "   ").unwrap_err().contains("empty"));
    // A null error pointer is allowed.
    let empty = CString::new("").unwrap();
    assert!(unsafe { ptts_text_split(h, empty.as_ptr(), 0, 0.0, std::ptr::null_mut()) }.is_null());
    unsafe { ptts_text_free(h) };
}

#[test]
fn lang_none_and_rewrites_are_passed_through() {
    let path = tokenizer_path();
    let tok = ptts::tok::Tok::open(std::path::Path::new(&path)).unwrap();
    let cpath = CString::new(path).unwrap();
    let text = "Call +1 555 123 4567 at 10:30 on 2024-01-02, $5.";
    for (lang, rewrites, normalize) in [
        ("none", None, Normalize::OFF),
        ("en", Some("all"), Normalize::for_lang(Lang::En).with_rules("all".parse().unwrap())),
        ("fr", None, Normalize::for_lang(Lang::Fr)),
    ] {
        let l = CString::new(lang).unwrap();
        let r = rewrites.map(|r| CString::new(r).unwrap());
        let h = unsafe {
            ptts_text_new_with_rewrites(
                cpath.as_ptr(),
                l.as_ptr(),
                r.as_ref().map_or(std::ptr::null(), |r| r.as_ptr()),
                std::ptr::null_mut(),
            )
        };
        assert!(!h.is_null());
        let want = ptts::plan::chunks(&tok, text, normalize, 50, 12.5).unwrap();
        let got = via_c(h, text).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!((&g.text, &g.tokens), (&w.text, &w.tokens), "{lang}");
        }
        let c = CString::new(text).unwrap();
        let n = unsafe { ptts_text_normalize(h, c.as_ptr(), std::ptr::null_mut()) };
        assert_eq!(unsafe { CStr::from_ptr(n) }.to_str().unwrap(), normalize.apply(text));
        unsafe { ptts_text_string_free(n) };
        unsafe { ptts_text_free(h) };
    }
}
