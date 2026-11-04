//! Decoding text tokens with the SentencePiece pieces the export writes to `tokenizer.json`.
//! Only decoding: the model reads its own output, never text from outside.

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum Kind {
    Normal,
    Byte,
    Control,
}

#[derive(serde::Deserialize)]
struct Piece {
    piece: String,
    kind: Kind,
}

#[derive(serde::Deserialize)]
struct File {
    pieces: Vec<Piece>,
}

pub struct Pieces {
    pieces: Vec<Piece>,
}

impl Pieces {
    pub fn load(path: &std::path::Path) -> xn::Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| xn::Error::msg(format!("reading {}: {e}", path.display())))?;
        let file: File = serde_json::from_str(&json)
            .map_err(|e| xn::Error::msg(format!("parsing {}: {e}", path.display())))?;
        Ok(Self { pieces: file.pieces })
    }
}

impl ptts::Tokenizer for Pieces {
    fn encode(&self, _text: &str) -> xn::Result<Vec<u32>> {
        xn::bail!("graphon's tokenizer only decodes")
    }

    fn decode(&self, tokens: &[u32]) -> xn::Result<String> {
        let mut bytes = Vec::new();
        for &t in tokens {
            let Some(p) = self.pieces.get(t as usize) else {
                xn::bail!("token {t} is out of the {}-piece vocabulary", self.pieces.len())
            };
            match p.kind {
                Kind::Normal => {
                    bytes.extend_from_slice(p.piece.replace('\u{2581}', " ").as_bytes())
                }
                // `<0xAB>`
                Kind::Byte => match u8::from_str_radix(&p.piece[3..p.piece.len() - 1], 16) {
                    Ok(b) => bytes.push(b),
                    Err(_) => xn::bail!("malformed byte piece {:?}", p.piece),
                },
                Kind::Control => {}
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
