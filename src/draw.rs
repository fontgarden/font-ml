//! Draw a shape from a sentence.
//!
//! A `draw` model is trained from scratch on one UFO. Every drawn
//! glyph is a shape; its name is the word for it. The trainer measures
//! each shape, scales it to many widths, puts it at many places in a
//! box, and writes a sentence for each one from a few templates:
//! "draw a circle 64 units wide in the lower left". The model then
//! learns that tiny language and the outlines that go with it.
//!
//! A sequence is `BOS <words> SEP ADV <advance> <ops> EOS`. The words
//! are ordinary name tokens, so the token table needs nothing new;
//! numbers are words too, one per width the trainer can produce.
//!
//! At run time the sentence is split into words, unknown words become
//! `UNK`, the advance of the target glyph goes in the header, and the
//! model continues greedily until `EOS`.

use crate::checkpoint::{ModelConfig, ModelKind};
use crate::error::{Error, Result};
use crate::outline::OutlineModel;
use crate::tokenizer::{snap, Op, Vocab, GRID};
use crate::train::{
    load_master, lr_at, pad_batch, today, write_json, write_vocab, Report, Rng, TrainConfig,
};
use candle_core::DType;
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

/// Space between a shape and the side of its box when it sits left or right.
pub const SIDEBEARING: f64 = 60.0;
/// The height of the box a shape is placed in: the cap height.
pub const BOX_HEIGHT: f64 = 700.0;
/// The narrowest width the trainer produces.
pub const WIDTH_MIN: i32 = 24;
/// The widest width the trainer produces.
pub const WIDTH_MAX: i32 = 720;
/// Widths come in steps of this many units, so every width is one word.
pub const WIDTH_STEP: i32 = 8;
/// How many fixed sequences the validation loss is taken over.
pub const VAL_SEQUENCES: usize = 48;

/// Where a shape sits in its box, and the phrases for it. The first
/// entry is the default: a sentence with no position means bottom left,
/// which is where a drawn glyph sits anyway.
pub const POSITIONS: &[(&str, &[&str])] = &[
    (
        "bottom-left",
        &["", "in the lower left", "at the bottom left", "bottom left"],
    ),
    (
        "bottom-center",
        &["at the bottom", "bottom center", "at the bottom, centered"],
    ),
    (
        "bottom-right",
        &["in the lower right", "at the bottom right", "bottom right"],
    ),
    (
        "middle-left",
        &["on the left", "middle left", "on the left, centered"],
    ),
    ("center", &["in the center", "centered", "in the middle"]),
    (
        "middle-right",
        &["on the right", "middle right", "on the right, centered"],
    ),
    (
        "top-left",
        &["in the upper left", "at the top left", "top left"],
    ),
    (
        "top-center",
        &["at the top", "top center", "at the top, centered"],
    ),
    (
        "top-right",
        &["in the upper right", "at the top right", "top right"],
    ),
];

/// The sentence shapes a model learns. `{name}`, `{w}` and `{pos}` are
/// filled in; a trailing empty `{pos}` leaves no trace.
pub const TEMPLATES: &[&str] = &[
    "draw a {name} {w} units wide {pos}",
    "a {name} {w} wide {pos}",
    "{name} {w} {pos}",
    "draw a {w} unit wide {name} {pos}",
    "make a {name} that is {w} units wide {pos}",
    "{name}, width {w}, {pos}",
];

/// The words of a sentence: lower case, letters and digits only.
pub fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// The word for a glyph: the name before any `.suffix`, with `-` and
/// `_` as spaces. `circle.300` and `circle` are both "circle";
/// `five-star` is "five star".
pub fn shape_words(glyph_name: &str) -> Vec<String> {
    let base = glyph_name.split('.').next().unwrap_or(glyph_name);
    words(&base.replace(['-', '_'], " "))
}

/// A width the trainer could have produced: the nearest step in range.
pub fn snap_width(w: f64) -> i32 {
    let q = (w / WIDTH_STEP as f64).round() as i32 * WIDTH_STEP;
    q.clamp(WIDTH_MIN, WIDTH_MAX)
}

/// One drawn shape, moved so its box starts at the origin.
#[derive(Debug, Clone)]
pub struct Shape {
    pub glyph: String,
    pub words: Vec<String>,
    pub ops: Vec<Op>,
    pub width: f64,
    pub height: f64,
}

fn op_points(op: &Op) -> Vec<(f64, f64)> {
    match op {
        Op::MoveTo(x, y) | Op::LineTo(x, y) => vec![(*x, *y)],
        Op::CurveTo(p) => p.to_vec(),
        Op::ClosePath => vec![],
    }
}

fn map_ops(ops: &[Op], f: impl Fn((f64, f64)) -> (f64, f64)) -> Vec<Op> {
    ops.iter()
        .map(|op| match op {
            Op::MoveTo(x, y) => {
                let (x, y) = f((*x, *y));
                Op::MoveTo(x, y)
            }
            Op::LineTo(x, y) => {
                let (x, y) = f((*x, *y));
                Op::LineTo(x, y)
            }
            Op::CurveTo(p) => Op::CurveTo([f(p[0]), f(p[1]), f(p[2])]),
            Op::ClosePath => Op::ClosePath,
        })
        .collect()
}

impl Shape {
    fn new(glyph: &str, ops: &[Op]) -> Option<Self> {
        let pts: Vec<(f64, f64)> = ops.iter().flat_map(op_points).collect();
        if pts.is_empty() {
            return None;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for (x, y) in &pts {
            x0 = x0.min(*x);
            y0 = y0.min(*y);
            x1 = x1.max(*x);
            y1 = y1.max(*y);
        }
        let width = x1 - x0;
        let height = y1 - y0;
        if width < 1.0 || height < 1.0 {
            return None;
        }
        Some(Self {
            glyph: glyph.to_string(),
            words: shape_words(glyph),
            ops: map_ops(ops, |(x, y)| (x - x0, y - y0)),
            width,
            height,
        })
    }

    /// The shape scaled to `width` units wide and placed at `(x, y)`,
    /// every point snapped to the grid.
    pub fn placed(&self, width: f64, x: f64, y: f64) -> Vec<Op> {
        let s = width / self.width;
        map_ops(&self.ops, |(px, py)| {
            (f64::from(snap(px * s + x)), f64::from(snap(py * s + y)))
        })
    }
}

/// Where a shape of `w` by `h` goes for a position index, in a box
/// `advance` wide and [`BOX_HEIGHT`] tall.
pub fn place(position: usize, w: f64, h: f64, advance: f64) -> (f64, f64) {
    let x = match position % 3 {
        0 => SIDEBEARING,
        1 => (advance - w) / 2.0,
        _ => advance - SIDEBEARING - w,
    };
    let y = match position / 3 {
        0 => 0.0,
        1 => (BOX_HEIGHT - h) / 2.0,
        _ => BOX_HEIGHT - h,
    };
    (x, y)
}

/// Every word a draw model can read.
pub fn vocabulary(shapes: &[Shape]) -> Vec<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for t in TEMPLATES {
        set.extend(words(&t.replace(['{', '}'], " ")));
    }
    for (_, phrases) in POSITIONS {
        for p in *phrases {
            set.extend(words(p));
        }
    }
    for s in shapes {
        set.extend(s.words.iter().cloned());
    }
    let mut w = WIDTH_MIN;
    while w <= WIDTH_MAX {
        set.insert(w.to_string());
        w += WIDTH_STEP;
    }
    // Placeholders are not words.
    for ph in ["name", "w", "pos"] {
        set.remove(ph);
    }
    set.into_iter().collect()
}

/// The header of a sequence: the sentence, then the advance.
pub fn encode_prompt(v: &Vocab, text: &str, advance: f64) -> (Vec<u32>, Vec<String>) {
    let unk = v.special("UNK").unwrap_or(14) as u32;
    let mut unknown = Vec::new();
    let mut toks = vec![v.special("BOS").unwrap_or(1) as u32];
    for w in words(text) {
        // A number the trainer never produced becomes the nearest one
        // it did: "300" reads as "304".
        let w = match w.parse::<f64>() {
            Ok(n) if v.name(&w).is_err() => snap_width(n).to_string(),
            _ => w,
        };
        match v.name(&w) {
            Ok(id) => toks.push(id as u32),
            Err(_) => {
                unknown.push(w);
                toks.push(unk);
            }
        }
    }
    toks.push(v.special("SEP").unwrap_or(3) as u32);
    toks.push(v.special("ADV").unwrap_or(8) as u32);
    toks.push(v.coord(advance) as u32);
    (toks, unknown)
}

/// The shapes of one UFO, ready to sample.
pub struct Corpus {
    pub vocab: Vocab,
    pub shapes: Vec<Shape>,
}

impl Corpus {
    pub fn from_ufo(path: &Path) -> Result<Self> {
        let master = load_master(path)?;
        let mut shapes: Vec<Shape> = master
            .glyphs
            .iter()
            .filter_map(|(name, (_, ops))| Shape::new(name, ops))
            .collect();
        shapes.sort_by(|a, b| a.glyph.cmp(&b.glyph));
        if shapes.is_empty() {
            return Err(Error::Io {
                path: path.to_path_buf(),
                source: std::io::Error::other("no drawn glyph to learn a shape from"),
            });
        }
        let vocab = Vocab::new(vocabulary(&shapes), Vec::new());
        Ok(Self { vocab, shapes })
    }

    /// One training sequence: a shape at a random width and place,
    /// with a sentence that says so.
    pub fn sample(&self, rng: &mut Rng) -> Vec<u32> {
        let shape = &self.shapes[rng.below(self.shapes.len())];
        let steps = ((WIDTH_MAX - WIDTH_MIN) / WIDTH_STEP) as usize + 1;
        let width = WIDTH_MIN + WIDTH_STEP * rng.below(steps) as i32;
        let w = f64::from(width);
        let h = shape.height * w / shape.width;
        let position = rng.below(POSITIONS.len());
        let advance = f64::from(snap(
            w + 2.0 * SIDEBEARING + f64::from(8 * rng.below(40) as i32),
        ));
        let (x, y) = place(position, w, h, advance);
        let ops = shape.placed(w, x, y);

        let template = TEMPLATES[rng.below(TEMPLATES.len())];
        let phrases = POSITIONS[position].1;
        let pos = phrases[rng.below(phrases.len())];
        let text = template
            .replace("{name}", &shape.words.join(" "))
            .replace("{w}", &width.to_string())
            .replace("{pos}", pos);
        let (mut toks, _) = encode_prompt(&self.vocab, &text, advance);
        toks.extend(self.vocab.encode_ops(&ops).into_iter().map(|t| t as u32));
        toks.push(self.vocab.special("EOS").unwrap_or(2) as u32);
        toks
    }
}

/// Next-token cross entropy over the drawing only. The sentence and
/// the advance are given at run time, so predicting them is wasted
/// capacity: every target up to and including the advance coordinate
/// is masked, along with padding.
fn drawing_loss(
    model: &OutlineModel,
    seqs: &[Vec<u32>],
    pad: u32,
    adv: u32,
    max_len: usize,
    device: &candle_core::Device,
    train: bool,
) -> Result<candle_core::Tensor> {
    use candle_core::{Tensor, D};
    let batch = pad_batch(seqs, pad, max_len, device)?;
    let (b, l) = batch.dims2()?;
    // Targets with the header blanked to pad, so the pad mask drops them.
    let width = l;
    let mut flat = Vec::with_capacity(b * width);
    for s in seqs {
        let n = s.len().min(width);
        let header_end = s.iter().position(|t| *t == adv).map_or(0, |i| i + 2);
        for (i, t) in s[..n].iter().enumerate() {
            flat.push(if i < header_end { pad } else { *t });
        }
        flat.extend(std::iter::repeat_n(pad, width - n));
    }
    let masked = Tensor::from_vec(flat, (b, width), device)?;
    let inputs = batch.narrow(1, 0, l - 1)?;
    let targets = masked.narrow(1, 1, l - 1)?;
    let logits = model.forward_batch(&inputs, train)?;
    let vocab = logits.dim(D::Minus1)?;
    let logits = logits.reshape((b * (l - 1), vocab))?;
    let targets = targets.reshape((b * (l - 1),))?;
    let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let picked = log_probs.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?;
    let mask = targets.ne(pad)?.to_dtype(DType::F32)?;
    let count = mask.sum_all()?;
    let loss = (picked.neg()? * &mask)?.sum_all()?;
    Ok((loss / count)?)
}

/// Trains a draw model on the shapes in `source` and writes it to `out`.
pub fn train_draw(
    source: &Path,
    out: &Path,
    cfg: &TrainConfig,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<Report> {
    let started = Instant::now();
    let corpus = Corpus::from_ufo(source)?;
    let vocab_size = corpus.vocab.len();
    let pad = corpus.vocab.special("PAD").unwrap_or(0) as u32;
    let adv = corpus.vocab.special("ADV").unwrap_or(8) as u32;

    let mut val_rng = Rng::new(cfg.seed.wrapping_add(2));
    let val: Vec<Vec<u32>> = (0..VAL_SEQUENCES)
        .map(|_| corpus.sample(&mut val_rng))
        .collect();
    let longest = {
        let mut probe = Rng::new(cfg.seed.wrapping_add(3));
        (0..256)
            .map(|_| corpus.sample(&mut probe).len())
            .max()
            .unwrap_or(64)
    };
    let max_len = if cfg.max_len > 0 {
        cfg.max_len
    } else {
        (longest + longest / 4).max(256)
    };
    let mut extra = serde_json::Map::new();
    extra.insert("task".into(), serde_json::Value::String("draw".into()));
    let model_cfg = ModelConfig {
        kind: ModelKind::Outline,
        dims: cfg.dims,
        layers: cfg.layers,
        heads: cfg.heads,
        vocab_size,
        max_len,
        delta_center: None,
        trim_close: false,
        extra,
    };

    let choice: crate::device::Choice = cfg.device.parse().map_err(|e: String| Error::Io {
        path: out.to_path_buf(),
        source: std::io::Error::other(e),
    })?;
    let device = crate::device::pick(choice)?;
    let device_name = crate::device::name(&device);
    eprintln!("device {device_name}");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = OutlineModel::build(
        vb,
        &model_cfg,
        corpus.vocab.clone(),
        device.clone(),
        cfg.dropout,
    )?;
    let params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();

    std::fs::create_dir_all(out).map_err(|source| Error::Io {
        path: out.to_path_buf(),
        source,
    })?;
    write_vocab(out, &corpus.vocab)?;
    write_json(&out.join("config.json"), &model_cfg)?;

    let mut opt = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: lr_at(1, cfg),
            ..Default::default()
        },
    )?;
    let mut rng = Rng::new(cfg.seed.wrapping_add(1));
    let deadline = (cfg.minutes > 0.0).then_some(cfg.minutes * 60.0);
    let mut best_val = f64::INFINITY;
    let mut init_loss = 0.0;
    let mut steps_done = 0;

    let val_loss = |model: &OutlineModel| -> Result<f64> {
        let mut total = 0.0;
        let mut count = 0;
        for chunk in val.chunks(cfg.batch.max(1)) {
            total += drawing_loss(model, chunk, pad, adv, max_len, &device, false)?
                .to_scalar::<f32>()? as f64;
            count += 1;
        }
        Ok(if count == 0 {
            0.0
        } else {
            total / count as f64
        })
    };
    let checkpoint = |model: &OutlineModel, tag: &str, best: &mut f64| -> Result<()> {
        let v = val_loss(model)?;
        let mut note = format!("checkpoint {tag}: val loss {v:.4}");
        if v < *best {
            *best = v;
            varmap.save(out.join("weights.safetensors"))?;
            note.push_str(" (best, saved)");
        }
        eprintln!("{note}");
        Ok(())
    };

    for step in 1..=cfg.steps {
        let seqs: Vec<Vec<u32>> = (0..cfg.batch).map(|_| corpus.sample(&mut rng)).collect();
        let loss = drawing_loss(&model, &seqs, pad, adv, max_len, &device, true)?;
        opt.set_learning_rate(lr_at(step, cfg));
        opt.backward_step(&loss)?;
        let value = loss.to_scalar::<f32>()? as f64;
        steps_done = step;
        if step == 1 {
            init_loss = value;
            eprintln!(
                "init loss {value:.3}, a random model gives {:.3}",
                (vocab_size as f64).ln()
            );
        }
        if step == 1 || step % 10 == 0 {
            progress(step, cfg.steps, &format!("loss {value:.4}"));
        }
        if step % cfg.ckpt_every == 0 {
            checkpoint(&model, &format!("step {step}"), &mut best_val)?;
        }
        if deadline.is_some_and(|limit| started.elapsed().as_secs_f64() > limit) {
            eprintln!("time budget reached at step {step}");
            break;
        }
    }
    checkpoint(&model, "final", &mut best_val)?;

    let report = Report {
        out: out.to_path_buf(),
        steps: steps_done,
        best_val,
        init_loss,
        params,
        seconds: started.elapsed().as_secs_f64(),
        vocab: vocab_size,
        pairs: 0,
        train_glyphs: corpus.shapes.len(),
        val_sequences: val.len(),
        center: [0, 0],
        device: device_name.to_string(),
    };
    let name = out
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();
    let shapes: Vec<String> = corpus.shapes.iter().map(|s| s.words.join(" ")).collect();
    write_json(
        &out.join("manifest.json"),
        &serde_json::json!({
            "name": name,
            "task": "draw",
            "kind": "model",
            "trained": today(),
            "source": source,
            "steps": steps_done,
            "best_val_loss": best_val,
            "params": params,
            "seconds": report.seconds,
            "device": device_name,
            "shapes": shapes,
            "notes": "Trained by font-ml train-draw. Validation loss picked the checkpoint.",
        }),
    )?;
    Ok(report)
}

/// What a draw call produced.
#[derive(Debug, Clone)]
pub struct Drawn {
    /// The outline, as far as it was well formed.
    pub ops: Vec<Op>,
    /// Words in the sentence the model has no token for.
    pub unknown: Vec<String>,
    /// Tokens the model produced, including the end token if it reached it.
    pub tokens: usize,
    /// Whether the model ended the drawing itself rather than hitting the limit.
    pub ended: bool,
}

/// Draws what `prompt` asks for, in a box `advance` wide.
pub fn draw(model: &OutlineModel, prompt: &str, advance: f64, max_len: usize) -> Result<Drawn> {
    let v = model.vocab();
    let eos = v.special("EOS").unwrap_or(2) as u32;
    let (mut ids, unknown) = encode_prompt(v, prompt, advance);
    let header = ids.len();
    let mut ended = false;
    while ids.len() < max_len {
        let next = model.next_token(&ids)?;
        ids.push(next);
        if next == eos {
            ended = true;
            break;
        }
    }
    let body: Vec<usize> = ids[header..].iter().map(|t| *t as usize).collect();
    Ok(Drawn {
        ops: v.decode_ops(&body),
        unknown,
        tokens: body.len(),
        ended,
    })
}

/// Drawing commands as UFO contours. A contour that returns to its
/// start keeps the start once, with the closing segment's type.
pub fn ops_to_contours(ops: &[Op]) -> Vec<norad::Contour> {
    use norad::{Contour, ContourPoint, PointType};
    let mut out = Vec::new();
    let mut points: Vec<ContourPoint> = Vec::new();
    let pt = |x: f64, y: f64, typ: PointType, smooth: bool| {
        ContourPoint::new(x, y, typ, smooth, None, None)
    };
    let mut finish = |points: &mut Vec<ContourPoint>| {
        if points.len() < 2 {
            points.clear();
            return;
        }
        let first = (points[0].x, points[0].y);
        let last = points.last().map(|p| (p.x, p.y, p.typ)).unwrap();
        if (last.0 - first.0).abs() < 0.5
            && (last.1 - first.1).abs() < 0.5
            && last.2 != PointType::OffCurve
        {
            points[0].typ = last.2;
            points[0].smooth = last.2 == PointType::Curve;
            points.pop();
        } else {
            points[0].typ = PointType::Line;
        }
        if points.len() >= 2 {
            out.push(Contour::new(std::mem::take(points), None));
        } else {
            points.clear();
        }
    };
    for op in ops {
        match op {
            Op::MoveTo(x, y) => {
                finish(&mut points);
                points.push(pt(*x, *y, PointType::Move, false));
            }
            Op::LineTo(x, y) => points.push(pt(*x, *y, PointType::Line, false)),
            Op::CurveTo(p) => {
                points.push(pt(p[0].0, p[0].1, PointType::OffCurve, false));
                points.push(pt(p[1].0, p[1].1, PointType::OffCurve, false));
                points.push(pt(p[2].0, p[2].1, PointType::Curve, true));
            }
            Op::ClosePath => finish(&mut points),
        }
    }
    finish(&mut points);
    out
}

/// The grid every coordinate lands on, for callers that report it.
pub const fn grid() -> i32 {
    GRID
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_lower_case_letters_and_digits() {
        assert_eq!(
            words("Draw a Circle, 64 units wide!"),
            vec!["draw", "a", "circle", "64", "units", "wide"]
        );
    }

    #[test]
    fn a_glyph_name_gives_its_shape_words() {
        assert_eq!(shape_words("circle.300"), vec!["circle"]);
        assert_eq!(shape_words("five-star"), vec!["five", "star"]);
    }

    #[test]
    fn a_square_round_trips_through_contours() {
        let ops = vec![
            Op::MoveTo(0.0, 0.0),
            Op::LineTo(10.0, 0.0),
            Op::LineTo(10.0, 10.0),
            Op::LineTo(0.0, 10.0),
            Op::LineTo(0.0, 0.0),
            Op::ClosePath,
        ];
        let contours = ops_to_contours(&ops);
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].points.len(), 4);
        assert_eq!(contours[0].points[0].typ, norad::PointType::Line);
    }

    #[test]
    fn the_vocabulary_holds_every_width_and_template_word() {
        let shape = Shape::new(
            "circle",
            &[
                Op::MoveTo(0.0, 0.0),
                Op::LineTo(10.0, 0.0),
                Op::LineTo(10.0, 10.0),
                Op::ClosePath,
            ],
        )
        .unwrap();
        let v = vocabulary(&[shape]);
        assert!(v.contains(&"circle".to_string()));
        assert!(v.contains(&"64".to_string()));
        assert!(v.contains(&"lower".to_string()));
        assert!(!v.contains(&"name".to_string()));
    }

    #[test]
    fn placement_covers_the_box() {
        assert_eq!(place(0, 100.0, 100.0, 400.0), (60.0, 0.0));
        assert_eq!(place(4, 100.0, 100.0, 400.0), (150.0, 300.0));
        assert_eq!(place(8, 100.0, 100.0, 400.0), (240.0, 600.0));
    }
}

#[cfg(test)]
mod sample_tests {
    use super::*;

    #[test]
    fn a_sample_reads_back_as_words_then_an_outline() {
        let square = Shape::new(
            "square",
            &[
                Op::MoveTo(0.0, 0.0),
                Op::LineTo(100.0, 0.0),
                Op::LineTo(100.0, 100.0),
                Op::LineTo(0.0, 100.0),
                Op::LineTo(0.0, 0.0),
                Op::ClosePath,
            ],
        )
        .unwrap();
        let shapes = vec![square];
        let vocab = Vocab::new(vocabulary(&shapes), Vec::new());
        let corpus = Corpus { vocab, shapes };
        let mut rng = Rng::new(5);
        let seq = corpus.sample(&mut rng);
        let text: Vec<&str> = seq
            .iter()
            .map(|t| corpus.vocab.token(*t as usize).unwrap_or("?"))
            .collect();
        eprintln!("{}", text.join(" "));
        assert_eq!(text[0], "BOS");
        assert!(text.contains(&"SEP"));
        assert!(text.contains(&"MOVE"));
        assert_eq!(*text.last().unwrap(), "EOS");
        // The outline is a square: four lines back to the start.
        let sep = text.iter().position(|t| *t == "SEP").unwrap();
        let body: Vec<usize> = seq[sep..].iter().map(|t| *t as usize).collect();
        let ops = corpus.vocab.decode_ops(&body);
        assert_eq!(ops.len(), 6, "{ops:?}");
    }
}
