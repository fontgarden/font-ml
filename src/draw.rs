//! Draw a shape from a word into a box.
//!
//! A `draw` model is trained from scratch on one or more UFOs. Every
//! drawn glyph is a shape; its name is the word for it. The trainer
//! measures each shape, then places it in boxes of many sizes and
//! positions, so the model learns the word and the box together.
//!
//! A sequence is `BOS <words> SEP ADV <x> <y> <w> <h> <ops> EOS`: the
//! words, then the box the drawing fills, then the drawing. The words
//! are ordinary name tokens, so the token table needs nothing new; the
//! `ADV` token is borrowed to introduce the four box coordinates.
//!
//! A base is trained with spare word slots. A later run can continue
//! from it with new shapes: each new word takes a slot, the slot is
//! renamed in `vocab.txt`, and every other token keeps its place. That
//! is how a font adds its own shapes to the tool in minutes rather
//! than training from nothing.
//!
//! At run time the tool sends the word and the box. The model
//! continues greedily until `EOS`, and the drawing lands in the glyph.

use crate::checkpoint::{Checkpoint, ModelConfig, ModelKind};
use crate::error::{Error, Result};
use crate::outline::OutlineModel;
use crate::tokenizer::{snap, Op, Vocab, COORD_MAX, COORD_MIN};
use crate::train::{
    load_master, lr_at, pad_batch, today, write_json, write_vocab, Report, Rng, TrainConfig,
};
use candle_core::{DType, Tensor, D};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The narrowest box the trainer produces.
pub const WIDTH_MIN: i32 = 24;
/// The widest box the trainer produces.
pub const WIDTH_MAX: i32 = 720;
/// How many fixed sequences the validation loss is taken over.
pub const VAL_SEQUENCES: usize = 48;
/// Spare word slots a fresh base keeps for words learned later.
pub const DEFAULT_SPARE: usize = 16;
/// A slot's name starts with this. `words` strips it, so no typed
/// sentence can ever name a slot.
pub const SLOT_PREFIX: &str = "~";
/// Words the sentence templates use besides the shape words.
pub const TEMPLATE_WORDS: &[&str] = &["a", "an", "the", "draw", "make"];

/// The words of a sentence: lower case, letters and digits only.
pub fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// The word for a glyph: the name before any `.suffix`, with `-` and
/// `_` as spaces. `circle.big` and `circle` are both "circle";
/// `five-star` is "five star".
pub fn shape_words(glyph_name: &str) -> Vec<String> {
    let base = glyph_name.split('.').next().unwrap_or(glyph_name);
    words(&base.replace(['-', '_'], " "))
}

/// A box in font units: the left, the bottom, the width, the height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Box {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Box {
    /// Parses `x,y,w,h`.
    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        let parts: Vec<f64> = text
            .split(',')
            .map(|p| p.trim().parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| format!("box: {e}"))?;
        if parts.len() != 4 {
            return Err("box needs four numbers: x,y,w,h".into());
        }
        if parts[2] < 2.0 || parts[3] < 2.0 {
            return Err("box: width and height must be at least 2 units".into());
        }
        Ok(Self {
            x: parts[0],
            y: parts[1],
            w: parts[2],
            h: parts[3],
        })
    }
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
    pub fn new(glyph: &str, ops: &[Op]) -> Option<Self> {
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
        let words = shape_words(glyph);
        if words.is_empty() {
            return None;
        }
        Some(Self {
            glyph: glyph.to_string(),
            words,
            ops: map_ops(ops, |(x, y)| (x - x0, y - y0)),
            width,
            height,
        })
    }

    /// The shape fitted to a box, every point snapped to the grid.
    pub fn placed(&self, b: Box) -> Vec<Op> {
        let sx = b.w / self.width;
        let sy = b.h / self.height;
        map_ops(&self.ops, |(px, py)| {
            (
                f64::from(snap(px * sx + b.x)),
                f64::from(snap(py * sy + b.y)),
            )
        })
    }
}

/// Every word a fresh draw model can read: the template words, the
/// shape words, then `spare` empty slots.
pub fn vocabulary(shapes: &[Shape], spare: usize) -> Vec<String> {
    let mut names: Vec<String> = TEMPLATE_WORDS.iter().map(|w| (*w).to_string()).collect();
    let mut seen: BTreeSet<String> = names.iter().cloned().collect();
    let mut shape_words: Vec<String> = shapes.iter().flat_map(|s| s.words.clone()).collect();
    shape_words.sort();
    for w in shape_words {
        if seen.insert(w.clone()) {
            names.push(w);
        }
    }
    for i in 0..spare {
        names.push(format!("{SLOT_PREFIX}{i}"));
    }
    names
}

/// The words a model knows, without its spare slots.
pub fn known_words(names: &[String]) -> Vec<String> {
    names
        .iter()
        .filter(|n| !n.starts_with(SLOT_PREFIX) && !TEMPLATE_WORDS.contains(&n.as_str()))
        .cloned()
        .collect()
}

/// Gives every new word a spare slot of an existing table. Returns the
/// new table, or the words that found no slot.
pub fn assign_slots(
    mut names: Vec<String>,
    shapes: &[Shape],
) -> std::result::Result<Vec<String>, Vec<String>> {
    let mut wanted: Vec<String> = shapes.iter().flat_map(|s| s.words.clone()).collect();
    wanted.sort();
    wanted.dedup();
    let mut missing = Vec::new();
    for w in wanted {
        if names.contains(&w) {
            continue;
        }
        match names.iter().position(|n| n.starts_with(SLOT_PREFIX)) {
            Some(i) => names[i] = w,
            None => missing.push(w),
        }
    }
    if missing.is_empty() {
        Ok(names)
    } else {
        Err(missing)
    }
}

/// The header of a sequence: the words, then the box.
pub fn encode_prompt(v: &Vocab, text: &str, b: Box) -> (Vec<u32>, Vec<String>) {
    let unk = v.special("UNK").unwrap_or(14) as u32;
    let mut unknown = Vec::new();
    let mut toks = vec![v.special("BOS").unwrap_or(1) as u32];
    for w in words(text) {
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
    for c in [b.x, b.y, b.w, b.h] {
        toks.push(v.coord(c) as u32);
    }
    (toks, unknown)
}

/// The shapes of one or more UFOs, ready to sample.
pub struct Corpus {
    pub vocab: Vocab,
    pub shapes: Vec<Shape>,
}

/// Reads every drawn glyph of a UFO as a shape.
pub fn load_shapes(path: &Path) -> Result<Vec<Shape>> {
    let master = load_master(path)?;
    let mut shapes: Vec<Shape> = master
        .glyphs
        .iter()
        .filter_map(|(name, (_, ops))| Shape::new(name, ops))
        .collect();
    shapes.sort_by(|a, b| a.glyph.cmp(&b.glyph));
    Ok(shapes)
}

impl Corpus {
    /// One training sequence: a shape in a random box, with a short
    /// sentence that names it.
    pub fn sample(&self, rng: &mut Rng) -> Vec<u32> {
        let shape = &self.shapes[rng.below(self.shapes.len())];
        let steps = ((WIDTH_MAX - WIDTH_MIN) / 8) as usize + 1;
        let w = f64::from(WIDTH_MIN + 8 * rng.below(steps) as i32);
        // The drawn aspect, stretched a little either way, so a wide
        // box gives a wide shape rather than a surprise.
        let stretch = 0.8 + 0.45 * rng.f64();
        let h = (w * shape.height / shape.width * stretch).clamp(16.0, 980.0);
        let x = -100.0 + rng.f64() * (1100.0 - w);
        let y = -200.0 + rng.f64() * (1000.0 - h);
        let b = Box {
            x: f64::from(snap(x)),
            y: f64::from(snap(y)),
            w: f64::from(snap(w)),
            h: f64::from(snap(h)),
        };
        let ops = shape.placed(b);
        let name = shape.words.join(" ");
        let text = match rng.below(5) {
            0 => format!("a {name}"),
            1 => format!("draw a {name}"),
            _ => name,
        };
        let (mut toks, _) = encode_prompt(&self.vocab, &text, b);
        toks.extend(self.vocab.encode_ops(&ops).into_iter().map(|t| t as u32));
        toks.push(self.vocab.special("EOS").unwrap_or(2) as u32);
        toks
    }
}

/// Next-token cross entropy over the drawing only. The words and the
/// box are given at run time, so predicting them is wasted capacity:
/// every target up to the last box coordinate is masked, with padding.
fn drawing_loss(
    model: &OutlineModel,
    seqs: &[Vec<u32>],
    pad: u32,
    adv: u32,
    max_len: usize,
    device: &candle_core::Device,
    train: bool,
) -> Result<Tensor> {
    let batch = pad_batch(seqs, pad, max_len, device)?;
    let (b, l) = batch.dims2()?;
    let mut flat = Vec::with_capacity(b * l);
    for s in seqs {
        let n = s.len().min(l);
        let header_end = s.iter().position(|t| *t == adv).map_or(0, |i| i + 5);
        for (i, t) in s[..n].iter().enumerate() {
            flat.push(if i < header_end { pad } else { *t });
        }
        flat.extend(std::iter::repeat_n(pad, l - n));
    }
    let masked = Tensor::from_vec(flat, (b, l), device)?;
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

/// What a training run is told beyond the shared config.
#[derive(Debug, Clone, Default)]
pub struct DrawOptions {
    /// Continue from this model: keep its words, give new words its
    /// spare slots, start from its weights.
    pub init: Option<PathBuf>,
    /// Spare slots for a fresh base.
    pub spare: usize,
    /// Words to draw at every checkpoint, printed as `sample` lines.
    pub sample_words: Vec<String>,
}

/// The box sample drawings use: a square in the middle of a glyph.
pub const SAMPLE_BOX: Box = Box {
    x: 150.0,
    y: 100.0,
    w: 400.0,
    h: 400.0,
};

/// Drawing commands as SVG path data, for a sample line or a proof.
pub fn svg_path(ops: &[Op]) -> String {
    let mut d = String::new();
    for op in ops {
        match op {
            Op::MoveTo(x, y) => d.push_str(&format!("M{x} {y} ")),
            Op::LineTo(x, y) => d.push_str(&format!("L{x} {y} ")),
            Op::CurveTo(p) => d.push_str(&format!(
                "C{} {} {} {} {} {} ",
                p[0].0, p[0].1, p[1].0, p[1].1, p[2].0, p[2].1
            )),
            Op::ClosePath => d.push_str("Z "),
        }
    }
    d.trim_end().to_string()
}

/// Trains a draw model on the shapes in `sources` and writes it to
/// `out`. With `init`, the run continues from that model and new words
/// take its spare slots.
pub fn train_draw(
    sources: &[PathBuf],
    out: &Path,
    cfg: &TrainConfig,
    opts: &DrawOptions,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<Report> {
    let started = Instant::now();
    let mut shapes = Vec::new();
    for s in sources {
        shapes.extend(load_shapes(s)?);
    }
    if shapes.is_empty() {
        return Err(Error::Io {
            path: sources.first().cloned().unwrap_or_default(),
            source: std::io::Error::other("no drawn glyph to learn a shape from"),
        });
    }

    let init_ckpt = opts.init.as_deref().map(Checkpoint::open).transpose()?;
    let names = match &init_ckpt {
        Some(c) => assign_slots(c.glyph_names.clone(), &shapes).map_err(|missing| Error::Io {
            path: out.to_path_buf(),
            source: std::io::Error::other(format!(
                "the base has no spare slot left for: {}. Train a base with more --spare.",
                missing.join(", ")
            )),
        })?,
        None => vocabulary(&shapes, opts.spare),
    };
    let vocab = Vocab::new(names, Vec::new());
    let corpus = Corpus { vocab, shapes };
    let vocab_size = corpus.vocab.len();
    let pad = corpus.vocab.special("PAD").unwrap_or(0) as u32;
    let adv = corpus.vocab.special("ADV").unwrap_or(8) as u32;

    let mut val_rng = Rng::new(cfg.seed.wrapping_add(2));
    let val: Vec<Vec<u32>> = (0..VAL_SEQUENCES)
        .map(|_| corpus.sample(&mut val_rng))
        .collect();
    let max_len = match (&init_ckpt, cfg.max_len) {
        (Some(c), _) => c.config.max_len,
        (None, 0) => {
            let mut probe = Rng::new(cfg.seed.wrapping_add(3));
            let longest = (0..256)
                .map(|_| corpus.sample(&mut probe).len())
                .max()
                .unwrap_or(64);
            (longest + longest / 4).max(256)
        }
        (None, n) => n,
    };
    let mut extra = serde_json::Map::new();
    extra.insert("task".into(), serde_json::Value::String("draw".into()));
    extra.insert("format".into(), serde_json::Value::String("box".into()));
    let model_cfg = ModelConfig {
        kind: ModelKind::Outline,
        dims: init_ckpt.as_ref().map_or(cfg.dims, |c| c.config.dims),
        layers: init_ckpt.as_ref().map_or(cfg.layers, |c| c.config.layers),
        heads: init_ckpt.as_ref().map_or(cfg.heads, |c| c.config.heads),
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
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = OutlineModel::build(
        vb,
        &model_cfg,
        corpus.vocab.clone(),
        device.clone(),
        cfg.dropout,
    )?;
    if let Some(c) = &init_ckpt {
        varmap.load(c.weights_path())?;
    }
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
    let samples = |model: &OutlineModel, step: usize| -> Result<()> {
        for w in &opts.sample_words {
            let drawn = draw(model, w, SAMPLE_BOX, max_len)?;
            eprintln!("sample {step} {w} {}", svg_path(&drawn.ops));
        }
        Ok(())
    };
    let checkpoint = |model: &OutlineModel, tag: &str, step: usize, best: &mut f64| -> Result<()> {
        let v = val_loss(model)?;
        let mut note = format!("checkpoint {tag}: val loss {v:.4}");
        if v < *best {
            *best = v;
            varmap.save(out.join("weights.safetensors"))?;
            note.push_str(" (best, saved)");
        }
        eprintln!("{note}");
        samples(model, step)
    };

    if init_ckpt.is_some() {
        checkpoint(&model, "start", 0, &mut best_val)?;
    }
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
            checkpoint(&model, &format!("step {step}"), step, &mut best_val)?;
        }
        if deadline.is_some_and(|limit| started.elapsed().as_secs_f64() > limit) {
            eprintln!("time budget reached at step {step}");
            break;
        }
    }
    checkpoint(&model, "final", steps_done, &mut best_val)?;

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
    let known = known_words(corpus.vocab.glyph_names());
    let spare = corpus
        .vocab
        .glyph_names()
        .iter()
        .filter(|n| n.starts_with(SLOT_PREFIX))
        .count();
    write_json(
        &out.join("manifest.json"),
        &serde_json::json!({
            "name": name,
            "task": "draw",
            "format": "box",
            "kind": "model",
            "trained": today(),
            "sources": sources,
            "init": opts.init,
            "steps": steps_done,
            "best_val_loss": best_val,
            "params": params,
            "seconds": report.seconds,
            "device": device_name,
            "words": known,
            "spare": spare,
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

/// Draws what `prompt` names, filling `b`.
pub fn draw(model: &OutlineModel, prompt: &str, b: Box, max_len: usize) -> Result<Drawn> {
    let v = model.vocab();
    let eos = v.special("EOS").unwrap_or(2) as u32;
    let b = Box {
        x: b.x.clamp(f64::from(COORD_MIN), f64::from(COORD_MAX)),
        y: b.y.clamp(f64::from(COORD_MIN), f64::from(COORD_MAX)),
        w: b.w.clamp(2.0, f64::from(COORD_MAX)),
        h: b.h.clamp(2.0, f64::from(COORD_MAX)),
    };
    let (mut ids, unknown) = encode_prompt(v, prompt, b);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Shape {
        Shape::new(
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
        .unwrap()
    }

    #[test]
    fn words_are_lower_case_letters_and_digits() {
        assert_eq!(
            words("Draw a Circle, 64 units wide!"),
            vec!["draw", "a", "circle", "64", "units", "wide"]
        );
        // A slot's name keeps its prefix, which no typed word has.
        assert_eq!(words("~3"), vec!["3"]);
    }

    #[test]
    fn a_glyph_name_gives_its_shape_words() {
        assert_eq!(shape_words("circle.300"), vec!["circle"]);
        assert_eq!(shape_words("five-star"), vec!["five", "star"]);
    }

    #[test]
    fn a_square_round_trips_through_contours() {
        let ops = square().placed(Box {
            x: 10.0,
            y: 10.0,
            w: 50.0,
            h: 50.0,
        });
        let contours = ops_to_contours(&ops);
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].points.len(), 4);
        assert_eq!(contours[0].points[0].typ, norad::PointType::Line);
        assert_eq!(
            (contours[0].points[2].x, contours[0].points[2].y),
            (60.0, 60.0)
        );
    }

    #[test]
    fn a_fresh_vocabulary_has_templates_words_and_slots() {
        let v = vocabulary(&[square()], 4);
        assert_eq!(&v[..5], TEMPLATE_WORDS);
        assert!(v.contains(&"square".to_string()));
        assert_eq!(v.iter().filter(|n| n.starts_with(SLOT_PREFIX)).count(), 4);
        assert_eq!(known_words(&v), vec!["square"]);
    }

    #[test]
    fn a_new_word_takes_a_slot_and_nothing_else_moves() {
        let base = vocabulary(&[square()], 2);
        let heart = Shape::new("heart", &square().ops).unwrap();
        let next = assign_slots(base.clone(), &[square(), heart]).unwrap();
        assert_eq!(next.len(), base.len());
        let slot = base.iter().position(|n| n == "~0").unwrap();
        assert_eq!(next[slot], "heart");
        assert!(next.iter().zip(&base).filter(|(a, b)| a != b).count() == 1);
        let star = Shape::new("star", &square().ops).unwrap();
        let moon = Shape::new("moon", &square().ops).unwrap();
        let ring = Shape::new("ring", &square().ops).unwrap();
        assert_eq!(
            assign_slots(base, &[star, moon, ring]).unwrap_err(),
            vec!["star"]
        );
    }

    #[test]
    fn a_sample_is_words_then_box_then_outline() {
        let shapes = vec![square()];
        let vocab = Vocab::new(vocabulary(&shapes, 1), Vec::new());
        let corpus = Corpus { vocab, shapes };
        let mut rng = Rng::new(5);
        let seq = corpus.sample(&mut rng);
        let text: Vec<&str> = seq
            .iter()
            .map(|t| corpus.vocab.token(*t as usize).unwrap_or("?"))
            .collect();
        assert_eq!(text[0], "BOS");
        let sep = text.iter().position(|t| *t == "SEP").unwrap();
        assert_eq!(text[sep + 1], "ADV");
        assert!(text[sep + 2..sep + 6].iter().all(|t| t.starts_with('C')));
        assert_eq!(text[sep + 6], "MOVE");
        assert_eq!(*text.last().unwrap(), "EOS");
        let body: Vec<usize> = seq[sep..].iter().map(|t| *t as usize).collect();
        assert_eq!(corpus.vocab.decode_ops(&body).len(), 6);
    }

    #[test]
    fn a_box_parses_four_numbers() {
        let b = Box::parse("10, 20, 300, 400").unwrap();
        assert_eq!((b.x, b.y, b.w, b.h), (10.0, 20.0, 300.0, 400.0));
        assert!(Box::parse("1,2,3").is_err());
        assert!(Box::parse("1,2,0,4").is_err());
    }
}
