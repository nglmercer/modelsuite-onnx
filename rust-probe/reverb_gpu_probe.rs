//! Feasibility probe for Reverb-TTS (OmniVoice-derived masked-diffusion TTS) on ORT's
//! WebGPU EP (Vulkan on Linux): runs the exported `embed_prefix`, `core`, `embed_target`,
//! `heads` and `codec_decoder` graphs on CPU and on the GPU with identical inputs and
//! compares outputs and latency.
//!
//! ```sh
//! REVERB_ONNX_DIR=/path/to/onnx_fp16 cargo run --release --example reverb_gpu_probe
//! ```
//! `REVERB_PROBE_FRAMES` (default 62) is the number of target frames,
//! `REVERB_PROBE_PASSES` (default 3) the timed step passes per EP.

use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::{Tensor, Value};

const LAYERS: usize = 24;
const KV_HEADS: usize = 6;
const HEAD_DIM: usize = 128;
const MASK: i64 = 1024;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> i64 {
        (self.next() % n) as i64
    }
}

fn session(dir: &std::path::Path, name: &str, gpu: bool) -> Result<Session> {
    onnx_init()?;
    let path = dir.join(format!("{name}.onnx"));
    let builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
        .with_intra_threads(6)
        .map_err(|e| anyhow::anyhow!("threads: {e}"))?;
    let mut builder = if gpu {
        sonicboom::onnx_runtime::with_gpu(builder, &path)
    } else {
        builder
    };
    builder
        .commit_from_file(&path)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))
}

fn onnx_init() -> Result<()> {
    sonicboom::onnx_runtime::initialize()
}

fn f32_out(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let (shape, data) = outputs[name].try_extract_tensor::<f32>()?;
    Ok((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
}

struct Graphs {
    embed_prefix: Session,
    core: Session,
    embed_target: Session,
    heads: Session,
    decoder: Session,
}

impl Graphs {
    fn load(dir: &std::path::Path, gpu: bool) -> Result<Self> {
        let started = Instant::now();
        let g = Self {
            embed_prefix: session(dir, "embed_prefix", gpu)?,
            core: session(dir, "core", gpu)?,
            embed_target: session(dir, "embed_target", gpu)?,
            heads: session(dir, "heads", gpu)?,
            decoder: session(dir, "codec_decoder", gpu)?,
        };
        println!(
            "[{}] load all graphs: {:.1}s",
            if gpu { "gpu" } else { "cpu" },
            started.elapsed().as_secs_f64()
        );
        Ok(g)
    }

    fn core_run(
        &mut self,
        x: (Vec<usize>, Vec<f32>),
        past: &(usize, Vec<f32>, Vec<f32>),
        bias: Vec<f32>,
        bias_shape: (usize, usize),
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (pp, pk, pv) = past;
        let shape = [LAYERS, 1, KV_HEADS, *pp, HEAD_DIM];
        let inputs: Vec<(String, Value)> = vec![
            ("x".into(), Tensor::from_array((x.0.clone(), x.1))?.into()),
            (
                "past_k".into(),
                Tensor::from_array((shape, pk.clone()))?.into(),
            ),
            (
                "past_v".into(),
                Tensor::from_array((shape, pv.clone()))?.into(),
            ),
            (
                "bias".into(),
                Tensor::from_array(([bias_shape.0, bias_shape.1], bias))?.into(),
            ),
        ];
        let outputs = self.core.run(inputs)?;
        Ok((
            f32_out(&outputs, "hidden")?.1,
            f32_out(&outputs, "new_k")?.1,
            f32_out(&outputs, "new_v")?.1,
        ))
    }

    /// Prefix pass then `passes` timed target passes. Returns the last log-probs and the codec audio.
    fn run(
        &mut self,
        text_ids: &[i64],
        ref_codes: &[i64],
        tokens: &[i64],
        frames: usize,
        passes: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f64>)> {
        let r = ref_codes.len() / 8;
        let t = Instant::now();
        let inputs: Vec<(String, Value)> = vec![
            (
                "text_ids".into(),
                Tensor::from_array(([1, text_ids.len()], text_ids.to_vec()))?.into(),
            ),
            (
                "ref_codes".into(),
                Tensor::from_array(([1, 8, r], ref_codes.to_vec()))?.into(),
            ),
        ];
        let outputs = self.embed_prefix.run(inputs)?;
        let x = f32_out(&outputs, "x")?;
        drop(outputs);
        let p = x.0[1];
        let empty = (0usize, Vec::new(), Vec::new());
        let (_, k, v) = self.core_run(x, &empty, vec![0.0; p * p], (p, p))?;
        println!("  prefix pass (P={p}): {:.2}s", t.elapsed().as_secs_f64());
        let past = (p, k, v);

        let mut times = Vec::new();
        let mut last = Vec::new();
        for i in 0..passes {
            let started = Instant::now();
            let tokens_in: Vec<(String, Value)> = vec![(
                "tokens".to_string(),
                Tensor::from_array(([1, 8, frames], tokens.to_vec()))?.into(),
            )];
            let outputs = self.embed_target.run(tokens_in)?;
            let x = f32_out(&outputs, "x")?;
            drop(outputs);
            let mut bias = vec![0.0f32; frames * (p + frames)];
            if i % 2 == 1 {
                // polish pass: bias the reference frames (the last R prefix keys)
                for row in 0..frames {
                    for col in (p - r)..p {
                        bias[row * (p + frames) + col] = 1.0;
                    }
                }
            }
            let (hidden, _, _) = self.core_run(x, &past, bias, (frames, p + frames))?;
            let heads_in: Vec<(String, Value)> = vec![(
                "hidden".to_string(),
                Tensor::from_array(([1, frames, 1024], hidden))?.into(),
            )];
            let outputs = self.heads.run(heads_in)?;
            last = f32_out(&outputs, "log_probs")?.1;
            times.push(started.elapsed().as_secs_f64());
        }

        let codes: Vec<i64> = tokens.iter().map(|&c| c.min(1023)).collect();
        let codes_in: Vec<(String, Value)> = vec![(
            "codes".to_string(),
            Tensor::from_array(([1, 8, frames], codes))?.into(),
        )];
        let outputs = self.decoder.run(codes_in)?;
        let audio = f32_out(&outputs, "audio")?.1;
        Ok((last, audio, times))
    }
}

fn compare(label: &str, a: &[f32], b: &[f32]) {
    let n = a.len().min(b.len());
    let (mut max, mut sum, mut bad) = (0.0f32, 0.0f64, 0usize);
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if !d.is_finite() {
            bad += 1;
            continue;
        }
        max = max.max(d);
        sum += d as f64;
    }
    println!(
        "  {label}: n={n} max_abs_diff={max:.5} mean_abs_diff={:.6} non_finite={bad}",
        sum / n as f64
    );
}

fn argmax_agreement(a: &[f32], b: &[f32], width: usize) -> f64 {
    let rows = a.len().min(b.len()) / width;
    let am = |s: &[f32]| {
        s.iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let same = (0..rows)
        .filter(|r| am(&a[r * width..(r + 1) * width]) == am(&b[r * width..(r + 1) * width]))
        .count();
    same as f64 / rows.max(1) as f64
}

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var("REVERB_ONNX_DIR").context("set REVERB_ONNX_DIR to the onnx_* folder")?,
    );
    let frames: usize = std::env::var("REVERB_PROBE_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(62);
    let passes: usize = std::env::var("REVERB_PROBE_PASSES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let text_ids: Vec<i64> = (0..53).map(|_| rng.below(1000)).collect();
    let ref_codes: Vec<i64> = (0..8 * 130).map(|_| rng.below(1024)).collect();
    let tokens: Vec<i64> = (0..8 * frames)
        .map(|_| {
            if rng.below(10) < 6 {
                MASK
            } else {
                rng.below(1024)
            }
        })
        .collect();

    let only_gpu = std::env::var("REVERB_PROBE_ONLY_GPU").is_ok();
    let mut cpu = None;
    if !only_gpu {
        println!("== CPU");
        let mut g = Graphs::load(&dir, false)?;
        let r = g.run(&text_ids, &ref_codes, &tokens, frames, passes)?;
        println!(
            "  step pass times (embed+core+heads): {:?}",
            r.2.iter().map(|t| format!("{t:.2}s")).collect::<Vec<_>>()
        );
        cpu = Some(r);
        drop(g);
    }
    println!("== GPU (WebGPU EP / Vulkan)");
    let mut g = Graphs::load(&dir, true)?;
    let gpu = g.run(&text_ids, &ref_codes, &tokens, frames, passes)?;
    println!(
        "  step pass times (embed+core+heads): {:?}",
        gpu.2.iter().map(|t| format!("{t:.2}s")).collect::<Vec<_>>()
    );
    let rms_of = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32).sqrt();
    println!(
        "  codec audio rms gpu={:.5} (healthy CPU decoder ~0.017 on this input)",
        rms_of(&gpu.1)
    );
    if let Some(cpu) = cpu {
        println!("== CPU vs GPU");
        compare("log_probs", &cpu.0, &gpu.0);
        println!(
            "  log_probs argmax agreement: {:.2}%",
            100.0 * argmax_agreement(&cpu.0, &gpu.0, 1024)
        );
        compare("codec audio", &cpu.1, &gpu.1);
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32).sqrt();
        println!(
            "  codec audio rms cpu={:.5} gpu={:.5}",
            rms(&cpu.1),
            rms(&gpu.1)
        );
    }
    Ok(())
}
