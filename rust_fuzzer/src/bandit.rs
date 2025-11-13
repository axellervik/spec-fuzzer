use crate::romu::RomuPrng;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BanditAlgorithm {
  RoundRobin,
  Exp3,
  Exp3IX,
}

impl BanditAlgorithm {
  pub fn from_env_or_default() -> Self {
    let algo = std::env::var("NYX_BANDIT_ALGO").unwrap_or_else(|_| "exp3ix".to_string());
    match algo.to_lowercase().as_str() {
      "roundrobin" => BanditAlgorithm::RoundRobin,
      "exp3" => BanditAlgorithm::Exp3,
      "exp3ix" => BanditAlgorithm::Exp3IX,
      _ => BanditAlgorithm::Exp3IX,
    }
  }
}

pub struct BanditScheduler {
  algorithm: BanditAlgorithm,
  weights: Vec<f64>,
  gamma: f64,          // exploration parameter
  last_choice: usize,
  last_probs: Vec<f64>,
  log_file: std::fs::File,
  rng: RomuPrng,
}

pub struct BanditUpdateStats {
  pub corpus_before: usize,
  pub corpus_after: usize,
  pub crashes_before: usize,
  pub crashes_after: usize,
}

impl BanditScheduler {
  pub fn new(workdir_path: &str, thread_id: usize, seed: u64) -> Self {
    let algorithm = BanditAlgorithm::from_env_or_default();
    let gamma = std::env::var("NYX_BANDIT_GAMMA").ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.1);

    let log_path = std::env::var("NYX_BANDIT_LOG")
      .unwrap_or_else(|| format!("{}/bandit_log{}.csv", workdir_path, thread_id));

    let mut log_file = OpenOptions::new().write(true).create(true).truncate(true)
      .open(&log_path).expect("Failed to create bandit log file");

    writeln!(log_file, "timestamp_ms,algo,k,chosen_idx,p_selected,reward,w_before,w_after,sum_w_after,corpus_before,corpus_after,crashes_before,crashes_after").unwrap();

    Self {
      algorithm,
      weights: Vec::new(),
      gamma,
      last_choice: 0,
      last_probs: Vec::new(),
      log_file,
      rng: RomuPrng::new_from_u64(seed),
    }
  }

  pub fn add_arm(&mut self) {
    self.weights.push(1.0);
  }

  pub fn k(&self) -> usize { self.weights.len() }

  fn eta(&self) -> f64 {
    // critical for stability
    if self.k() == 0 { 0.1 } else { self.gamma / self.k() as f64 }
  }

  pub fn choose(&mut self) -> usize {
    if self.k() == 0 { return 0; }

    let mut sum_w: f64 = self.weights.iter().sum();
    if !sum_w.is_finite() || sum_w <= 0.0 {
      self.weights.iter_mut().for_each(|w| *w = 1.0);
      sum_w = self.k() as f64;
    }

    match self.algorithm {
      BanditAlgorithm::RoundRobin => {
        self.last_probs = vec![1.0 / self.k() as f64; self.k()];
        self.last_choice = (self.last_choice + 1) % self.k();
        self.last_choice
      }
      BanditAlgorithm::Exp3 => {
        let k = self.k() as f64;
        self.last_probs = self.weights.iter().map(|w| {
          (1.0 - self.gamma) * (w / sum_w) + (self.gamma / k)
        }).collect();
        self.sample_from_probs()
      }
      BanditAlgorithm::Exp3IX => {
        self.last_probs = self.weights.iter().map(|w| w / sum_w).collect();
        self.sample_from_probs()
      }
    }
  }

  fn sample_from_probs(&mut self) -> usize {
    let r: f64 = (self.rng.next_u64() as f64) / (u64::MAX as f64);
    let mut acc = 0.0;
    for (i, p) in self.last_probs.iter().enumerate() {
      acc += *p;
      if r < acc { self.last_choice = i; return i; }
    }
    self.last_choice = self.k() - 1;
    self.last_choice
  }

  pub fn update(&mut self, reward: f64, stats: &BanditUpdateStats) {
    if self.algorithm == BanditAlgorithm::RoundRobin || self.k() == 0 { return; }
    let i = self.last_choice;
    if i >= self.k() { return; }

    let p = self.last_probs[i];
    let w_before = self.weights[i];

    let denom = match self.algorithm {
      BanditAlgorithm::Exp3 => p.max(1e-12),
      BanditAlgorithm::Exp3IX => (p + self.gamma).max(1e-12),
      _ => 1.0,
    };
    let xhat = reward / denom;

    self.weights[i] *= (self.eta() * xhat).exp();

    // Normalize weights to prevent blow-up and preserve distribution meaning
    let mut sum_w: f64 = self.weights.iter().sum();
    if !sum_w.is_finite() || sum_w <= 0.0 {
      self.weights.iter_mut().for_each(|w| *w = 1.0);
      sum_w = self.k() as f64;
    } else {
      for w in self.weights.iter_mut() { *w /= sum_w; }
      sum_w = 1.0; // after normalization
    }

    let w_after = self.weights[i];
    let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    writeln!(
      self.log_file,
      "{},{},{},{},{:.6},{:.3},{:.6},{:.6},{:.6},{},{},{},{},{}",
      now_ms,
      match self.algorithm { BanditAlgorithm::RoundRobin=>"roundrobin", BanditAlgorithm::Exp3=>"exp3", BanditAlgorithm::Exp3IX=>"exp3ix" },
      self.k(),
      i,
      p,
      reward,
      w_before,
      w_after,
      sum_w,
      stats.corpus_before,
      stats.corpus_after,
      stats.crashes_before,
      stats.crashes_after
    ).unwrap();
    let _ = self.log_file.flush();
  }
}