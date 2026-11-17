//! Exhaustive numerical comparison against Magenta.js DrumsRNN reference.
//!
//! The reference JSON comes from experiments/drumsrnn/reference.html with
//! "From scratch" + "Argmax" selected. Check exactly 16 × 512 values
//! (not "512 time steps"). Stochastic generations cannot be compared
//! stepwise unless both implementations consume identical chosen classes.

use crate::attention::{softmax, AttentionRnn};
use serde::Deserialize;
use std::io::{self, ErrorKind};

pub const STEPS: usize = 16;
pub const CLASSES: usize = 512;
pub const DEFAULT_TOLERANCE: f64 = 1e-5;

/// Shape of the actual `Save JSON` output from the pinned reference.html.
#[derive(Debug, Deserialize)]
pub struct MagentaReference {
    pub implementation: String,
    pub checkpoint: String,
    pub sampling: String,
    #[serde(rename = "generationMode")]
    pub generation_mode: String,
    #[serde(rename = "primerId")]
    pub primer_id: String,
    #[serde(rename = "noUserInput")]
    pub no_user_input: bool,
    #[serde(rename = "tfjsBackend", default)]
    pub tfjs_backend: Option<String>,
    #[serde(rename = "sampledClassIds")]
    pub classes: Vec<usize>,
    pub probabilities: Vec<Vec<f32>>,
    /// Missing in legacy (scratch Argmax) reference exports.
    #[serde(rename = "traceMode", default)]
    pub trace_mode: Option<String>,
    /// Forced previous outputs for the 15 transitions between 16 predictions.
    #[serde(rename = "forcedClassIds", default)]
    pub forced_classes: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceMode {
    ScratchArgmax,
    TeacherForced,
}

/// The computed Rust probability matrix and independent Argmax predictions.
/// These are DIAGNOSTICS only; they do not control normal music generation.
pub struct TeacherForcedTrace {
    pub probabilities: Vec<Vec<f32>>,
    pub predicted_classes: Vec<usize>,
}

/// Drive both model implementations with one silent bootstrap, then the SAME
/// 15 chosen previous classes, while observing 16 predicted distributions.
/// There is no user primer in normal generation; this is a numerical test.
pub fn infer_teacher_forced(
    network: &AttentionRnn<'_>,
    forced_classes: &[usize],
) -> io::Result<TeacherForcedTrace> {
    validate_forced_classes(forced_classes)?;

    let mut state = network.zero_state();
    network.step(0, 0, &mut state)?;
    let mut trace = TeacherForcedTrace {
        probabilities: Vec::with_capacity(STEPS),
        predicted_classes: Vec::with_capacity(STEPS),
    };

    for step in 0..STEPS {
        let probabilities = softmax(state.logits(), 1.0)?;
        trace.predicted_classes.push(argmax_first(state.logits()));
        trace.probabilities.push(probabilities);
        if step + 1 < STEPS {
            network.step(forced_classes[step], step + 1, &mut state)?;
        }
    }
    Ok(trace)
}

fn validate_forced_classes(classes: &[usize]) -> io::Result<()> {
    if classes.len() != STEPS - 1 {
        return Err(invalid(format!(
            "Teacher forcing requires {} previous class IDs, got {}",
            STEPS - 1, classes.len()
        )));
    }
    if classes.iter().any(|&class| class >= CLASSES) {
        return Err(invalid("Forced class ID must be in 0..512"));
    }
    if classes.iter().all(|&class| class == 0) {
        return Err(invalid("Teacher-forcing fixture must contain nonzero drum classes"));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct StepComparison {
    pub step: usize,
    pub max_abs: f64,
    pub worst_class: usize,
    pub mean_abs: f64,
    /// Total variation distance = half of L1 for normalized distributions.
    pub total_variation: f64,
    pub over_tolerance: usize,
}

#[derive(Debug)]
pub struct ComparisonReport {
    pub steps: Vec<StepComparison>,
    pub max_abs: f64,
    pub worst_step: usize,
    pub worst_class: usize,
    pub mean_abs: f64,
    pub rms_abs: f64,
    pub over_tolerance: usize,
    pub mismatched_classes: usize,
    pub tolerance: f64,
}

impl ComparisonReport {
    pub fn passed(&self) -> bool {
        self.over_tolerance == 0 && self.mismatched_classes == 0
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

fn check_probability_row(values: &[f32], label: &str, step: usize) -> io::Result<()> {
    if values.len() != CLASSES {
        return Err(invalid(format!(
            "{label} step {step}: expected {CLASSES} probabilities, found {}",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
        return Err(invalid(format!("{label} step {step}: non-finite or out-of-range probability")));
    }
    let sum: f64 = values.iter().map(|&value| f64::from(value)).sum();
    if (sum - 1.0).abs() > 1e-3 {
        return Err(invalid(format!(
            "{label} step {step}: probability sum {sum:.9} is not close to 1"
        )));
    }
    Ok(())
}

fn argmax_first(values: &[f32]) -> usize {
    let mut best_index = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best_index] {
            best_index = index;
        }
    }
    best_index
}

impl MagentaReference {
    pub fn mode(&self) -> io::Result<TraceMode> {
        match self.trace_mode.as_deref() {
            None | Some("scratch_argmax") => Ok(TraceMode::ScratchArgmax),
            Some("teacher_forced") => Ok(TraceMode::TeacherForced),
            Some(other) => Err(invalid(format!("Unknown reference traceMode: {other}"))),
        }
    }

    pub fn validate(&self) -> io::Result<()> {
        if self.implementation != "@magenta/music@1.23.1" {
            return Err(invalid("Expected @magenta/music@1.23.1; use the pinned reference.html"));
        }
        if self.checkpoint != "drum_kit_rnn_tfjs" {
            return Err(invalid("Wrong checkpoint: expected drum_kit_rnn_tfjs"));
        }
        if self.sampling != "argmax" {
            return Err(invalid("Use Argmax, not stochastic Temperature sampling"));
        }
        match self.mode()? {
            TraceMode::ScratchArgmax => {
                if self.generation_mode != "scratch"
                    || self.primer_id != "scratch"
                    || !self.no_user_input
                    || self.forced_classes.is_some()
                {
                    return Err(invalid("Use From scratch (one silent internal bootstrap step)"));
                }
            }
            TraceMode::TeacherForced => {
                if self.generation_mode != "teacher_forced"
                    || self.primer_id != "teacher_forced"
                    || self.no_user_input
                {
                    return Err(invalid("Expected diagnostic teacher_forced metadata"));
                }
                let inputs = self.forced_classes.as_deref().ok_or_else(|| {
                    invalid("Teacher-forced reference is missing forcedClassIds")
                })?;
                validate_forced_classes(inputs)?;
            }
        }
        if self.classes.len() != STEPS || self.probabilities.len() != STEPS {
            return Err(invalid(format!(
                "Expected {STEPS} steps, got {} class IDs and {} probability rows",
                self.classes.len(), self.probabilities.len()
            )));
        }
        for step in 0..STEPS {
            let selected = self.classes[step];
            if selected >= CLASSES {
                return Err(invalid(format!("Reference step {step}: invalid class {selected}")));
            }
            check_probability_row(&self.probabilities[step], "Magenta.js", step)?;
            let max_index = argmax_first(&self.probabilities[step]);
            if selected != max_index {
                return Err(invalid(format!(
                    "Reference step {step}: selected class {selected}, but argmax of distribution is {max_index}"
                )));
            }
        }
        Ok(())
    }
}

/// Compare each corresponding probability value; reject malformed or
/// incomparable inputs rather than silently truncating to a common prefix.
/// Returned report carries numerical failures; the caller prints it and
/// exits nonzero when `passed()` is false.
pub fn compare_probabilities(
    reference: &MagentaReference,
    rust_probabilities: &[Vec<f32>],
    rust_classes: &[usize],
    tolerance: f64,
) -> io::Result<ComparisonReport> {
    reference.validate()?;
    if !tolerance.is_finite() || tolerance <= 0.0 || tolerance > 0.1 {
        return Err(invalid("Tolerance must be finite, > 0 and <= 0.1"));
    }
    if rust_probabilities.len() != STEPS || rust_classes.len() != STEPS {
        return Err(invalid(format!(
            "Rust trace must have {STEPS} steps; got {} probability rows and {} class IDs",
            rust_probabilities.len(), rust_classes.len()
        )));
    }

    let mut steps = Vec::with_capacity(STEPS);
    let mut worst_step = 0;
    let mut worst_class = 0;
    let mut max_abs = 0.0_f64;
    let mut total_abs = 0.0_f64;
    let mut total_squared = 0.0_f64;
    let mut over_tolerance = 0usize;
    let mut mismatched_classes = 0usize;

    for step in 0..STEPS {
        let rust_row = &rust_probabilities[step];
        let ref_row = &reference.probabilities[step];
        check_probability_row(rust_row, "Rust", step)?;
        if rust_classes[step] >= CLASSES {
            return Err(invalid(format!("Rust step {step}: invalid class {}", rust_classes[step])));
        }
        if rust_classes[step] != argmax_first(rust_row) {
            return Err(invalid(format!("Rust step {step}: class is not argmax of its distribution")));
        }
        if rust_classes[step] != reference.classes[step] {
            mismatched_classes += 1;
        }

        let mut row_max = 0.0_f64;
        let mut row_worst_class = 0usize;
        let mut row_total = 0.0_f64;
        let mut row_over_tolerance = 0usize;
        for class in 0..CLASSES {
            let difference = (f64::from(rust_row[class]) - f64::from(ref_row[class])).abs();
            row_total += difference;
            total_squared += difference * difference;
            if difference > tolerance {
                row_over_tolerance += 1;
                over_tolerance += 1;
            }
            if difference > row_max {
                row_max = difference;
                row_worst_class = class;
            }
            if difference > max_abs {
                max_abs = difference;
                worst_step = step;
                worst_class = class;
            }
        }
        total_abs += row_total;
        steps.push(StepComparison {
            step,
            max_abs: row_max,
            worst_class: row_worst_class,
            mean_abs: row_total / CLASSES as f64,
            total_variation: row_total / 2.0,
            over_tolerance: row_over_tolerance,
        });
    }

    let count = (STEPS * CLASSES) as f64;
    Ok(ComparisonReport {
        steps,
        max_abs,
        worst_step,
        worst_class,
        mean_abs: total_abs / count,
        rms_abs: (total_squared / count).sqrt(),
        over_tolerance,
        mismatched_classes,
        tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> MagentaReference {
        let mut row = vec![0.0_f32; CLASSES];
        row[0] = 1.0;
        MagentaReference {
            implementation: "@magenta/music@1.23.1".into(),
            checkpoint: "drum_kit_rnn_tfjs".into(),
            sampling: "argmax".into(),
            generation_mode: "scratch".into(),
            primer_id: "scratch".into(),
            no_user_input: true,
            tfjs_backend: Some("cpu".into()),
            classes: vec![0; STEPS],
            probabilities: vec![row; STEPS],
            trace_mode: None,
            forced_classes: None,
        }
    }

    #[test]
    fn exact_trace_compares_all_8192_values() {
        let reference = reference();
        let report = compare_probabilities(
            &reference, &reference.probabilities, &reference.classes, DEFAULT_TOLERANCE,
        ).unwrap();
        assert!(report.passed());
        assert_eq!(report.steps.len(), STEPS);
        assert_eq!(report.max_abs, 0.0);
    }

    #[test]
    fn detects_difference_on_last_class_and_last_step() {
        let reference = reference();
        let mut rust = reference.probabilities.clone();
        rust[15][0] = 0.99;
        rust[15][511] = 0.01;
        let report = compare_probabilities(
            &reference, &rust, &reference.classes, DEFAULT_TOLERANCE,
        ).unwrap();
        assert!(!report.passed());
        assert_eq!(report.over_tolerance, 2);
        assert_eq!(report.worst_step, 15);
        assert!(report.max_abs > 0.009);
    }

    #[test]
    fn near_roundoff_difference_passes() {
        let reference = reference();
        let mut rust = reference.probabilities.clone();
        rust[7][0] = 1.0 - 1e-7;
        rust[7][511] = 1e-7;
        let report = compare_probabilities(
            &reference, &rust, &reference.classes, DEFAULT_TOLERANCE,
        ).unwrap();
        assert!(report.passed());
    }

    #[test]
    fn validates_teacher_forced_reference_and_checks_input_shape() {
        let mut fixture = reference();
        fixture.trace_mode = Some("teacher_forced".into());
        fixture.generation_mode = "teacher_forced".into();
        fixture.primer_id = "teacher_forced".into();
        fixture.no_user_input = false;
        fixture.forced_classes = Some(vec![5; STEPS - 1]);
        assert_eq!(fixture.mode().unwrap(), TraceMode::TeacherForced);
        fixture.validate().unwrap();
        let report = compare_probabilities(
            &fixture, &fixture.probabilities, &fixture.classes, DEFAULT_TOLERANCE,
        ).unwrap();
        assert!(report.passed());

        fixture.forced_classes = Some(vec![5; STEPS - 2]);
        assert!(fixture.validate().is_err());
        fixture.forced_classes = Some(vec![5; STEPS - 1]);
        fixture.forced_classes.as_mut().unwrap()[14] = 512;
        assert!(fixture.validate().is_err());
        fixture.forced_classes = Some(vec![0; STEPS - 1]);
        assert!(fixture.validate().is_err());
        fixture.forced_classes = None;
        assert!(fixture.validate().is_err());
        fixture.forced_classes = Some(vec![5; STEPS - 1]);
        fixture.no_user_input = true;
        assert!(fixture.validate().is_err());
    }

    #[test]
    fn legacy_scratch_references_still_work_and_reject_forced_ids() {
        let mut fixture = reference();
        fixture.validate().unwrap();
        fixture.forced_classes = Some(vec![5; STEPS - 1]);
        assert!(fixture.validate().is_err());
        fixture.forced_classes = None;
        fixture.trace_mode = Some("unknown".into());
        assert!(fixture.validate().is_err());
    }

    #[test]
    fn rejects_stochastic_reference_without_matching_trajectory() {
        let mut reference = reference();
        reference.sampling = "1.2".into();
        assert!(reference.validate().is_err());
        reference.sampling = "argmax".into();
        reference.primer_id = "regular".into();
        assert!(reference.validate().is_err());
    }

    #[test]
    fn refuses_malformed_shape_and_unnormalized_distribution() {
        let mut malformed = reference();
        malformed.probabilities[15].pop();
        assert!(malformed.validate().is_err());

        let mut unnormalized = reference();
        unnormalized.probabilities[2][0] = 0.5;
        assert!(unnormalized.validate().is_err());

        let mut wrong_length = reference();
        wrong_length.classes.truncate(15);
        assert!(wrong_length.validate().is_err());
    }

    #[test]
    fn rejects_inconsistent_rust_class_and_invalid_tolerance() {
        let reference = reference();
        let mut wrong_classes = reference.classes.clone();
        wrong_classes[2] = 511;
        assert!(compare_probabilities(
            &reference, &reference.probabilities, &wrong_classes, DEFAULT_TOLERANCE,
        ).is_err());
        assert!(compare_probabilities(
            &reference, &reference.probabilities, &reference.classes, f64::NAN,
        ).is_err());
    }
}
