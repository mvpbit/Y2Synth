//! Read TensorFlow.js Magenta MusicRNN weight shards as little-endian f32.
//! The crate includes native Rust LSTM+attention inference, autonomous drum generation,
//! and an optional synthesized WAV preview; Y2 hardware support is not implemented.

pub mod lstm;
pub mod attention;
pub mod generation;
pub mod verification;
pub mod audio;
pub mod kit;
pub mod dsp;
pub mod sequencer;
pub mod synth1;
pub mod project;
pub mod application;
pub mod ui;
#[cfg(feature = "live-audio")]
pub mod realtime;

use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl Tensor {
    pub fn element_count(&self) -> usize {
        self.values.len()
    }
}

#[derive(Debug)]
pub struct Checkpoint {
    pub attention_length: usize,
    pub tensors: BTreeMap<String, Tensor>,
}

impl Checkpoint {
    pub fn parameter_count(&self) -> usize {
        self.tensors.values().map(Tensor::element_count).sum()
    }

    pub fn weights_bytes(&self) -> usize {
        self.parameter_count() * std::mem::size_of::<f32>()
    }
}

#[derive(Deserialize)]
struct Config {
    #[serde(rename = "type")]
    model_type: String,
    #[serde(rename = "attentionLength")]
    attention_length: usize,
}

#[derive(Deserialize)]
struct ManifestGroup {
    paths: Vec<String>,
    weights: Vec<WeightSpec>,
}

#[derive(Deserialize)]
struct WeightSpec {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    #[serde(default)]
    quantization: Option<serde_json::Value>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> io::Result<T> {
    let file = File::open(path)?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|err| invalid(format!("{}: {err}", path.display())))
}

/// Treat manifest paths as filenames only; do not allow escaping the checkpoint directory.
fn valid_shard_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains('\\')
        && !name.contains(':')
        && Path::new(name).components().count() == 1
        && Path::new(name).file_name().is_some_and(|part| part.to_str() == Some(name))
}

fn weight_count(spec: &WeightSpec) -> io::Result<usize> {
    if spec.name.is_empty() {
        return Err(invalid("Empty tensor name"));
    }
    if spec.dtype != "float32" || spec.quantization.is_some() {
        return Err(invalid(format!(
            "{}: only unquantized float32 weights are supported (found {})",
            spec.name, spec.dtype
        )));
    }
    if spec.shape.iter().any(|&dim| dim == 0) {
        return Err(invalid(format!("{}: zero dimension", spec.name)));
    }
    spec.shape.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| invalid(format!("{}: shape overflows usize", spec.name)))
    })
}

/// Stream a single manifest group through shards, including tensors split across boundaries.
/// The stream does not concatenate all shard bytes in memory.
struct ShardStream {
    paths: Vec<PathBuf>,
    next: usize,
    current: Option<BufReader<File>>,
}

impl ShardStream {
    fn new(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            next: 0,
            current: None,
        }
    }

    fn read_exact_across_shards(&mut self, mut destination: &mut [u8]) -> io::Result<()> {
        while !destination.is_empty() {
            if self.current.is_none() {
                let path = self.paths.get(self.next).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "Shard data ended early")
                })?;
                self.current = Some(BufReader::new(File::open(path)?));
                self.next += 1;
            }
            match self.current.as_mut().expect("opened above").read(destination) {
                Ok(0) => self.current = None,
                Ok(read) => destination = &mut destination[read..],
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    fn read_f32_vec(&mut self, count: usize) -> io::Result<Vec<f32>> {
        // Decode small chunks, avoiding a second full-size buffer for the entire model.
        const FLOATS_PER_CHUNK: usize = 8192;
        let mut bytes = [0u8; FLOATS_PER_CHUNK * 4];
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            let elements = (count - values.len()).min(FLOATS_PER_CHUNK);
            let chunk = &mut bytes[..elements * 4];
            self.read_exact_across_shards(chunk)?;
            for value in chunk.chunks_exact(4) {
                let f = f32::from_le_bytes(value.try_into().expect("four bytes"));
                if !f.is_finite() {
                    return Err(invalid("Non-finite float32 value in checkpoint"));
                }
                values.push(f);
            }
        }
        Ok(values)
    }
}

/// Read the *already downloaded* Magenta.js converted checkpoint in `model_dir`.
///
/// Supports only unquantized float32 tensor groups. Tensor names and shapes are
/// taken from weights_manifest.json; bytes are stored in the manifest's order.
/// A group may contain multiple shards and a tensor may span shard boundaries.
///
/// Does not verify an upstream checksum or implement the inference graph.
pub fn load_checkpoint(model_dir: &Path) -> io::Result<Checkpoint> {
    let config: Config = read_json(&model_dir.join("config.json"))?;
    if config.model_type != "MusicRNN" {
        return Err(invalid(format!(
            "Expected config.type=MusicRNN, got {:?}",
            config.model_type
        )));
    }

    let groups: Vec<ManifestGroup> =
        read_json(&model_dir.join("weights_manifest.json"))?;
    if groups.is_empty() {
        return Err(invalid("Empty weights manifest"));
    }

    let mut tensors = BTreeMap::new();
    let mut visited_shards = HashSet::new();
    let mut visited_tensors = HashSet::new();

    for (group_index, group) in groups.iter().enumerate() {
        if group.paths.is_empty() || group.weights.is_empty() {
            return Err(invalid(format!(
                "Group {} needs both weights and shard paths",
                group_index + 1
            )));
        }

        let mut paths = Vec::with_capacity(group.paths.len());
        let mut actual_bytes = 0u64;
        for name in &group.paths {
            if !valid_shard_name(name) || !visited_shards.insert(name.clone()) {
                return Err(invalid(format!("Invalid or duplicated shard: {name:?}")));
            }
            let path = model_dir.join(name);
            // Validate shard lengths before allocating tensors.
            let len = fs::metadata(&path)?.len();
            actual_bytes = actual_bytes.checked_add(len).ok_or_else(|| {
                invalid(format!("Group {}: shard sizes overflow", group_index + 1))
            })?;
            paths.push(path);
        }

        let mut expected_bytes = 0u64;
        let mut counts = Vec::with_capacity(group.weights.len());
        for spec in &group.weights {
            if !visited_tensors.insert(spec.name.clone()) {
                return Err(invalid(format!("Duplicate tensor: {}", spec.name)));
            }
            let count = weight_count(spec)?;
            let bytes = count
                .checked_mul(4)
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| invalid(format!("{}: byte size overflows", spec.name)))?;
            expected_bytes = expected_bytes.checked_add(bytes).ok_or_else(|| {
                invalid(format!("Group {}: expected size overflows", group_index + 1))
            })?;
            counts.push(count);
        }

        if actual_bytes != expected_bytes {
            return Err(invalid(format!(
                "Group {} byte length mismatch: manifest expects {}, shards contain {}",
                group_index + 1,
                expected_bytes,
                actual_bytes
            )));
        }

        let mut stream = ShardStream::new(paths);
        for (spec, &count) in group.weights.iter().zip(&counts) {
            let values = stream.read_f32_vec(count)?;
            tensors.insert(
                spec.name.clone(),
                Tensor {
                    shape: spec.shape.clone(),
                    values,
                },
            );
        }
    }

    Ok(Checkpoint {
        attention_length: config.attention_length,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST_DIR: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> io::Result<Self> {
            let unique = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "y2synth-checkpoint-test-{}-{}", std::process::id(), unique
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_fixture(dir: &Path, shape: &str, shard1: &[u8], shard2: &[u8]) {
        fs::write(dir.join("config.json"), r#"{"type":"MusicRNN","attentionLength":32}"#)
            .unwrap();
        let manifest = format!(
            r#"[{{"paths":["part1","part2"],"weights":[{{"name":"test/kernel","shape":{shape},"dtype":"float32"}}]}}]"#
        );
        fs::write(dir.join("weights_manifest.json"), manifest).unwrap();
        fs::write(dir.join("part1"), shard1).unwrap();
        fs::write(dir.join("part2"), shard2).unwrap();
    }

    #[test]
    fn reads_tensor_across_shard_boundary_even_inside_a_float() {
        let folder = TempDir::new().unwrap();
        let mut bytes = Vec::new();
        for value in [1.5f32, -2.25, 0.125] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        write_fixture(&folder.0, "[3]", &bytes[..5], &bytes[5..]);
        let model = load_checkpoint(&folder.0).unwrap();
        assert_eq!(model.attention_length, 32);
        assert_eq!(model.parameter_count(), 3);
        assert_eq!(model.weights_bytes(), 12);
        assert_eq!(model.tensors["test/kernel"].shape, vec![3]);
        assert_eq!(model.tensors["test/kernel"].values, vec![1.5, -2.25, 0.125]);
    }

    #[test]
    fn keeps_manifest_tensor_order_with_multiple_tensors() {
        let folder = TempDir::new().unwrap();
        let mut bytes = Vec::new();
        for value in [10.0f32, -3.0, 7.5] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        write_fixture(&folder.0, "[3]", &bytes[..7], &bytes[7..]);
        fs::write(
            folder.0.join("weights_manifest.json"),
            r#"[{"paths":["part1","part2"],"weights":[
                {"name":"first","shape":[1],"dtype":"float32"},
                {"name":"second","shape":[2],"dtype":"float32"}
            ]}]"#,
        )
        .unwrap();
        let checkpoint = load_checkpoint(&folder.0).unwrap();
        assert_eq!(checkpoint.tensors["first"].values, vec![10.0]);
        assert_eq!(checkpoint.tensors["second"].values, vec![-3.0, 7.5]);
    }

    #[test]
    fn rejects_manifest_with_incorrect_byte_count() {
        let folder = TempDir::new().unwrap();
        write_fixture(&folder.0, "[4]", &[0; 5], &[0; 7]);
        let err = load_checkpoint(&folder.0).unwrap_err();
        assert!(err.to_string().contains("byte length mismatch"));
    }

    #[test]
    fn rejects_path_traversal() {
        let folder = TempDir::new().unwrap();
        assert!(!valid_shard_name("../secrets"));
        assert!(!valid_shard_name("nested/file"));
        assert!(!valid_shard_name(".."));
        assert!(valid_shard_name("group1-shard1of3"));
        write_fixture(&folder.0, "[3]", &[0; 5], &[0; 7]);
        let manifest = fs::read_to_string(folder.0.join("weights_manifest.json"))
            .unwrap()
            .replace("\"part1\"", "\"../secrets\"");
        fs::write(folder.0.join("weights_manifest.json"), manifest).unwrap();
        assert!(load_checkpoint(&folder.0).is_err());
    }

    #[test]
    fn rejects_nan_weights() {
        let folder = TempDir::new().unwrap();
        let mut bytes = Vec::new();
        for value in [1.0f32, f32::NAN, -1.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        write_fixture(&folder.0, "[3]", &bytes[..4], &bytes[4..]);
        assert!(load_checkpoint(&folder.0)
            .unwrap_err()
            .to_string()
            .contains("Non-finite"));
    }
}
