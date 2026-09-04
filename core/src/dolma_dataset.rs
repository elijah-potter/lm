use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use burn::data::dataset::Dataset;
use flate2::read::GzDecoder;
use serde_json::Value;

/// A dataset that retrieves document text from a Dolma gzip-compressed JSONL shard.
///
/// Dolma shards are `*.json.gz` files containing one JSON object per line. This
/// loader extracts the `text` field from each row and stores it as characters so
/// it can be consumed by the existing character-level training batcher.
pub struct DolmaDataset {
    contents: Vec<Vec<char>>,
}

impl DolmaDataset {
    /// Returns true when a directory tree contains at least one Dolma shard.
    pub fn folder_contains_shards(path: impl AsRef<Path>) -> bool {
        !collect_json_gz_files(path.as_ref()).is_empty()
    }

    /// Loads every Dolma shard in a directory tree.
    ///
    /// Files are discovered recursively and sorted by path for deterministic
    /// loading. Any file that does not end in `.json.gz` is ignored.
    pub fn load_from_folder(path: impl AsRef<Path>) -> Self {
        let mut contents = Vec::new();

        for shard in collect_json_gz_files(path.as_ref()) {
            Self::append_json_gz(&shard, &mut contents);
        }

        Self { contents }
    }

    /// Loads a Dolma shard by reading each JSONL row's `text` field as UTF-8.
    ///
    /// Rows that cannot be parsed, or that do not contain a string `text` field,
    /// are skipped.
    pub fn load_from_json_gz(path: impl AsRef<Path>) -> Self {
        let mut contents = Vec::new();
        Self::append_json_gz(path.as_ref(), &mut contents);
        Self { contents }
    }

    fn append_json_gz(path: &Path, contents: &mut Vec<Vec<char>>) {
        let Ok(file) = File::open(path) else {
            return;
        };

        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);

        for line in reader.lines().map_while(Result::ok) {
            let Ok(row) = serde_json::from_str::<Value>(&line) else {
                continue;
            };

            let Some(text) = row.get("text").and_then(Value::as_str) else {
                continue;
            };

            contents.push(text.chars().collect());
        }
    }
}

fn collect_json_gz_files(path: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_json_gz_files_inner(path, &mut files);
    files.sort();
    files
}

fn collect_json_gz_files_inner(path: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            collect_json_gz_files_inner(&path, files);
            continue;
        }

        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".json.gz"))
        {
            files.push(path);
        }
    }
}

impl Dataset<Vec<char>> for DolmaDataset {
    fn get(&self, index: usize) -> Option<Vec<char>> {
        self.contents.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.contents.len()
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::data::dataset::Dataset;
    use flate2::Compression;
    use flate2::write::GzEncoder;

    use super::DolmaDataset;

    #[test]
    fn loads_text_from_dolma_json_gz() {
        let shard_path = unique_shard_path();
        write_shard(
            &shard_path,
            &[
                r#"{"id":"1","text":"hello world","source":"test"}"#,
                r#"{"id":"2","text":"second document"}"#,
                r#"{"id":"3","source":"missing text"}"#,
                "not json",
            ],
        );

        let dataset = DolmaDataset::load_from_json_gz(&shard_path);

        assert_eq!(dataset.len(), 2);
        assert_eq!(
            dataset.get(0).unwrap().into_iter().collect::<String>(),
            "hello world"
        );
        assert_eq!(
            dataset.get(1).unwrap().into_iter().collect::<String>(),
            "second document"
        );

        let _ = fs::remove_file(shard_path);
    }

    #[test]
    fn loads_text_from_dolma_folder_recursively() {
        let dir = unique_dir_path();
        let nested = dir.join("nested");
        fs::create_dir_all(&nested).unwrap();

        write_shard(&dir.join("a.json.gz"), &[r#"{"id":"1","text":"first"}"#]);
        write_shard(
            &nested.join("b.json.gz"),
            &[r#"{"id":"2","text":"second"}"#],
        );
        fs::write(dir.join("not-a-shard.txt"), "ignored").unwrap();

        assert!(DolmaDataset::folder_contains_shards(&dir));

        let dataset = DolmaDataset::load_from_folder(&dir);
        let texts = (0..dataset.len())
            .map(|index| dataset.get(index).unwrap().into_iter().collect::<String>())
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["first".to_string(), "second".to_string()]);

        let _ = fs::remove_dir_all(dir);
    }

    fn unique_shard_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("lm-dolma-{nanos}.json.gz"))
    }

    fn unique_dir_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("lm-dolma-dir-{nanos}"))
    }

    fn write_shard(path: &PathBuf, lines: &[&str]) {
        let file = File::create(path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());

        for line in lines {
            writeln!(encoder, "{line}").unwrap();
        }

        encoder.finish().unwrap();
    }
}
