use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use burn::data::dataset::Dataset;
use flate2::read::GzDecoder;
use tar::Archive;

/// A dataset that retrieves data from a collection of files in a folder.
pub struct FileFolderDataset {
    contents: Vec<Vec<char>>,
}

impl FileFolderDataset {
    /// Loads a dataset from a directory by reading each regular file as UTF-8 text.
    pub fn load_from_folder(path: impl AsRef<Path>) -> Self {
        let mut contents = Vec::new();

        if let Ok(entries) = fs::read_dir(path.as_ref()) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_file() {
                    if let Ok(text) = fs::read_to_string(&entry_path) {
                        contents.push(text.chars().collect());
                    }
                }
            }
        }

        Self { contents }
    }

    /// Loads a dataset directly from a gzip-compressed tar archive.
    ///
    /// Each regular file entry is read as UTF-8 text and stored as a sequence of
    /// characters. Directory entries and non-UTF-8 files are skipped.
    pub fn load_from_tar_gz(path: impl AsRef<Path>) -> Self {
        let mut contents = Vec::new();

        let Ok(file) = File::open(path.as_ref()) else {
            return Self { contents };
        };

        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        let Ok(entries) = archive.entries() else {
            return Self { contents };
        };

        for mut entry in entries.flatten() {
            if !entry.header().entry_type().is_file() {
                continue;
            }

            let mut text = String::new();
            if entry.read_to_string(&mut text).is_ok() {
                contents.push(text.chars().collect());
            }
        }

        Self { contents }
    }
}

impl Dataset<Vec<char>> for FileFolderDataset {
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
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::data::dataset::Dataset;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use tar::{Builder, EntryType, Header};

    use super::FileFolderDataset;

    #[test]
    fn loads_text_files_from_tar_gz() {
        let archive_path = unique_archive_path();
        write_archive(
            &archive_path,
            &[
                ("train/first.txt", "hello"),
                ("train/second.txt", "world"),
                ("train/subdir/", ""),
            ],
        );

        let dataset = FileFolderDataset::load_from_tar_gz(&archive_path);
        let mut texts = (0..dataset.len())
            .map(|index| dataset.get(index).unwrap().into_iter().collect::<String>())
            .collect::<Vec<_>>();
        texts.sort();

        assert_eq!(texts, vec!["hello".to_string(), "world".to_string()]);

        let _ = fs::remove_file(archive_path);
    }

    fn unique_archive_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("lm-dataset-{nanos}.tar.gz"))
    }

    fn write_archive(path: &PathBuf, files: &[(&str, &str)]) {
        let file = File::create(path).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut builder = Builder::new(encoder);

        for (name, contents) in files {
            if name.ends_with('/') {
                let mut header = Header::new_gnu();
                header.set_path(name).unwrap();
                header.set_entry_type(EntryType::Directory);
                header.set_size(0);
                header.set_mode(0o755);
                header.set_cksum();
                builder.append(&header, std::io::empty()).unwrap();
                continue;
            }

            let bytes = contents.as_bytes();
            let mut header = Header::new_gnu();
            header.set_path(name).unwrap();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, bytes).unwrap();
        }

        builder.into_inner().unwrap().finish().unwrap();
    }
}
