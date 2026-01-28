use std::fs;
use std::path::Path;

use burn::data::dataset::Dataset;

/// A dataset that retrieves data from a collection of files in a folder.
pub struct FileFolderDataset {
    contents: Vec<Vec<char>>,
}

impl FileFolderDataset {
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
}

impl Dataset<Vec<char>> for FileFolderDataset {
    fn get(&self, index: usize) -> Option<Vec<char>> {
        self.contents.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.contents.len()
    }
}
