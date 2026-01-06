//! Dataset loading and preprocessing.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{DatasetConfig, DatasetFormat};
use crate::error::{AxolotlError, Result};

/// A single training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Input text (instruction/prompt).
    pub input: String,
    /// Target output.
    pub output: String,
    /// Full formatted text for training.
    pub text: String,
}

/// Dataset for training.
pub struct Dataset {
    /// Training examples.
    pub train: Vec<Example>,
    /// Validation examples.
    pub validation: Vec<Example>,
    /// Configuration used.
    pub config: DatasetConfig,
}

impl Dataset {
    /// Load dataset from configuration.
    pub fn load(config: &DatasetConfig) -> Result<Self> {
        let path = Path::new(&config.path);
        
        if !path.exists() {
            return Err(AxolotlError::Dataset(format!(
                "Dataset not found: {}",
                config.path
            )));
        }

        let examples = match config.format {
            DatasetFormat::Alpaca => load_alpaca(path, config)?,
            DatasetFormat::Sharegpt => load_sharegpt(path, config)?,
            DatasetFormat::Completion => load_completion(path, config)?,
            DatasetFormat::Custom => load_custom(path, config)?,
        };

        // Split into train/validation
        let split_idx = ((1.0 - config.val_split) * examples.len() as f32) as usize;
        let (train, validation) = examples.split_at(split_idx);

        Ok(Self {
            train: train.to_vec(),
            validation: validation.to_vec(),
            config: config.clone(),
        })
    }

    /// Get number of training examples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.train.len()
    }

    /// Check if dataset is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.train.is_empty()
    }
}

/// Alpaca format: {"instruction": "", "input": "", "output": ""}
#[derive(Deserialize)]
struct AlpacaExample {
    instruction: String,
    #[serde(default)]
    input: String,
    output: String,
}

fn load_alpaca(path: &Path, _config: &DatasetConfig) -> Result<Vec<Example>> {
    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        
        let alpaca: AlpacaExample = serde_json::from_str(line)
            .map_err(|e| AxolotlError::Dataset(format!("Failed to parse line: {}", e)))?;

        let input = if alpaca.input.is_empty() {
            alpaca.instruction.clone()
        } else {
            format!("{}\n\n{}", alpaca.instruction, alpaca.input)
        };

        let text = format!(
            "### Instruction:\n{}\n\n### Response:\n{}",
            input, alpaca.output
        );

        examples.push(Example {
            input,
            output: alpaca.output,
            text,
        });
    }

    Ok(examples)
}

/// ShareGPT format: {"conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]}
#[derive(Deserialize)]
struct ShareGptExample {
    conversations: Vec<ShareGptMessage>,
}

#[derive(Deserialize)]
struct ShareGptMessage {
    from: String,
    value: String,
}

fn load_sharegpt(path: &Path, _config: &DatasetConfig) -> Result<Vec<Example>> {
    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        
        let sharegpt: ShareGptExample = serde_json::from_str(line)
            .map_err(|e| AxolotlError::Dataset(format!("Failed to parse line: {}", e)))?;

        let mut text = String::new();
        let mut input = String::new();
        let mut output = String::new();

        for msg in &sharegpt.conversations {
            match msg.from.as_str() {
                "human" | "user" => {
                    text.push_str(&format!("### Human:\n{}\n\n", msg.value));
                    input = msg.value.clone();
                }
                "gpt" | "assistant" => {
                    text.push_str(&format!("### Assistant:\n{}\n\n", msg.value));
                    output = msg.value.clone();
                }
                _ => {}
            }
        }

        if !output.is_empty() {
            examples.push(Example { input, output, text });
        }
    }

    Ok(examples)
}

/// Completion format: {"text": ""}
fn load_completion(path: &Path, _config: &DatasetConfig) -> Result<Vec<Example>> {
    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        #[derive(Deserialize)]
        struct CompletionExample {
            text: String,
        }
        
        let completion: CompletionExample = serde_json::from_str(line)
            .map_err(|e| AxolotlError::Dataset(format!("Failed to parse line: {}", e)))?;

        examples.push(Example {
            input: String::new(),
            output: completion.text.clone(),
            text: completion.text,
        });
    }

    Ok(examples)
}

/// Custom format with configurable fields.
fn load_custom(path: &Path, config: &DatasetConfig) -> Result<Vec<Example>> {
    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        
        let obj: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| AxolotlError::Dataset(format!("Failed to parse line: {}", e)))?;

        let input = obj.get(&config.input_field)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let output = obj.get(&config.output_field)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let text = format!("### Input:\n{}\n\n### Output:\n{}", input, output);

        examples.push(Example { input, output, text });
    }

    Ok(examples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_alpaca() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"instruction": "Test", "input": "", "output": "Response"}}"#).unwrap();
        
        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            ..Default::default()
        };
        
        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        assert_eq!(dataset.train[0].output, "Response");
    }
}
